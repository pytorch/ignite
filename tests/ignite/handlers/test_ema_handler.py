import os
from typing import Any, Callable, Union

import pytest
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset

import ignite.distributed as idist
from ignite.engine import Engine, Events
from ignite.handlers import EMAHandler


def _get_dummy_dataloader() -> DataLoader:
    """Get a dummy data loader with length of 2"""
    x = torch.randint(0, 10, (2, 2))
    data_set = TensorDataset(x)
    data_loader = DataLoader(data_set, batch_size=1)
    return data_loader


def _get_dummy_model() -> nn.Module:
    model = nn.Linear(2, 1, bias=False)
    model.weight.data.fill_(1)
    return model


def _get_dummy_step_fn(model: Union[nn.Module, DataParallel, DistributedDataParallel]) -> Callable:
    """Get a dummy step function, given model is a (wrapper of) dummy model returned from _get_dummy_model"""

    def step_fn(engine, batch):
        """Increment the weight by 1 at each iteration"""
        model.weight.data.add_(1)
        return 0

    return step_fn


@pytest.mark.parametrize("momentum_end", [-1, 2])
def test_ema_invalid_momentum_end(momentum_end):
    with pytest.raises(ValueError):
        model = _get_dummy_model()
        EMAHandler(model, momentum_end=momentum_end)


@pytest.mark.parametrize("momentum_start", [-1, 2])
def test_ema_invalid_momentum_start(momentum_start):
    with pytest.raises(ValueError):
        model = _get_dummy_model()
        EMAHandler(model, momentum_start=momentum_start)


def test_ema_invalid_momentum_start_end():
    """Test momentum_end > momentum_start"""
    momentum_end = 0.001
    momentum_start = 0.1
    with pytest.raises(ValueError):
        model = _get_dummy_model()
        EMAHandler(model, momentum_start=momentum_start, momentum_end=momentum_end)


def test_ema_invalid_model():
    with pytest.raises(ValueError):
        model = "Invalid Model"
        EMAHandler(model)  # type: ignore


@pytest.mark.distributed
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_ema_ema_model():
    """Test if ema_handler.ema_model is nn.Module and under eval mode"""
    model = _get_dummy_model().to(idist.device())
    model = idist.auto_model(model)
    ema_handler = EMAHandler(model)
    ema_model = ema_handler.ema_model
    assert (
        isinstance(ema_model, nn.Module)
        and not isinstance(ema_model, nn.parallel.DistributedDataParallel)
        and not isinstance(ema_model, nn.parallel.DataParallel)
    )
    assert not ema_model.training


def test_ema_load_state_dict():
    model_1 = _get_dummy_model()
    model_1.weight.data.fill_(2)
    state_dict_1 = model_1.state_dict()

    model_2 = _get_dummy_model()
    ema_handler = EMAHandler(model_2)
    ema_model = ema_handler.ema_model
    ema_model.load_state_dict(state_dict_1)
    torch.testing.assert_allclose(ema_model.weight.data, model_1.weight.data)


def test_ema_update_ema_momentum():
    data_loader = _get_dummy_dataloader()  # noqa
    model = _get_dummy_model()
    step_fn = _get_dummy_step_fn(model)
    engine = Engine(step_fn)

    warmup_iters = 4
    momentum_start = 0.1
    momentum_end = 0.2
    ema_handler = EMAHandler(model, momentum_start=momentum_start, momentum_end=momentum_end, warmup_iters=warmup_iters)
    ema_handler.attach(engine, "model")

    # add handlers to check momentum at each iteration
    @engine.on(Events.ITERATION_COMPLETED)
    def assert_momentum(engine: Engine):
        curr_iter = engine.state.iteration
        curr_momentum = engine.state.ema_momentum["model"]
        if curr_iter == 1:
            assert curr_momentum == momentum_start
        elif 1 < curr_iter < warmup_iters:
            assert momentum_start < curr_momentum < momentum_end
        else:
            assert curr_momentum == momentum_end

    engine.run(data_loader, max_epochs=5)


def test_ema_buffer():
    """Test if the tensors in buffer are also synchronized"""
    model = nn.BatchNorm2d(2)
    model.running_mean.data.fill_(1.5)
    model.running_var.data.fill_(1.5)
    ema_handler = EMAHandler(model)

    def _bn_step_fn(engine, batch):
        return 1

    engine = Engine(_bn_step_fn)
    # engine will run 4 iterations
    engine.run(_get_dummy_dataloader(), max_epochs=2)

    ema_model = ema_handler.ema_model
    torch.testing.assert_allclose(ema_model.running_mean, model.running_mean)
    torch.testing.assert_allclose(ema_model.running_var, model.running_var)


def test_ema_two_handlers():
    """Test when two EMA handlers are attached to a trainer"""
    data_loader = _get_dummy_dataloader()
    model_1 = _get_dummy_model()
    # momentum will be constantly 0.5
    ema_handler_1 = EMAHandler(model_1, momentum_start=0.5, momentum_end=0.5, warmup_iters=1)

    model_2 = _get_dummy_model()
    ema_handler_2 = EMAHandler(model_2, momentum_start=0.5, momentum_end=0.5, warmup_iters=1)

    def _step_fn(engine: Engine, batch: Any):
        model_1.weight.data.add_(1)
        model_2.weight.data.add_(1)
        return 0

    engine = Engine(_step_fn)
    assert not hasattr(engine.state, "ema_momentum")
    # handler_1 update EMA model of model_1 every 1 iteration
    ema_handler_1.attach(engine, "model_1", event=Events.ITERATION_COMPLETED)
    assert isinstance(engine.state.ema_momentum, dict) and len(engine.state.ema_momentum) == 1
    assert engine.state.ema_momentum["model_1"] is None

    with pytest.raises(ValueError, match="Key: 'model_1' is already in Engine.state.ema_momentum."):
        ema_handler_2.attach(engine, "model_1")
    # handler_2 update EMA model for model_2 every 2 iterations
    ema_handler_2.attach(engine, "model_2", event=Events.ITERATION_COMPLETED(every=2))
    assert engine.state.ema_momentum["model_2"] is None

    # engine will run 4 iterations
    engine.run(data_loader, max_epochs=2)
    ema_weight_1 = ema_handler_1.ema_model.weight.data
    ema_weight_2 = ema_handler_2.ema_model.weight.data
    torch.testing.assert_allclose(ema_weight_1, torch.full((1, 2), 4.0625))
    torch.testing.assert_allclose(ema_weight_2, torch.full((1, 2), 3.5))

    assert engine.state.ema_momentum["model_1"] == 0.5
    assert engine.state.ema_momentum["model_2"] == 0.5


def _test_ema_final_weight(device, ddp=False, interval=1):
    """Test if final smoothed weights are correct"""
    if isinstance(device, str):
        device = torch.device(device)
    data_loader = _get_dummy_dataloader()
    model = _get_dummy_model().to(device)
    if ddp:
        model = idist.auto_model(model)
    step_fn = _get_dummy_step_fn(model)
    engine = Engine(step_fn)

    # momentum will be constantly 0.5
    ema_handler = EMAHandler(model, momentum_start=0.5, momentum_end=0.5, warmup_iters=1)
    ema_handler.attach(engine, "model", event=Events.ITERATION_COMPLETED(every=interval))

    # engine will run 4 iterations
    engine.run(data_loader, max_epochs=2)

    ema_weight = ema_handler.ema_model.weight.data
    model_weight = model.weight.data
    assert ema_weight.device == device
    assert model_weight.device == device
    if interval == 1:
        torch.testing.assert_allclose(ema_weight, torch.full((1, 2), 4.0625, device=device))
    elif interval == 2:
        torch.testing.assert_allclose(ema_weight, torch.full((1, 2), 3.5, device=device))
    else:
        pass
    torch.testing.assert_allclose(model_weight, torch.full((1, 2), 5.0, device=device))


@pytest.mark.parametrize("interval", [1, 2])
def test_ema_final_weight_cpu(interval):
    device = torch.device("cpu")
    _test_ema_final_weight(device=device, ddp=False, interval=interval)


@pytest.mark.parametrize("interval", [1, 2])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_ema_final_weight_cuda(interval):
    device = torch.device("cuda:0")
    _test_ema_final_weight(device=device, ddp=False, interval=interval)


@pytest.mark.distributed
@pytest.mark.parametrize("interval", [1, 2])
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_ema_final_weight_nccl_cuda(distributed_context_single_node_nccl, interval):
    device = idist.device()
    _test_ema_final_weight(device=device, ddp=True, interval=interval)


@pytest.mark.distributed
@pytest.mark.parametrize("interval", [1, 2])
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_ema_final_weight_gloo_cuda(distributed_context_single_node_gloo, interval):
    device = idist.device()
    _test_ema_final_weight(device=device, ddp=True, interval=interval)


@pytest.mark.distributed
@pytest.mark.parametrize("interval", [1, 2])
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_ema_final_weight_hvd_cuda(gloo_hvd_executor, interval):
    device = torch.device("cuda")
    nproc = torch.cuda.device_count()

    gloo_hvd_executor(_test_ema_final_weight, (device, True, interval), np=nproc, do_init=True)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_ema_final_weight_single_device_xla():
    device = idist.device()
    _test_ema_final_weight(device=device, ddp=True)


def _test_ema_final_weight_xla_nprocs(index):
    device = idist.device()
    _test_ema_final_weight(device=device, ddp=True)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_ema_final_weight_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_ema_final_weight_xla_nprocs, args=(), nprocs=n)


@pytest.mark.multinode_distributed
@pytest.mark.parametrize("interval", [1, 2])
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_ema_final_weight_multinode_gloo_cuda(distributed_context_multi_node_gloo, interval):
    device = idist.device()
    _test_ema_final_weight(device=device, ddp=True, interval=interval)


@pytest.mark.multinode_distributed
@pytest.mark.parametrize("interval", [1, 2])
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_ema_final_weight_multinode_nccl_cuda(distributed_context_multi_node_nccl, interval):
    device = idist.device()
    _test_ema_final_weight(device=device, ddp=True, interval=interval)
