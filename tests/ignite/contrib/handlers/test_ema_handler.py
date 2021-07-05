import os
from typing import Callable, Union

import pytest
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset

import ignite.distributed as idist
from ignite.contrib.handlers import EMAHandler
from ignite.engine import Engine, Events


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
        return model.weight.data.mean().cpu().item()

    return step_fn


@pytest.mark.parametrize("momentum", [-1, 2])
def test_ema_invalid_momentum(momentum):
    with pytest.raises(ValueError):
        model = _get_dummy_model()
        EMAHandler(model, momentum=momentum)


@pytest.mark.parametrize("momentum_warmup", [-1, 2])
def test_ema_invalid_momentum_warmup(momentum_warmup):
    with pytest.raises(ValueError):
        model = _get_dummy_model()
        EMAHandler(model, momentum_warmup=momentum_warmup)


def test_ema_smaller_momentum():
    """Test momentum_warmup > momentum"""
    momentum = 0.001
    momentum_warmup = 0.1
    with pytest.raises(ValueError):
        model = _get_dummy_model()
        EMAHandler(model, momentum=momentum, momentum_warmup=momentum_warmup)


@pytest.mark.parametrize("interval", [-1, 1.05])
def test_ema_invalid_interval(interval):
    with pytest.raises(ValueError):
        model = _get_dummy_model()
        EMAHandler(model, interval=interval)


def test_ema_invalid_model():
    with pytest.raises(ValueError):
        model = "Invalid Model"
        EMAHandler(model)  # type: ignore


def test_ema_load_state_dict():
    model_1 = _get_dummy_model()
    model_1.weight.data.fill_(2)
    state_dict_1 = model_1.state_dict()

    model_2 = _get_dummy_model()
    ema_handler = EMAHandler(model_2)
    ema_handler.load_state_dict(state_dict_1)  # type: ignore
    torch.testing.assert_allclose(ema_handler.ema.weight.data, model_1.weight.data)

    state_dict_2 = ema_handler.state_dict()
    assert "weight" in state_dict_2
    torch.testing.assert_allclose(state_dict_2["weight"], model_1.weight.data)


def test_ema_get_momentum():
    data_loader = _get_dummy_dataloader()  # noqa
    model = _get_dummy_model()
    step_fn = _get_dummy_step_fn(model)
    engine = Engine(step_fn)

    warmup_iters = 4
    momentum = 0.2
    momentum_warmup = 0.1
    ema_handler = EMAHandler(model, momentum=momentum, momentum_warmup=momentum_warmup, warmup_iters=warmup_iters)
    ema_handler.attach(engine)

    # add handlers to check momentum at each iteration
    @engine.on(Events.ITERATION_COMPLETED)
    def assert_momentum(engine: Engine):
        curr_iter = engine.state.iteration
        curr_momentum = ema_handler._get_momentum(curr_iter)
        if curr_iter == 1:
            assert curr_momentum == momentum_warmup
        elif 1 < curr_iter < warmup_iters:
            assert momentum_warmup < curr_momentum < momentum
        else:
            assert curr_momentum == momentum

    engine.run(data_loader, max_epochs=5)


def _test_ema_final_weight(device, ddp=False):
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
    ema_handler = EMAHandler(model, momentum=0.5, momentum_warmup=0.5, warmup_iters=1, device=device)
    ema_handler.attach(engine)

    # engine run 4 iterations
    engine.run(data_loader, max_epochs=2)

    ema_handler_weight = ema_handler.ema.weight.data
    model_weight = model.weight.data
    assert ema_handler_weight.device == device
    assert model_weight.device == device
    torch.testing.assert_allclose(ema_handler_weight, torch.full((1, 2), 4.0625, device=device))
    torch.testing.assert_allclose(model_weight, torch.full((1, 2), 5.0, device=device))


def test_ema_final_weight_cpu():
    device = torch.device("cpu")
    _test_ema_final_weight(device, ddp=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_ema_final_weight_cuda():
    device = torch.device("cuda:0")
    _test_ema_final_weight(device, ddp=False)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_ema_final_weight_nccl_cuda(distributed_context_single_node_nccl):
    device = idist.device()
    _test_ema_final_weight(device, ddp=True)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_ema_final_weight_gloo_cuda(distributed_context_single_node_gloo):
    device = idist.device()
    _test_ema_final_weight(device, ddp=True)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_ema_final_weight_hvd_cuda(gloo_hvd_executor):
    device = torch.device("cuda")
    nproc = torch.cuda.device_count()

    gloo_hvd_executor(_test_ema_final_weight, (device, True), np=nproc, do_init=True)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_ema_final_weight_single_device_xla():
    device = idist.device()
    _test_ema_final_weight(device, ddp=True)


def _test_ema_final_weight_xla_nprocs(index):
    device = idist.device()
    _test_ema_final_weight(device, ddp=True)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_ema_final_weight_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_ema_final_weight_xla_nprocs, args=(), nprocs=n)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_ema_final_weight_multinode_gloo_cuda(distributed_context_multi_node_gloo):
    device = idist.device()
    _test_ema_final_weight(device, ddp=True)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_ema_final_weight_multinode_nccl_cuda(distributed_context_multi_node_nccl):
    device = idist.device()
    _test_ema_final_weight(device, ddp=True)
