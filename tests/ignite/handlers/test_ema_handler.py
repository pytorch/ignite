import os
from typing import Any, Callable, Union

import pytest
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

import ignite.distributed as idist
from ignite.engine import Engine, Events
from ignite.handlers import EMAHandler


def _get_dummy_model() -> nn.Module:
    model = nn.Linear(2, 1, bias=False)
    model.weight.data.fill_(1)
    return model


def _unwrap_model(model):
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        return model.module
    else:
        return model


@pytest.fixture(scope="module")
def get_dummy_model():
    """Returns a function since the fixture is needed multiple times in a single test"""
    yield _get_dummy_model


def _get_dummy_step_fn(model: Union[nn.Module, DataParallel, DistributedDataParallel]) -> Callable:
    """Get a dummy step function, given model is a (wrapper of) dummy model returned from _get_dummy_model"""

    def step_fn(engine, batch):
        """Increment the weight by 1 at each iteration"""
        _unwrap_model(model).weight.data.add_(1)
        return 0

    return step_fn


@pytest.mark.parametrize("momentum", [-1, 2])
def test_ema_invalid_momentum(get_dummy_model, momentum):
    with pytest.raises(ValueError, match="Invalid momentum"):
        EMAHandler(get_dummy_model(), momentum=momentum)


@pytest.mark.parametrize("momentum_warmup", [-1, 2])
def test_ema_invalid_momentum_warmup(get_dummy_model, momentum_warmup):
    with pytest.raises(ValueError, match="Invalid momentum_warmup"):
        EMAHandler(get_dummy_model, momentum_warmup=momentum_warmup)


def test_ema_invalid_momentum_start_end(get_dummy_model):
    """Test momentum_end > momentum_start"""
    momentum = 0.001
    momentum_warmup = 0.1
    with pytest.raises(ValueError, match="momentum_warmup should be less than or equal to momentum"):
        EMAHandler(get_dummy_model(), momentum_warmup=momentum_warmup, momentum=momentum)


def test_ema_invalid_model():
    with pytest.raises(ValueError, match="model should be an instance of nn.Module or its subclasses"):
        model = "Invalid Model"
        EMAHandler(model)  # type: ignore


@pytest.mark.distributed
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_ema_ema_model_on_cuda(get_dummy_model):
    """Test if ema_handler.ema_model is nn.Module or nn.DataParallel and under eval mode"""
    model = get_dummy_model().to(idist.device())
    model = idist.auto_model(model)
    ema_handler = EMAHandler(model)
    ema_model = ema_handler.ema_model
    assert not ema_model.training
    if isinstance(model, DataParallel):
        assert isinstance(ema_model, DataParallel)
    else:
        assert (
            isinstance(ema_model, nn.Module)
            and (not isinstance(ema_model, DataParallel))
            and (not isinstance(ema_model, DistributedDataParallel))
        )


def test_ema_load_state_dict(get_dummy_model):
    model_1 = get_dummy_model()
    model_1.weight.data.fill_(2)
    state_dict_1 = model_1.state_dict()

    model_2 = get_dummy_model()
    ema_handler = EMAHandler(model_2)
    ema_model = ema_handler.ema_model
    ema_model.load_state_dict(state_dict_1)
    assert ema_model.weight.data.allclose(model_1.weight.data)


def test_ema_no_warmup_momentum(get_dummy_model):
    model = get_dummy_model()
    step_fn = _get_dummy_step_fn(model)
    engine = Engine(step_fn)

    def assert_const_momentum(engine: Engine, const_momentum):
        assert engine.state.ema_momentum == const_momentum

    # no momentum_warmup
    ema_handler = EMAHandler(model, momentum=0.002, momentum_warmup=None, warmup_iters=1)
    ema_handler.attach(engine)
    # attach the assertion handler after ema_handler, so the momentum is first updated and then tested
    engine.add_event_handler(Events.ITERATION_COMPLETED, assert_const_momentum, ema_handler.momentum)
    engine.run(range(2))

    # no warmup_iters
    engine = Engine(step_fn)
    ema_handler = EMAHandler(model, momentum=0.002, momentum_warmup=0.001, warmup_iters=None)
    ema_handler.attach(engine)
    # attach the assertion handler after ema_handler, so the momentum is first updated and then tested
    engine.add_event_handler(Events.ITERATION_COMPLETED, assert_const_momentum, ema_handler.momentum)
    engine.run(range(2))


def test_ema_update_ema_momentum(get_dummy_model):
    model = get_dummy_model()
    step_fn = _get_dummy_step_fn(model)
    engine = Engine(step_fn)

    warmup_iters = 4
    momentum_warmup = 0.1
    momentum = 0.2
    ema_handler = EMAHandler(model, momentum_warmup=momentum_warmup, momentum=momentum, warmup_iters=warmup_iters)
    ema_handler.attach(engine)

    # add handlers to check momentum at each iteration
    @engine.on(Events.ITERATION_COMPLETED)
    def assert_momentum(engine: Engine):
        curr_iter = engine.state.iteration
        curr_momentum = engine.state.ema_momentum
        if curr_iter == 1:
            assert curr_momentum == momentum_warmup
        elif 1 < curr_iter < warmup_iters:
            assert momentum_warmup < curr_momentum < momentum
        else:
            assert curr_momentum == momentum

    engine.run(range(2), max_epochs=5)


def test_ema_buffer():
    """Test if the tensors in buffer are also synchronized"""
    model = nn.BatchNorm2d(2)
    model.running_mean.data.fill_(1.5)
    model.running_var.data.fill_(1.5)
    ema_handler = EMAHandler(model)

    def _bn_step_fn(engine, batch):
        x = torch.rand(4, 2, 32, 32)
        _ = model(x)
        return 1

    engine = Engine(_bn_step_fn)
    ema_handler.attach(engine)

    ema_model = ema_handler.ema_model

    @engine.on(Events.ITERATION_COMPLETED)
    def check_buffers():
        assert ema_model.running_mean.allclose(model.running_mean)
        assert ema_model.running_var.allclose(model.running_var)

    # engine will run 4 iterations
    engine.run([0, 1], max_epochs=2)

    assert ema_model.running_mean.allclose(model.running_mean)
    assert ema_model.running_var.allclose(model.running_var)


def test_ema_two_handlers(get_dummy_model):
    """Test when two EMA handlers are attached to a trainer"""
    model_1 = get_dummy_model()
    # momentum will be constantly 0.5
    ema_handler_1 = EMAHandler(model_1, momentum_warmup=0.5, momentum=0.5, warmup_iters=1)

    model_2 = get_dummy_model()
    ema_handler_2 = EMAHandler(model_2, momentum_warmup=0.5, momentum=0.5, warmup_iters=1)

    def _step_fn(engine: Engine, batch: Any):
        model_1.weight.data.add_(1)
        model_2.weight.data.add_(1)
        return 0

    engine = Engine(_step_fn)
    assert not hasattr(engine.state, "ema_momentum_1")
    # handler_1 update EMA model of model_1 every 1 iteration
    ema_handler_1.attach(engine, "ema_momentum_1", event=Events.ITERATION_COMPLETED)
    assert hasattr(engine.state, "ema_momentum_1")

    # handler_2 update EMA model for model_2 every 2 iterations
    ema_handler_2.attach(engine, "ema_momentum_2", event=Events.ITERATION_COMPLETED(every=2))
    assert hasattr(engine.state, "ema_momentum_2")

    # engine will run 4 iterations
    engine.run(range(2), max_epochs=2)
    # explicitly cast to float32 to avoid test failure on XLA devices
    ema_weight_1 = ema_handler_1.ema_model.weight.data.to(torch.float32)
    ema_weight_2 = ema_handler_2.ema_model.weight.data.to(torch.float32)
    assert ema_weight_1.allclose(ema_weight_1.new_full((1, 2), 4.0625))
    assert ema_weight_2.allclose(ema_weight_2.new_full((1, 2), 3.5))

    assert engine.state.ema_momentum_1 == 0.5
    assert engine.state.ema_momentum_2 == 0.5

    model_3 = get_dummy_model()
    ema_handler_3 = EMAHandler(model_3)
    with pytest.raises(ValueError, match="Please select another name"):
        ema_handler_3.attach(engine, "ema_momentum_2")


def _test_ema_final_weight(model, device=None, ddp=False, interval=1):
    """Test if final smoothed weights are correct"""
    if device is None:
        # let horovod decide the device
        device = idist.device()
    if isinstance(device, str):
        device = torch.device(device)
    model = model.to(device)
    if ddp:
        model = idist.auto_model(model)
    step_fn = _get_dummy_step_fn(model)
    engine = Engine(step_fn)

    # momentum will be constantly 0.5
    ema_handler = EMAHandler(model, momentum_warmup=0.5, momentum=0.5, warmup_iters=1)
    ema_handler.attach(engine, "model", event=Events.ITERATION_COMPLETED(every=interval))

    # engine will run 4 iterations
    engine.run(range(2), max_epochs=2)

    # ema_model and model can be DP or DDP
    # explicitly cast to float32 to avoid test failure on XLA devices
    ema_weight = _unwrap_model(ema_handler.ema_model).weight.data.to(torch.float32)
    model_weight = _unwrap_model(model).weight.data.to(torch.float32)
    assert ema_weight.device == device
    assert model_weight.device == device
    if interval == 1:
        assert ema_weight.allclose(ema_weight.new_full((1, 2), 4.0625))
    elif interval == 2:
        assert ema_weight.allclose(ema_weight.new_full((1, 2), 3.5))
    else:
        pass
    assert model_weight.allclose(model_weight.new_full((1, 2), 5.0))


@pytest.mark.parametrize("interval", [1, 2])
def test_ema_final_weight_cpu(get_dummy_model, interval):
    device = torch.device("cpu")
    _test_ema_final_weight(get_dummy_model(), device=device, ddp=False, interval=interval)


@pytest.mark.parametrize("interval", [1, 2])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_ema_final_weight_cuda(get_dummy_model, interval):
    device = torch.device("cuda:0")
    _test_ema_final_weight(get_dummy_model(), device=device, ddp=False, interval=interval)


@pytest.mark.distributed
@pytest.mark.parametrize("interval", [1, 2])
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_ema_final_weight_distrib_nccl_gpu(get_dummy_model, distributed_context_single_node_nccl, interval):
    device = idist.device()
    _test_ema_final_weight(get_dummy_model(), device=device, ddp=True, interval=interval)


@pytest.mark.distributed
@pytest.mark.parametrize("interval", [1, 2])
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_ema_final_weight_distrib_gloo_cpu_or_gpu(get_dummy_model, distributed_context_single_node_gloo, interval):
    device = idist.device()
    _test_ema_final_weight(get_dummy_model(), device=device, ddp=True, interval=interval)


@pytest.mark.distributed
@pytest.mark.parametrize("interval", [1, 2])
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_ema_final_weight_distrib_hvd(get_dummy_model, gloo_hvd_executor, interval):
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()
    # pass device = None to the executor. Different from other distributed tests where the processes are
    # already spawn in the context, the processes here will be explicitly spawn by the executor, so we
    # pass None to the function, and call idist.device() in side the function to get the corresponding device
    gloo_hvd_executor(_test_ema_final_weight, (get_dummy_model(), None, True, interval), np=nproc, do_init=True)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_ema_final_weight_distrib_single_device_xla(get_dummy_model):
    device = idist.device()
    _test_ema_final_weight(get_dummy_model(), device=device, ddp=True)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_ema_final_weight_distrib_xla_nprocs(get_dummy_model, xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])

    def _test_ema_final_weight_xla_nprocs(index):
        device = idist.device()
        _test_ema_final_weight(get_dummy_model(), device=device, ddp=True)

    xmp_executor(_test_ema_final_weight_xla_nprocs, args=(), nprocs=n)


@pytest.mark.multinode_distributed
@pytest.mark.parametrize("interval", [1, 2])
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_ema_final_weight_distrib_multinode_gloo_cpu_or_gpu(
    get_dummy_model, distributed_context_multi_node_gloo, interval
):
    device = idist.device()
    _test_ema_final_weight(get_dummy_model(), device=device, ddp=True, interval=interval)


@pytest.mark.multinode_distributed
@pytest.mark.parametrize("interval", [1, 2])
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_ema_final_weight_distrib_multinode_nccl_gpu(get_dummy_model, distributed_context_multi_node_nccl, interval):
    device = idist.device()
    _test_ema_final_weight(get_dummy_model(), device=device, ddp=True, interval=interval)
