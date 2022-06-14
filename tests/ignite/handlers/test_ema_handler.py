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


def test_has_momentum_scheduler(get_dummy_model):
    """Test the handler has attribute `momentum_scheduler` and `_momentum_lambda_obj`"""
    momentum_warmup = 0.0
    warmup_iters = 10
    ema_handler = EMAHandler(get_dummy_model(), momentum_warmup=momentum_warmup, warmup_iters=warmup_iters)
    assert hasattr(ema_handler, "momentum_scheduler")
    assert hasattr(ema_handler, "_momentum_lambda_obj")


def test_ema_warmup_func(get_dummy_model):
    """Test the built-in linear warmup function for the EMA momentum"""
    momentum = 0.5
    momentum_warmup_1 = 0.0
    momentum_warmup_2 = 1.0
    warmup_iters = 5

    def check_ema_momentum(engine: Engine, momentum_warmup, final_momentum, warmup_iters):
        if engine.state.iteration == 1:
            assert engine.state.ema_momentum == momentum_warmup
        elif engine.state.iteration >= 1 + warmup_iters:
            assert engine.state.ema_momentum == final_momentum
        else:
            min_momentum = min(momentum, momentum_warmup)
            max_momentum = max(momentum, momentum_warmup)
            assert min_momentum <= engine.state.ema_momentum <= max_momentum

    # momentum_warmup < momentum
    model_1 = get_dummy_model()
    engine_1 = Engine(_get_dummy_step_fn(model_1))
    ema_handler_1 = EMAHandler(model_1, momentum, momentum_warmup_1, warmup_iters)
    ema_handler_1.attach(engine_1)
    engine_1.add_event_handler(
        Events.ITERATION_COMPLETED, check_ema_momentum, momentum_warmup_1, momentum, warmup_iters
    )
    engine_1.run(range(10))

    # momentum_warmup > momentum
    model_2 = get_dummy_model()
    engine_2 = Engine(_get_dummy_step_fn(model_2))
    ema_handler_2 = EMAHandler(model_2, momentum, momentum_warmup_2, warmup_iters)
    ema_handler_2.attach(engine_2)
    engine_2.add_event_handler(
        Events.ITERATION_COMPLETED, check_ema_momentum, momentum_warmup_2, momentum, warmup_iters
    )
    engine_2.run(range(10))


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


def test_ema_get_const_momentum(get_dummy_model):
    """Test if momentum retrieved from the engine is constant and equal to the handler's momentum"""
    model = get_dummy_model()
    step_fn = _get_dummy_step_fn(model)
    engine = Engine(step_fn)

    def assert_const_momentum(engine: Engine, const_momentum):
        assert engine.state.ema_momentum == const_momentum

    ema_handler = EMAHandler(model, momentum=0.002)
    ema_handler.attach(engine)
    engine.add_event_handler(Events.ITERATION_COMPLETED, assert_const_momentum, ema_handler.momentum)
    engine.run(range(10))


@pytest.mark.parametrize("handle_buffers", ["copy", "update", "ema_train", "invalid"])
def test_ema_buffer(handle_buffers):
    """Test if the tensors in buffer are also correctly updated"""
    model = nn.BatchNorm2d(2)
    model.running_mean.data.fill_(1.5)
    model.running_var.data.fill_(1.5)

    # manually register a buffer to test if it will be correctly updated
    model.register_buffer("dummy_buffer", tensor=torch.tensor(1.0, dtype=torch.float32))

    if handle_buffers == "invalid":
        with pytest.raises(ValueError, match="handle_buffers can only"):
            _ = EMAHandler(model, momentum=0.5, handle_buffers=handle_buffers)
    else:
        ema_handler = EMAHandler(model, momentum=0.5, handle_buffers=handle_buffers)

        def _bn_step_fn(engine, batch):
            x = torch.rand(4, 2, 32, 32)
            _ = model(x)
            # manually increment the dummy_buffer at every step
            model.dummy_buffer += 1.0
            return 1

        engine = Engine(_bn_step_fn)
        ema_handler.attach(engine)

        ema_model = ema_handler.ema_model
        if handle_buffers == "ema_train":
            assert ema_model.training
        else:
            assert not ema_model.training

        @engine.on(Events.ITERATION_COMPLETED)
        def check_buffers():
            if handle_buffers == "update":
                # the buffers with torch.int64 data type should be directly copied
                assert ema_model.num_batches_tracked.allclose(model.num_batches_tracked)

                # buffers with floating type will be updated rather than copied
                assert not ema_model.dummy_buffer.allclose(model.dummy_buffer)
                assert not ema_model.running_mean.allclose(model.running_mean)
                assert not ema_model.running_var.allclose(model.running_var)
            elif handle_buffers == "copy":
                # the buffers with torch.int64 data type should be directly copied
                assert ema_model.num_batches_tracked.allclose(model.num_batches_tracked)

                assert ema_model.dummy_buffer.allclose(model.dummy_buffer)
                assert ema_model.running_mean.allclose(model.running_mean)
                assert ema_model.running_var.allclose(model.running_var)
            else:
                # buffers will not be copied or EMA updated
                assert ema_model.num_batches_tracked.allclose(torch.tensor(0, dtype=torch.int64))
                assert ema_model.dummy_buffer.allclose(torch.tensor(1.0, dtype=torch.float32))

        # engine will run 4 iterations
        engine.run([0, 1], max_epochs=2)

        if handle_buffers == "update":
            assert ema_model.num_batches_tracked.allclose(model.num_batches_tracked)
            assert ema_model.dummy_buffer.allclose(torch.tensor(4.0625, dtype=torch.float32))
            assert not ema_model.dummy_buffer.allclose(model.dummy_buffer)
            assert not ema_model.running_mean.allclose(model.running_mean)
            assert not ema_model.running_var.allclose(model.running_var)
        elif handle_buffers == "copy":
            assert ema_model.num_batches_tracked.allclose(model.num_batches_tracked)
            assert ema_model.dummy_buffer.allclose(model.dummy_buffer)
            assert ema_model.running_mean.allclose(model.running_mean)
            assert ema_model.running_var.allclose(model.running_var)
        else:
            # buffers will not be copied or EMA updated
            assert ema_model.num_batches_tracked.allclose(torch.tensor(0, dtype=torch.int64))
            assert ema_model.dummy_buffer.allclose(torch.tensor(1.0, dtype=torch.float32))


def test_ema_two_handlers(get_dummy_model):
    """Test when two EMA handlers are attached to a trainer"""
    model_1 = get_dummy_model()
    ema_handler_1 = EMAHandler(model_1, momentum=0.5)

    model_2 = get_dummy_model()
    ema_handler_2 = EMAHandler(model_2, momentum=0.5)

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
    with pytest.warns(UserWarning, match="Attribute 'ema_momentum_1' already exists"):
        ema_handler_3.attach(engine, name="ema_momentum_1")


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

    ema_handler = EMAHandler(model, momentum=0.5)
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
