import os

import pytest
import torch

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics import Rouge


def test_zero_div():
    rouge = Rouge()
    with pytest.raises(NotComputableError):
        rouge.compute()


def test_input():
    with pytest.raises(TypeError):
        rouge = Rouge(beta="l", n=1)

    with pytest.raises(ValueError):
        rouge = Rouge(n="m")

    with pytest.raises(ValueError):
        rouge = Rouge(n=-1)


def test_compute():
    rouge = Rouge()

    y_pred = "the cat was found under the bed"
    y = "the cat was under the bed"
    rouge.update([y_pred.split(), [y.split()]])
    assert isinstance(rouge.compute(), torch.Tensor)
    assert rouge.compute() == 0.8571428656578064

    y_pred = "the tiny little cat was found under the big funny bed"
    y = "the cat was under the bed"
    rouge.update([y_pred.split(), [y.split()]])
    assert isinstance(rouge.compute(), torch.Tensor)
    assert rouge.compute() == 0.701298713684082

    rouge = Rouge(n="l")

    y_pred = "the cat was found under the bed"
    y = "the cat was not under the bed"
    rouge.update([y_pred.split(), [y.split()]])

    assert isinstance(rouge.compute(), torch.Tensor)
    assert rouge.compute() == 0.8571428656578064

    y_pred = "the cat was found under the big funny bed"
    y = "the tiny little cat was under the bed"
    rouge.update([y_pred.split(), [y.split()]])

    assert isinstance(rouge.compute(), torch.Tensor)
    assert rouge.compute() == 0.761904776096344


def _test_distrib_integration(device):
    import ignite.distributed as idist
    from ignite.engine import Engine
    from ignite.metrics import Rouge

    n_iters = 2

    y_true = ["Hi", "Hello"]
    y_preds = [["Hi there!", "Hi , How are you?"], ["Hello there", "Hello , How are you?"]]

    def update(engine, i):
        return (y_true[i].split(), [s.split() for s in y_preds[i]])

    def _test_n(metric_device):
        engine = Engine(update)
        m = Rouge(device=metric_device)
        m.attach(engine, "rouge")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=1)

        assert "rouge" in engine.state.metrics

    def _test_l(metric_device):
        engine = Engine(update)
        m = Rouge(n="l", device=metric_device)
        m.attach(engine, "rouge")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=1)

        assert "rouge" in engine.state.metrics

    _test_n("cpu")
    _test_l("cpu")
    if device.type != "xla":
        _test_n(idist.device())
        _test_l(idist.device())


def _test_distrib_accumulator_device(device):

    metric_devices = [torch.device("cpu")]
    if torch.device(device).type != "xla":
        metric_devices.append(idist.device())

    for metric_device in metric_devices:
        rouge = Rouge(device=metric_device)
        dev = rouge._device
        assert dev == metric_device, f"{dev} vs {metric_device}"

        y_pred = "the tiny little cat was found under the big funny bed"
        y = "the cat was under the bed"
        rouge.update([y_pred.split(), [y.split()]])
        dev = rouge._rougetotal.device
        assert dev == metric_device, f"{dev} vs {metric_device}"


def test_accumulator_detached():
    rouge = Rouge()

    y_pred = "the cat was found under the big funny bed"
    y = "the tiny little cat was under the bed"
    rouge.update([y_pred.split(), [y.split()]])

    assert not rouge._rougetotal.requires_grad


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_gpu(local_rank, distributed_context_single_node_nccl):
    device = torch.device(f"cuda:{local_rank}")
    _test_distrib_integration(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_cpu(distributed_context_single_node_gloo):
    device = torch.device("cpu")
    _test_distrib_integration(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_distrib_integration, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_accumulator_device, (device,), np=nproc, do_init=True)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):
    device = torch.device("cpu")
    _test_distrib_integration(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gpu(distributed_context_multi_node_nccl):
    device = torch.device(f"cuda:{distributed_context_multi_node_nccl['local_rank']}")
    _test_distrib_integration(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():
    device = idist.device()
    _test_distrib_integration(device)
    _test_distrib_accumulator_device(device)


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_distrib_integration(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)
