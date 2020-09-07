import os

import pytest
import torch

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics import RootMeanSquaredError


def test_zero_div():
    rmse = RootMeanSquaredError()
    with pytest.raises(NotComputableError):
        rmse.compute()


def test_compute():
    rmse = RootMeanSquaredError()

    y_pred = torch.Tensor([[2.0], [-2.0]])
    y = torch.zeros(2)
    rmse.update((y_pred, y))
    assert isinstance(rmse.compute(), float)
    assert rmse.compute() == 2.0

    rmse.reset()
    y_pred = torch.Tensor([[3.0], [-3.0]])
    y = torch.zeros(2)
    rmse.update((y_pred, y))
    assert isinstance(rmse.compute(), float)
    assert rmse.compute() == 3.0


def _test_distrib_integration(device, tol=1e-6):
    import numpy as np
    from ignite.engine import Engine

    rank = idist.get_rank()
    n_iters = 100
    s = 10
    offset = n_iters * s

    y_true = torch.arange(0, offset * idist.get_world_size(), dtype=torch.float).to(device)
    y_preds = (rank + 1) * torch.ones(offset, dtype=torch.float).to(device)

    def update(engine, i):
        return y_preds[i * s : (i + 1) * s], y_true[i * s + offset * rank : (i + 1) * s + offset * rank]

    def _test(metric_device):
        engine = Engine(update)

        m = RootMeanSquaredError(device=metric_device)
        m.attach(engine, "rmse")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=1)

        assert "rmse" in engine.state.metrics
        res = engine.state.metrics["rmse"]

        y_preds_full = []
        for i in range(idist.get_world_size()):
            y_preds_full.append((i + 1) * torch.ones(offset))
        y_preds_full = torch.stack(y_preds_full).to(device).flatten()

        true_res = np.sqrt(np.mean(np.square((y_true - y_preds_full).cpu().numpy())))

        assert pytest.approx(res, rel=tol) == true_res

    _test("cpu")
    if device.type != "xla":
        _test(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_gpu(local_rank, distributed_context_single_node_nccl):

    device = torch.device("cuda:{}".format(local_rank))
    _test_distrib_integration(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_cpu(local_rank, distributed_context_single_node_gloo):

    device = torch.device("cpu")
    _test_distrib_integration(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_distrib_integration, (device,), np=nproc, do_init=True)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):
    device = torch.device("cpu")
    _test_distrib_integration(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gpu(distributed_context_multi_node_nccl):
    device = torch.device("cuda:{}".format(distributed_context_multi_node_nccl["local_rank"]))
    _test_distrib_integration(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():
    device = idist.device()
    _test_distrib_integration(device, tol=1e-4)


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_distrib_integration(device, tol=1e-4)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)
