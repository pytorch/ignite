import os

import numpy as np
import pytest
import torch

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics import RootMeanSquaredError


def test_zero_sample():
    rmse = RootMeanSquaredError()
    with pytest.raises(
        NotComputableError, match=r"MeanSquaredError must have at least one example before it can be computed"
    ):
        rmse.compute()


def test_compute():

    rmse = RootMeanSquaredError()

    def _test(y_pred, y, batch_size):
        rmse.reset()
        if batch_size > 1:
            n_iters = y.shape[0] // batch_size + 1
            for i in range(n_iters):
                idx = i * batch_size
                rmse.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))
        else:
            rmse.update((y_pred, y))

        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()

        np_res = np.sqrt(np.power((np_y - np_y_pred), 2.0).sum() / np_y.shape[0])
        res = rmse.compute()

        assert isinstance(res, float)
        assert pytest.approx(res) == np_res

    def get_test_cases():

        test_cases = [
            (torch.empty(10,).uniform_(0, 10), torch.empty(10,).uniform_(0, 10), 1,),
            (torch.empty(10, 1).uniform_(-10, 10), torch.empty(10, 1).uniform_(-10, 10), 1),
            # updated batches
            (torch.empty(50,).uniform_(0, 10), torch.empty(50).uniform_(0, 10), 16,),
            (torch.empty(50, 1).uniform_(-10, 10), torch.empty(50, 1).uniform_(-10, 10), 16),
        ]

        return test_cases

    for _ in range(5):
        # check multiple random inputs as random exact occurencies are rare
        test_cases = get_test_cases()
        for y_pred, y, batch_size in test_cases:
            _test(y_pred, y, batch_size)


def _test_distrib_integration(device, tol=1e-6):

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
        _test(idist.device())


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_nccl_gpu(distributed_context_single_node_nccl):

    device = idist.device()
    _test_distrib_integration(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_gloo_cpu_or_gpu(distributed_context_single_node_gloo):

    device = idist.device()
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
def test_multinode_distrib_gloo_cpu_or_gpu(distributed_context_multi_node_gloo):

    device = idist.device()
    _test_distrib_integration(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_nccl_gpu(distributed_context_multi_node_nccl):

    device = idist.device()
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
