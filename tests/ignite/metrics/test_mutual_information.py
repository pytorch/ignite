from typing import Tuple
import os

import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
import pytest
import torch
from torch import Tensor

from ignite.engine import Engine
import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics import MutualInformation


def np_mutual_information(np_y_pred: np.ndarray) -> float:
    prob = softmax(np_y_pred, axis=1)
    marginal_ent = entropy(np.mean(prob, axis=0))
    conditional_ent = np.mean(entropy(prob, axis=1))
    return max(0.0, marginal_ent - conditional_ent)


def test_zero_sample():
    mi = MutualInformation()
    with pytest.raises(
        NotComputableError, match=r"MutualInformation must have at least one example before it can be computed"
    ):
        mi.compute()


def test_invalid_shape():
    mi = MutualInformation()
    y_pred = torch.randn(10).float()
    with pytest.raises(ValueError, match=r"y_pred must be in the shape of \(B, C\) or \(B, C, ...\), got"):
        mi.update((y_pred, None))


@pytest.fixture(params=[item for item in range(4)])
def test_case(request):
    return [
        (torch.randn((100, 10)).float(), torch.randint(0, 10, size=[100]), 1),
        (torch.rand((100, 500)).float(), torch.randint(0, 500, size=[100]), 1),
        # updated batches
        (torch.normal(0.0, 5.0, size=(100, 10)).float(), torch.randint(0, 10, size=[100]), 16),
        (torch.normal(5.0, 3.0, size=(100, 200)).float(), torch.randint(0, 200, size=[100]), 16),
        # image segmentation
        (torch.randn((100, 5, 32, 32)).float(), torch.randint(0, 5, size=(100, 32, 32)), 16),
        (torch.randn((100, 5, 224, 224)).float(), torch.randint(0, 5, size=(100, 224, 224)), 16),
    ][request.param]


@pytest.mark.parametrize("n_times", range(5))
def test_compute(n_times, test_case: Tuple[Tensor, Tensor, int]):
    mi = MutualInformation()

    y_pred, y, batch_size = test_case

    mi.reset()
    if batch_size > 1:
        n_iters = y.shape[0] // batch_size + 1
        for i in range(n_iters):
            idx = i * batch_size
            mi.update((y_pred[idx: idx + batch_size], y[idx: idx + batch_size]))
    else:
        mi.update((y_pred, y))

    np_res = np_mutual_information(y_pred.numpy())
    res = mi.compute()

    assert isinstance(res, float)
    assert pytest.approx(np_res, rel=1e-4) == res


def _test_distrib_integration(device, tol=1e-4):
    rank = idist.get_rank()
    torch.manual_seed(12 + rank)

    def _test(metric_device):
        n_iters = 100
        batch_size = 10
        n_cls = 50

        y_true = torch.randint(0, n_cls, size=[n_iters * batch_size], dtype=torch.long).to(device)
        y_preds = torch.normal(0.0, 3.0, size=(n_iters * batch_size, n_cls), dtype=torch.float).to(device)

        def update(engine, i):
            return (
                y_preds[i * batch_size: (i + 1) * batch_size],
                y_true[i * batch_size: (i + 1) * batch_size],
            )

        engine = Engine(update)

        m = MutualInformation(device=metric_device)
        m.attach(engine, "mutual_information")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=1)

        y_preds = idist.all_gather(y_preds)
        y_true = idist.all_gather(y_true)

        assert "mutual_information" in engine.state.metrics
        res = engine.state.metrics["mutual_information"]

        true_res = np_mutual_information(y_preds.cpu().numpy())

        assert pytest.approx(true_res, rel=tol) == res

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
