import os

import numpy as np
import pytest
import torch
from skimage.measure import compare_ssim as ski_ssim

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics import SSIM


def test_zero_div():
    ssim = SSIM(data_range=1.0)
    with pytest.raises(NotComputableError):
        ssim.compute()


def test_invalid_ssim():
    y_pred = torch.rand(16, 1, 32, 32)
    y = y_pred + 0.125
    with pytest.raises(ValueError, match=r"Expected kernel_size to have odd positive number. Got 10."):
        ssim = SSIM(data_range=1.0, kernel_size=10)
        ssim.update((y_pred, y))
        ssim.compute()

    with pytest.raises(ValueError, match=r"Expected kernel_size to have odd positive number. Got -1."):
        ssim = SSIM(data_range=1.0, kernel_size=-1)
        ssim.update((y_pred, y))
        ssim.compute()

    with pytest.raises(ValueError, match=r"Argument kernel_size should be either int or a sequence of int."):
        ssim = SSIM(data_range=1.0, kernel_size=1.0)
        ssim.update((y_pred, y))
        ssim.compute()

    with pytest.raises(ValueError, match=r"Argument sigma should be either float or a sequence of float."):
        ssim = SSIM(data_range=1.0, sigma=-1)
        ssim.update((y_pred, y))
        ssim.compute()

    with pytest.raises(ValueError, match=r"Argument sigma should be either float or a sequence of float."):
        ssim = SSIM(data_range=1.0, sigma=1)
        ssim.update((y_pred, y))
        ssim.compute()


def test_ssim():
    ssim = SSIM(data_range=1.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    y_pred = torch.rand(16, 3, 32, 32, device=device)
    y = y_pred + 0.125
    ssim.update((y_pred, y))

    np_pred = y_pred.permute(0, 2, 3, 1).numpy()
    np_y = np.add(np_pred, 0.125)
    np_ssim = ski_ssim(np_pred, np_y, win_size=11, multichannel=True, gaussian_weights=True)

    assert isinstance(ssim.compute(), torch.Tensor)
    assert torch.allclose(
        ssim.compute(), torch.tensor(np_ssim, dtype=torch.float32, device=device), atol=1e-3, rtol=1e-3
    )


def _test_distrib_integration(device, tol=1e-3):
    from ignite.engine import Engine

    rank = idist.get_rank()
    n_iters = 100
    s = 10
    offset = n_iters * s

    y_pred = torch.rand(offset * idist.get_world_size(), 3, 28, 28, dtype=torch.float, device=device)
    y = y_pred + 0.125

    def update(engine, i):
        return (
            y_pred[i * s + offset * rank : (i + 1) * s + offset * rank],
            y[i * s + offset * rank : (i + 1) * s + offset * rank],
        )

    engine = Engine(update)
    SSIM(data_range=1.0).attach(engine, "ssim")

    data = list(range(n_iters))
    engine.run(data=data, max_epochs=1)

    assert "ssim" in engine.state.metrics
    res = engine.state.metrics["ssim"]

    np_pred = y_pred.permute(0, 2, 3, 1).numpy()
    np_true = np.add(np_pred, 0.125)
    true_res = ski_ssim(np_pred, np_true, win_size=11, multichannel=True, gaussian_weights=True)

    assert pytest.approx(res, rel=tol) == true_res


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_gpu(local_rank, distributed_context_single_node_nccl):

    device = "cuda:{}".format(local_rank)
    _test_distrib_integration(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_cpu(distributed_context_single_node_gloo):
    device = "cpu"
    _test_distrib_integration(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):
    device = "cpu"
    _test_distrib_integration(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gpu(distributed_context_multi_node_nccl):
    device = "cuda:{}".format(distributed_context_multi_node_nccl["local_rank"])
    _test_distrib_integration(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():
    device = idist.device()
    _test_distrib_integration(device, tol=1e-3)


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_distrib_integration(device, tol=1e-3)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)
