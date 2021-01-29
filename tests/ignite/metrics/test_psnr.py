import os

import numpy as np
import pytest
import torch
from skimage.metrics import peak_signal_noise_ratio as ski_psnr

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics import PSNR
from ignite.utils import manual_seed


def test_zero_div():
    psnr = PSNR()
    with pytest.raises(NotComputableError, match="PSNR must have at least one example before it can be computed"):
        psnr.compute()


def test_invalid_psnr():
    y_pred = torch.rand(1, 3, 8, 8)
    y = torch.rand(1, 3, 8, 8)

    psnr = PSNR()
    with pytest.raises(TypeError, match="Expected y_pred and y to have the same data type."):
        psnr.update((y_pred, y.double()))

    with pytest.raises(ValueError, match="Expected y_pred and y to have the same shape."):
        # psnr = PSNR()
        psnr.update((y_pred, y.squeeze(dim=0)))

    with pytest.raises(ValueError, match="y has intensity values outside the range expected for its data type."):
        # psnr = PSNR()
        psnr.update((y_pred, y))
        # to catch ValueError for this batch
        psnr.update((y_pred, y + 1.0))

    with pytest.raises(ValueError, match="Range for this dtype cannot be automatically estimated."):
        psnr.update((y_pred.long(), y.long()))


def _test_psnr_per_image(y_pred, y, data_range, device):
    psnr = PSNR(data_range=data_range, device=device)
    psnr.update((y_pred, y))

    np_y_pred = y_pred.cpu().numpy()
    np_y = y.cpu().numpy()
    np_psnr = 0
    for np_y_pred_, np_y_ in zip(np_y_pred, np_y):
        np_psnr += ski_psnr(np_y_, np_y_pred_, data_range=data_range)

    assert np.allclose(psnr._sum_of_batchwise_psnr.cpu().numpy(), np_psnr)


def test_psnr_per_image():
    device = idist.device()
    manual_seed(42)
    y_pred = torch.rand(8, 3, 28, 28, device=device)
    y = y_pred * 0.8

    _test_psnr_per_image(y_pred, y, None, device)


def _test_psnr(y_pred, y, data_range, device):
    psnr = PSNR(data_range=data_range, device=device)
    psnr.update((y_pred, y))
    psnr_compute = psnr.compute()

    np_y_pred = y_pred.cpu().numpy()
    np_y = y.cpu().numpy()
    np_psnr = 0
    for np_y_pred_, np_y_ in zip(np_y_pred, np_y):
        np_psnr += ski_psnr(np_y_, np_y_pred_, data_range=data_range)

    assert isinstance(psnr_compute, torch.Tensor)
    assert psnr_compute.dtype == torch.float64
    assert psnr_compute.device == torch.device(device)
    assert np.allclose(psnr_compute.numpy(), np_psnr / np_y.shape[0])


def test_psnr():
    device = idist.device()
    manual_seed(42)
    y_pred = torch.rand(8, 3, 28, 28, device=device)
    y = y_pred * 0.8
    _test_psnr(y_pred, y, None, device)
    _test_psnr(y_pred, y, 0.8, device)
    _test_psnr(y_pred, y, 1.0, device)

    # test for true_min < 0
    manual_seed(42)
    y_pred = torch.empty(2, 3, 12, 12, device=device).random_(-1, 2)
    y = torch.empty(2, 3, 12, 12, device=device).random_(-1, 2)
    _test_psnr(y_pred, y, None, device)
    _test_psnr(y_pred, y, 0.5, device)
    _test_psnr(y_pred, y, 1, device)

    manual_seed(42)
    y_pred = torch.randint(0, 256, (4, 3, 16, 16), dtype=torch.uint8, device=device)
    y = (y_pred * 0.8).to(torch.uint8)
    _test_psnr(y_pred, y, None, device)
    _test_psnr(y_pred, y, 240, device)
    _test_psnr(y_pred, y, 256, device)

    # test with NHW shape
    manual_seed(42)
    y_pred = torch.rand(8, 28, 28, device=device)
    y = y_pred * 0.8
    _test_psnr(y_pred, y, None, device)
    _test_psnr(y_pred, y, 0.3, device)
    _test_psnr(y_pred, y, 1.0, device)


def _test_distrib_integration(device, atol=1e-8):
    from ignite.engine import Engine

    rank = idist.get_rank()
    n_iters = 100
    s = 10
    offset = n_iters * s

    def _test(y_pred, y, data_range, metric_device):
        def update(engine, i):
            return (
                y_pred[i * s + offset * rank : (i + 1) * s + offset * rank],
                y[i * s + offset * rank : (i + 1) * s + offset * rank],
            )

        engine = Engine(update)
        PSNR(data_range=data_range, device=metric_device).attach(engine, "psnr")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=1)
        result = engine.state.metrics["psnr"]
        assert "psnr" in engine.state.metrics

        np_y_pred = y_pred.cpu().numpy()
        np_y = y.cpu().numpy()
        np_psnr = 0
        for np_y_pred_, np_y_ in zip(np_y_pred, np_y):
            np_psnr += ski_psnr(np_y_, np_y_pred_, data_range=data_range)

        assert np.allclose(result, np_psnr / np_y.shape[0], atol=atol)

    manual_seed(42)
    y_pred = torch.rand(offset * idist.get_world_size(), 3, 28, 28, device=device)
    y = y_pred * 0.65
    _test(y_pred, y, None, "cpu")
    _test(y_pred, y, 0.5, "cpu")
    _test(y_pred, y, 1, "cpu")

    # test for true_min < 0
    manual_seed(42)
    y_pred = torch.empty(offset * idist.get_world_size(), 3, 12, 12, device=device).random_(-1, 2)
    y = -1 * y_pred
    _test(y_pred, y, None, "cpu")
    _test(y_pred, y, 0.5, "cpu")
    _test(y_pred, y, 1, "cpu")

    manual_seed(42)
    y_pred = torch.randint(0, 256, (offset * idist.get_world_size(), 3, 16, 16), device=device, dtype=torch.uint8)
    y = (y_pred * 0.65).to(torch.uint8)
    _test(y_pred, y, None, "cpu")
    _test(y_pred, y, 240, "cpu")
    _test(y_pred, y, 256, "cpu")

    # test with NHW shape
    manual_seed(42)
    y_pred = torch.rand(offset * idist.get_world_size(), 28, 28, device=device)
    y = y_pred * 0.8
    _test(y_pred, y, None, "cpu")
    _test(y_pred, y, 0.3, "cpu")
    _test(y_pred, y, 1, "cpu")

    if torch.device(device).type != "xla":
        manual_seed(42)
        y_pred = torch.rand(offset * idist.get_world_size(), 3, 28, 28, device=device)
        y = y_pred * 0.65
        _test(y_pred, y, None, idist.device())
        _test(y_pred, y, 0.5, idist.device())
        _test(y_pred, y, 1, idist.device())

        # test for true_min < 0
        manual_seed(42)
        y_pred = torch.empty(offset * idist.get_world_size(), 3, 12, 12, device=device).random_(-1, 2)
        y = -1 * y_pred
        _test(y_pred, y, None, idist.device())
        _test(y_pred, y, 0.5, idist.device())
        _test(y_pred, y, 1, idist.device())

        manual_seed(42)
        y_pred = torch.randint(0, 256, (offset * idist.get_world_size(), 3, 16, 16), device=device, dtype=torch.uint8)
        y = (y_pred * 0.65).to(torch.uint8)
        _test(y_pred, y, None, idist.device())
        _test(y_pred, y, 240, idist.device())
        _test(y_pred, y, 256, idist.device())

        # test with NHW shape
        manual_seed(42)
        y_pred = torch.rand(offset * idist.get_world_size(), 28, 28, device=device)
        y = y_pred * 0.8
        _test(y_pred, y, None, idist.device())
        _test(y_pred, y, 0.3, idist.device())
        _test(y_pred, y, 1, idist.device())


def _test_distrib_accumulator_device(device):

    metric_devices = [torch.device("cpu")]
    if torch.device(device).type != "xla":
        metric_devices.append(idist.device())

    for metric_device in metric_devices:
        psnr = PSNR(data_range=1.0, device=metric_device)
        dev = psnr._device
        assert dev == metric_device, f"{dev} vs {metric_device}"

        y_pred = torch.rand(2, 3, 28, 28, dtype=torch.float, device=device)
        y = y_pred * 0.65
        psnr.update((y_pred, y))
        dev = psnr._sum_of_batchwise_psnr.device
        assert dev == metric_device, f"{dev} vs {metric_device}"


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_cpu(distributed_context_single_node_gloo):
    device = "cpu"
    _test_distrib_integration(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_gpu(local_rank, distributed_context_single_node_nccl):
    device = f"cuda:{local_rank}"
    _test_distrib_integration(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):
    device = "cpu"
    _test_distrib_integration(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gpu(distributed_context_multi_node_nccl):
    device = f"cuda:{distributed_context_multi_node_nccl['local_rank']}"
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


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_distrib_integration, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_accumulator_device, (device,), np=nproc, do_init=True)
