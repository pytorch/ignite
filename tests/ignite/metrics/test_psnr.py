import os

import numpy as np
import pytest
import torch
from skimage.metrics import peak_signal_noise_ratio as ski_psnr

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics import PSNR
from ignite.utils import manual_seed

from tests.ignite import cpu_and_maybe_cuda


def test_zero_div():
    psnr = PSNR(1.0)
    with pytest.raises(NotComputableError, match="PSNR must have at least one example before it can be computed"):
        psnr.compute()


def test_invalid_psnr():
    y_pred = torch.rand(1, 3, 8, 8)
    y = torch.rand(1, 3, 8, 8)

    psnr = PSNR(1.0)
    with pytest.raises(TypeError, match="Expected y_pred and y to have the same data type."):
        psnr.update((y_pred, y.double()))

    with pytest.raises(ValueError, match="Expected y_pred and y to have the same shape."):
        psnr.update((y_pred, y.squeeze(dim=0)))


def _test_psnr(y_pred, y, data_range, device):
    psnr = PSNR(data_range=data_range, device=device)
    psnr.update((y_pred, y))
    psnr_compute = psnr.compute()

    np_y_pred = y_pred.cpu().numpy()
    np_y = y.cpu().numpy()
    np_psnr = 0
    for np_y_pred_, np_y_ in zip(np_y_pred, np_y):
        np_psnr += ski_psnr(np_y_, np_y_pred_, data_range=data_range)

    assert torch.gt(psnr_compute, 0.0)
    assert isinstance(psnr_compute, torch.Tensor)
    assert psnr_compute.dtype == torch.float64
    assert psnr_compute.device.type == torch.device(device).type
    assert np.allclose(psnr_compute.cpu().numpy(), np_psnr / np_y.shape[0])


@pytest.mark.parametrize("device", cpu_and_maybe_cuda())
def test_psnr(device):

    # test for float
    manual_seed(42)
    y_pred = torch.rand(8, 3, 28, 28, device=device)
    y = y_pred * 0.8
    data_range = (y.max() - y.min()).cpu().item()
    _test_psnr(y_pred, y, data_range, device)

    # test for YCbCr
    manual_seed(42)
    y_pred = torch.randint(16, 236, (4, 1, 12, 12), dtype=torch.uint8, device=device)
    y = torch.randint(16, 236, (4, 1, 12, 12), dtype=torch.uint8, device=device)
    data_range = (y.max() - y.min()).cpu().item()
    _test_psnr(y_pred, y, data_range, device)

    # test for uint8
    manual_seed(42)
    y_pred = torch.randint(0, 256, (4, 3, 16, 16), dtype=torch.uint8, device=device)
    y = (y_pred * 0.8).to(torch.uint8)
    data_range = (y.max() - y.min()).cpu().item()
    _test_psnr(y_pred, y, data_range, device)

    # test with NHW shape
    manual_seed(42)
    y_pred = torch.rand(8, 28, 28, device=device)
    y = y_pred * 0.8
    data_range = (y.max() - y.min()).cpu().item()
    _test_psnr(y_pred, y, data_range, device)


def _test(
    y_pred,
    y,
    data_range,
    metric_device,
    n_iters,
    batch_size,
    atol,
    output_transform=lambda x: x,
    compute_y_channel=False,
):
    def update(engine, i):
        return (
            y_pred[i * batch_size : (i + 1) * batch_size],
            y[i * batch_size : (i + 1) * batch_size],
        )

    engine = Engine(update)
    psnr = PSNR(data_range=data_range, output_transform=output_transform, device=metric_device)
    psnr.attach(engine, "psnr")
    data = list(range(n_iters))

    engine.run(data=data, max_epochs=1)

    y = idist.all_gather(y)
    y_pred = idist.all_gather(y_pred)

    result = engine.state.metrics["psnr"]
    assert result > 0.0
    assert "psnr" in engine.state.metrics

    if compute_y_channel:
        np_y_pred = y_pred[:, 0, ...].cpu().numpy()
        np_y = y[:, 0, ...].cpu().numpy()
    else:
        np_y_pred = y_pred.cpu().numpy()
        np_y = y.cpu().numpy()

    np_psnr = 0
    for np_y_pred_, np_y_ in zip(np_y_pred, np_y):
        np_psnr += ski_psnr(np_y_, np_y_pred_, data_range=data_range)

    assert np.allclose(result, np_psnr / np_y.shape[0], atol=atol)


def _test_distrib_input_float(device, atol=1e-8):
    def get_test_cases():

        y_pred = torch.rand(n_iters * batch_size, 2, 2, device=device)
        y = y_pred * 0.65

        return y_pred, y

    n_iters = 100
    batch_size = 10

    rank = idist.get_rank()
    for i in range(3):
        # check multiple random inputs as random exact occurencies are rare
        torch.manual_seed(42 + rank + i)
        y_pred, y = get_test_cases()
        _test(y_pred, y, 1, "cpu", n_iters, batch_size, atol=atol)
        if device.type != "xla":
            _test(y_pred, y, 1, idist.device(), n_iters, batch_size, atol=atol)


def _test_distrib_multilabel_input_YCbCr(device, atol=1e-8):
    def get_test_cases():

        y_pred = torch.randint(16, 236, (n_iters * batch_size, 1, 12, 12), dtype=torch.uint8, device=device)
        cbcr_pred = torch.randint(16, 241, (n_iters * batch_size, 2, 12, 12), dtype=torch.uint8, device=device)
        y = torch.randint(16, 236, (n_iters * batch_size, 1, 12, 12), dtype=torch.uint8, device=device)
        cbcr = torch.randint(16, 241, (n_iters * batch_size, 2, 12, 12), dtype=torch.uint8, device=device)

        y_pred, y = torch.cat((y_pred, cbcr_pred), dim=1), torch.cat((y, cbcr), dim=1)

        return y_pred, y

    n_iters = 100
    batch_size = 10

    def out_fn(x):
        return x[0][:, 0, ...], x[1][:, 0, ...]

    rank = idist.get_rank()
    for i in range(3):
        # check multiple random inputs as random exact occurencies are rare
        torch.manual_seed(42 + rank + i)
        y_pred, y = get_test_cases()
        _test(y_pred, y, 220, "cpu", n_iters, batch_size, atol, output_transform=out_fn, compute_y_channel=True)
        if device.type != "xla":
            dev = idist.device()
            _test(y_pred, y, 220, dev, n_iters, batch_size, atol, output_transform=out_fn, compute_y_channel=True)


def _test_distrib_multilabel_input_uint8(device, atol=1e-8):
    def get_test_cases():

        y_pred = torch.randint(0, 256, (n_iters * batch_size, 3, 16, 16), device=device, dtype=torch.uint8)
        y = (y_pred * 0.65).to(torch.uint8)

        return y_pred, y

    n_iters = 100
    batch_size = 10

    rank = idist.get_rank()
    for i in range(3):
        # check multiple random inputs as random exact occurencies are rare
        torch.manual_seed(42 + rank + i)
        y_pred, y = get_test_cases()
        _test(y_pred, y, 100, "cpu", n_iters, batch_size, atol)
        if device.type != "xla":
            _test(y_pred, y, 100, idist.device(), n_iters, batch_size, atol)


def _test_distrib_multilabel_input_NHW(device, atol=1e-8):
    def get_test_cases():

        y_pred = torch.rand(n_iters * batch_size, 28, 28, device=device)
        y = y_pred * 0.8

        return y_pred, y

    n_iters = 100
    batch_size = 10

    rank = idist.get_rank()
    for i in range(3):
        # check multiple random inputs as random exact occurencies are rare
        torch.manual_seed(42 + rank + i)
        y_pred, y = get_test_cases()
        _test(y_pred, y, 10, "cpu", n_iters, batch_size, atol)
        if device.type != "xla":
            _test(y_pred, y, 10, idist.device(), n_iters, batch_size, atol)


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
def test_distrib_gloo_cpu_or_gpu(distributed_context_single_node_gloo):

    device = idist.device()
    _test_distrib_input_float(device)
    _test_distrib_multilabel_input_YCbCr(device)
    _test_distrib_multilabel_input_uint8(device)
    _test_distrib_multilabel_input_NHW(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_nccl_gpu(distributed_context_single_node_nccl):

    device = idist.device()
    _test_distrib_input_float(device)
    _test_distrib_multilabel_input_YCbCr(device)
    _test_distrib_multilabel_input_uint8(device)
    _test_distrib_multilabel_input_NHW(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gloo_cpu_or_gpu(distributed_context_multi_node_gloo):

    device = idist.device()
    _test_distrib_input_float(device)
    _test_distrib_multilabel_input_YCbCr(device)
    _test_distrib_multilabel_input_uint8(device)
    _test_distrib_multilabel_input_NHW(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_nccl_gpu(distributed_context_multi_node_nccl):

    device = idist.device()
    _test_distrib_input_float(device)
    _test_distrib_multilabel_input_YCbCr(device)
    _test_distrib_multilabel_input_uint8(device)
    _test_distrib_multilabel_input_NHW(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():

    device = idist.device()
    _test_distrib_input_float(device)
    _test_distrib_multilabel_input_YCbCr(device)
    _test_distrib_multilabel_input_uint8(device)
    _test_distrib_multilabel_input_NHW(device)
    _test_distrib_accumulator_device(device)


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_distrib_input_float(device)
    _test_distrib_multilabel_input_YCbCr(device)
    _test_distrib_multilabel_input_uint8(device)
    _test_distrib_multilabel_input_NHW(device)
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

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_distrib_input_float, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_multilabel_input_YCbCr, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_multilabel_input_uint8, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_multilabel_input_NHW, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_accumulator_device, (device,), np=nproc, do_init=True)
