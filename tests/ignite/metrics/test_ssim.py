import os

import numpy as np
import pytest
import torch
from skimage.metrics import structural_similarity as ski_ssim

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics import SSIM


def test_zero_div():
    ssim = SSIM(data_range=1.0)
    with pytest.raises(NotComputableError):
        ssim.compute()


def test_invalid_ssim():
    y_pred = torch.rand(1, 1, 4, 4)
    y = y_pred + 0.125
    with pytest.raises(ValueError, match=r"Expected kernel_size to have odd positive number."):
        ssim = SSIM(data_range=1.0, kernel_size=2)
        ssim.update((y_pred, y))
        ssim.compute()

    with pytest.raises(ValueError, match=r"Expected kernel_size to have odd positive number."):
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

    with pytest.raises(ValueError, match=r"Expected sigma to have positive number."):
        ssim = SSIM(data_range=1.0, sigma=(-1, -1))
        ssim.update((y_pred, y))
        ssim.compute()

    with pytest.raises(ValueError, match=r"Argument sigma should be either float or a sequence of float."):
        ssim = SSIM(data_range=1.0, sigma=1)
        ssim.update((y_pred, y))
        ssim.compute()

    with pytest.raises(ValueError, match=r"Expected y_pred and y to have the same shape."):
        y = y.squeeze(dim=0)
        ssim = SSIM(data_range=1.0)
        ssim.update((y_pred, y))
        ssim.compute()

    with pytest.raises(ValueError, match=r"Expected y_pred and y to have BxCxHxW shape."):
        y = y.squeeze(dim=0)
        ssim = SSIM(data_range=1.0)
        ssim.update((y, y))
        ssim.compute()

    with pytest.raises(TypeError, match=r"Expected y_pred and y to have the same data type."):
        y = y.double()
        ssim = SSIM(data_range=1.0)
        ssim.update((y_pred, y))
        ssim.compute()


@pytest.mark.parametrize(
    "shape, kernel_size, gaussian, use_sample_covariance",
    [[(8, 3, 224, 224), 7, False, True], [(12, 3, 28, 28), 11, True, False]],
)
def test_ssim(available_device, shape, kernel_size, gaussian, use_sample_covariance):
    y_pred = torch.rand(shape, device=available_device)
    y = y_pred * 0.8

    sigma = 1.5
    data_range = 1.0
    ssim = SSIM(data_range=data_range, sigma=sigma, device=available_device)
    ssim.update((y_pred, y))
    ignite_ssim = ssim.compute()

    skimg_pred = y_pred.cpu().numpy()
    skimg_y = skimg_pred * 0.8
    skimg_ssim = ski_ssim(
        skimg_pred,
        skimg_y,
        win_size=kernel_size,
        sigma=sigma,
        channel_axis=1,
        gaussian_weights=gaussian,
        data_range=data_range,
        use_sample_covariance=use_sample_covariance,
    )

    assert isinstance(ignite_ssim, torch.Tensor)
    assert np.allclose(ignite_ssim.item(), skimg_ssim, atol=7e-5)


def test_ssim_variable_batchsize(available_device):
    # Checks https://github.com/pytorch/ignite/issues/2532
    sigma = 1.5
    data_range = 1.0
    ssim = SSIM(data_range=data_range, sigma=sigma)

    y_preds = [
        torch.rand(12, 3, 28, 28, device=available_device),
        torch.rand(12, 3, 28, 28, device=available_device),
        torch.rand(8, 3, 28, 28, device=available_device),
        torch.rand(16, 3, 28, 28, device=available_device),
        torch.rand(1, 3, 28, 28, device=available_device),
        torch.rand(30, 3, 28, 28, device=available_device),
    ]
    y_true = [v * 0.8 for v in y_preds]

    for y_pred, y in zip(y_preds, y_true):
        ssim.update((y_pred, y))

    out = ssim.compute()
    ssim.reset()
    ssim.update((torch.cat(y_preds), torch.cat(y_true)))
    expected = ssim.compute()
    assert np.allclose(out.item(), expected.item())


def _test_distrib_integration(device, tol=1e-4):
    from ignite.engine import Engine

    rank = idist.get_rank()
    torch.manual_seed(12 + rank)
    n_iters = 100
    batch_size = 10

    def _test(metric_device):
        y_pred = torch.rand(n_iters * batch_size, 3, 28, 28, dtype=torch.float, device=device)
        y = y_pred * 0.65

        def update(engine, i):
            return (
                y_pred[i * batch_size : (i + 1) * batch_size, ...],
                y[i * batch_size : (i + 1) * batch_size, ...],
            )

        engine = Engine(update)
        SSIM(data_range=1.0, device=metric_device).attach(engine, "ssim")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=1)

        y_pred = idist.all_gather(y_pred)
        y = idist.all_gather(y)

        assert "ssim" in engine.state.metrics
        res = engine.state.metrics["ssim"]

        np_pred = y_pred.cpu().numpy()
        np_true = np_pred * 0.65
        true_res = ski_ssim(
            np_pred,
            np_true,
            win_size=11,
            sigma=1.5,
            channel_axis=1,
            gaussian_weights=True,
            data_range=1.0,
            use_sample_covariance=False,
        )

        assert pytest.approx(res, abs=tol) == true_res

        engine = Engine(update)
        SSIM(data_range=1.0, gaussian=False, kernel_size=7, device=metric_device).attach(engine, "ssim")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=1)

        assert "ssim" in engine.state.metrics
        res = engine.state.metrics["ssim"]

        np_pred = y_pred.cpu().numpy()
        np_true = np_pred * 0.65
        true_res = ski_ssim(np_pred, np_true, win_size=7, channel_axis=1, gaussian_weights=False, data_range=1.0)

        assert pytest.approx(res, abs=tol) == true_res

    _test("cpu")
    if torch.device(device).type != "xla":
        _test(idist.device())


def _test_distrib_accumulator_device(device):

    metric_devices = [torch.device("cpu")]
    if torch.device(device).type != "xla":
        metric_devices.append(idist.device())
    for metric_device in metric_devices:

        ssim = SSIM(data_range=1.0, device=metric_device)

        for dev in [ssim._device, ssim._kernel.device]:
            assert dev == metric_device, f"{type(dev)}:{dev} vs {type(metric_device)}:{metric_device}"

        y_pred = torch.rand(2, 3, 28, 28, dtype=torch.float, device=device)
        y = y_pred * 0.65
        ssim.update((y_pred, y))

        dev = ssim._sum_of_ssim.device
        assert dev == metric_device, f"{type(dev)}:{dev} vs {type(metric_device)}:{metric_device}"


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_nccl_gpu(distributed_context_single_node_nccl):

    device = idist.device()
    _test_distrib_integration(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_gloo_cpu_or_gpu(distributed_context_single_node_gloo):

    device = idist.device()
    _test_distrib_integration(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gloo_cpu_or_gpu(distributed_context_multi_node_gloo):

    device = idist.device()
    _test_distrib_integration(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_nccl_gpu(distributed_context_multi_node_nccl):

    device = idist.device()
    _test_distrib_integration(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():
    device = idist.device()
    _test_distrib_integration(device, tol=1e-3)
    _test_distrib_accumulator_device(device)


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_distrib_integration(device, tol=1e-3)
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
