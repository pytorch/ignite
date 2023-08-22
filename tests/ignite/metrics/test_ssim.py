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
def test_ssim(available_device, shape, kernel_size, gaussian, use_sample_covariance, dtype=torch.float32, precision=7e-5):
    y_pred = torch.rand(shape, device=available_device, dtype=dtype)
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

    assert isinstance(ignite_ssim, float)
    assert np.allclose(ignite_ssim, skimg_ssim, atol=precision)


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
    assert np.allclose(out, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_cuda_ssim_dtypes(available_device, dtype):
    # Checks https://github.com/pytorch/ignite/pull/3034
    # this test should not be run on CPU
    if available_device == "cpu" and dtype == torch.float16:
        pytest.skip(reason="Unsupported dtype float16 on CPU device")

    test_ssim(
        available_device,
        (12, 3, 28, 28),
        11,
        True,
        False,
        dtype=dtype,
        precision=4e-4
    )


@pytest.mark.parametrize("metric_device", ["cpu", "process_device"])
def test_distrib_integration(distributed, metric_device):
    from ignite.engine import Engine

    rank = idist.get_rank()
    torch.manual_seed(12 + rank)
    n_iters = 100
    batch_size = 10
    device = idist.device()
    if metric_device == "process_device":
        metric_device = device if device.type != "xla" else "cpu"

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

    tol = 1e-3 if device.type == "xla" else 1e-4  # Isn't better to ask `distributed` about backend info?

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


@pytest.mark.parametrize("metric_device", [torch.device("cpu"), "process_device"])
def test_distrib_accumulator_device(distributed, metric_device):
    device = idist.device()
    if metric_device == "process_device":
        metric_device = torch.device(device if device.type != "xla" else "cpu")

    ssim = SSIM(data_range=1.0, device=metric_device)

    for dev in [ssim._device, ssim._kernel.device]:
        assert dev == metric_device, f"{type(dev)}:{dev} vs {type(metric_device)}:{metric_device}"

    y_pred = torch.rand(2, 3, 28, 28, dtype=torch.float, device=device)
    y = y_pred * 0.65
    ssim.update((y_pred, y))

    dev = ssim._sum_of_ssim.device
    assert dev == metric_device, f"{type(dev)}:{dev} vs {type(metric_device)}:{metric_device}"
