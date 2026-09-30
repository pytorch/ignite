import numpy as np
import pytest
import torch
from skimage.metrics import peak_signal_noise_ratio as ski_psnr

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics import PSNR
from ignite.utils import manual_seed


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


@pytest.fixture(params=["float", "YCbCr", "uint8", "NHW shape"])
def test_data(request, available_device):
    manual_seed(42)
    if request.param == "float":
        y_pred = torch.rand(8, 3, 28, 28, device=available_device)
        y = y_pred * 0.8
    elif request.param == "YCbCr":
        y_pred = torch.randint(16, 236, (4, 1, 12, 12), dtype=torch.uint8, device=available_device)
        y = torch.randint(16, 236, (4, 1, 12, 12), dtype=torch.uint8, device=available_device)
    elif request.param == "uint8":
        y_pred = torch.randint(0, 256, (4, 3, 16, 16), dtype=torch.uint8, device=available_device)
        y = (y_pred * 0.8).to(torch.uint8)
    elif request.param == "NHW shape":
        y_pred = torch.rand(8, 28, 28, device=available_device)
        y = y_pred * 0.8
    else:
        raise ValueError(f"Wrong fixture parameter, given {request.param}")
    return (y_pred, y)


def test_psnr(test_data, available_device):
    y_pred, y = test_data
    data_range = (y.max() - y.min()).cpu().item()

    psnr = PSNR(data_range=data_range, device=available_device)
    assert psnr._device == torch.device(available_device)
    psnr.update(test_data)
    psnr_compute = psnr.compute()

    np_y_pred = y_pred.cpu().numpy()
    np_y = y.cpu().numpy()
    np_psnr = 0
    for np_y_pred_, np_y_ in zip(np_y_pred, np_y):
        np_psnr += ski_psnr(np_y_, np_y_pred_, data_range=data_range)

    assert psnr_compute > 0.0
    assert isinstance(psnr_compute, float)
    assert np.allclose(psnr_compute, np_psnr / np_y.shape[0])


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

    assert "psnr" in engine.state.metrics
    result = engine.state.metrics["psnr"]
    assert result > 0.0

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


@pytest.mark.usefixtures("distributed")
class TestDistributed:
    def test_input_float(self):
        device = idist.device()

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
            _test(y_pred, y, 1, "cpu", n_iters, batch_size, atol=1e-8)
            if device.type != "xla":
                _test(y_pred, y, 1, idist.device(), n_iters, batch_size, atol=1e-8)

    def test_multilabel_input_YCbCr(self):
        device = idist.device()

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
            _test(
                y_pred, y, 220, "cpu", n_iters, batch_size, atol=1e-8, output_transform=out_fn, compute_y_channel=True
            )
            if device.type != "xla":
                dev = idist.device()
                _test(
                    y_pred, y, 220, dev, n_iters, batch_size, atol=1e-8, output_transform=out_fn, compute_y_channel=True
                )

    def test_multilabel_input_uint8(self):
        device = idist.device()

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
            _test(y_pred, y, 100, "cpu", n_iters, batch_size, atol=1e-8)
            if device.type != "xla":
                _test(y_pred, y, 100, idist.device(), n_iters, batch_size, atol=1e-8)

    def test_multilabel_input_NHW(self):
        device = idist.device()

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
            _test(y_pred, y, 10, "cpu", n_iters, batch_size, atol=1e-8)
            if device.type != "xla":
                _test(y_pred, y, 10, idist.device(), n_iters, batch_size, atol=1e-8)

    def test_accumulator_device(self):
        device = idist.device()
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
