import math
from typing import Tuple

import numpy as np
import pytest

import torch
from torch import nn, Tensor

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics import HSIC


def np_hsic(x: Tensor, y: Tensor, sigma_x: float = -1, sigma_y: float = -1) -> float:
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    b = x_np.shape[0]

    ii, jj = np.meshgrid(np.arange(b), np.arange(b), indexing="ij")
    mask = 1.0 - np.eye(b)

    dxx = np.square(x_np[ii] - x_np[jj]).sum(axis=2)
    if sigma_x < 0:
        vx = np.median(dxx)
    else:
        vx = sigma_x * sigma_x
    K = np.exp(-0.5 * dxx / vx) * mask

    dyy = np.square(y_np[ii] - y_np[jj]).sum(axis=2)
    if sigma_y < 0:
        vy = np.median(dyy)
    else:
        vy = sigma_y * sigma_y
    L = np.exp(-0.5 * dyy / vy) * mask

    KL = K @ L
    ones = np.ones(b)
    hsic = np.trace(KL) + (ones @ K @ ones) * (ones @ L @ ones) / ((b - 1) * (b - 2)) - ones @ KL @ ones * 2 / (b - 2)
    hsic /= b * (b - 3)
    hsic = np.clip(hsic, 0.0, None)
    return hsic


def test_zero_batch():
    hsic = HSIC()
    with pytest.raises(NotComputableError, match=r"HSIC must have at least one batch before it can be computed"):
        hsic.compute()


def test_invalid_batch():
    hsic = HSIC(ignore_invalid_batch=False)
    X = torch.tensor([[1, 2, 3]]).float()
    Y = torch.tensor([[4, 5, 6]]).float()
    with pytest.raises(ValueError, match=r"A batch must contain more than four samples, got only"):
        hsic.update((X, Y))


@pytest.fixture(params=[0, 1, 2])
def test_case(request) -> Tuple[Tensor, Tensor, int]:
    if request.param == 0:
        # independent
        N = 100
        b = 10
        x, y = torch.randn((N, 50)), torch.randn((N, 30))
    elif request.param == 1:
        # linearly dependent
        N = 100
        b = 10
        x = torch.normal(1.0, 2.0, size=(N, 10))
        y = x @ torch.rand(10, 15) * 3 + torch.randn(N, 15) * 1e-4
    else:
        # non-linearly dependent
        N = 200
        b = 20
        x = torch.randn(N, 5)
        y = x @ torch.normal(0.0, math.pi, size=(5, 3))
        y = (
            torch.stack([torch.sin(y[:, 0]), torch.cos(y[:, 1]), torch.exp(y[:, 2]) / 10], dim=1)
            + torch.randn_like(y) * 1e-4
        )

    return x, y, b


@pytest.mark.parametrize("n_times", range(3))
@pytest.mark.parametrize("sigma_x", [-1.0, 1.0])
@pytest.mark.parametrize("sigma_y", [-1.0, 1.0])
def test_compute(n_times, sigma_x: float, sigma_y: float, test_case: Tuple[Tensor, Tensor, int], available_device):
    x, y, batch_size = test_case

    hsic = HSIC(sigma_x=sigma_x, sigma_y=sigma_y, device=available_device)
    assert hsic._device == torch.device(available_device)

    hsic.reset()

    np_hsic_sum = 0.0
    n_iters = y.shape[0] // batch_size
    for i in range(n_iters):
        idx = i * batch_size
        x_batch = x[idx : idx + batch_size]
        y_batch = y[idx : idx + batch_size]

        hsic.update((x_batch, y_batch))
        np_hsic_sum += np_hsic(x_batch, y_batch, sigma_x, sigma_y)
    expected_hsic = np_hsic_sum / n_iters

    assert isinstance(hsic.compute(), float)
    assert pytest.approx(expected_hsic, abs=2e-5) == hsic.compute()


def test_accumulator_detached(available_device):
    hsic = HSIC(device=available_device)
    assert hsic._device == torch.device(available_device)

    x = torch.rand(10, 10, dtype=torch.float)
    y = torch.rand(10, 10, dtype=torch.float)
    hsic.update((x, y))

    assert not hsic._sum_of_hsic.requires_grad


@pytest.mark.usefixtures("distributed")
class TestDistributed:
    @pytest.mark.parametrize("sigma_x", [-1.0, 1.0])
    @pytest.mark.parametrize("sigma_y", [-1.0, 1.0])
    def test_integration(self, sigma_x: float, sigma_y: float):
        tol = 2e-5
        n_iters = 100
        batch_size = 20
        n_dims_x = 100
        n_dims_y = 50

        rank = idist.get_rank()
        torch.manual_seed(12 + rank)

        device = idist.device()
        metric_devices = [torch.device("cpu")]
        if device.type != "xla":
            metric_devices.append(device)

        for metric_device in metric_devices:
            x = torch.randn((n_iters * batch_size, n_dims_x), device=device).float()

            lin = nn.Linear(n_dims_x, n_dims_y).to(device)
            y = torch.sin(lin(x) * 100) + torch.randn(n_iters * batch_size, n_dims_y, device=x.device) * 1e-4

            def data_loader(i, input_x, input_y):
                return input_x[i * batch_size : (i + 1) * batch_size], input_y[i * batch_size : (i + 1) * batch_size]

            engine = Engine(lambda e, i: data_loader(i, x, y))

            m = HSIC(sigma_x=sigma_x, sigma_y=sigma_y, device=metric_device)
            m.attach(engine, "hsic")

            data = list(range(n_iters))
            engine.run(data=data, max_epochs=1)

            assert "hsic" in engine.state.metrics
            res = engine.state.metrics["hsic"]

            x = idist.all_gather(x)
            y = idist.all_gather(y)
            total_n_iters = idist.all_reduce(n_iters)

            np_res = 0.0
            for i in range(total_n_iters):
                x_batch, y_batch = data_loader(i, x, y)
                np_res += np_hsic(x_batch, y_batch, sigma_x, sigma_y)

            expected_hsic = np_res / total_n_iters
            assert pytest.approx(expected_hsic, abs=tol) == res

    def test_accumulator_device(self):
        device = idist.device()
        metric_devices = [torch.device("cpu")]
        if device.type != "xla":
            metric_devices.append(device)
        for metric_device in metric_devices:
            hsic = HSIC(device=metric_device)

            for dev in (hsic._device, hsic._sum_of_hsic.device):
                assert dev == metric_device, f"{type(dev)}:{dev} vs {type(metric_device)}:{metric_device}"

            x = torch.zeros(10, 10).float()
            y = torch.ones(10, 10).float()
            hsic.update((x, y))

            for dev in (hsic._device, hsic._sum_of_hsic.device):
                assert dev == metric_device, f"{type(dev)}:{dev} vs {type(metric_device)}:{metric_device}"
