from typing import Tuple

import numpy as np
import pytest
import torch
from torch import Tensor

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics import MaximumMeanDiscrepancy


def np_mmd(x: np.ndarray, y: np.ndarray, var: float = 1.0):
    n = x.shape[0]
    x = x.reshape(n, -1)
    y = y.reshape(n, -1)

    a = np.arange(n)
    ii, jj = np.meshgrid(a, a, indexing="ij")
    XX = np.exp(-np.square(x[ii] - x[jj]).sum(axis=2) / (var * 2))
    XX = (np.sum(XX) - n) / (n * (n - 1))

    XY = np.exp(-np.square(x[ii] - y[jj]).sum(axis=2) / (var * 2))
    XY = np.sum(XY) / (n * n)

    YY = np.exp(-np.square(y[ii] - y[jj]).sum(axis=2) / (var * 2))
    YY = (np.sum(YY) - n) / (n * (n - 1))

    mmd2 = np.clip(XX + YY - XY * 2, 0.0, None)

    return np.sqrt(mmd2)


def test_zero_sample():
    mmd = MaximumMeanDiscrepancy()
    with pytest.raises(
        NotComputableError, match=r"MaximumMeanDiscrepacy must have at least one batch before it can be computed"
    ):
        mmd.compute()


def test_shape_mismatch():
    mmd = MaximumMeanDiscrepancy()
    x = torch.tensor([[2.0, 3.0], [-2.0, 1.0]], dtype=torch.float)
    y = torch.tensor([[-2.0, 1.0]], dtype=torch.float)
    with pytest.raises(ValueError, match=r"x and y must be in the same shape, got"):
        mmd.update((x, y))


def test_invalid_shape():
    mmd = MaximumMeanDiscrepancy()
    x = torch.tensor([2.0, 3.0], dtype=torch.float)
    y = torch.tensor([4.0, 5.0], dtype=torch.float)
    with pytest.raises(ValueError, match=r"x must be in the shape of \(B, ...\), got"):
        mmd.update((x, y))


@pytest.fixture(params=list(range(4)))
def test_case(request):
    return [
        (torch.randn((100, 10)), torch.rand((100, 10)), 10 ** np.random.uniform(-1.0, 0.0), 1),
        (torch.rand((100, 500)), torch.randn((100, 500)), 10 ** np.random.uniform(-1.0, 0.0), 1),
        # updated batches
        (torch.normal(0.0, 5.0, size=(100, 10)), torch.rand((100, 10)), 10 ** np.random.uniform(-1.0, 0.0), 16),
        (torch.normal(5.0, 3.0, size=(100, 200)), torch.rand((100, 200)), 10 ** np.random.uniform(-1.0, 0.0), 16),
        # image segmentation
        (torch.randn((100, 5, 32, 32)), torch.rand((100, 5, 32, 32)), 10 ** np.random.uniform(-1.0, 0.0), 32),
        (torch.rand((100, 5, 224, 224)), torch.randn((100, 5, 224, 224)), 10 ** np.random.uniform(-1.0, 0.0), 32),
    ][request.param]


@pytest.mark.parametrize("n_times", range(5))
def test_compute(n_times, test_case: Tuple[Tensor, Tensor, float, int]):
    x, y, var, batch_size = test_case

    mmd = MaximumMeanDiscrepancy(var=var)
    mmd.reset()

    if batch_size > 1:
        np_mmd_sum = 0.0
        n_iters = y.shape[0] // batch_size + 1
        for i in range(n_iters):
            idx = i * batch_size
            x_batch, y_batch = x[idx : idx + batch_size], y[idx : idx + batch_size]
            mmd.update((x_batch, y_batch))

            np_mmd_sum += np_mmd(x_batch.cpu().numpy(), y_batch.cpu().numpy(), var)

        np_res = np_mmd_sum / n_iters
    else:
        mmd.update((x, y))
        np_res = np_mmd(x.cpu().numpy(), y.cpu().numpy(), var)

    res = mmd.compute()

    assert isinstance(res, float)
    assert pytest.approx(np_res, abs=1e-4) == res


def test_accumulator_detached():
    mmd = MaximumMeanDiscrepancy()

    x = torch.tensor([[2.0, 3.0], [-2.0, 1.0]], dtype=torch.float)
    y = torch.tensor([[-2.0, 1.0], [2.0, 3.0]], dtype=torch.float)
    mmd.update((x, y))

    assert not mmd._sum_of_mmd.requires_grad


@pytest.mark.usefixtures("distributed")
class TestDistributed:
    def test_integration(self):
        tol = 1e-4
        n_iters = 100
        batch_size = 10
        n_dims = 100

        rank = idist.get_rank()
        torch.manual_seed(12 + rank)

        device = idist.device()
        metric_devices = [torch.device("cpu")]
        if device.type != "xla":
            metric_devices.append(device)

        for metric_device in metric_devices:
            y = torch.randn((n_iters * batch_size, n_dims)).float().to(device)
            x = torch.normal(2.0, 3.0, size=(n_iters * batch_size, n_dims)).float().to(device)

            def data_loader(i):
                return x[i * batch_size : (i + 1) * batch_size], y[i * batch_size : (i + 1) * batch_size]

            engine = Engine(lambda e, i: data_loader(i))

            m = MaximumMeanDiscrepancy(device=metric_device)
            m.attach(engine, "mmd")

            data = list(range(n_iters))
            engine.run(data=data, max_epochs=1)

            x = idist.all_gather(x)
            y = idist.all_gather(y)

            assert "mmd" in engine.state.metrics
            res = engine.state.metrics["mmd"]

            # compute numpy mmd
            true_res = 0.0
            for i in range(n_iters):
                x_batch, y_batch = data_loader(i)
                x_np = x_batch.cpu().numpy()
                y_np = y_batch.cpu().numpy()
                true_res += np_mmd(x_np, y_np)

            true_res /= n_iters
            assert pytest.approx(true_res, abs=tol) == res

    def test_accumulator_device(self):
        device = idist.device()
        metric_devices = [torch.device("cpu")]
        if device.type != "xla":
            metric_devices.append(device)
        for metric_device in metric_devices:
            mmd = MaximumMeanDiscrepancy(device=metric_device)

            for dev in (mmd._device, mmd._sum_of_mmd.device):
                assert dev == metric_device, f"{type(dev)}:{dev} vs {type(metric_device)}:{metric_device}"

            x = torch.tensor([[2.0, 3.0], [-2.0, 1.0]]).float()
            y = torch.ones(2, 2).float()
            mmd.update((x, y))

            for dev in (mmd._device, mmd._sum_of_mmd.device):
                assert dev == metric_device, f"{type(dev)}:{dev} vs {type(metric_device)}:{metric_device}"
