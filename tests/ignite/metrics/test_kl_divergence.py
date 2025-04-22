from typing import Tuple

import numpy as np
import pytest
import torch
from scipy.special import softmax
from scipy.stats import entropy
from torch import Tensor

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics import KLDivergence


def scipy_kl_div(np_y_pred: np.ndarray, np_y: np.ndarray) -> float:
    y_pred_prob = softmax(np_y_pred, axis=1)
    y_prob = softmax(np_y, axis=1)
    kl_mean = entropy(y_prob, y_pred_prob, axis=1).mean()
    return kl_mean


def test_zero_sample():
    kl_div = KLDivergence()
    with pytest.raises(
        NotComputableError, match=r"KLDivergence must have at least one example before it can be computed"
    ):
        kl_div.compute()


def test_shape_mismatch():
    kl_div = KLDivergence()
    y_pred = torch.tensor([[2.0, 3.0], [-2.0, 1.0]], dtype=torch.float)
    y = torch.tensor([[-2.0, 1.0]], dtype=torch.float)
    with pytest.raises(ValueError, match=r"y_pred and y must be in the same shape, got"):
        kl_div.update((y_pred, y))


def test_invalid_shape():
    kl_div = KLDivergence()
    y_pred = torch.tensor([2.0, 3.0], dtype=torch.float)
    y = torch.tensor([4.0, 5.0], dtype=torch.float)
    with pytest.raises(ValueError, match=r"y_pred must be in the shape of \(B, C\) or \(B, C, ...\), got"):
        kl_div.update((y_pred, y))


@pytest.fixture(params=list(range(4)))
def test_case(request):
    return [
        (torch.randn((100, 10)), torch.rand((100, 10)), 1),
        (torch.rand((100, 500)), torch.randn((100, 500)), 1),
        # updated batches
        (torch.normal(0.0, 5.0, size=(100, 10)), torch.rand((100, 10)), 16),
        (torch.normal(5.0, 3.0, size=(100, 200)), torch.rand((100, 200)), 16),
        # image segmentation
        (torch.randn((100, 5, 32, 32)), torch.rand((100, 5, 32, 32)), 16),
        (torch.rand((100, 5, 224, 224)), torch.randn((100, 5, 224, 224)), 16),
    ][request.param]


@pytest.mark.parametrize("n_times", range(5))
def test_compute(n_times, test_case: Tuple[Tensor, Tensor, int], available_device):
    y_pred, y, batch_size = test_case

    kl_div = KLDivergence(device=available_device)
    assert kl_div._device == torch.device(available_device)

    kl_div.reset()
    if batch_size > 1:
        n_iters = y.shape[0] // batch_size + 1
        for i in range(n_iters):
            idx = i * batch_size
            kl_div.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))
    else:
        kl_div.update((y_pred, y))

    res = kl_div.compute()

    np_y_pred = y_pred.numpy()
    np_y = y.numpy()

    np_res = scipy_kl_div(np_y_pred, np_y)

    assert isinstance(res, float)
    assert pytest.approx(np_res, rel=1e-4) == res


def test_accumulator_detached(available_device):
    kl_div = KLDivergence(device=available_device)
    assert kl_div._device == torch.device(available_device)

    y_pred = torch.tensor([[2.0, 3.0], [-2.0, 1.0]], dtype=torch.float)
    y = torch.tensor([[-2.0, 1.0], [2.0, 3.0]], dtype=torch.float)
    kl_div.update((y_pred, y))

    assert not kl_div._sum_of_kl.requires_grad


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
            y_true = torch.randn((n_iters * batch_size, n_dims)).float().to(device)
            y_preds = torch.normal(2.0, 3.0, size=(n_iters * batch_size, n_dims)).float().to(device)

            engine = Engine(
                lambda e, i: (
                    y_preds[i * batch_size : (i + 1) * batch_size],
                    y_true[i * batch_size : (i + 1) * batch_size],
                )
            )

            m = KLDivergence(device=metric_device)
            m.attach(engine, "kl_div")

            data = list(range(n_iters))
            engine.run(data=data, max_epochs=1)

            y_preds = idist.all_gather(y_preds)
            y_true = idist.all_gather(y_true)

            assert "kl_div" in engine.state.metrics
            res = engine.state.metrics["kl_div"]

            y_true_np = y_true.cpu().numpy()
            y_preds_np = y_preds.cpu().numpy()
            true_res = scipy_kl_div(y_preds_np, y_true_np)

            assert pytest.approx(true_res, rel=tol) == res

    def test_accumulator_device(self):
        device = idist.device()
        metric_devices = [torch.device("cpu")]
        if device.type != "xla":
            metric_devices.append(device)
        for metric_device in metric_devices:
            kl_div = KLDivergence(device=metric_device)

            for dev in (kl_div._device, kl_div._sum_of_kl.device):
                assert dev == metric_device, f"{type(dev)}:{dev} vs {type(metric_device)}:{metric_device}"

            y_pred = torch.tensor([[2.0, 3.0], [-2.0, 1.0]]).float()
            y = torch.ones(2, 2).float()
            kl_div.update((y_pred, y))

            for dev in (kl_div._device, kl_div._sum_of_kl.device):
                assert dev == metric_device, f"{type(dev)}:{dev} vs {type(metric_device)}:{metric_device}"
