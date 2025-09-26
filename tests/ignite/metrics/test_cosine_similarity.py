from typing import Tuple

import numpy as np
import pytest
import torch
from torch import Tensor

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics import CosineSimilarity


def test_zero_sample():
    cos_sim = CosineSimilarity()
    with pytest.raises(
        NotComputableError, match=r"CosineSimilarity must have at least one example before it can be computed"
    ):
        cos_sim.compute()


@pytest.fixture(params=list(range(4)))
def test_case(request):
    torch.manual_seed(0)  # For reproducibility

    eps = float(torch.empty(1).uniform_(-8, 0).exp())  # 10 ** uniform(-8, 0)

    return [
        (torch.randn((100, 50)), torch.randn((100, 50)), eps, 1),
        (torch.normal(1.0, 2.0, size=(100, 10)), torch.normal(3.0, 4.0, size=(100, 10)), eps, 1),
        (torch.rand((100, 128)), torch.rand((100, 128)), eps, 16),
        (torch.normal(0.0, 5.0, size=(100, 30)), torch.normal(5.0, 1.0, size=(100, 30)), eps, 16),
    ][request.param]


@pytest.mark.parametrize("n_times", range(5))
def test_compute(n_times, test_case: Tuple[Tensor, Tensor, float, int], available_device):
    y_pred, y, eps, batch_size = test_case

    cos = CosineSimilarity(eps=eps, device=available_device)
    assert cos._device == torch.device(available_device)

    cos.reset()
    if batch_size > 1:
        n_iters = y.shape[0] // batch_size + 1
        for i in range(n_iters):
            idx = i * batch_size
            cos.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))
    else:
        cos.update((y_pred, y))

    y_norm = torch.clamp(torch.norm(y, dim=1, keepdim=True), min=eps)
    y_pred_norm = torch.clamp(torch.norm(y_pred, dim=1, keepdim=True), min=eps)

    cosine_sim = torch.sum((y / y_norm) * (y_pred / y_pred_norm), dim=1)
    expected = cosine_sim.mean().item()

    result = cos.compute()

    assert isinstance(result, float)
    assert pytest.approx(expected, rel=2e-5) == result


def test_accumulator_detached(available_device):
    cos = CosineSimilarity(device=available_device)
    assert cos._device == torch.device(available_device)

    y_pred = torch.tensor([[2.0, 3.0], [-2.0, 1.0]], dtype=torch.float)
    y = torch.ones(2, 2, dtype=torch.float)
    cos.update((y_pred, y))

    assert not cos._sum_of_cos_similarities.requires_grad


@pytest.mark.usefixtures("distributed")
class TestDistributed:
    def test_integration(self):
        tol = 2e-5
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

            m = CosineSimilarity(device=metric_device)
            m.attach(engine, "cosine_similarity")

            data = list(range(n_iters))
            engine.run(data=data, max_epochs=1)

            y_preds = idist.all_gather(y_preds)
            y_true = idist.all_gather(y_true)

            assert "cosine_similarity" in engine.state.metrics
            res = engine.state.metrics["cosine_similarity"]

            y_true_np = y_true.cpu().numpy()
            y_preds_np = y_preds.cpu().numpy()
            y_true_norm = np.clip(np.linalg.norm(y_true_np, axis=1, keepdims=True), 1e-8, None)
            y_preds_norm = np.clip(np.linalg.norm(y_preds_np, axis=1, keepdims=True), 1e-8, None)
            true_res = np.sum((y_true_np / y_true_norm) * (y_preds_np / y_preds_norm), axis=1)
            true_res = np.mean(true_res)

            assert pytest.approx(res, rel=tol) == true_res

    def test_accumulator_device(self):
        device = idist.device()
        metric_devices = [torch.device("cpu")]
        if device.type != "xla":
            metric_devices.append(device)
        for metric_device in metric_devices:
            cos = CosineSimilarity(device=metric_device)

            for dev in (cos._device, cos._sum_of_cos_similarities.device):
                assert dev == metric_device, f"{type(dev)}:{dev} vs {type(metric_device)}:{metric_device}"

            y_pred = torch.tensor([[2.0, 3.0], [-2.0, 1.0]]).float()
            y = torch.ones(2, 2).float()
            cos.update((y_pred, y))

            for dev in (cos._device, cos._sum_of_cos_similarities.device):
                assert dev == metric_device, f"{type(dev)}:{dev} vs {type(metric_device)}:{metric_device}"
