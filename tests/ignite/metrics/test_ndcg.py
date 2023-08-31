import os

import numpy as np
import pytest
import torch
from sklearn.metrics import ndcg_score
from sklearn.metrics._ranking import _dcg_sample_scores

import ignite.distributed as idist
from ignite.engine import Engine

from ignite.exceptions import NotComputableError
from ignite.metrics.recsys.ndcg import NDCG


@pytest.fixture(params=[item for item in range(6)])
def test_case(request):
    return [
        (torch.tensor([[3.7, 4.8, 3.9, 4.3, 4.9]]), torch.tensor([[2.9, 5.6, 3.8, 7.9, 6.2]])),
        (
            torch.tensor([[3.7, 3.7, 3.7, 3.7, 3.7], [3.7, 3.7, 3.7, 3.7, 3.9]]),
            torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]]),
        ),
    ][request.param % 2]


@pytest.mark.parametrize("k", [None, 2, 3])
@pytest.mark.parametrize("exponential", [True, False])
@pytest.mark.parametrize("ignore_ties, replacement", [(True, False), (False, True), (False, False)])
def test_output(available_device, test_case, k, exponential, ignore_ties, replacement):
    device = available_device
    y_pred_distribution, y = test_case

    y_pred = torch.multinomial(y_pred_distribution, 5, replacement=replacement)

    y_pred = y_pred.to(device)
    y = y.to(device)

    ndcg = NDCG(k=k, device=device, exponential=exponential, ignore_ties=ignore_ties)
    ndcg.update([y_pred, y])
    result_ignite = ndcg.compute()

    if exponential:
        y = 2**y - 1

    result_sklearn = ndcg_score(y.cpu().numpy(), y_pred.cpu().numpy(), k=k, ignore_ties=ignore_ties)

    np.testing.assert_allclose(np.array(result_ignite), result_sklearn, rtol=2e-6)


def test_reset():
    y = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    y_pred = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    ndcg = NDCG()
    ndcg.update([y_pred, y])
    ndcg.reset()

    with pytest.raises(NotComputableError, match=r"NGCD must have at least one example before it can be computed."):
        ndcg.compute()


def _ndcg_sample_scores(y, y_score, k=None, ignore_ties=False):
    gain = _dcg_sample_scores(y, y_score, k, ignore_ties=ignore_ties)
    normalizing_gain = _dcg_sample_scores(y, y, k, ignore_ties=True)
    all_irrelevant = normalizing_gain == 0
    gain[all_irrelevant] = 0
    gain[~all_irrelevant] /= normalizing_gain[~all_irrelevant]
    return gain


@pytest.mark.parametrize("log_base", [2, 3, 10])
def test_log_base(log_base):
    def ndcg_score_with_log_base(y, y_score, *, k=None, sample_weight=None, ignore_ties=False, log_base=2):
        gain = _ndcg_sample_scores(y, y_score, k=k, ignore_ties=ignore_ties)
        return np.average(gain, weights=sample_weight)

    y = torch.tensor([[3.7, 4.8, 3.9, 4.3, 4.9]])
    y_pred = torch.tensor([[2.9, 5.6, 3.8, 7.9, 6.2]])

    ndcg = NDCG(log_base=log_base)
    ndcg.update([y_pred, y])

    result_ignite = ndcg.compute()
    result_sklearn = ndcg_score_with_log_base(y.numpy(), y_pred.numpy(), log_base=log_base)

    np.testing.assert_allclose(np.array(result_ignite), result_sklearn, rtol=2e-6)


def test_update(test_case):
    y_pred, y = test_case

    y_pred = y_pred
    y = y

    y1_pred = torch.multinomial(y_pred, 5, replacement=True)
    y1_true = torch.multinomial(y, 5, replacement=True)

    y2_pred = torch.multinomial(y_pred, 5, replacement=True)
    y2_true = torch.multinomial(y, 5, replacement=True)

    y_pred_combined = torch.cat((y1_pred, y2_pred))
    y_combined = torch.cat((y1_true, y2_true))

    ndcg = NDCG()

    ndcg.update([y1_pred, y1_true])
    ndcg.update([y2_pred, y2_true])

    result_ignite = ndcg.compute()

    result_sklearn = ndcg_score(y_combined.numpy(), y_pred_combined.numpy())

    np.testing.assert_allclose(np.array(result_ignite), result_sklearn, rtol=2e-6)


@pytest.mark.parametrize("metric_device", ["cpu", "process_device"])
@pytest.mark.parametrize("num_epochs", [1, 2])
def test_distrib_integration(distributed, num_epochs, metric_device):
    from ignite.engine import Engine

    rank = idist.get_rank()
    torch.manual_seed(12 + rank)
    n_iters = 5
    batch_size = 8
    device = idist.device()
    if metric_device == "process_device":
        metric_device = device if device.type != "xla" else "cpu"

    # 10 items
    y = torch.rand((n_iters * batch_size, 10)).to(device)
    y_preds = torch.rand((n_iters * batch_size, 10)).to(device)

    def update(engine, i):
        return (
            y_preds[i * batch_size : (i + 1) * batch_size, ...],
            y[i * batch_size : (i + 1) * batch_size, ...],
        )

    engine = Engine(update)
    NDCG(device=metric_device).attach(engine, "ndcg")

    data = list(range(n_iters))
    engine.run(data=data, max_epochs=num_epochs)

    y_preds = idist.all_gather(y_preds)
    y = idist.all_gather(y)

    assert "ndcg" in engine.state.metrics
    res = engine.state.metrics["ndcg"]

    true_res = ndcg_score(y.cpu().numpy(), y_preds.cpu().numpy())

    tol = 1e-3 if device.type == "xla" else 1e-4  # Isn't better to ask `distributed` about backend info?

    assert pytest.approx(res, abs=tol) == true_res


@pytest.mark.parametrize("metric_device", [torch.device("cpu"), "process_device"])
def test_distrib_accumulator_device(distributed, metric_device):
    device = idist.device()
    if metric_device == "process_device":
        metric_device = torch.device(device if device.type != "xla" else "cpu")

    ndcg = NDCG(device=metric_device)

    assert ndcg._device == metric_device, f"{type(dev)}:{dev} vs {type(metric_device)}:{metric_device}"

    y_pred = torch.rand((2, 10)).to(device)
    y = torch.rand((2, 10)).to(device)
    ndcg.update((y_pred, y))

    dev = ndcg.ndcg.device
    assert dev == metric_device, f"{type(dev)}:{dev} vs {type(metric_device)}:{metric_device}"
