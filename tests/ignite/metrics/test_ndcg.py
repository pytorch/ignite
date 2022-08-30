import numpy as np
import pytest
import torch
from sklearn.metrics import ndcg_score
from sklearn.metrics._ranking import _dcg_sample_scores

from ignite.exceptions import NotComputableError
from ignite.metrics.recsys.ndcg import NDCG


@pytest.fixture(params=[item for item in range(2)])
def test_case(request):

    return [
        (torch.tensor([[3.7, 4.8, 3.9, 4.3, 4.9]]), torch.tensor([[2.9, 5.6, 3.8, 7.9, 6.2]])),
        (
            torch.tensor([[3.7, 3.7, 3.7, 3.7, 3.7], [3.7, 3.7, 3.7, 3.7, 3.9]]),
            torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]]),
        ),
    ][request.param]


@pytest.mark.parametrize("k", [None, 2, 3])
@pytest.mark.parametrize("exponential", [True, False])
@pytest.mark.parametrize("ignore_ties, replacement", [(True, False), (False, True), (False, False)])
def test_output_cpu(test_case, k, exponential, ignore_ties, replacement):

    device = "cpu"
    y_pred_distribution, y_true = test_case

    y_pred = torch.multinomial(y_pred_distribution, 5, replacement=replacement)

    ndcg = NDCG(k=k, device=device, exponential=exponential, ignore_ties=ignore_ties)
    ndcg.update([y_pred, y_true])
    result_ignite = ndcg.compute()

    if exponential:
        y_true = 2 ** y_true - 1

    result_sklearn = ndcg_score(y_true.numpy(), y_pred.numpy(), k=k, ignore_ties=ignore_ties)

    np.testing.assert_allclose(np.array(result_ignite), result_sklearn, rtol=2e-6)


@pytest.mark.parametrize("k", [None, 2, 3])
@pytest.mark.parametrize("exponential", [True, False])
@pytest.mark.parametrize("ignore_ties, replacement", [(True, False), (False, True), (False, False)])
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_output_cuda(test_case, k, exponential, ignore_ties, replacement):

    device = "cuda"
    y_pred_distribution, y_true = test_case

    y_pred = torch.multinomial(y_pred_distribution, 5, replacement=replacement)

    y_pred = y_pred.to(device)
    y_true = y_true.to(device)

    ndcg = NDCG(k=k, device=device, exponential=exponential, ignore_ties=ignore_ties)
    ndcg.update([y_pred, y_true])
    result_ignite = ndcg.compute()

    if exponential:
        y_true = 2 ** y_true - 1

    result_sklearn = ndcg_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), k=k, ignore_ties=ignore_ties)

    np.testing.assert_allclose(np.array(result_ignite), result_sklearn, rtol=2e-6)


def test_reset():

    y_true = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    y_pred = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    ndcg = NDCG()
    ndcg.update([y_pred, y_true])
    ndcg.reset()

    with pytest.raises(NotComputableError, match=r"NGCD must have at least one example before it can be computed."):
        ndcg.compute()


def _ndcg_sample_scores(y_true, y_score, k=None, ignore_ties=False):

    gain = _dcg_sample_scores(y_true, y_score, k, ignore_ties=ignore_ties)
    normalizing_gain = _dcg_sample_scores(y_true, y_true, k, ignore_ties=True)
    all_irrelevant = normalizing_gain == 0
    gain[all_irrelevant] = 0
    gain[~all_irrelevant] /= normalizing_gain[~all_irrelevant]
    return gain


@pytest.mark.parametrize("log_base", [2, 3, 10])
def test_log_base(log_base):
    def ndcg_score_with_log_base(y_true, y_score, *, k=None, sample_weight=None, ignore_ties=False, log_base=2):

        gain = _ndcg_sample_scores(y_true, y_score, k=k, ignore_ties=ignore_ties)
        return np.average(gain, weights=sample_weight)

    y_true = torch.tensor([[3.7, 4.8, 3.9, 4.3, 4.9]])
    y_pred = torch.tensor([[2.9, 5.6, 3.8, 7.9, 6.2]])

    ndcg = NDCG(log_base=log_base)
    ndcg.update([y_pred, y_true])

    result_ignite = ndcg.compute()
    result_sklearn = ndcg_score_with_log_base(y_true.numpy(), y_pred.numpy(), log_base=log_base)

    np.testing.assert_allclose(np.array(result_ignite), result_sklearn, rtol=2e-6)


def test_update(test_case):

    y_pred, y_true = test_case

    y_pred = y_pred
    y_true = y_true

    y1_pred = torch.multinomial(y_pred, 5, replacement=True)
    y1_true = torch.multinomial(y_true, 5, replacement=True)

    y2_pred = torch.multinomial(y_pred, 5, replacement=True)
    y2_true = torch.multinomial(y_true, 5, replacement=True)

    y_pred_combined = torch.cat((y1_pred, y2_pred))
    y_true_combined = torch.cat((y1_true, y2_true))

    ndcg = NDCG()

    ndcg.update([y1_pred, y1_true])
    ndcg.update([y2_pred, y2_true])

    result_ignite = ndcg.compute()

    result_sklearn = ndcg_score(y_true_combined.numpy(), y_pred_combined.numpy())

    np.testing.assert_allclose(np.array(result_ignite), result_sklearn, rtol=2e-6)
