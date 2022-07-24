import numpy as np
import pytest
import torch
from sklearn.metrics import ndcg_score

from ignite.exceptions import NotComputableError
from ignite.metrics.recsys.ndcg import NDCG


@pytest.mark.parametrize(
    "true_score, pred_score",
    [
        (torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]), torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])),
        (torch.tensor([[2.9, 5.6, 3.8, 7.9, 6.2]]), torch.tensor([[3.7, 4.8, 3.9, 4.3, 4.9]])),
        (
            torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [1.2, 4.5, 8.9, 5.6, 7.2], [2.9, 5.6, 3.8, 7.9, 6.2]]),
            torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [3.7, 4.8, 3.9, 4.3, 4.9], [3.7, 4.8, 3.9, 4.3, 4.9]]),
        ),
    ],
)
@pytest.mark.parametrize("k", [None, 2, 3])
def test_output_cpu(true_score, pred_score, k):

    device = "cpu"

    ndcg = NDCG(k=k, device=device)
    ndcg.update([true_score, pred_score])
    result_ignite = ndcg.compute()
    result_sklearn = ndcg_score(true_score.numpy(), pred_score.numpy(), k=k)

    np.testing.assert_allclose(np.array(result_ignite), result_sklearn, rtol=2e-7)


@pytest.mark.parametrize(
    "true_score, pred_score",
    [
        (torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]), torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])),
        (torch.tensor([[2.9, 5.6, 3.8, 7.9, 6.2]]), torch.tensor([[3.7, 4.8, 3.9, 4.3, 4.9]])),
        (
            torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [1.2, 4.5, 8.9, 5.6, 7.2], [2.9, 5.6, 3.8, 7.9, 6.2]]),
            torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [3.7, 4.8, 3.9, 4.3, 4.9], [3.7, 4.8, 3.9, 4.3, 4.9]]),
        ),
    ],
)
@pytest.mark.parametrize("k", [None, 2, 3])
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_output_gpu(true_score, pred_score, k):

    device = "cuda"
    ndcg = NDCG(k=k, device=device)
    ndcg.update([true_score, pred_score])
    result_ignite = ndcg.compute()
    result_sklearn = ndcg_score(true_score.numpy(), pred_score.numpy(), k=k)

    np.testing.assert_allclose(np.array(result_ignite), result_sklearn, rtol=2e-7)


def test_reset():

    true_score = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    pred_score = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    ndcg = NDCG()
    ndcg.update([true_score, pred_score])
    ndcg.reset()

    with pytest.raises(NotComputableError, match=r"NGCD must have at least one example before it can be computed."):
        ndcg.compute()
