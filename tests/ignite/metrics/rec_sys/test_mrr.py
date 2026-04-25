import numpy as np
import pytest
import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.rec_sys.mrr import MRR


def manual_mrr(
    y_pred: np.ndarray,
    y: np.ndarray,
    top_k: list[int],
    ignore_zero_hits: bool = True,
) -> list[float]:
    """Reference MRR implementation in numpy for verification."""
    sorted_top_k = sorted(top_k)

    if ignore_zero_hits:
        valid_mask = np.any(y > 0, axis=-1)
        y_pred = y_pred[valid_mask]
        y = y[valid_mask]

    n_samples = y.shape[0]
    if n_samples == 0:
        raise ValueError("No valid samples for manual MRR computation.")

    sorted_indices = np.argsort(-y_pred, axis=-1)

    results = []
    for k in sorted_top_k:
        rr_sum = 0.0
        for i in range(n_samples):
            top_k_relevance = y[i, sorted_indices[i, :k]]
            relevant_positions = np.where(top_k_relevance > 0)[0]
            if relevant_positions.size > 0:
                rr_sum += 1.0 / (relevant_positions[0] + 1)
        results.append(rr_sum / n_samples)

    return results


def test_zero_sample():
    metric = MRR(top_k=[1, 5])
    with pytest.raises(NotComputableError, match=r"MRR must have at least one example"):
        metric.compute()


def test_shape_mismatch():
    metric = MRR(top_k=[1])
    y_pred = torch.randn(4, 10)
    y = torch.ones(4, 5)
    with pytest.raises(ValueError, match="y_pred and y must be in the same shape"):
        metric.update((y_pred, y))


def test_empty_top_k():
    with pytest.raises(ValueError, match="top_k must have at least one positive value"):
        MRR(top_k=[])


def test_invalid_top_k_type():
    with pytest.raises(ValueError, match="top_k must be either int or a list"):
        MRR(top_k="invalid")


def test_negative_top_k():
    with pytest.raises(ValueError, match="top_k must be list of positive integers only"):
        MRR(top_k=[1, -2])


def test_int_top_k():
    metric = MRR(top_k=2)
    y_pred = torch.tensor([[4.0, 2.0, 3.0, 1.0]])
    y_true = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    metric.update((y_pred, y_true))
    res = metric.compute()
    expected = manual_mrr(y_pred.numpy(), y_true.numpy(), [2])
    np.testing.assert_allclose(res, expected)


def test_bad_output_length():
    metric = MRR(top_k=[1])
    with pytest.raises(ValueError, match="output should be in format"):
        metric.update((torch.randn(2, 4),))


@pytest.mark.parametrize("top_k", [[1], [1, 2, 4], [3, 5]])
@pytest.mark.parametrize("ignore_zero_hits", [True, False])
def test_compute(top_k, ignore_zero_hits):
    metric = MRR(top_k=top_k, ignore_zero_hits=ignore_zero_hits)

    y_pred = torch.tensor(
        [
            [4.0, 2.0, 3.0, 1.0, 0.5],
            [1.0, 2.0, 3.0, 4.0, 0.0],
            [0.5, 0.5, 0.5, 0.5, 0.5],
        ]
    )
    y_true = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )

    metric.update((y_pred, y_true))
    res = metric.compute()

    expected = manual_mrr(
        y_pred.numpy(),
        y_true.numpy(),
        top_k,
        ignore_zero_hits=ignore_zero_hits,
    )

    assert isinstance(res, list)
    assert len(res) == len(top_k)
    np.testing.assert_allclose(res, expected, rtol=1e-5)


def test_perfect_ranking():
    # First relevant always at rank 1 -> MRR = 1.0 at any k
    metric = MRR(top_k=[1, 3])
    y_pred = torch.tensor([[10.0, 1.0, 2.0, 3.0]])
    y_true = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    metric.update((y_pred, y_true))
    res = metric.compute()
    np.testing.assert_allclose(res, [1.0, 1.0])


def test_no_hits_in_top_k():
    # Only relevant item is the lowest-scored one; with top_k=2, no hit -> MRR=0
    # but ignore_zero_hits is False so it counts as 0.
    metric = MRR(top_k=[2], ignore_zero_hits=False)
    y_pred = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    y_true = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    metric.update((y_pred, y_true))
    res = metric.compute()
    np.testing.assert_allclose(res, [0.0])


def test_all_zero_y_ignored():
    metric = MRR(top_k=[2], ignore_zero_hits=True)
    y_pred = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    y_true = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    # First update has no valid sample
    metric.update((y_pred, y_true))
    # Second update has a valid sample
    y_pred2 = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    y_true2 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    metric.update((y_pred2, y_true2))
    res = metric.compute()
    np.testing.assert_allclose(res, [1.0])


def test_reset():
    metric = MRR(top_k=[1])
    y_pred = torch.tensor([[4.0, 2.0, 3.0, 1.0]])
    y_true = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    metric.update((y_pred, y_true))
    metric.reset()
    with pytest.raises(NotComputableError):
        metric.compute()


def test_multiple_updates():
    metric = MRR(top_k=[1, 3])
    all_y_pred = []
    all_y = []

    torch.manual_seed(0)
    for _ in range(4):
        y_pred = torch.rand(5, 8)
        y = (torch.rand(5, 8) > 0.5).float()
        metric.update((y_pred, y))
        all_y_pred.append(y_pred.numpy())
        all_y.append(y.numpy())

    res = metric.compute()
    expected = manual_mrr(
        np.concatenate(all_y_pred, axis=0),
        np.concatenate(all_y, axis=0),
        [1, 3],
    )
    np.testing.assert_allclose(res, expected, rtol=1e-5)
