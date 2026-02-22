import pytest
import torch
import numpy as np

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics.rec_sys.mrr import MRR


def manual_mrr(
    y_pred: np.ndarray,
    y: np.ndarray,
    top_k: list[int],
    ignore_zero_hits: bool = True,
) -> list[float]:
    """Manual implementation of MRR using numpy for verification."""
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
        k_indices = sorted_indices[:, :k]
        rr_sum = 0.0
        for i in range(n_samples):
            relevance = y[i, k_indices[i]]
            hits = np.where(relevance > 0)[0]
            if len(hits) > 0:
                rr_sum += 1.0 / (hits[0] + 1)
        results.append(rr_sum / n_samples)

    return results


def test_zero_sample():
    metric = MRR(top_k=[1, 5])
    with pytest.raises(NotComputableError, match=r"MRR must have at least one example"):
        metric.compute()


def test_shape_mismatch():
    metric = MRR(top_k=[1])
    y_pred = torch.randn(4, 10)
    y = torch.ones(4, 5)  # Mismatched items count
    with pytest.raises(ValueError, match="y_pred and y must be in the same shape"):
        metric.update((y_pred, y))


def test_invalid_top_k():
    with pytest.raises(ValueError, match="top_k must be list of positive integers"):
        MRR(top_k=[0])
    with pytest.raises(ValueError, match="top_k must be list of positive integers"):
        MRR(top_k=[-1, 5])


@pytest.mark.parametrize("top_k", [[1], [1, 2, 4]])
@pytest.mark.parametrize("ignore_zero_hits", [True, False])
def test_compute(top_k, ignore_zero_hits, available_device):
    metric = MRR(
        top_k=top_k,
        ignore_zero_hits=ignore_zero_hits,
        device=available_device,
    )

    y_pred = torch.tensor([[4.0, 2.0, 3.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
    y_true = torch.tensor([[0, 0, 1.0, 1.0], [0, 0, 0.0, 0.0]])

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
    np.testing.assert_allclose(res, expected)


def test_known_values():
    """Test with manually computed expected values."""
    metric = MRR(top_k=[1, 2, 3, 4])

    # User 1: y_pred=[4,2,3,1] -> sorted indices [0,2,1,3]
    #          y=[0,0,1,1] -> relevance at sorted positions: [0,1,0,1]
    #   MRR@1: no hit -> 0
    #   MRR@2: first hit at position 2 -> 1/2 = 0.5
    #   MRR@3: first hit at position 2 -> 1/2 = 0.5
    #   MRR@4: first hit at position 2 -> 1/2 = 0.5
    y_pred = torch.tensor([[4.0, 2.0, 3.0, 1.0]])
    y_true = torch.tensor([[0.0, 0.0, 1.0, 1.0]])

    metric.update((y_pred, y_true))
    res = metric.compute()

    assert res == pytest.approx([0.0, 0.5, 0.5, 0.5])


def test_perfect_prediction():
    """Test when the most relevant item is the top prediction."""
    metric = MRR(top_k=[1, 3])

    y_pred = torch.tensor([[5.0, 1.0, 2.0]])
    y_true = torch.tensor([[1.0, 0.0, 0.0]])

    metric.update((y_pred, y_true))
    res = metric.compute()

    assert res == pytest.approx([1.0, 1.0])


def test_multiple_batches():
    """Test accumulation across multiple update calls."""
    metric = MRR(top_k=[2])

    # Batch 1: hit at position 2 -> RR = 0.5
    y_pred1 = torch.tensor([[4.0, 2.0, 3.0, 1.0]])
    y_true1 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    metric.update((y_pred1, y_true1))

    # Batch 2: hit at position 1 -> RR = 1.0
    y_pred2 = torch.tensor([[5.0, 1.0, 2.0, 3.0]])
    y_true2 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    metric.update((y_pred2, y_true2))

    res = metric.compute()
    # MRR = (0.5 + 1.0) / 2 = 0.75
    assert res == pytest.approx([0.75])


def test_accumulator_detached(available_device):
    metric = MRR(top_k=[1], device=available_device)
    y_pred = torch.randn(4, 5, requires_grad=True)
    y = torch.randint(0, 2, (4, 5)).float()
    metric.update((y_pred, y))

    assert metric._sum_reciprocal_ranks_per_k.requires_grad is False
    assert metric._sum_reciprocal_ranks_per_k.is_leaf is True


def test_all_zero_targets_ignore():
    metric = MRR(top_k=[1, 3], ignore_zero_hits=True)

    y_pred = torch.randn(4, 5)
    y = torch.zeros(4, 5)

    metric.update((y_pred, y))

    with pytest.raises(NotComputableError):
        metric.compute()


@pytest.mark.usefixtures("distributed")
class TestDistributed:
    def test_integration(self):
        n_iters = 10
        batch_size = 4
        num_items = 20
        top_k = [1, 5]

        rank = idist.get_rank()
        torch.manual_seed(42 + rank)
        device = idist.device()

        metric_devices = [torch.device("cpu")]
        if device.type != "xla":
            metric_devices.append(device)

        for metric_device in metric_devices:
            all_y_true = torch.randint(0, 2, (n_iters * batch_size, num_items)).float().to(device)
            all_y_pred = torch.randn((n_iters * batch_size, num_items)).to(device)

            for ignore_zero_hits in [True, False]:
                engine = Engine(
                    lambda e, i: (
                        all_y_pred[i * batch_size : (i + 1) * batch_size],
                        all_y_true[i * batch_size : (i + 1) * batch_size],
                    )
                )
                m = MRR(
                    top_k=top_k,
                    ignore_zero_hits=ignore_zero_hits,
                    device=metric_device,
                )
                m.attach(engine, "mrr")

                engine.run(range(n_iters), max_epochs=1)

                global_y_true = idist.all_gather(all_y_true).cpu().numpy()
                global_y_pred = idist.all_gather(all_y_pred).cpu().numpy()

                res = engine.state.metrics["mrr"]

                true_res = manual_mrr(
                    global_y_pred,
                    global_y_true,
                    top_k,
                    ignore_zero_hits=ignore_zero_hits,
                )

                assert isinstance(res, list)
                assert res == pytest.approx(true_res)

                engine.state.metrics.clear()

    def test_accumulator_device(self):
        device = idist.device()
        metric = MRR(top_k=[1, 5], device=device)

        assert metric._device == device
        assert metric._sum_reciprocal_ranks_per_k.device == device

        y_pred = torch.randn(2, 10)
        y = torch.zeros(2, 10)
        metric.update((y_pred, y))

        assert metric._sum_reciprocal_ranks_per_k.device == device
