import numpy as np
import pytest
import torch

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics.rec_sys.mrr import MRR


def ranx_mrr(
    y_pred: np.ndarray,
    y: np.ndarray,
    top_k: list[int],
    ignore_zero_hits: bool = True,
) -> list[float]:
    """Reference MRR implementation using ranx for verification. https://github.com/AmenRa/ranx """
    from ranx import Qrels, Run, evaluate

    sorted_top_k = sorted(top_k)
    results = []

    for k in sorted_top_k:
        qrels_dict = {}
        run_dict = {}

        for i, (scores, labels) in enumerate(zip(y_pred, y)):
            qid = f"q{i}"
            relevant = {f"d{j}": int(label) for j, label in enumerate(labels) if label > 0}

            if ignore_zero_hits and not relevant:
                continue

            qrels_dict[qid] = relevant if relevant else {f"d0": 0}
            run_dict[qid] = {f"d{j}": float(s) for j, s in enumerate(scores)}

        if not qrels_dict:
            results.append(0.0)
            continue

        run_dict = {q: run_dict[q] for q in qrels_dict}
        results.append(float(evaluate(Qrels(qrels_dict), Run(run_dict), f"mrr@{k}")))

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
    with pytest.raises(ValueError, match="positive integers"):
        MRR(top_k=[0])
    with pytest.raises(ValueError, match="positive integers"):
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

    expected = ranx_mrr(
        y_pred.numpy(),
        y_true.numpy(),
        top_k,
        ignore_zero_hits=ignore_zero_hits,
    )

    assert isinstance(res, list)
    assert len(res) == len(top_k)
    np.testing.assert_allclose(res, expected)


@pytest.mark.parametrize("num_queries", [1, 10, 100])
@pytest.mark.parametrize("num_items", [5, 20, 100])
@pytest.mark.parametrize("k", [1, 5, 10])
@pytest.mark.parametrize("ignore_zero_hits", [True, False])
def test_compute_vs_ranx(num_queries, num_items, k, ignore_zero_hits, available_device):
    """Verify MRR matches ranx across a wide range of input shapes and k values."""
    torch.manual_seed(42)
    y_pred = torch.randn(num_queries, num_items)
    y_true = torch.randint(0, 2, (num_queries, num_items)).float()

    metric = MRR(
        top_k=[k],
        ignore_zero_hits=ignore_zero_hits,
        device=available_device,
    )
    metric.update((y_pred, y_true))

    try:
        res = metric.compute()
    except NotComputableError:
        res = [0.0]

    expected = ranx_mrr(
        y_pred.numpy(),
        y_true.numpy(),
        top_k=[k],
        ignore_zero_hits=ignore_zero_hits,
    )

    np.testing.assert_allclose(res, expected, rtol=1e-5)


def test_known_values():
    """Test with manually computed expected values."""
    metric = MRR(top_k=[1, 2, 3, 4])
    # y_pred=[4,2,3,1] -> rank order: doc0, doc2, doc1, doc3
    # y=[0,0,1,1]      -> relevance at ranked positions: [0,1,0,1]
    # MRR@1: no hit at rank 1 -> 0
    # MRR@2: first hit at rank 2 -> 1/2 = 0.5
    # MRR@3: first hit at rank 2 -> 1/2 = 0.5
    # MRR@4: first hit at rank 2 -> 1/2 = 0.5
    y_pred = torch.tensor([[4.0, 2.0, 3.0, 1.0]])
    y_true = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    metric.update((y_pred, y_true))
    assert metric.compute() == pytest.approx([0.0, 0.5, 0.5, 0.5])


def test_perfect_prediction():
    """Relevant item is top-ranked -> RR = 1.0."""
    metric = MRR(top_k=[1, 3])
    y_pred = torch.tensor([[5.0, 1.0, 2.0]])
    y_true = torch.tensor([[1.0, 0.0, 0.0]])
    metric.update((y_pred, y_true))
    assert metric.compute() == pytest.approx([1.0, 1.0])


def test_multiple_batches():
    """RR accumulates correctly across multiple update() calls."""
    metric = MRR(top_k=[2])
    # batch 1: relevant at rank 2 -> RR = 0.5
    metric.update((
        torch.tensor([[4.0, 2.0, 3.0, 1.0]]),
        torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
    ))
    # batch 2: relevant at rank 1 -> RR = 1.0
    metric.update((
        torch.tensor([[5.0, 1.0, 2.0, 3.0]]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    ))
    # MRR = (0.5 + 1.0) / 2 = 0.75
    assert metric.compute() == pytest.approx([0.75])


def test_graded_relevance():
    """Labels >= relevance_threshold are treated as relevant."""
    # relevance_threshold=2: label=1 is NOT relevant, label=2 IS relevant
    metric = MRR(top_k=[3], relevance_threshold=2.0)
    y_pred = torch.tensor([[0.9, 0.7, 0.3]])  # rank order: d0, d1, d2
    y_true = torch.tensor([[1.0, 0.0, 2.0]])  # only d2 (rank 3) is relevant
    metric.update((y_pred, y_true))
    assert metric.compute() == pytest.approx([1 / 3])


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

                true_res = ranx_mrr(
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
