import numpy as np
import pytest
import torch

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics.rec_sys.ndcg import NDCG


def ranx_ndcg(
    y_pred: np.ndarray,
    y: np.ndarray,
    top_k: list[int],
    ignore_zero_hits: bool = True,
) -> list[float]:
    """Reference NDCG implementation using ranx for verification. https://github.com/AmenRa/ranx """
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
        results.append(float(evaluate(Qrels(qrels_dict), Run(run_dict), f"ndcg@{k}")))

    return results


def test_zero_sample():
    metric = NDCG(top_k=[1, 5])
    with pytest.raises(NotComputableError, match=r"NDCG must have at least one example"):
        metric.compute()


def test_shape_mismatch():
    metric = NDCG(top_k=[1])
    y_pred = torch.randn(4, 10)
    y = torch.ones(4, 5)  # Mismatched items count
    with pytest.raises(ValueError, match="y_pred and y must be in the same shape"):
        metric.update((y_pred, y))


def test_invalid_top_k():
    with pytest.raises(ValueError, match="positive integers"):
        NDCG(top_k=[0])
    with pytest.raises(ValueError, match="positive integers"):
        NDCG(top_k=[-1, 5])


@pytest.mark.parametrize("top_k", [[1], [1, 2, 4]])
@pytest.mark.parametrize("ignore_zero_hits", [True, False])
def test_compute(top_k, ignore_zero_hits, available_device):
    metric = NDCG(
        top_k=top_k,
        ignore_zero_hits=ignore_zero_hits,
        device=available_device,
    )

    y_pred = torch.tensor([[4.0, 2.0, 3.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
    y_true = torch.tensor([[0, 0, 1.0, 1.0], [0, 0, 0.0, 0.0]])

    metric.update((y_pred, y_true))
    res = metric.compute()

    expected = ranx_ndcg(
        y_pred.numpy(),
        y_true.numpy(),
        top_k,
        ignore_zero_hits=ignore_zero_hits,
    )

    assert isinstance(res, list)
    assert len(res) == len(top_k)
    np.testing.assert_allclose(res, expected, rtol=1e-5)


@pytest.mark.parametrize("num_queries", [10, 100])
@pytest.mark.parametrize("num_items", [20, 100])
@pytest.mark.parametrize("k", [1, 5])
@pytest.mark.parametrize("ignore_zero_hits", [True, False])
def test_compute_vs_ranx(num_queries, num_items, k, ignore_zero_hits, available_device):
    """Verify NDCG matches ranx across a wide range of input shapes and k values."""
    torch.manual_seed(42)
    y_pred = torch.randn(num_queries, num_items)
    y_true = torch.randint(0, 2, (num_queries, num_items)).float()

    metric = NDCG(
        top_k=[k],
        ignore_zero_hits=ignore_zero_hits,
        device=available_device,
    )
    metric.update((y_pred, y_true))

    try:
        res = metric.compute()
    except NotComputableError:
        res = [0.0]

    expected = ranx_ndcg(
        y_pred.numpy(),
        y_true.numpy(),
        top_k=[k],
        ignore_zero_hits=ignore_zero_hits,
    )

    np.testing.assert_allclose(res, expected, rtol=1e-5)


def test_perfect_prediction():
    """Perfect ranking -> NDCG = 1.0."""
    metric = NDCG(top_k=[1, 3])
    y_pred = torch.tensor([[5.0, 3.0, 4.0, 1.0]])
    y_true = torch.tensor([[3.0, 1.0, 2.0, 0.0]])  # Matches ranking order
    metric.update((y_pred, y_true))
    assert metric.compute() == pytest.approx([1.0, 1.0])


def test_all_zeros_relevance():
    """When all relevance is 0, IDCG=0, so NDCG should be 0 (or ignored if ignore_zero_hits=True)."""
    metric = NDCG(top_k=[2], ignore_zero_hits=False)
    y_pred = torch.tensor([[5.0, 3.0, 4.0]])
    y_true = torch.tensor([[0.0, 0.0, 0.0]])
    metric.update((y_pred, y_true))
    # NDCG should be 0 when there are no relevant items
    assert metric.compute() == pytest.approx([0.0])


def test_graded_relevance_threshold():
    """Labels >= relevance_threshold are considered, but contribute their full value to DCG."""
    # relevance_threshold=2: labels < 2 are zeroed out
    metric = NDCG(top_k=[3], relevance_threshold=2.0)
    
    # Predictions rank: doc0, doc2, doc1
    # True relevance: [3, 1, 2] -> After threshold: [3, 0, 2]
    # Ranked relevance (after threshold): [3, 2, 0]
    y_pred = torch.tensor([[0.9, 0.3, 0.7]])
    y_true = torch.tensor([[3.0, 1.0, 2.0]])
    metric.update((y_pred, y_true))
    
    # DCG:  (2^3-1)/log2(2) + (2^2-1)/log2(3) + (2^0-1)/log2(4)
    #     = 7/1 + 3/1.585 + 0/2 = 7 + 1.893 = 8.893
    # IDCG: (2^3-1)/log2(2) + (2^2-1)/log2(3) + (2^0-1)/log2(4)
    #     = 7/1 + 3/1.585 + 0/2 = 8.893
    # NDCG = 1.0 (perfect ranking of items that meet threshold)
    
    result = metric.compute()
    assert result[0] == pytest.approx(1.0, rel=1e-5)


@pytest.mark.usefixtures("distributed")
class TestDistributed:
    def test_integration(self):
        n_iters = 10
        batch_size = 4
        num_items = 20
        top_k = [1, 5, 10]

        rank = idist.get_rank()
        torch.manual_seed(42 + rank)
        device = idist.device()

        metric_devices = [torch.device("cpu")]
        if device.type != "xla":
            metric_devices.append(device)

        for metric_device in metric_devices:
            all_y_true = torch.randint(0, 4, (n_iters * batch_size, num_items)).float().to(device)
            all_y_pred = torch.randn((n_iters * batch_size, num_items)).to(device)

            for ignore_zero_hits in [True, False]:
                engine = Engine(
                    lambda e, i: (
                        all_y_pred[i * batch_size : (i + 1) * batch_size],
                        all_y_true[i * batch_size : (i + 1) * batch_size],
                    )
                )
                m = NDCG(
                    top_k=top_k,
                    ignore_zero_hits=ignore_zero_hits,
                    device=metric_device,
                )
                m.attach(engine, "ndcg")

                engine.run(range(n_iters), max_epochs=1)

                global_y_true = idist.all_gather(all_y_true).cpu().numpy()
                global_y_pred = idist.all_gather(all_y_pred).cpu().numpy()

                res = engine.state.metrics["ndcg"]

                true_res = ranx_ndcg(
                    global_y_pred,
                    global_y_true,
                    top_k,
                    ignore_zero_hits=ignore_zero_hits,
                )

                assert isinstance(res, list)
                np.testing.assert_allclose(res, true_res, rtol=1e-5)

                engine.state.metrics.clear()

    def test_accumulator_device(self):
        device = idist.device()
        metric = NDCG(top_k=[1, 5], device=device)

        assert metric._device == device
        assert metric._sum_ndcg_per_k.device == device

        y_pred = torch.randn(2, 10)
        y = torch.zeros(2, 10)
        metric.update((y_pred, y))

        assert metric._sum_ndcg_per_k.device == device
