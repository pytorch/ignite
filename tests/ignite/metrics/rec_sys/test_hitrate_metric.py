import pytest
import torch
import numpy as np

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics.rec_sys.hitrate import HitRate
from ranx import Qrels, Run, evaluate


def ranx_hit_rate(
    y_pred: np.ndarray,
    y: np.ndarray,
    top_k: list[int],
    ignore_zero_hits: bool = True,
) -> list[float]:
    """Reference HitRate implementation using ranx for verification. https://github.com/AmenRa/ranx"""

    sorted_top_k = sorted(top_k)
    results = []

    for k in sorted_top_k:
        qrels_dict = {}
        run_dict = {}

        for i, (scores, labels) in enumerate(zip(y_pred, y)):
            qid = f"q{i}"
            relevant = {f"d{j}": 1 for j, label in enumerate(labels) if label > 0}

            if ignore_zero_hits and not relevant:
                continue

            qrels_dict[qid] = relevant if relevant else {"d0": 0}
            run_dict[qid] = {f"d{j}": float(s) for j, s in enumerate(scores)}

        results.append(float(evaluate(Qrels(qrels_dict), Run(run_dict), f"hit_rate@{k}")))
    return results


def test_zero_sample():
    metric = HitRate(top_k=[1, 5])
    with pytest.raises(NotComputableError, match=r"HitRate must have at least one example"):
        metric.compute()


def test_shape_mismatch():
    metric = HitRate(top_k=[1])
    y_pred = torch.randn(4, 10)
    y = torch.ones(4, 5)  # Mismatched items count
    with pytest.raises(ValueError, match="y_pred and y must be in the same shape"):
        metric.update((y_pred, y))


def test_empty_top_k():
    with pytest.raises(ValueError, match="top_k must have at least one positive value"):
        HitRate(top_k=[])


def test_invalid_top_k_type():
    with pytest.raises(ValueError, match="top_k must be either int or a list"):
        HitRate(top_k="invalid")


def test_int_top_k(available_device):
    metric = HitRate(top_k=2, device=available_device)
    y_pred = torch.tensor([[4.0, 2.0, 3.0, 1.0]])
    y_true = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    metric.update((y_pred, y_true))
    res = metric.compute()
    expected = ranx_hit_rate(y_pred.numpy(), y_true.numpy(), [2])
    np.testing.assert_allclose(res, expected)


@pytest.mark.parametrize("top_k", [[1], [1, 2, 4, 10]])
@pytest.mark.parametrize("ignore_zero_hits", [True, False])
def test_compute(top_k, ignore_zero_hits, available_device):
    metric = HitRate(
        top_k=top_k,
        ignore_zero_hits=ignore_zero_hits,
        device=available_device,
    )

    y_pred = torch.tensor([[4.0, 2.0, 3.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
    y_true = torch.tensor([[0, 0, 1.0, 1.0], [0, 0, 0.0, 0.0]])

    metric.update((y_pred, y_true))
    res = metric.compute()

    expected = ranx_hit_rate(
        y_pred.numpy(),
        y_true.numpy(),
        top_k,
        ignore_zero_hits=ignore_zero_hits,
    )

    assert isinstance(res, list)
    assert len(res) == len(top_k)
    np.testing.assert_allclose(res, expected)


def test_accumulator_detached(available_device):
    metric = HitRate(top_k=[1], device=available_device)
    y_pred = torch.randn(4, 5, requires_grad=True)
    y = torch.randint(0, 2, (4, 5)).float()
    metric.update((y_pred, y))

    assert metric._hits_per_k.requires_grad is False
    assert metric._hits_per_k.is_leaf is True


def test_all_zero_targets_ignore():
    metric = HitRate(top_k=[1, 3], ignore_zero_hits=True)

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
                m = HitRate(
                    top_k=top_k,
                    ignore_zero_hits=ignore_zero_hits,
                    device=metric_device,
                )
                m.attach(engine, "hitrate")

                engine.run(range(n_iters), max_epochs=1)

                global_y_true = idist.all_gather(all_y_true).cpu().numpy()
                global_y_pred = idist.all_gather(all_y_pred).cpu().numpy()

                res = engine.state.metrics["hitrate"]

                true_res = ranx_hit_rate(
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
        metric = HitRate(top_k=[1, 5], device=device)

        assert metric._device == device
        assert metric._hits_per_k.device == device

        y_pred = torch.randn(2, 10)
        y = torch.zeros(2, 10)
        metric.update((y_pred, y))

        assert metric._hits_per_k.device == device
