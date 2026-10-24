import pytest
import torch

from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics.rec_sys.mrr import MRR


def test_zero_sample():
    mrr = MRR(top_k=[1, 3])
    with pytest.raises(
        NotComputableError,
        match="MRR must have at least one valid example before it can be computed",
    ):
        mrr.compute()


def test_invalid_inputs():
    with pytest.raises(ValueError, match="top_k must be a non-empty sequence"):
        MRR(top_k=[])

    with pytest.raises(ValueError, match="top_k must contain only positive integers"):
        MRR(top_k=[0, -2])

    mrr = MRR(top_k=[2])
    with pytest.raises(ValueError, match="y_pred and y must be 2D tensors"):
        mrr.update((torch.tensor([1.0, 2.0]), torch.tensor([0.0, 1.0])))

    with pytest.raises(ValueError, match="y_pred and y must have the same shape"):
        mrr.update((torch.ones(2, 4), torch.ones(2, 3)))

    with pytest.raises(ValueError, match="y_pred must be a floating point tensor"):
        mrr.update((torch.ones(2, 4, dtype=torch.int64), torch.ones(2, 4)))


@pytest.mark.parametrize(
    "ignore_zero_hits, expected",
    [
        (True, [0.0, 0.5, 0.5]),
        (False, [0.0, 0.25, 0.25]),
    ],
)
def test_mrr_computation(ignore_zero_hits, expected):
    metric = MRR(top_k=[4, 1, 2], ignore_zero_hits=ignore_zero_hits)

    y_pred = torch.tensor(
        [
            [4.0, 2.0, 3.0, 1.0],  # rank of first hit (index 2) is 2 -> RR = 0.5
            [1.0, 2.0, 3.0, 4.0],  # no hits
        ]
    )
    y_true = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    metric.update((y_pred, y_true))
    res = metric.compute()
    assert pytest.approx(res) == expected


def test_mrr_with_engine():
    def process_fn(engine, batch):
        return batch

    engine = Engine(process_fn)
    mrr = MRR(top_k=[1, 3])
    mrr.attach(engine, "mrr")

    y_pred = torch.tensor([[0.1, 0.9, 0.4], [0.8, 0.2, 0.5]])
    y_true = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    state = engine.run([(y_pred, y_true)])
    assert pytest.approx(state.metrics["mrr"]) == [0.5, 0.75]
