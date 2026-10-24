import pytest
import torch

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics.rec_sys import MRR


def test_zero_sample():
    mrr = MRR(top_k=[1, 3])
    with pytest.raises(
        NotComputableError,
        match="MRR must have at least one valid example before it can be computed",
    ):
        mrr.compute()


def test_invalid_inputs():
    with pytest.raises(ValueError, match="top_k must be either int or a list"):
        MRR(top_k={})

    with pytest.raises(ValueError, match="top_k must have at least one positive value"):
        MRR(top_k=[])

    with pytest.raises(ValueError, match="top_k must be list of positive integers only"):
        MRR(top_k=[0, -2])

    mrr = MRR(top_k=[2])
    # Shape mismatch
    with pytest.raises(ValueError, match="y_pred and y must be in the same shape"):
        mrr.update((torch.ones(2, 4), torch.ones(2, 3)))

    # Output format validation
    with pytest.raises(ValueError, match="output should be in format"):
        mrr.update((torch.ones(2, 4),))


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
            [4.0, 2.0, 3.0, 1.0],  # sorted item indices: [0, 2, 1, 3] -> first hit is index 2 at rank 2 (RR = 0.5)
            [1.0, 2.0, 3.0, 4.0],  # no relevant items in y_true
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


def test_int_top_k(available_device):
    metric = MRR(top_k=2, device=available_device)
    y_pred = torch.tensor([[4.0, 2.0, 3.0, 1.0]])
    y_true = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    metric.update((y_pred, y_true))
    res = metric.compute()
    # sorted: [0, 2, 1, 3], hit at rank 2 -> RR = 0.5
    assert pytest.approx(res) == [0.5]


def test_mrr_with_engine(available_device):
    def process_fn(engine, batch):
        return batch

    engine = Engine(process_fn)
    mrr = MRR(top_k=[1, 3], device=available_device)
    mrr.attach(engine, "mrr")

    y_pred = torch.tensor([[0.1, 0.9, 0.4], [0.8, 0.2, 0.5]])
    y_true = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    state = engine.run([(y_pred, y_true)])
    # User 0: topk=[1, 2, 0], hit at rank 1 -> RR@1 = 1.0, RR@3 = 1.0
    # User 1: topk=[0, 2, 1], hit at rank 2 -> RR@1 = 0.0, RR@3 = 0.5
    assert pytest.approx(state.metrics["mrr"]) == [0.5, 0.75]


def _test_distrib_integration(device):
    rank = idist.get_rank()
    torch.manual_seed(12 + rank)

    metric_devices = [torch.device("cpu")]
    if device.type != "xla":
        metric_devices.append(idist.device())

    for metric_device in metric_devices:
        mrr = MRR(top_k=[1, 3, 5], device=metric_device)

        y_pred = torch.rand(16, 10, device=device)
        y_true = torch.randint(0, 2, size=(16, 10), device=device).float()

        mrr.update((y_pred, y_true))

        # Gather all tensors across ranks to verify against non-distributed calculation
        y_pred_all = idist.all_gather(y_pred)
        y_true_all = idist.all_gather(y_true)

        res_distrib = mrr.compute()

        ref_mrr = MRR(top_k=[1, 3, 5], device="cpu")
        ref_mrr.update((y_pred_all.cpu(), y_true_all.cpu()))
        res_ref = ref_mrr.compute()

        assert pytest.approx(res_distrib) == res_ref


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_nccl(distributed_context_single_node_nccl):
    device = idist.device()
    _test_distrib_integration(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_gloo_cpu_or_gpu(distributed_context_single_node_gloo):
    device = idist.device()
    _test_distrib_integration(device)
