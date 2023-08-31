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


def _test_distrib_output(device):

    rank = idist.get_rank()

    def _test(n_epochs, metric_device):

        metric_device = torch.device(metric_device)

        n_iters = 5
        batch_size = 8
        n_items = 5

        torch.manual_seed(12 + rank)

        y_true = torch.rand((n_iters * batch_size, n_items)).to(device)
        y_preds = torch.rand((n_iters * batch_size, n_items)).to(device)

        def update(_, i):
            return (
                [v for v in y_preds[i * batch_size : (i + 1) * batch_size, ...]],
                [v for v in y_true[i * batch_size : (i + 1) * batch_size]],
            )

        engine = Engine(update)

        ndcg = NDCG(device=metric_device)
        ndcg.attach(engine, "ndcg")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        y_true = idist.all_gather(y_true)
        y_preds = idist.all_gather(y_preds)

        assert (
            ndcg._device == metric_device
        ), f"{type(ndcg._device)}:{ndcg._device} vs {type(metric_device)}:{metric_device}"

        assert "ndcg" in engine.state.metrics
        res = engine.state.metrics["ndcg"]
        if isinstance(res, torch.Tensor):
            res = res.cpu().numpy()

        true_res = ndcg_score(y_true.cpu().numpy(), y_preds.cpu().numpy())
        assert pytest.approx(res) == true_res

    metric_devices = ["cpu"]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for metric_device in metric_devices:
        for _ in range(2):
            _test(n_epochs=1, metric_device=metric_device)
            _test(n_epochs=2, metric_device=metric_device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gloo_cpu_or_gpu(distributed_context_multi_node_gloo):

    device = idist.device()
    _test_distrib_output(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_nccl_gpu(distributed_context_multi_node_nccl):

    device = idist.device()
    _test_distrib_output(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_nccl_gpu(distributed_context_single_node_nccl):

    device = idist.device()
    _test_distrib_output(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_gloo_cpu_or_gpu(distributed_context_single_node_gloo):

    device = idist.device()
    _test_distrib_output(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_distrib_output, (device,), np=nproc, do_init=True)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():

    device = idist.device()
    _test_distrib_output(device)


def _test_distrib_xla_nprocs(index):

    device = idist.device()
    _test_distrib_output(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)