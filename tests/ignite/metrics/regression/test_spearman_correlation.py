from typing import Tuple

import numpy as np
import pytest

import torch
from scipy.stats import spearmanr
from torch import Tensor

from ignite import distributed as idist
from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics.regression import SpearmanRankCorrelation


def test_zero_sample():
    with pytest.raises(
        NotComputableError, match="SpearmanRankCorrelation must have at least one example before it can be computed"
    ):
        metric = SpearmanRankCorrelation()
        metric.compute()


def test_wrong_y_pred_shape():
    with pytest.raises(ValueError, match=r"Input y_pred should have shape \(N,\) or \(N, 1\), but given"):
        metric = SpearmanRankCorrelation()
        y_pred = torch.arange(9).reshape(3, 3).float()
        y = torch.arange(3).unsqueeze(1).float()
        metric.update((y_pred, y))


def test_wrong_y_shape():
    with pytest.raises(ValueError, match=r"Input y should have shape \(N,\) or \(N, 1\), but given"):
        metric = SpearmanRankCorrelation()
        y_pred = torch.arange(3).unsqueeze(1).float()
        y = torch.arange(9).reshape(3, 3).float()
        metric.update((y_pred, y))


def test_wrong_y_pred_dtype():
    with pytest.raises(TypeError, match="Input y_pred dtype should be float 16, 32 or 64, but given"):
        metric = SpearmanRankCorrelation()
        y_pred = torch.arange(3).unsqueeze(1).long()
        y = torch.arange(3).unsqueeze(1).float()
        metric.update((y_pred, y))


def test_wrong_y_dtype():
    with pytest.raises(TypeError, match="Input y dtype should be float 16, 32 or 64, but given"):
        metric = SpearmanRankCorrelation()
        y_pred = torch.arange(3).unsqueeze(1).float()
        y = torch.arange(3).unsqueeze(1).long()
        metric.update((y_pred, y))


def test_spearman_correlation():
    a = np.random.randn(4).astype(np.float32)
    b = np.random.randn(4).astype(np.float32)
    c = np.random.randn(4).astype(np.float32)
    d = np.random.randn(4).astype(np.float32)
    ground_truth = np.random.randn(4).astype(np.float32)

    m = SpearmanRankCorrelation()

    m.update((torch.from_numpy(a), torch.from_numpy(ground_truth)))
    np_ans = spearmanr(a, ground_truth).statistic
    assert m.compute() == pytest.approx(np_ans, rel=1e-4)

    m.update((torch.from_numpy(b), torch.from_numpy(ground_truth)))
    np_ans = spearmanr(np.concatenate([a, b]), np.concatenate([ground_truth] * 2)).statistic
    assert m.compute() == pytest.approx(np_ans, rel=1e-4)

    m.update((torch.from_numpy(c), torch.from_numpy(ground_truth)))
    np_ans = spearmanr(np.concatenate([a, b, c]), np.concatenate([ground_truth] * 3)).statistic
    assert m.compute() == pytest.approx(np_ans, rel=1e-4)

    m.update((torch.from_numpy(d), torch.from_numpy(ground_truth)))
    np_ans = spearmanr(np.concatenate([a, b, c, d]), np.concatenate([ground_truth] * 4)).statistic
    assert m.compute() == pytest.approx(np_ans, rel=1e-4)


@pytest.fixture(params=list(range(2)))
def test_case(request):
    # correlated sample
    x = torch.randn(size=[50]).float()
    y = x + torch.randn_like(x) * 0.1

    return [
        (x, y, 1),
        (torch.rand(size=(50, 1)).float(), torch.rand(size=(50, 1)).float(), 10),
    ][request.param]


@pytest.mark.parametrize("n_times", range(5))
def test_integration(n_times, test_case: Tuple[Tensor, Tensor, int]):
    y_pred, y, batch_size = test_case

    np_y = y.numpy().ravel()
    np_y_pred = y_pred.numpy().ravel()

    def update_fn(engine: Engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        y_true_batch = np_y[idx : idx + batch_size]
        y_pred_batch = np_y_pred[idx : idx + batch_size]
        return torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

    engine = Engine(update_fn)

    m = SpearmanRankCorrelation()
    m.attach(engine, "spearman_corr")

    data = list(range(y_pred.shape[0] // batch_size))
    corr = engine.run(data, max_epochs=1).metrics["spearman_corr"]

    np_ans = spearmanr(np_y_pred, np_y).statistic

    assert pytest.approx(np_ans, rel=2e-4) == corr


@pytest.mark.usefixtures("distributed")
class TestDistributed:
    def test_compute(self):
        rank = idist.get_rank()
        device = idist.device()
        metric_devices = [torch.device("cpu")]
        if device.type != "xla":
            metric_devices.append(device)

        torch.manual_seed(10 + rank)
        for metric_device in metric_devices:
            m = SpearmanRankCorrelation(device=metric_device)

            y_pred = torch.rand(size=[100], device=device)
            y = torch.rand(size=[100], device=device)

            m.update((y_pred, y))

            y_pred = idist.all_gather(y_pred)
            y = idist.all_gather(y)

            np_y = y.cpu().numpy()
            np_y_pred = y_pred.cpu().numpy()

            np_ans = spearmanr(np_y_pred, np_y).statistic

            assert pytest.approx(np_ans, rel=2e-4) == m.compute()

    @pytest.mark.parametrize("n_epochs", [1, 2])
    def test_integration(self, n_epochs: int):
        tol = 2e-4
        rank = idist.get_rank()
        device = idist.device()
        metric_devices = [torch.device("cpu")]
        if device.type != "xla":
            metric_devices.append(device)

        n_iters = 80
        batch_size = 16

        for metric_device in metric_devices:
            torch.manual_seed(12 + rank)

            y_true = torch.rand(size=(n_iters * batch_size,)).to(device)
            y_preds = torch.rand(size=(n_iters * batch_size,)).to(device)

            engine = Engine(
                lambda e, i: (
                    y_preds[i * batch_size : (i + 1) * batch_size],
                    y_true[i * batch_size : (i + 1) * batch_size],
                )
            )

            corr = SpearmanRankCorrelation(device=metric_device)
            corr.attach(engine, "spearman_corr")

            data = list(range(n_iters))
            engine.run(data=data, max_epochs=n_epochs)

            y_preds = idist.all_gather(y_preds)
            y_true = idist.all_gather(y_true)

            assert "spearman_corr" in engine.state.metrics

            res = engine.state.metrics["spearman_corr"]

            np_y = y_true.cpu().numpy()
            np_y_pred = y_preds.cpu().numpy()

            np_ans = spearmanr(np_y_pred, np_y).statistic

            assert pytest.approx(np_ans, rel=tol) == res
