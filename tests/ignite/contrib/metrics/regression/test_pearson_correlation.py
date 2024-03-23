import os

import numpy as np
import pytest
import torch

import ignite.distributed as idist
from ignite.contrib.metrics.regression import PearsonCorrelation
from ignite.engine import Engine
from ignite.exceptions import NotComputableError


def np_corr(np_y_pred: np.ndarray, np_y: np.ndarray, eps: float = 1e-8):
    cov = np.cov(np_y_pred, np_y, ddof=0)[0, 1]
    std_y_pred = np.std(np_y_pred, ddof=0)
    std_y = np.std(np_y, ddof=0)
    corr = cov / np.clip(std_y_pred * std_y, eps, None)
    return corr


def test_zero_sample():
    m = PearsonCorrelation()
    with pytest.raises(
        NotComputableError, match=r"PearsonCorrelation must have at least one example before it can be computed"
    ):
        m.compute()


def test_wrong_input_shapes():
    m = PearsonCorrelation()

    with pytest.raises(ValueError, match=r"Input data shapes should be the same, but given"):
        m.update((torch.rand(4), torch.rand(4, 1)))

    with pytest.raises(ValueError, match=r"Input data shapes should be the same, but given"):
        m.update((torch.rand(4, 1), torch.rand(4)))


def test_pearson_correlation():
    a = np.random.randn(4).astype(np.float32)
    b = np.random.randn(4).astype(np.float32)
    c = np.random.randn(4).astype(np.float32)
    d = np.random.randn(4).astype(np.float32)
    ground_truth = np.random.randn(4).astype(np.float32)

    m = PearsonCorrelation()

    m.update((torch.from_numpy(a), torch.from_numpy(ground_truth)))
    np_ans = np_corr(a, ground_truth)
    assert m.compute() == pytest.approx(np_ans, rel=1e-4)

    m.update((torch.from_numpy(b), torch.from_numpy(ground_truth)))
    np_ans = np_corr(np.concatenate([a, b]), np.concatenate([ground_truth] * 2))
    assert m.compute() == pytest.approx(np_ans, rel=1e-4)

    m.update((torch.from_numpy(c), torch.from_numpy(ground_truth)))
    np_ans = np_corr(np.concatenate([a, b, c]), np.concatenate([ground_truth] * 3))
    assert m.compute() == pytest.approx(np_ans, rel=1e-4)

    m.update((torch.from_numpy(d), torch.from_numpy(ground_truth)))
    np_ans = np_corr(np.concatenate([a, b, c, d]), np.concatenate([ground_truth] * 4))
    assert m.compute() == pytest.approx(np_ans, rel=1e-4)


def test_integration():
    def _test(y_pred, y, eps, batch_size):
        def update_fn(engine, batch):
            idx = (engine.state.iteration - 1) * batch_size
            y_true_batch = np_y[idx : idx + batch_size]
            y_pred_batch = np_y_pred[idx : idx + batch_size]
            return torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

        engine = Engine(update_fn)

        m = PearsonCorrelation(eps=eps)
        m.attach(engine, "corr")

        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()

        data = list(range(y_pred.shape[0] // batch_size))
        corr = engine.run(data, max_epochs=1).metrics["corr"]

        np_ans = np_corr(np_y_pred, np_y, eps=eps)

        assert pytest.approx(np_ans, rel=1e-4) == corr

    def get_test_cases():
        test_cases = [
            (torch.rand(size=(50,)).float(), torch.rand(size=(50,)).float(), 10 ** np.random.normal(-8, 0), 1),
            (torch.rand(size=(50, 1)).float(), torch.rand(size=(50, 1)).float(), 10 ** np.random.normal(-8, 0), 10),
        ]
        return test_cases

    for _ in range(5):
        test_cases = get_test_cases()
        for y_pred, y, eps, batch_size in test_cases:
            _test(y_pred, y, eps, batch_size)


@pytest.mark.usefixtures("distributed")
class TestDistributed:
    def test_compute(self):
        rank = idist.get_rank()
        device = idist.device()

        def _test(metric_device):
            metric_device = torch.device(metric_device)
            m = PearsonCorrelation(device=metric_device)

            y_pred = torch.rand(size=(100,), device=device)
            y = torch.rand(size=(100,), device=device)

            m.update((y_pred, y))

            y_pred = idist.all_gather(y_pred)
            y = idist.all_gather(y)

            np_y = y.cpu().numpy()
            np_y_pred = y_pred.cpu().numpy()

            np_ans = np_corr(np_y_pred, np_y)

            assert pytest.approx(np_ans) == m.compute()

        for i in range(3):
            torch.manual_seed(10 + rank + i)
            _test("cpu")
            if device.type != "xla":
                _test(idist.device())

    def test_integration(self, tol=1e-5):
        rank = idist.get_rank()
        device = idist.device()

        def _test(n_epochs, metric_device):
            metric_device = torch.device(metric_device)
            n_iters = 80
            batch_size = 16

            y_true = torch.rand(size=(n_iters * batch_size,)).to(device)
            y_preds = torch.rand(size=(n_iters * batch_size,)).to(device)

            def update(engine, i):
                return (
                    y_preds[i * batch_size : (i + 1) * batch_size],
                    y_true[i * batch_size : (i + 1) * batch_size],
                )

            engine = Engine(update)

            corr = PearsonCorrelation(device=metric_device)
            corr.attach(engine, "corr")

            data = list(range(n_iters))
            engine.run(data=data, max_epochs=n_epochs)

            y_preds = idist.all_gather(y_preds)
            y_true = idist.all_gather(y_true)

            assert "corr" in engine.state.metrics

            res = engine.state.metrics["corr"]

            np_y = y_true.cpu().numpy()
            np_y_pred = y_preds.cpu().numpy()

            np_ans = np_corr(np_y_pred, np_y)

            assert pytest.approx(np_ans, rel=tol) == res

        metric_devices = ["cpu"]
        if device.type != "xla":
            metric_devices.append(idist.device())
        for metric_device in metric_devices:
            for i in range(2):
                torch.manual_seed(12 + rank + i)
                _test(n_epochs=1, metric_device=metric_device)
                _test(n_epochs=2, metric_device=metric_device)
