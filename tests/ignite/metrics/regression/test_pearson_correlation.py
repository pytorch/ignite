from typing import Tuple

import numpy as np
import pytest
import torch
from scipy.stats import pearsonr
from torch import Tensor

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics.regression import PearsonCorrelation


def np_corr_eps(np_y_pred: np.ndarray, np_y: np.ndarray, eps: float = 1e-8):
    cov = np.cov(np_y_pred, np_y, ddof=0)[0, 1]
    std_y_pred = np.std(np_y_pred, ddof=0)
    std_y = np.std(np_y, ddof=0)
    corr = cov / np.clip(std_y_pred * std_y, eps, None)
    return corr


def scipy_corr(np_y_pred: np.ndarray, np_y: np.ndarray):
    corr = pearsonr(np_y_pred, np_y)
    return corr.statistic


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


def test_degenerated_sample(available_device):
    if available_device == "mps":
        pytest.skip(reason="PearsonCorrelation.compute returns nan on mps")
        # r = cov / torch.clamp(torch.sqrt(y_pred_var * y_var), min=self.eps)

    # one sample
    m = PearsonCorrelation(device=available_device)
    assert m._device == torch.device(available_device)
    y_pred = torch.tensor([1.0])
    y = torch.tensor([1.0])
    m.update((y_pred, y))

    np_y_pred = y_pred.cpu().numpy()
    np_y = y_pred.cpu().numpy()
    expected = np_corr_eps(np_y_pred, np_y)
    actual = m.compute()

    assert pytest.approx(expected) == actual

    # constant samples
    m.reset()
    y_pred = torch.ones(10).float()
    y = torch.zeros(10).float()
    m.update((y_pred, y))

    np_y_pred = y_pred.cpu().numpy()
    np_y = y_pred.cpu().numpy()
    expected = np_corr_eps(np_y_pred, np_y)
    actual = m.compute()

    assert pytest.approx(expected) == actual


def test_pearson_correlation(available_device):
    torch.manual_seed(1)

    inputs = [torch.randn(4) for _ in range(4)]
    ground_truth = torch.randn(4)

    m = PearsonCorrelation(device=available_device)
    assert m._device == torch.device(available_device)

    all_preds = []
    all_targets = []

    for i, pred in enumerate(inputs, 1):
        m.update((pred, ground_truth))
        all_preds.append(pred)
        all_targets.append(ground_truth)

        pred_concat = torch.cat(all_preds).cpu().numpy()
        target_concat = torch.cat(all_targets).cpu().numpy()
        expected = pearsonr(pred_concat, target_concat)[0]

        assert m.compute() == pytest.approx(expected, rel=1e-4)


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
def test_integration_pearson_correlation(n_times, test_case: Tuple[Tensor, Tensor, int], available_device):
    y_pred, y, batch_size = test_case

    def update_fn(engine: Engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        y_true_batch = np_y[idx : idx + batch_size]
        y_pred_batch = np_y_pred[idx : idx + batch_size]
        return torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

    engine = Engine(update_fn)

    m = PearsonCorrelation(device=available_device)
    assert m._device == torch.device(available_device)
    m.attach(engine, "corr")

    np_y = y.numpy().ravel()
    np_y_pred = y_pred.numpy().ravel()

    data = list(range(y_pred.shape[0] // batch_size))
    corr = engine.run(data, max_epochs=1).metrics["corr"]

    np_ans = scipy_corr(np_y_pred, np_y)

    assert pytest.approx(np_ans, rel=2e-4) == corr


def test_accumulator_detached(available_device):
    corr = PearsonCorrelation(device=available_device)
    assert corr._device == torch.device(available_device)

    y_pred = torch.tensor([2.0, 3.0], requires_grad=True)
    y = torch.tensor([-2.0, -1.0])
    corr.update((y_pred, y))

    assert all(
        (not accumulator.requires_grad)
        for accumulator in (
            corr._sum_of_products,
            corr._sum_of_y_pred_squares,
            corr._sum_of_y_preds,
            corr._sum_of_y_squares,
            corr._sum_of_ys,
        )
    )


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
            m = PearsonCorrelation(device=metric_device)

            y_pred = torch.rand(size=[100], device=device)
            y = torch.rand(size=[100], device=device)

            m.update((y_pred, y))

            y_pred = idist.all_gather(y_pred)
            y = idist.all_gather(y)

            np_y = y.cpu().numpy()
            np_y_pred = y_pred.cpu().numpy()

            np_ans = scipy_corr(np_y_pred, np_y)

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

            np_ans = scipy_corr(np_y_pred, np_y)

            assert pytest.approx(np_ans, rel=tol) == res

    def test_accumulator_device(self):
        device = idist.device()
        metric_devices = [torch.device("cpu")]
        if device.type != "xla":
            metric_devices.append(device)
        for metric_device in metric_devices:
            corr = PearsonCorrelation(device=metric_device)

            devices = (
                corr._device,
                corr._sum_of_products.device,
                corr._sum_of_y_pred_squares.device,
                corr._sum_of_y_preds.device,
                corr._sum_of_y_squares.device,
                corr._sum_of_ys.device,
            )
            for dev in devices:
                assert dev == metric_device, f"{type(dev)}:{dev} vs {type(metric_device)}:{metric_device}"

            y_pred = torch.tensor([2.0, 3.0])
            y = torch.tensor([-1.0, 1.0])
            corr.update((y_pred, y))

            devices = (
                corr._device,
                corr._sum_of_products.device,
                corr._sum_of_y_pred_squares.device,
                corr._sum_of_y_preds.device,
                corr._sum_of_y_squares.device,
                corr._sum_of_ys.device,
            )
            for dev in devices:
                assert dev == metric_device, f"{type(dev)}:{dev} vs {type(metric_device)}:{metric_device}"
