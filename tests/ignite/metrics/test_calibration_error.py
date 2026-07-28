import numpy as np
import pytest
import torch
from scipy.special import softmax

import ignite.distributed as idist

from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics import ExpectedCalibrationError, MaximumCalibrationError


def _bin_stats(conf: np.ndarray, correct: np.ndarray, num_bins: int):
    """Reference per-bin aggregation mirroring the Baal binning (floor(conf * num_bins))."""
    bin_idx = np.clip((conf * num_bins).astype(np.int64), 0, num_bins - 1)
    count = np.zeros(num_bins)
    conf_sum = np.zeros(num_bins)
    correct_sum = np.zeros(num_bins)
    np.add.at(count, bin_idx, 1.0)
    np.add.at(conf_sum, bin_idx, conf)
    np.add.at(correct_sum, bin_idx, correct.astype(np.float64))
    nonempty = count > 0
    acc = np.zeros(num_bins)
    avg_conf = np.zeros(num_bins)
    acc[nonempty] = correct_sum[nonempty] / count[nonempty]
    avg_conf[nonempty] = conf_sum[nonempty] / count[nonempty]
    return count, np.abs(acc - avg_conf), nonempty


def np_ece(conf: np.ndarray, correct: np.ndarray, num_bins: int) -> float:
    count, gap, _ = _bin_stats(conf, correct, num_bins)
    return float((count / count.sum() * gap).sum())


def np_mce(conf: np.ndarray, correct: np.ndarray, num_bins: int) -> float:
    count, gap, nonempty = _bin_stats(conf, correct, num_bins)
    return float(gap[nonempty].max())


def _conf_correct(np_y_pred: np.ndarray, np_y: np.ndarray):
    """Extract per-sample confidence and correctness from probabilities, matching the metric."""
    if np_y_pred.ndim == np_y.ndim + 1 and np_y_pred.shape[1] >= 2:
        probs = np_y_pred.transpose(0, *range(2, np_y_pred.ndim), 1).reshape(-1, np_y_pred.shape[1])
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        y = np_y.reshape(-1)
    else:
        p = np_y_pred.reshape(-1)
        conf = np.maximum(p, 1.0 - p)
        pred = (p >= 0.5).astype(np.int64)
        y = np_y.reshape(-1)
    return conf, (pred == y)


def test_zero_sample():
    for cls, name in [
        (ExpectedCalibrationError, "ExpectedCalibrationError"),
        (MaximumCalibrationError, "MaximumCalibrationError"),
    ]:
        metric = cls()
        with pytest.raises(
            NotComputableError, match=rf"{name} must have at least one example before it can be computed"
        ):
            metric.compute()


@pytest.mark.parametrize("num_bins", [0, -3, 2.5, "10"])
def test_invalid_num_bins(num_bins):
    with pytest.raises(ValueError, match=r"Argument num_bins must be a positive integer"):
        ExpectedCalibrationError(num_bins=num_bins)


def test_invalid_shape():
    metric = ExpectedCalibrationError()
    y_pred = torch.rand(4, 3, 2)
    y = torch.randint(0, 3, size=(4,))
    with pytest.raises(ValueError, match=r"y_pred must have shape \(B, C\) or \(B, C, ...\)"):
        metric.update((y_pred, y))


@pytest.fixture(params=range(6))
def test_case(request):
    return [
        # multiclass (N, C), single update
        (softmax(torch.randn(100, 10).numpy(), axis=1), torch.randint(0, 10, size=[100]).numpy(), 1),
        (softmax(torch.randn(100, 4).numpy(), axis=1), torch.randint(0, 4, size=[100]).numpy(), 1),
        # multiclass, batched updates
        (softmax(torch.randn(100, 10).numpy(), axis=1), torch.randint(0, 10, size=[100]).numpy(), 16),
        # binary (N,)
        (torch.rand(100).numpy(), torch.randint(0, 2, size=[100]).numpy(), 1),
        (torch.rand(100).numpy(), torch.randint(0, 2, size=[100]).numpy(), 16),
        # multiclass image segmentation (N, C, H, W)
        (softmax(torch.randn(50, 5, 8, 8).numpy(), axis=1), torch.randint(0, 5, size=(50, 8, 8)).numpy(), 16),
    ][request.param]


@pytest.mark.parametrize("metric_cls, np_ref", [(ExpectedCalibrationError, np_ece), (MaximumCalibrationError, np_mce)])
@pytest.mark.parametrize("num_bins", [5, 10, 15])
def test_compute(metric_cls, np_ref, num_bins, test_case, available_device):
    np_y_pred, np_y, batch_size = test_case
    y_pred = torch.tensor(np_y_pred)
    y = torch.tensor(np_y)

    metric = metric_cls(num_bins=num_bins, device=available_device)
    assert metric._device == torch.device(available_device)

    metric.reset()
    if batch_size > 1:
        n_iters = y.shape[0] // batch_size + 1
        for i in range(n_iters):
            idx = i * batch_size
            metric.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))
    else:
        metric.update((y_pred, y))

    conf, correct = _conf_correct(np_y_pred, np_y)
    expected = np_ref(conf, correct, num_bins)

    res = metric.compute()
    assert isinstance(res, float)
    assert res == pytest.approx(expected, abs=1e-5)


def test_calibration_extremes(available_device):
    # Confidently correct -> perfect calibration -> ECE = MCE = 0
    y_pred = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    y = torch.tensor([1, 1])
    ece = ExpectedCalibrationError(device=available_device)
    ece.update((y_pred, y))
    assert ece.compute() == pytest.approx(0.0)

    # Confidently wrong -> worst calibration -> ECE = MCE = 1
    y_wrong = torch.tensor([0, 0])
    mce = MaximumCalibrationError(device=available_device)
    mce.update((y_pred, y_wrong))
    assert mce.compute() == pytest.approx(1.0)


def test_batched_vs_single_shot(available_device):
    torch.manual_seed(0)
    y_pred = softmax(torch.randn(120, 6).numpy(), axis=1)
    y = torch.randint(0, 6, size=[120])

    single = ExpectedCalibrationError(num_bins=10, device=available_device)
    single.update((torch.tensor(y_pred), y))

    batched = ExpectedCalibrationError(num_bins=10, device=available_device)
    for i in range(0, 120, 16):
        batched.update((torch.tensor(y_pred[i : i + 16]), y[i : i + 16]))

    assert single.compute() == pytest.approx(batched.compute(), abs=1e-6)


def test_accumulator_detached(available_device):
    metric = ExpectedCalibrationError(device=available_device)
    y_pred = torch.tensor([[0.9, 0.1], [0.3, 0.7]], requires_grad=True)
    y = torch.tensor([0, 1])
    metric.update((y_pred, y))

    assert not metric._bin_conf.requires_grad
    assert not metric._bin_correct.requires_grad
    assert not metric._bin_count.requires_grad


@pytest.mark.usefixtures("distributed")
class TestDistributed:
    @pytest.mark.parametrize(
        "metric_cls, np_ref", [(ExpectedCalibrationError, np_ece), (MaximumCalibrationError, np_mce)]
    )
    def test_integration(self, metric_cls, np_ref):
        tol = 1e-5
        num_bins = 10
        device = idist.device()
        rank = idist.get_rank()
        torch.manual_seed(12 + rank)

        n_iters = 80
        batch_size = 16
        n_cls = 10

        metric_devices = [torch.device("cpu")]
        if device.type != "xla":
            metric_devices.append(idist.device())

        for metric_device in metric_devices:
            y_true = torch.randint(0, n_cls, size=[n_iters * batch_size], dtype=torch.long).to(device)
            y_preds = torch.softmax(torch.randn(n_iters * batch_size, n_cls), dim=1).to(device)

            def update(engine, i):
                return (
                    y_preds[i * batch_size : (i + 1) * batch_size],
                    y_true[i * batch_size : (i + 1) * batch_size],
                )

            engine = Engine(update)
            m = metric_cls(num_bins=num_bins, device=metric_device)
            m.attach(engine, "cal_err")

            engine.run(data=list(range(n_iters)), max_epochs=1)

            y_preds = idist.all_gather(y_preds)
            y_true = idist.all_gather(y_true)

            assert "cal_err" in engine.state.metrics
            res = engine.state.metrics["cal_err"]

            conf, correct = _conf_correct(y_preds.cpu().numpy(), y_true.cpu().numpy())
            true_res = np_ref(conf, correct, num_bins)

            assert res == pytest.approx(true_res, abs=tol)

    def test_accumulator_device(self):
        device = idist.device()
        metric_devices = [torch.device("cpu")]
        if device.type != "xla":
            metric_devices.append(idist.device())

        for metric_device in metric_devices:
            metric = ExpectedCalibrationError(device=metric_device)
            for dev in [metric._device, metric._bin_count.device, metric._bin_conf.device, metric._bin_correct.device]:
                assert dev == metric_device, f"{type(dev)}:{dev} vs {type(metric_device)}:{metric_device}"

            y_pred = torch.tensor([[0.9, 0.1], [0.2, 0.8]]).to(device)
            y = torch.tensor([0, 1]).to(device)
            metric.update((y_pred, y))

            for dev in [metric._device, metric._bin_count.device, metric._bin_conf.device, metric._bin_correct.device]:
                assert dev == metric_device, f"{type(dev)}:{dev} vs {type(metric_device)}:{metric_device}"
