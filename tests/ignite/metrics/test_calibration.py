import numpy as np
import pytest
import torch

import ignite.distributed as idist

from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics import ExpectedCalibrationError


def np_ece(y_pred, y_true, n_bins=15):
    """Reference ECE implementation in numpy."""
    confidences = np.max(y_pred, axis=1)
    predicted = np.argmax(y_pred, axis=1)
    corrects = (predicted == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)
        bin_size = in_bin.sum()
        if bin_size > 0:
            bin_acc = corrects[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            ece += (bin_size / n) * abs(bin_acc - bin_conf)
    return ece


def test_zero_sample():
    ece = ExpectedCalibrationError()
    with pytest.raises(
        NotComputableError,
        match=r"ExpectedCalibrationError must have at least one example before it can be computed",
    ):
        ece.compute()


def test_invalid_n_bins():
    with pytest.raises(ValueError, match=r"n_bins must be a positive integer"):
        ExpectedCalibrationError(n_bins=0)


def test_invalid_y_pred_shape():
    ece = ExpectedCalibrationError()
    y_pred = torch.randn(10)
    y = torch.randint(0, 2, (10,))
    with pytest.raises(ValueError, match=r"y_pred must be of shape"):
        ece.update((y_pred, y))


def test_invalid_y_shape():
    ece = ExpectedCalibrationError()
    y_pred = torch.randn(10, 3)
    y = torch.randint(0, 3, (10, 1))
    with pytest.raises(ValueError, match=r"y must be of shape"):
        ece.update((y_pred, y))


def test_perfect_calibration():
    """When predictions are perfectly calibrated, ECE should be 0."""
    ece = ExpectedCalibrationError(n_bins=10)
    # Perfectly confident and correct
    y_pred = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    y = torch.tensor([0, 1, 2])
    ece.update((y_pred, y))
    result = ece.compute()
    # confidence=1.0, accuracy=1.0 -> ECE=0
    assert result == pytest.approx(0.0, abs=1e-6)


def test_known_value():
    """Test with hand-computed ECE."""
    ece = ExpectedCalibrationError(n_bins=10)
    # 4 samples, all land in the 0.6-0.8 range
    y_pred = torch.tensor([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.6, 0.3, 0.1],
    ])
    y = torch.tensor([0, 1, 2, 0])

    ece.update((y_pred, y))
    result = ece.compute()
    expected = np_ece(y_pred.numpy(), y.numpy(), n_bins=10)
    assert result == pytest.approx(expected, abs=1e-6)


def test_reset():
    ece = ExpectedCalibrationError(n_bins=10)
    y_pred = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    y = torch.tensor([1, 0])
    ece.update((y_pred, y))
    ece.reset()
    with pytest.raises(NotComputableError):
        ece.compute()


@pytest.fixture(params=list(range(4)))
def test_case(request):
    return [
        (torch.softmax(torch.randn(100, 10), dim=1), torch.randint(0, 10, (100,)), 1),
        (torch.softmax(torch.randn(100, 5), dim=1), torch.randint(0, 5, (100,)), 1),
        # batched updates
        (torch.softmax(torch.randn(100, 10), dim=1), torch.randint(0, 10, (100,)), 16),
        (torch.softmax(torch.randn(100, 5), dim=1), torch.randint(0, 5, (100,)), 16),
    ][request.param]


@pytest.mark.parametrize("n_times", range(3))
def test_compute(n_times, test_case, available_device):
    n_bins = 15
    ece = ExpectedCalibrationError(n_bins=n_bins, device=available_device)

    y_pred, y, batch_size = test_case

    ece.reset()
    if batch_size > 1:
        n_iters = y.shape[0] // batch_size + 1
        for i in range(n_iters):
            idx = i * batch_size
            ece.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))
    else:
        ece.update((y_pred, y))

    np_res = np_ece(y_pred.cpu().numpy(), y.cpu().numpy(), n_bins=n_bins)

    assert isinstance(ece.compute(), float)
    assert ece.compute() == pytest.approx(np_res, abs=1e-6)


def test_accumulator_detached(available_device):
    ece = ExpectedCalibrationError(device=available_device)

    y_pred = torch.tensor([[0.7, 0.3], [0.4, 0.6]], requires_grad=True)
    y = torch.tensor([0, 1])
    ece.update((y_pred, y))

    assert not ece._confidences.requires_grad
    assert not ece._corrects.requires_grad


@pytest.mark.usefixtures("distributed")
class TestDistributed:
    def test_integration(self):
        tol = 1e-6
        device = idist.device()
        rank = idist.get_rank()
        torch.manual_seed(12 + rank)

        n_iters = 100
        batch_size = 10
        n_cls = 10
        n_bins = 15

        metric_devices = [torch.device("cpu")]
        if device.type != "xla":
            metric_devices.append(idist.device())

        for metric_device in metric_devices:
            y_true = torch.randint(0, n_cls, size=[n_iters * batch_size], dtype=torch.long).to(device)
            y_preds = torch.softmax(
                torch.normal(0.0, 1.0, size=(n_iters * batch_size, n_cls), dtype=torch.float), dim=1
            ).to(device)

            def update(engine, i):
                return (
                    y_preds[i * batch_size : (i + 1) * batch_size],
                    y_true[i * batch_size : (i + 1) * batch_size],
                )

            engine = Engine(update)

            m = ExpectedCalibrationError(n_bins=n_bins, device=metric_device)
            m.attach(engine, "ece")

            data = list(range(n_iters))
            engine.run(data=data, max_epochs=1)

            y_preds_all = idist.all_gather(y_preds)
            y_true_all = idist.all_gather(y_true)

            assert "ece" in engine.state.metrics
            res = engine.state.metrics["ece"]

            true_res = np_ece(y_preds_all.cpu().numpy(), y_true_all.cpu().numpy(), n_bins=n_bins)

            assert res == pytest.approx(true_res, rel=tol)

    def test_accumulator_device(self):
        device = idist.device()
        metric_devices = [torch.device("cpu")]
        if device.type != "xla":
            metric_devices.append(idist.device())

        for metric_device in metric_devices:
            ece = ExpectedCalibrationError(device=metric_device)

            assert ece._device == metric_device
            assert ece._confidences.device == metric_device
            assert ece._corrects.device == metric_device

            y_pred = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
            y = torch.tensor([0, 1])
            ece.update((y_pred, y))

            assert ece._confidences.device == metric_device
            assert ece._corrects.device == metric_device
