import numpy as np
import pytest
import torch
from pytest import approx, raises

from ignite.contrib.metrics.regression import MeanAbsoluteRelativeError
from ignite.engine import Engine
from ignite.exceptions import NotComputableError


def test_wrong_input_shapes():
    m = MeanAbsoluteRelativeError()

    with pytest.raises(ValueError, match=r"Input data shapes should be the same, but given"):
        m.update((torch.rand(4), torch.rand(4, 1)))

    with pytest.raises(ValueError, match=r"Input data shapes should be the same, but given"):
        m.update((torch.rand(4, 1), torch.rand(4,)))


def test_mean_absolute_relative_error():
    a = torch.rand(4)
    b = torch.rand(4)
    c = torch.rand(4)
    d = torch.rand(4)
    ground_truth = torch.rand(4)

    m = MeanAbsoluteRelativeError()

    m.update((a, ground_truth))
    abs_error_a = torch.sum(torch.abs(ground_truth - a) / torch.abs(ground_truth))
    num_samples_a = a.size()[0]
    sum_error = abs_error_a
    sum_samples = num_samples_a
    MARE_a = sum_error / sum_samples
    assert m.compute() == approx(MARE_a.item())

    m.update((b, ground_truth))
    abs_error_b = torch.sum(torch.abs(ground_truth - b) / torch.abs(ground_truth))
    num_samples_b = b.size()[0]
    sum_error += abs_error_b
    sum_samples += num_samples_b
    MARE_b = sum_error / sum_samples
    assert m.compute() == approx(MARE_b.item())

    m.update((c, ground_truth))
    abs_error_c = torch.sum(torch.abs(ground_truth - c) / torch.abs(ground_truth))
    num_samples_c = c.size()[0]
    sum_error += abs_error_c
    sum_samples += num_samples_c
    MARE_c = sum_error / sum_samples
    assert m.compute() == approx(MARE_c.item())

    m.update((d, ground_truth))
    abs_error_d = torch.sum(torch.abs(ground_truth - d) / torch.abs(ground_truth))
    num_samples_d = d.size()[0]
    sum_error += abs_error_d
    sum_samples += num_samples_d
    MARE_d = sum_error / sum_samples
    assert m.compute() == approx(MARE_d.item())


def test_zero_div():
    a = torch.tensor([2.0, -1.0, -1.0, 2.0])
    ground_truth = torch.tensor([0.0, 0.5, 0.2, 1.0])

    m = MeanAbsoluteRelativeError()
    with raises(NotComputableError, match=r"The ground truth has 0"):
        m.update((a, ground_truth))


def test_zero_sample():
    m = MeanAbsoluteRelativeError()
    with raises(NotComputableError, match=r"MeanAbsoluteRelativeError must have at least one sample"):
        m.compute()


def test_integration():
    def _test(y_pred, y, batch_size):
        def update_fn(engine, batch):
            idx = (engine.state.iteration - 1) * batch_size
            y_true_batch = np_y[idx : idx + batch_size]
            y_pred_batch = np_y_pred[idx : idx + batch_size]
            return torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

        engine = Engine(update_fn)

        m = MeanAbsoluteRelativeError()
        m.attach(engine, "mare")

        np_y = y.numpy()
        np_y_pred = y_pred.numpy()

        data = list(range(y_pred.shape[0] // batch_size))
        mare = engine.run(data, max_epochs=1).metrics["mare"]

        abs_error = np.sum(abs(np_y - np_y_pred) / abs(np_y))
        num_samples = len(y_pred)
        res = abs_error / num_samples

        assert res == approx(mare)

    def get_test_cases():
        test_cases = [
            (torch.rand(size=(100,)), torch.rand(size=(100,)), 10),
            (torch.rand(size=(200,)), torch.rand(size=(200,)), 10),
            (torch.rand(size=(100,)), torch.rand(size=(100,)), 20),
            (torch.rand(size=(200,)), torch.rand(size=(200,)), 20),
        ]
        return test_cases

    for _ in range(10):
        # check multiple random inputs as random exact occurencies are rare
        test_cases = get_test_cases()
        for y_pred, y, batch_size in test_cases:
            _test(y_pred, y, batch_size)
