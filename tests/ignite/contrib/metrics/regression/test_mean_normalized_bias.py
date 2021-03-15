import numpy as np
import pytest
import torch

from ignite.contrib.metrics.regression import MeanNormalizedBias
from ignite.engine import Engine
from ignite.exceptions import NotComputableError


def test_zero_sample():
    m = MeanNormalizedBias()
    with pytest.raises(
        NotComputableError, match=r"MeanNormalizedBias must have at least one example before it can be computed"
    ):
        m.compute()


def test_zero_gt():
    a = np.random.randn(4)
    ground_truth = np.zeros(4)

    m = MeanNormalizedBias()

    with pytest.raises(NotComputableError, match=r"The ground truth has 0."):
        m.update((torch.from_numpy(a), torch.from_numpy(ground_truth)))


def test_wrong_input_shapes():
    m = MeanNormalizedBias()

    with pytest.raises(ValueError, match=r"Input data shapes should be the same, but given"):
        m.update((torch.rand(4, 1, 2), torch.rand(4, 1)))

    with pytest.raises(ValueError, match=r"Input data shapes should be the same, but given"):
        m.update((torch.rand(4, 1), torch.rand(4, 1, 2)))

    with pytest.raises(ValueError, match=r"Input data shapes should be the same, but given"):
        m.update((torch.rand(4, 1, 2), torch.rand(4,),))

    with pytest.raises(ValueError, match=r"Input data shapes should be the same, but given"):
        m.update((torch.rand(4,), torch.rand(4, 1, 2),))


def test_mean_error():
    a = np.random.randn(4)
    b = np.random.randn(4)
    c = np.random.randn(4)
    d = np.random.randn(4)
    ground_truth = np.random.randn(4)

    m = MeanNormalizedBias()

    m.update((torch.from_numpy(a), torch.from_numpy(ground_truth)))
    np_sum = ((ground_truth - a) / ground_truth).sum()
    np_len = len(a)
    np_ans = np_sum / np_len
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(b), torch.from_numpy(ground_truth)))
    np_sum += ((ground_truth - b) / ground_truth).sum()
    np_len += len(b)
    np_ans = np_sum / np_len
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(c), torch.from_numpy(ground_truth)))
    np_sum += ((ground_truth - c) / ground_truth).sum()
    np_len += len(c)
    np_ans = np_sum / np_len
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(d), torch.from_numpy(ground_truth)))
    np_sum += ((ground_truth - d) / ground_truth).sum()
    np_len += len(d)
    np_ans = np_sum / np_len
    assert m.compute() == pytest.approx(np_ans)


def test_integration():
    def _test(y_pred, y, batch_size):
        def update_fn(engine, batch):
            idx = (engine.state.iteration - 1) * batch_size
            y_true_batch = np_y[idx : idx + batch_size]
            y_pred_batch = np_y_pred[idx : idx + batch_size]
            return torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

        engine = Engine(update_fn)

        m = MeanNormalizedBias()
        m.attach(engine, "mnb")

        np_y = y.numpy()
        np_y_pred = y_pred.numpy()

        data = list(range(y_pred.shape[0] // batch_size))
        mnb = engine.run(data, max_epochs=1).metrics["mnb"]

        np_sum = ((np_y - np_y_pred) / np_y).sum()
        np_len = len(np_y_pred)
        np_ans = np_sum / np_len

        assert np_ans == pytest.approx(mnb)

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
