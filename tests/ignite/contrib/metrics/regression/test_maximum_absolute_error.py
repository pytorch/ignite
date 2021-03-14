import numpy as np
import pytest
import torch

from ignite.contrib.metrics.regression import MaximumAbsoluteError
from ignite.engine import Engine
from ignite.exceptions import NotComputableError


def test_zero_sample():
    m = MaximumAbsoluteError()
    with pytest.raises(
        NotComputableError, match=r"MaximumAbsoluteError must have at least one example before it can be computed"
    ):
        m.compute()


def test_wrong_input_shapes():
    m = MaximumAbsoluteError()

    with pytest.raises(ValueError, match=r"Input data shapes should be the same, but given"):
        m.update((torch.rand(4, 1, 2), torch.rand(4, 1)))

    with pytest.raises(ValueError, match=r"Input data shapes should be the same, but given"):
        m.update((torch.rand(4, 1), torch.rand(4, 1, 2)))

    with pytest.raises(ValueError, match=r"Input data shapes should be the same, but given"):
        m.update((torch.rand(4, 1, 2), torch.rand(4,),))

    with pytest.raises(ValueError, match=r"Input data shapes should be the same, but given"):
        m.update((torch.rand(4,), torch.rand(4, 1, 2),))


def test_maximum_absolute_error():
    a = np.random.randn(4)
    b = np.random.randn(4)
    c = np.random.randn(4)
    d = np.random.randn(4)
    ground_truth = np.random.randn(4)

    m = MaximumAbsoluteError()

    np_ans = -1

    m.update((torch.from_numpy(a), torch.from_numpy(ground_truth)))
    np_max = np.max(np.abs((a - ground_truth)))
    np_ans = np_max if np_max > np_ans else np_ans
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(b), torch.from_numpy(ground_truth)))
    np_max = np.max(np.abs((b - ground_truth)))
    np_ans = np_max if np_max > np_ans else np_ans
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(c), torch.from_numpy(ground_truth)))
    np_max = np.max(np.abs((c - ground_truth)))
    np_ans = np_max if np_max > np_ans else np_ans
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(d), torch.from_numpy(ground_truth)))
    np_max = np.max(np.abs((d - ground_truth)))
    np_ans = np_max if np_max > np_ans else np_ans
    assert m.compute() == pytest.approx(np_ans)


def test_integration():
    def _test(y_pred, y, batch_size):
        def update_fn(engine, batch):
            idx = (engine.state.iteration - 1) * batch_size
            y_true_batch = np_y[idx : idx + batch_size]
            y_pred_batch = np_y_pred[idx : idx + batch_size]
            return idx, torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

        engine = Engine(update_fn)

        m = MaximumAbsoluteError(output_transform=lambda x: (x[1], x[2]))
        m.attach(engine, "mae")

        np_y = y.numpy()
        np_y_pred = y_pred.numpy()

        data = list(range(y_pred.shape[0] // batch_size))
        mae = engine.run(data, max_epochs=1).metrics["mae"]

        np_max = np.max(np.abs((np_y_pred - np_y)))

        assert np_max == pytest.approx(mae)

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
