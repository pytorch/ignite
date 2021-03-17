import numpy as np
import pytest
import torch

from ignite.contrib.metrics.regression import WaveHedgesDistance
from ignite.engine import Engine


def test_wrong_input_shapes():
    m = WaveHedgesDistance()

    with pytest.raises(ValueError, match=r"Input data shapes should be the same, but given"):
        m.update((torch.rand(4, 1, 2), torch.rand(4, 1)))

    with pytest.raises(ValueError, match=r"Input data shapes should be the same, but given"):
        m.update((torch.rand(4, 1), torch.rand(4, 1, 2)))

    with pytest.raises(ValueError, match=r"Input data shapes should be the same, but given"):
        m.update((torch.rand(4, 1, 2), torch.rand(4,),))

    with pytest.raises(ValueError, match=r"Input data shapes should be the same, but given"):
        m.update((torch.rand(4,), torch.rand(4, 1, 2),))


def test_compute():
    a = np.random.randn(4)
    b = np.random.randn(4)
    c = np.random.randn(4)
    d = np.random.randn(4)
    ground_truth = np.random.randn(4)

    m = WaveHedgesDistance()

    m.update((torch.from_numpy(a), torch.from_numpy(ground_truth)))
    np_sum = (np.abs(ground_truth - a) / np.maximum.reduce([a, ground_truth])).sum()
    assert m.compute() == pytest.approx(np_sum)

    m.update((torch.from_numpy(b), torch.from_numpy(ground_truth)))
    np_sum += (np.abs(ground_truth - b) / np.maximum.reduce([b, ground_truth])).sum()
    assert m.compute() == pytest.approx(np_sum)

    m.update((torch.from_numpy(c), torch.from_numpy(ground_truth)))
    np_sum += (np.abs(ground_truth - c) / np.maximum.reduce([c, ground_truth])).sum()
    assert m.compute() == pytest.approx(np_sum)

    m.update((torch.from_numpy(d), torch.from_numpy(ground_truth)))
    np_sum += (np.abs(ground_truth - d) / np.maximum.reduce([d, ground_truth])).sum()
    assert m.compute() == pytest.approx(np_sum)


def test_integration():
    def _test(y_pred, y, batch_size):
        def update_fn(engine, batch):
            idx = (engine.state.iteration - 1) * batch_size
            y_true_batch = np_y[idx : idx + batch_size]
            y_pred_batch = np_y_pred[idx : idx + batch_size]
            return torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

        engine = Engine(update_fn)

        m = WaveHedgesDistance()
        m.attach(engine, "whd")

        np_y = y.numpy()
        np_y_pred = y_pred.numpy()

        data = list(range(y_pred.shape[0] // batch_size))
        whd = engine.run(data, max_epochs=1).metrics["whd"]

        np_sum = (np.abs(np_y - np_y_pred) / np.maximum.reduce([np_y_pred, np_y])).sum()

        assert np_sum == pytest.approx(whd)

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
