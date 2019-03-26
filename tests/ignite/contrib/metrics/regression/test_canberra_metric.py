import torch
import numpy as np
import pytest

from ignite.contrib.metrics.regression import CanberraMetric


def test_wrong_input_shapes():
    m = CanberraMetric()

    with pytest.raises(ValueError):
        m.update((torch.rand(4, 1, 2),
                  torch.rand(4, 1)))

    with pytest.raises(ValueError):
        m.update((torch.rand(4, 1),
                  torch.rand(4, 1, 2)))

    with pytest.raises(ValueError):
        m.update((torch.rand(4, 1, 2),
                  torch.rand(4,)))

    with pytest.raises(ValueError):
        m.update((torch.rand(4,),
                  torch.rand(4, 1, 2)))


def test_compute():
    a = np.random.randn(4)
    b = np.random.randn(4)
    c = np.random.randn(4)
    d = np.random.randn(4)
    ground_truth = np.random.randn(4)

    m = CanberraMetric()

    m.update((torch.from_numpy(a), torch.from_numpy(ground_truth)))
    np_sum = (np.abs(ground_truth - a) / (a + ground_truth)).sum()
    assert m.compute() == pytest.approx(np_sum)

    m.update((torch.from_numpy(b), torch.from_numpy(ground_truth)))
    np_sum += ((np.abs(ground_truth - b)) / (b + ground_truth)).sum()
    assert m.compute() == pytest.approx(np_sum)

    m.update((torch.from_numpy(c), torch.from_numpy(ground_truth)))
    np_sum += ((np.abs(ground_truth - c)) / (c + ground_truth)).sum()
    assert m.compute() == pytest.approx(np_sum)

    m.update((torch.from_numpy(d), torch.from_numpy(ground_truth)))
    np_sum += (np.abs(ground_truth - d) / (d + ground_truth)).sum()
    assert m.compute() == pytest.approx(np_sum)
