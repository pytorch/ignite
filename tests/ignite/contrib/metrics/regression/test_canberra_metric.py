import numpy as np
import pytest
import torch
from sklearn.neighbors import DistanceMetric

from ignite.contrib.metrics.regression import CanberraMetric


def test_wrong_input_shapes():
    m = CanberraMetric()

    with pytest.raises(ValueError):
        m.update((torch.rand(4, 1, 2), torch.rand(4, 1)))

    with pytest.raises(ValueError):
        m.update((torch.rand(4, 1), torch.rand(4, 1, 2)))

    with pytest.raises(ValueError):
        m.update((torch.rand(4, 1, 2), torch.rand(4,)))

    with pytest.raises(ValueError):
        m.update((torch.rand(4,), torch.rand(4, 1, 2)))


def test_compute():
    a = np.random.randn(4)
    b = np.random.randn(4)
    c = np.random.randn(4)
    d = np.random.randn(4)
    ground_truth = np.random.randn(4)

    m = CanberraMetric()

    canberra = DistanceMetric.get_metric("canberra")

    m.update((torch.from_numpy(a), torch.from_numpy(ground_truth)))
    np_sum = (np.abs(ground_truth - a) / (np.abs(a) + np.abs(ground_truth))).sum()
    assert m.compute() == pytest.approx(np_sum)
    assert canberra.pairwise([a, ground_truth])[0][1] == pytest.approx(np_sum)

    m.update((torch.from_numpy(b), torch.from_numpy(ground_truth)))
    value = ((np.abs(ground_truth - b)) / (np.abs(b) + np.abs(ground_truth))).sum()
    np_sum += value
    assert m.compute() == pytest.approx(np_sum)
    assert canberra.pairwise([b, ground_truth])[0][1] == pytest.approx(value)

    m.update((torch.from_numpy(c), torch.from_numpy(ground_truth)))
    value = ((np.abs(ground_truth - c)) / (np.abs(c) + np.abs(ground_truth))).sum()
    np_sum += value
    assert m.compute() == pytest.approx(np_sum)
    assert canberra.pairwise([c, ground_truth])[0][1] == pytest.approx(value)

    m.update((torch.from_numpy(d), torch.from_numpy(ground_truth)))
    value = (np.abs(ground_truth - d) / (np.abs(d) + np.abs(ground_truth))).sum()
    np_sum += value
    assert m.compute() == pytest.approx(np_sum)
    assert canberra.pairwise([d, ground_truth])[0][1] == pytest.approx(value)
