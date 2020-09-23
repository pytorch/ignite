import numpy as np
import pytest
import torch

from ignite.contrib.metrics.regression import CanberraMetric
from sklearn.neighbors import DistanceMetric


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

    canberra = DistanceMetric.get_metric('canberra')

    m.update((torch.from_numpy(a), torch.from_numpy(ground_truth)))
    np_sum = (np.abs(ground_truth - a) / (np.abs(a) + np.abs(ground_truth))).sum()
    assert m.compute() == pytest.approx(np_sum)
    assert canberra.pairwise([a, ground_truth])[0][1] == pytest.approx(np_sum)

    m.update((torch.from_numpy(b), torch.from_numpy(ground_truth)))
    np_sum += ((np.abs(ground_truth - b)) / (np.abs(b) + np.abs(ground_truth))).sum()
    assert m.compute() == pytest.approx(np_sum)
    v1 = np.hstack([a, b])
    v2 = np.hstack([ground_truth, ground_truth])
    assert canberra.pairwise([v1, v2])[0][1] == pytest.approx(np_sum)

    m.update((torch.from_numpy(c), torch.from_numpy(ground_truth)))
    np_sum += ((np.abs(ground_truth - c)) / (np.abs(c) + np.abs(ground_truth))).sum()
    assert m.compute() == pytest.approx(np_sum)
    v1 = np.hstack([v1, c])
    v2 = np.hstack([v2, ground_truth])
    assert canberra.pairwise([v1, v2])[0][1] == pytest.approx(np_sum)

    m.update((torch.from_numpy(d), torch.from_numpy(ground_truth)))
    np_sum += (np.abs(ground_truth - d) / (np.abs(d) + np.abs(ground_truth))).sum()
    assert m.compute() == pytest.approx(np_sum)
    v1 = np.hstack([v1, d])
    v2 = np.hstack([v2, ground_truth])
    assert canberra.pairwise([v1, v2])[0][1] == pytest.approx(np_sum)