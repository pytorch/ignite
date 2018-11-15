from ignite.exceptions import NotComputableError
from ignite.contrib.metrics.regression import WaveHedgesDistance
import torch
import numpy as np
import pytest


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
