from ignite.exceptions import NotComputableError
from ignite.contrib.metrics.regression import ManhattanDistance
import torch
import numpy as np
import pytest


def test_mahattan_distance():
    a = np.random.randn(4)
    b = np.random.randn(4)
    c = np.random.randn(4)
    d = np.random.randn(4)
    ground_truth = np.random.randn(4)

    m = ManhattanDistance()

    m.update((torch.from_numpy(a), torch.from_numpy(ground_truth)))
    np_ans = (ground_truth - a).sum()
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(b), torch.from_numpy(ground_truth)))
    np_ans += (ground_truth - b).sum()
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(c), torch.from_numpy(ground_truth)))
    np_ans += (ground_truth - c).sum()
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(d), torch.from_numpy(ground_truth)))
    np_ans += (ground_truth - d).sum()
    assert m.compute() == pytest.approx(np_ans)
