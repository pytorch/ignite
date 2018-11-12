from ignite.exceptions import NotComputableError
from ignite.contrib.metrics import ManhattanDistance
import torch
import numpy as np
import pytest


def test_zero_div():
    m = ManhattanDistance()
    with pytest.raises(NotComputableError):
        m.compute()


def test_mahattan_distance():
    a = np.random.randn(4)
    b = np.random.randn(4)
    c = np.random.randn(4)
    d = np.random.randn(4)
    ground_truth = np.random.randn(4)

    m = ManhattanDistance()
    m.reset()

    m.update((torch.from_numpy(a), torch.from_numpy(ground_truth)))
    np_ans = (a - ground_truth).sum()
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(b), torch.from_numpy(ground_truth)))
    np_ans += (b - ground_truth).sum()
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(c), torch.from_numpy(ground_truth)))
    np_ans += (c - ground_truth).sum()
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(d), torch.from_numpy(ground_truth)))
    np_ans += (d - ground_truth).sum()
    assert m.compute() == pytest.approx(np_ans)
