from ignite.exceptions import NotComputableError
from ignite.contrib.metrics.regression import FractionalAbsoluteError
import torch
import numpy as np
import pytest


def test_zero_div():
    m = FractionalAbsoluteError()
    with pytest.raises(NotComputableError):
        m.compute()


def test_wrong_input_shapes():
    m = FractionalAbsoluteError()

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

    m = FractionalAbsoluteError()

    m.update((torch.from_numpy(a), torch.from_numpy(ground_truth)))
    np_sum = (2 * np.abs((a - ground_truth)) / (np.abs(a) + np.abs(ground_truth))).sum()
    np_len = len(a)
    np_ans = np_sum / np_len
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(b), torch.from_numpy(ground_truth)))
    np_sum += (2 * np.abs((b - ground_truth)) / (np.abs(b) + np.abs(ground_truth))).sum()
    np_len += len(b)
    np_ans = np_sum / np_len
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(c), torch.from_numpy(ground_truth)))
    np_sum += (2 * np.abs((c - ground_truth)) / (np.abs(c) + np.abs(ground_truth))).sum()
    np_len += len(c)
    np_ans = np_sum / np_len
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(d), torch.from_numpy(ground_truth)))
    np_sum += (2 * np.abs((d - ground_truth)) / (np.abs(d) + np.abs(ground_truth))).sum()
    np_len += len(d)
    np_ans = np_sum / np_len
    assert m.compute() == pytest.approx(np_ans)
