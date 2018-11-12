from ignite.exceptions import NotComputableError
from ignite.contrib.metrics import MeanPercentageError
import torch
import numpy as np
import pytest


def test_zero_div():
    m = MeanPercentageError()
    with pytest.raises(NotComputableError):
        m.compute()


def test_zero_gt():
    a = np.random.randn(4)
    ground_truth = np.zeros(4)

    m = MeanPercentageError()
    m.reset()

    with pytest.raises(NotComputableError):
        m.update((torch.from_numpy(a), torch.from_numpy(ground_truth)))


def test_mean_error():
    a = np.random.randn(4)
    b = np.random.randn(4)
    c = np.random.randn(4)
    d = np.random.randn(4)
    ground_truth = np.random.randn(4)

    m = MeanPercentageError()
    m.reset()

    m.update((torch.from_numpy(a), torch.from_numpy(ground_truth)))
    np_sum = ((a - ground_truth) / ground_truth).sum()
    np_len = len(a)
    np_ans = 100.00 * np_sum / np_len
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(b), torch.from_numpy(ground_truth)))
    np_sum += ((b - ground_truth) / ground_truth).sum()
    np_len += len(b)
    np_ans = 100.00 * np_sum / np_len
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(c), torch.from_numpy(ground_truth)))
    np_sum += ((c - ground_truth) / ground_truth).sum()
    np_len += len(c)
    np_ans = 100.00 * np_sum / np_len
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(d), torch.from_numpy(ground_truth)))
    np_sum += ((d - ground_truth) / ground_truth).sum()
    np_len += len(d)
    np_ans = 100.00 * np_sum / np_len
    assert m.compute() == pytest.approx(np_ans)
