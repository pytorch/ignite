from ignite.exceptions import NotComputableError
from ignite.contrib.metrics.regression import MeanNormalizedBias
import torch
import numpy as np
import pytest


def test_zero_div():
    m = MeanNormalizedBias()
    with pytest.raises(NotComputableError):
        m.compute()


def test_zero_gt():
    a = np.random.randn(4)
    ground_truth = np.zeros(4)

    m = MeanNormalizedBias()

    with pytest.raises(NotComputableError):
        m.update((torch.from_numpy(a), torch.from_numpy(ground_truth)))


def test_mean_error():
    a = np.random.randn(4)
    b = np.random.randn(4)
    c = np.random.randn(4)
    d = np.random.randn(4)
    ground_truth = np.random.randn(4)

    m = MeanNormalizedBias()

    m.update((torch.from_numpy(a), torch.from_numpy(ground_truth)))
    np_sum = ((ground_truth - a) / ground_truth).sum()
    np_len = len(a)
    np_ans = np_sum / np_len
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(b), torch.from_numpy(ground_truth)))
    np_sum += ((ground_truth - b) / ground_truth).sum()
    np_len += len(b)
    np_ans = np_sum / np_len
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(c), torch.from_numpy(ground_truth)))
    np_sum += ((ground_truth - c) / ground_truth).sum()
    np_len += len(c)
    np_ans = np_sum / np_len
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(d), torch.from_numpy(ground_truth)))
    np_sum += ((ground_truth - d) / ground_truth).sum()
    np_len += len(d)
    np_ans = np_sum / np_len
    assert m.compute() == pytest.approx(np_ans)
