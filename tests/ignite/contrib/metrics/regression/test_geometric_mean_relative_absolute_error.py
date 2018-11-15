from ignite.exceptions import NotComputableError
from ignite.contrib.metrics.regression import GeometricMeanRelativeAbsoluteError
import torch
import numpy as np
import pytest


def test_zero_div():
    m = GeometricMeanRelativeAbsoluteError()
    with pytest.raises(NotComputableError):
        m.compute()


def test_compute():
    a = np.random.randn(4)
    b = np.random.randn(4)
    c = np.random.randn(4)
    d = np.random.randn(4)
    ground_truth = np.random.randn(4)
    np_prod = 1.0
    prev_mean = 0.0
    np_len = 0

    m = GeometricMeanRelativeAbsoluteError()

    m.update((torch.from_numpy(a), torch.from_numpy(ground_truth)))
    numerator = np.abs(ground_truth - a)
    prev_sum = prev_mean * np_len
    np_len += len(a)
    a_mean = (ground_truth.sum() + prev_sum) / np_len
    prev_mean = a_mean
    denominator = np.abs(ground_truth - a_mean)
    np_prod = np.prod(numerator / denominator) * np_prod
    np_ans = np.power(np_prod, 1.0 / np_len)
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(b), torch.from_numpy(ground_truth)))
    numerator = np.abs(ground_truth - b)
    prev_sum = prev_mean * np_len
    np_len += len(b)
    a_mean = (ground_truth.sum() + prev_sum) / np_len
    denominator = np.abs(ground_truth - a_mean)
    np_prod = np.prod(numerator / denominator) * np_prod
    np_ans = np.power(np_prod, 1.0 / np_len)
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(c), torch.from_numpy(ground_truth)))
    numerator = np.abs(ground_truth - c)
    prev_sum = prev_mean * np_len
    np_len += len(c)
    a_mean = (ground_truth.sum() + prev_sum) / np_len
    denominator = np.abs(ground_truth - a_mean)
    np_prod = np.prod(numerator / denominator) * np_prod
    np_ans = np.power(np_prod, 1.0 / np_len)
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(d), torch.from_numpy(ground_truth)))
    numerator = np.abs(ground_truth - d)
    prev_sum = prev_mean * np_len
    np_len += len(d)
    a_mean = (ground_truth.sum() + prev_sum) / np_len
    denominator = np.abs(ground_truth - a_mean)
    np_prod = np.prod(numerator / denominator) * np_prod
    np_ans = np.power(np_prod, 1.0 / np_len)
    assert m.compute() == pytest.approx(np_ans)
