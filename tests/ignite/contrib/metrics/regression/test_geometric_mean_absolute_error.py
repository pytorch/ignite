from ignite.exceptions import NotComputableError
from ignite.contrib.metrics.regression import GeometricMeanAbsoluteError
import torch
import numpy as np
import pytest


def test_zero_div():
    m = GeometricMeanAbsoluteError()
    with pytest.raises(NotComputableError):
        m.compute()


def test_wrong_input_shapes():
    m = GeometricMeanAbsoluteError()

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
    np_prod = 1.0

    m = GeometricMeanAbsoluteError()
    m.update((torch.from_numpy(a), torch.from_numpy(ground_truth)))

    errors = np.abs(ground_truth - a)
    np_prod = np.multiply.reduce(errors) * np_prod
    np_len = len(a)
    np_ans = np.power(np_prod, 1.0 / np_len)
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(b), torch.from_numpy(ground_truth)))
    errors = np.abs(ground_truth - b)
    np_prod = np.multiply.reduce(errors) * np_prod
    np_len += len(b)
    np_ans = np.power(np_prod, 1.0 / np_len)
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(c), torch.from_numpy(ground_truth)))
    errors = np.abs(ground_truth - c)
    np_prod = np.multiply.reduce(errors) * np_prod
    np_len += len(c)
    np_ans = np.power(np_prod, 1.0 / np_len)
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(d), torch.from_numpy(ground_truth)))
    errors = np.abs(ground_truth - d)
    np_prod = np.multiply.reduce(errors) * np_prod
    np_len += len(d)
    np_ans = np.power(np_prod, 1.0 / np_len)
    assert m.compute() == pytest.approx(np_ans)
