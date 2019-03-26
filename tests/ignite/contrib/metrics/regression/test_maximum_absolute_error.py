from ignite.exceptions import NotComputableError
from ignite.contrib.metrics.regression import MaximumAbsoluteError
import torch
import numpy as np
import pytest


def test_zero_div():
    m = MaximumAbsoluteError()
    with pytest.raises(NotComputableError):
        m.compute()


def test_wrong_input_shapes():
    m = MaximumAbsoluteError()

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


def test_maximum_absolute_error():
    a = np.random.randn(4)
    b = np.random.randn(4)
    c = np.random.randn(4)
    d = np.random.randn(4)
    ground_truth = np.random.randn(4)

    m = MaximumAbsoluteError()

    np_ans = -1

    m.update((torch.from_numpy(a), torch.from_numpy(ground_truth)))
    np_max = np.max(np.abs((a - ground_truth)))
    np_ans = np_max if np_max > np_ans else np_ans
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(b), torch.from_numpy(ground_truth)))
    np_max = np.max(np.abs((b - ground_truth)))
    np_ans = np_max if np_max > np_ans else np_ans
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(c), torch.from_numpy(ground_truth)))
    np_max = np.max(np.abs((c - ground_truth)))
    np_ans = np_max if np_max > np_ans else np_ans
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(d), torch.from_numpy(ground_truth)))
    np_max = np.max(np.abs((d - ground_truth)))
    np_ans = np_max if np_max > np_ans else np_ans
    assert m.compute() == pytest.approx(np_ans)
