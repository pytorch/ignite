from ignite.exceptions import NotComputableError
from ignite.contrib.metrics.regression import FractionalBias
import torch
import numpy as np
import pytest


def test_zero_div():
    m = FractionalBias()
    with pytest.raises(NotComputableError):
        m.compute()


def test_wrong_input_shapes():
    m = FractionalBias()

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


def test_fractional_bias():
    a = np.random.randn(4)
    b = np.random.randn(4)
    c = np.random.randn(4)
    d = np.random.randn(4)
    ground_truth = np.random.randn(4)

    m = FractionalBias()

    m.update((torch.from_numpy(a), torch.from_numpy(ground_truth)))
    np_sum = (2 * (ground_truth - a) / (a + ground_truth)).sum()
    np_len = len(a)
    np_ans = np_sum / np_len
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(b), torch.from_numpy(ground_truth)))
    np_sum += (2 * (ground_truth - b) / (b + ground_truth)).sum()
    np_len += len(b)
    np_ans = np_sum / np_len
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(c), torch.from_numpy(ground_truth)))
    np_sum += (2 * (ground_truth - c) / (c + ground_truth)).sum()
    np_len += len(c)
    np_ans = np_sum / np_len
    assert m.compute() == pytest.approx(np_ans)

    m.update((torch.from_numpy(d), torch.from_numpy(ground_truth)))
    np_sum += (2 * (ground_truth - d) / (d + ground_truth)).sum()
    np_len += len(d)
    np_ans = np_sum / np_len
    assert m.compute() == pytest.approx(np_ans)
