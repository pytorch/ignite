import torch
import numpy as np
import pytest

from ignite.contrib.metrics.regression import ManhattanDistance


def test_wrong_input_shapes():
    m = ManhattanDistance()

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
