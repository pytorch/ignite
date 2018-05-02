from ignite.exceptions import NotComputableError
from ignite.metrics import CategoricalAccuracy
import pytest
import torch


def test_zero_div():
    acc = CategoricalAccuracy()
    with pytest.raises(NotComputableError):
        acc.compute()


def test_compute():
    acc = CategoricalAccuracy()

    y_pred = torch.eye(4)
    y = torch.ones(4).type(torch.LongTensor)
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert acc.compute() == 0.25

    acc.reset()
    y_pred = torch.eye(2)
    y = torch.ones(2).type(torch.LongTensor)
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert acc.compute() == 0.5


def test_compute_batch_images():
    acc = CategoricalAccuracy()
    y_pred = torch.zeros(2, 3, 2, 2)
    y_pred[0, 1, :] = 1
    y_pred[0, 2, :] = 1

    y = torch.LongTensor([[[0, 1],
                           [0, 1]],
                          [[0, 2],
                           [0, 2]]])

    acc.update((y_pred, y))

    assert isinstance(acc.compute(), float)
    assert acc.compute() == 0.5

    acc.reset()
    y_pred = torch.zeros(2, 3, 2, 2)
    y_pred[0, 1, :] = 1
    y_pred[1, 2, :] = 1

    y = torch.LongTensor([[[2, 1],
                           [1, 1]],
                          [[2, 2],
                           [0, 2]]])

    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert acc.compute() == 0.75
