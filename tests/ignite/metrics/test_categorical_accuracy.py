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
    assert acc.compute() == 0.25

    acc.reset()
    y_pred = torch.eye(2)
    y = torch.ones(2).type(torch.LongTensor)
    acc.update((y_pred, y))
    assert acc.compute() == 0.5
