from ignite.exceptions import NotComputableError
from ignite.metrics import Precision
import pytest
import torch


def test_no_update():
    precision = Precision()
    with pytest.raises(NotComputableError):
        precision.compute()


def test_compute():
    precision = Precision()

    y_pred = torch.eye(4)
    y = torch.ones(4).type(torch.LongTensor)
    precision.update((y_pred, y))
    results = list(precision.compute())
    assert results[0] == 0.0
    assert results[1] == 1.0
    assert results[2] == 0.0
    assert results[3] == 0.0

    precision.reset()
    y_pred = torch.eye(2)
    y = torch.ones(2).type(torch.LongTensor)
    precision.update((y_pred, y))
    y = torch.zeros(2).type(torch.LongTensor)
    precision.update((y_pred, y))
    results = list(precision.compute())
    assert results[0] == 0.5
    assert results[1] == 0.5
