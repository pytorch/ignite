from ignite.exceptions import NotComputableError
from ignite.metrics import TopKCategoricalAccuracy
import pytest
import torch


def test_zero_div():
    acc = TopKCategoricalAccuracy(2)
    with pytest.raises(NotComputableError):
        acc.compute()


def test_compute():
    acc = TopKCategoricalAccuracy(2)

    y_pred = torch.FloatTensor([[0.2, 0.4, 0.6, 0.8], [0.8, 0.6, 0.4, 0.2]])
    y = torch.ones(2).type(torch.LongTensor)
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert acc.compute() == 0.5

    acc.reset()
    y_pred = torch.FloatTensor([[0.4, 0.8, 0.2, 0.6], [0.8, 0.6, 0.4, 0.2]])
    y = torch.ones(2).type(torch.LongTensor)
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert acc.compute() == 1.0
