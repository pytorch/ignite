from math import isnan
from ignite.exceptions import NotComputableError
from ignite.metrics import Recall
import pytest
import torch


def test_no_update():
    recall = Recall()
    with pytest.raises(NotComputableError):
        recall.compute()


def test_compute():
    recall = Recall()

    y_pred = torch.eye(4)
    y = torch.ones(4).type(torch.LongTensor)
    recall.update((y_pred, y))
    result = list(recall.compute())
    assert isnan(result[0])
    assert result[1] == 0.25
    assert isnan(result[2])
    assert isnan(result[3])

    recall.reset()
    y_pred = torch.eye(2)
    y = torch.ones(2).type(torch.LongTensor)
    recall.update((y_pred, y))
    y = torch.zeros(2).type(torch.LongTensor)
    recall.update((y_pred, y))
    result = list(recall.compute())
    assert result[0] == 0.5
    assert result[1] == 0.5
