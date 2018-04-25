from ignite.exceptions import NotComputableError
from ignite.metrics import BinaryAccuracy
import pytest
import torch


def test_zero_div():
    acc = BinaryAccuracy()
    with pytest.raises(NotComputableError):
        acc.compute()


def test_compute():
    acc = BinaryAccuracy()

    y_pred = torch.FloatTensor([0.2, 0.4, 0.6, 0.8])
    y = torch.ones(4).type(torch.LongTensor)
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert acc.compute() == 0.5

    acc.reset()
    y_pred = torch.FloatTensor([0.2, 0.7, 0.8, 0.9])
    y = torch.ones(4).type(torch.LongTensor)
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert acc.compute() == 0.75


def test_compute_batch_images():
    acc = BinaryAccuracy()

    y_pred = torch.FloatTensor([[[0.3, 0.7],
                                 [0.1, 0.6]],
                                [[0.2, 0.7],
                                 [0.2, 0.6]]])
    y = torch.ones(1, 2, 2).type(torch.LongTensor)
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert acc.compute() == 0.5

    acc.reset()
    y_pred = torch.FloatTensor([[[0.3, 0.7],
                                 [0.8, 0.6]],
                                [[0.2, 0.7],
                                 [0.9, 0.6]]])
    y = torch.ones(2, 2, 2).type(torch.LongTensor)
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert acc.compute() == 0.75
