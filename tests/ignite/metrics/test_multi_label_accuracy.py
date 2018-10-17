from ignite.exceptions import NotComputableError
from ignite.metrics import MultiLabelAccuracy
import pytest
import torch


def test_zero_div():
    acc = MultiLabelAccuracy()
    with pytest.raises(NotComputableError):
        acc.compute()


def test_incorrect_shape():
    acc = MultiLabelAccuracy()

    y_pred = torch.zeros(2, 6)
    y = torch.zeros(2, 4).type(torch.LongTensor)

    with pytest.raises(ValueError):
        acc.update((y_pred, y))

    y_pred = torch.zeros(2, 1)
    y = torch.zeros(2, 1).type(torch.LongTensor)

    with pytest.raises(ValueError):
        acc.update((y_pred, y))

    y_pred = torch.zeros(2)
    y = torch.zeros(2).type(torch.LongTensor)

    with pytest.raises(ValueError):
        acc.update((y_pred, y))


def test_compute():
    acc = MultiLabelAccuracy()

    y_pred = torch.eye(4)
    y = torch.Tensor([[1., 0., 1., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 1., 0.],
                      [1., 0., 0., 1.]]).type(torch.LongTensor)

    acc.update((y_pred, y))
    assert acc.compute() == 0.5
