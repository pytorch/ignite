from ignite.exceptions import NotComputableError
from ignite.metrics import CategoricalAccuracy
import pytest
import torch
from sklearn.metrics import accuracy_score


def test_zero_div():
    acc = CategoricalAccuracy()
    with pytest.raises(NotComputableError):
        acc.compute()


def test_compute():
    acc = CategoricalAccuracy()

    y_pred = torch.softmax(torch.rand(4, 4), dim=1)
    y = torch.ones(4).type(torch.LongTensor)
    indices = torch.max(y_pred, dim=1)[1]
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert accuracy_score(y.view(-1).data.numpy(), indices.view(-1).data.numpy()) == pytest.approx(acc.compute())

    acc.reset()
    y_pred = torch.softmax(torch.rand(2, 2), dim=1)
    y = torch.ones(2).type(torch.LongTensor)
    indices = torch.max(y_pred, dim=1)[1]
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert accuracy_score(y.view(-1).data.numpy(), indices.view(-1).data.numpy()) == pytest.approx(acc.compute())


def test_compute_batch_images():
    acc = CategoricalAccuracy()

    y_pred = torch.softmax(torch.rand(2, 3, 2, 2), dim=1)
    y = torch.LongTensor([[[0, 1],
                           [0, 1]],
                          [[0, 2],
                           [0, 2]]])
    indices = torch.max(y_pred, dim=1)[1]
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert accuracy_score(y.view(-1).data.numpy(), indices.view(-1).data.numpy()) == pytest.approx(acc.compute())
