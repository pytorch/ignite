from ignite.exceptions import NotComputableError
from ignite.metrics import BinaryAccuracy
import pytest
import torch
from sklearn.metrics import accuracy_score


def test_zero_div():
    acc = BinaryAccuracy()
    with pytest.raises(NotComputableError):
        acc.compute()


def test_compute():
    acc = BinaryAccuracy()

    y_pred = torch.sigmoid(torch.rand(4, 1))
    y = torch.ones(4).type(torch.LongTensor)
    indices = torch.max(torch.cat([1.0 - y_pred, y_pred], dim=1), dim=1)[1]
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert accuracy_score(y.data.numpy(), indices.data.numpy()) == pytest.approx(acc.compute())

    acc.reset()
    y_pred = torch.sigmoid(torch.rand(4))
    y = torch.ones(4).type(torch.LongTensor)
    y_pred = y_pred.unsqueeze(1)
    indices = torch.max(torch.cat([1.0 - y_pred, y_pred], dim=1), dim=1)[1]
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert accuracy_score(y.data.numpy(), indices.data.numpy()) == pytest.approx(acc.compute())


def test_compute_batch_images():
    acc = BinaryAccuracy()

    y_pred = torch.sigmoid(torch.rand(1, 2, 2))
    y = torch.ones(1, 2, 2).type(torch.LongTensor)
    y_pred = y_pred.unsqueeze(1)
    indices = torch.max(torch.cat([1.0 - y_pred, y_pred], dim=1), dim=1)[1]
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert accuracy_score(y.view(-1).data.numpy(), indices.view(-1).data.numpy()) == pytest.approx(acc.compute())

    acc.reset()
    y_pred = torch.sigmoid(torch.rand(2, 1, 2, 2))
    y = torch.ones(2, 2, 2).type(torch.LongTensor)
    indices = torch.max(torch.cat([1.0 - y_pred, y_pred], dim=1), dim=1)[1]
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert accuracy_score(y.view(-1).data.numpy(), indices.view(-1).data.numpy()) == pytest.approx(acc.compute())

    acc.reset()
    y_pred = torch.sigmoid(torch.rand(2, 1, 2, 2))
    y = torch.ones(2, 1, 2, 2).type(torch.LongTensor)
    indices = torch.max(torch.cat([1.0 - y_pred, y_pred], dim=1), dim=1)[1]
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert accuracy_score(y.view(-1).data.numpy(), indices.view(-1).data.numpy()) == pytest.approx(acc.compute())
