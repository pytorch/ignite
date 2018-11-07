from ignite.exceptions import NotComputableError
from ignite.metrics import Accuracy
import pytest
import torch
from sklearn.metrics import accuracy_score


def test_zero_div():
    acc = Accuracy()
    with pytest.raises(NotComputableError):
        acc.compute()


def test_binary_compute():
    acc = Accuracy()

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
    acc = Accuracy()

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


def test_categorical_compute():
    acc = Accuracy()

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


def test_categorical_compute_batch_images():
    acc = Accuracy()

    y_pred = torch.softmax(torch.rand(2, 3, 2, 2), dim=1)
    y = torch.LongTensor([[[0, 1],
                           [0, 1]],
                          [[0, 2],
                           [0, 2]]])
    indices = torch.max(y_pred, dim=1)[1]
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert accuracy_score(y.view(-1).data.numpy(), indices.view(-1).data.numpy()) == pytest.approx(acc.compute())


def test_ner_example():
    acc = Accuracy()

    y_pred = torch.softmax(torch.rand(2, 3, 8), dim=1)
    y = torch.Tensor([[1, 1, 1, 1, 1, 1, 1, 1],
                      [2, 2, 2, 2, 2, 2, 2, 2]]).type(torch.LongTensor)
    indices = torch.max(y_pred, dim=1)[1]
    acc.update((y_pred, y))
    assert accuracy_score(y.view(-1).data.numpy(), indices.view(-1).data.numpy()) == pytest.approx(acc.compute())


def test_incorrect_shape():
    acc = Accuracy()

    y_pred = torch.zeros(2, 3, 2, 2)
    y = torch.zeros(2, 3)

    with pytest.raises(ValueError):
        acc.update((y_pred, y))

    y_pred = torch.zeros(2, 3, 2, 2)
    y = torch.zeros(2, 3, 4, 4)

    with pytest.raises(ValueError):
        acc.update((y_pred, y))
