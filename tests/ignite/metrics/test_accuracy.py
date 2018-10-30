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

    acc.reset()
    y_pred = torch.FloatTensor([[0.2], [0.7], [0.8], [0.9]])
    y = torch.ones(4, 1).type(torch.LongTensor)
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert acc.compute() == 0.75


def test_binary_compute_batch_images():
    acc = Accuracy()

    y_pred = torch.FloatTensor([[[0.3, 0.7],
                                 [0.2, 0.8]]])
    y = torch.ones(1, 2, 2).type(torch.LongTensor)
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert acc.compute() == 0.5

    acc.reset()
    y_pred = torch.FloatTensor([[[0.3, 0.7],
                                 [0.9, 0.8]]])
    y = torch.ones(1, 2, 2).type(torch.LongTensor)
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert acc.compute() == 0.75

    acc.reset()
    y_pred = torch.FloatTensor([[[0.3, 0.7],
                                 [0.9, 0.8]],
                                [[0.8, 0.3],
                                 [0.9, 0.4]]])
    y = torch.ones(2, 2, 2).type(torch.LongTensor)
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert acc.compute() == 0.625

    acc.reset()
    y_pred = torch.zeros(2, 1, 2, 2).type(torch.FloatTensor)
    y = torch.ones(2, 2, 2).type(torch.LongTensor)
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert acc.compute() == 0.


def test_categorical_compute():
    acc = Accuracy()

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


def test_categorical_compute_batch_images():
    acc = Accuracy()
    y_pred = torch.zeros(2, 3, 2, 2)
    y_pred[0, 1, :] = 1
    y_pred[1, 2, :] = 1

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


def test_ner_example():
    acc = Accuracy()

    y_pred = torch.zeros(2, 3, 8)
    y_pred[0, 1, :] = 1
    y_pred[1, 2, :] = 1

    y = torch.Tensor([[1, 1, 1, 1, 1, 1, 1, 1],
                      [2, 2, 2, 2, 2, 2, 2, 2]]).type(torch.LongTensor)

    acc.update((y_pred, y))
    assert acc.compute() == 1.0


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


def test_sklearn_compare():
    acc = Accuracy()

    y = torch.Tensor(range(5)).type(torch.LongTensor)
    y_pred = torch.softmax(torch.rand(5, 5), dim=1)

    indices = torch.max(y_pred, dim=1)[1]
    acc.update((y_pred, y))
    assert accuracy_score(y.data.numpy(), indices.data.numpy()) == pytest.approx(acc.compute())
