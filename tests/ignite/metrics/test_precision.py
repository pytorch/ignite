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


def test_compute_average():
    precision = Precision(average=True)

    y_pred = torch.eye(4)
    y = torch.ones(4).type(torch.LongTensor)
    precision.update((y_pred, y))
    assert isinstance(precision.compute(), float)
    assert precision.compute() == 0.25


def test_compute_all_wrong():
    precision = Precision()

    y_pred = torch.FloatTensor([[1.0, 0.0], [1.0, 0.0]])
    y = torch.ones(2).type(torch.LongTensor)
    precision.update((y_pred, y))

    results = list(precision.compute())

    assert results[0] == 0.0
    assert results[1] == 0.0


def test_binary_vs_categorical():
    precision_sigmoid = Precision(average=True)
    precision_softmax = Precision(average=True)

    y_pred_binary = torch.FloatTensor([0.9, 0.2])
    y_pred_categorical = torch.FloatTensor([[0.1, 0.9], [0.8, 0.2]])
    y = torch.LongTensor([1, 0])

    precision_sigmoid.update((y_pred_binary, y))
    precision_softmax.update((y_pred_categorical, y))

    results_sigmoid = precision_sigmoid.compute()
    results_softmax = precision_softmax.compute()

    assert results_sigmoid == results_softmax
    assert results_sigmoid == 1.0
    assert results_softmax == 1.0


def test_binary_shapes():
    precision = Precision(average=True)

    y = torch.LongTensor([1, 0])
    y_pred_ndim = torch.FloatTensor([[0.9], [0.2]])
    y_pred_1dim = torch.FloatTensor([0.9, 0.2])

    precision.update((y_pred_1dim, y))
    results_1dim = precision.compute()
    precision.reset()

    precision.update((y_pred_ndim, y))
    results_ndim = precision.compute()

    assert results_1dim == results_ndim
    assert results_1dim == 1.0
    assert results_ndim == 1.0


def test_multilabel_average():
    precision = Precision(average=True)

    y = torch.eye(4).type(torch.LongTensor)
    y_pred = torch.Tensor([[1., 0., 1., 0.],
                           [0., 1., 0., 0.],
                           [0., 0., 1., 0.],
                           [1., 0., 0., 1.]])

    precision.update((y_pred, y))
    results = precision.compute()

    assert results == 0.75


def test_multilabel():
    precision = Precision()

    y = torch.eye(4).type(torch.LongTensor)
    y_pred = torch.Tensor([[1., 0., 1., 0.],
                           [0., 1., 0., 0.],
                           [0., 0., 1., 0.],
                           [1., 0., 0., 1.]])
    precision.update((y_pred, y))
    results = precision.compute()

    assert results[0] == 0.5
    assert results[1] == 1.0
    assert results[2] == 0.5
    assert results[3] == 1.0


def test_ner_example():
    precision = Precision()

    y = torch.Tensor([[1, 1, 1, 1, 1, 1, 1, 1],
                      [2, 2, 2, 2, 2, 2, 2, 2]]).type(torch.LongTensor)

    y_pred = torch.zeros(2, 3, 8)
    y_pred[0, 1, :] = 1
    y_pred[1, 2, :] = 1

    precision.update((y_pred, y))
    results = precision.compute()

    assert results[0] == 0.
    assert results[1] == 1.
    assert results[2] == 1.


def test_ner_multilabel_example():
    precision = Precision()

    y_pred = torch.zeros(2, 3, 8)
    y_pred[0, 1, :] = 1
    y_pred[0, 0, :] = 1
    y_pred[1, 2, :] = 1

    y = torch.zeros(2, 3, 8)
    y[0, 1, :] = 1
    y[0, 0, :] = 1
    y[1, 0, :] = 1
    y[1, 2, :] = 1

    precision.update((y_pred, y))
    results = precision.compute()

    assert results[0] == 1.
    assert results[1] == 1.
    assert results[2] == 1.
