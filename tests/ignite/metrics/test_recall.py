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

    assert result[0] == 0.0
    assert result[1] == 0.25
    assert result[2] == 0.0
    assert result[3] == 0.0

    recall.reset()
    y_pred = torch.eye(2)
    y = torch.ones(2).type(torch.LongTensor)
    recall.update((y_pred, y))
    y = torch.zeros(2).type(torch.LongTensor)
    recall.update((y_pred, y))

    result = list(recall.compute())

    assert result[0] == 0.5
    assert result[1] == 0.5


def test_compute_average():
    recall = Recall(average=True)

    y_pred = torch.eye(4)
    y = torch.ones(4).type(torch.LongTensor)
    recall.update((y_pred, y))

    assert isinstance(recall.compute(), float)
    assert recall.compute() == 0.0625


def test_compute_all_wrong():
    recall = Recall()

    y_pred = torch.FloatTensor([[1.0, 0.0], [1.0, 0.0]])
    y = torch.ones(2).type(torch.LongTensor)
    recall.update((y_pred, y))

    result = list(recall.compute())

    assert result[0] == 0.0
    assert result[1] == 0.0


def test_binary_vs_categorical():
    recall_sigmoid = Recall(average=True)
    recall_softmax = Recall(average=True)

    y_pred_binary = torch.FloatTensor([0.9, 0.2])
    y_pred_categorical = torch.FloatTensor([[0.1, 0.9], [0.8, 0.2]])
    y = torch.LongTensor([1, 0])

    recall_sigmoid.update((y_pred_binary, y))
    recall_softmax.update((y_pred_categorical, y))

    results_sigmoid = recall_sigmoid.compute()
    results_softmax = recall_softmax.compute()

    assert results_sigmoid == results_softmax
    assert results_sigmoid == 1.0
    assert results_softmax == 1.0


def test_binary_shapes():
    recall = Recall(average=True)

    y = torch.LongTensor([1, 0])
    y_pred_ndim = torch.FloatTensor([[0.9], [0.2]])
    y_pred_1dim = torch.FloatTensor([0.9, 0.2])

    recall.update((y_pred_1dim, y))
    results_1dim = recall.compute()
    recall.reset()

    recall.update((y_pred_ndim, y))
    results_ndim = recall.compute()

    assert results_1dim == results_ndim
    assert results_1dim == 1.0
    assert results_ndim == 1.0


def test_multilabel_average():
    recall = Recall(average=True)

    y = torch.eye(4).type(torch.LongTensor)
    y_pred = torch.Tensor([[1., 0., 1., 0.],
                           [0., 1., 0., 0.],
                           [0., 0., 1., 0.],
                           [1., 0., 0., 1.]])

    recall.update((y_pred, y))
    results = recall.compute()

    assert results == 1.0


def test_multilabel():
    recall = Recall()

    y = torch.eye(4).type(torch.LongTensor)
    y_pred = torch.Tensor([[1., 0., 1., 0.],
                           [0., 1., 0., 0.],
                           [0., 0., 1., 0.],
                           [1., 0., 0., 1.]])
    recall.update((y_pred, y))
    results = recall.compute()

    assert results[0] == 1.0
    assert results[1] == 1.0
    assert results[2] == 1.0
    assert results[3] == 1.0


def test_ner_example():
    recall = Recall()

    y = torch.Tensor([[1, 1, 1, 1, 1, 1, 1, 1],
                      [2, 2, 2, 2, 2, 2, 2, 2]]).type(torch.LongTensor)

    y_pred = torch.zeros(2, 3, 8)
    y_pred[0, 1, :] = 1
    y_pred[1, 2, :] = 1

    recall.update((y_pred, y))
    results = recall.compute()

    assert results[0] == 0.
    assert results[1] == 1.
    assert results[2] == 1.
