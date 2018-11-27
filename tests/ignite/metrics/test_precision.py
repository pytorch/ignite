from ignite.exceptions import NotComputableError
from ignite.metrics import Precision
import pytest
import torch
from sklearn.metrics import precision_score


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
    precision = Precision(average=True)

    y_pred = torch.FloatTensor([0.9, 0.2])
    y = torch.LongTensor([1, 0])
    y_pred = torch.round(y_pred)
    precision.update((y_pred, y))
    assert precision.compute() == pytest.approx(precision_score(y.numpy(), y_pred.numpy(), average='macro'))
    assert precision.compute() == 1.0

    precision.reset()
    y_pred = torch.FloatTensor([[0.1, 0.9], [0.8, 0.2]])
    y = torch.LongTensor([1, 0])
    indices = torch.max(y_pred, dim=1)[1]
    precision.update((y_pred, y))
    assert precision.compute() == pytest.approx(precision_score(y.numpy(), indices.numpy(), average='macro'))
    assert precision.compute() == 1.0


def test_binary_shapes():
    precision = Precision(average=True)

    y = torch.LongTensor([1, 0])
    y_pred = torch.FloatTensor([0.9, 0.2])
    y_pred = torch.round(y_pred)
    precision.update((y_pred, y))
    assert precision.compute() == pytest.approx(precision_score(y.numpy(), y_pred.numpy(), average='macro'))
    assert precision.compute() == 1.0

    y = torch.LongTensor([[1], [0]])
    y_pred = torch.FloatTensor([[0.9], [0.2]])
    y_pred = torch.round(y_pred)
    precision.reset()
    precision.update((y_pred, y))
    assert precision.compute() == pytest.approx(precision_score(y.numpy(), y_pred.numpy(), average='macro'))
    assert precision.compute() == 1.0


def test_ner_example():
    precision = Precision()

    y = torch.Tensor([[1, 1, 1, 1, 1, 1, 1, 1],
                      [2, 2, 2, 2, 2, 2, 2, 2]]).type(torch.LongTensor)
    y_pred = torch.softmax(torch.rand(2, 3, 8), dim=1)
    indices = torch.max(y_pred, dim=1)[1]
    y_pred_labels = list(set(indices.view(-1).tolist()))

    precision_sk = precision_score(y.view(-1).numpy(),
                                   indices.view(-1).numpy(),
                                   labels=y_pred_labels,
                                   average=None)
    precision.update((y_pred, y))
    precision_ig = precision.compute().tolist()
    precision_ig = [precision_ig[i] for i in y_pred_labels]

    assert all([a == pytest.approx(b) for a, b in zip(precision_sk, precision_ig)])


def test_incorrect_shape():
    precision = Precision()

    y_pred = torch.zeros(2, 3, 2, 2)
    y = torch.zeros(2, 3)

    with pytest.raises(ValueError):
        precision.update((y_pred, y))

    y_pred = torch.zeros(2, 3, 2, 2)
    y = torch.zeros(2, 3, 4, 4)

    with pytest.raises(ValueError):
        precision.update((y_pred, y))


def test_sklearn_compute():
    precision = Precision(average=False)

    y = torch.Tensor(range(5)).type(torch.LongTensor)
    y_pred = torch.softmax(torch.rand(5, 5), dim=1)

    indices = torch.max(y_pred, dim=1)[1]
    precision.update((y_pred, y))

    y_pred_labels = list(set(indices.tolist()))

    precision_sk = precision_score(y.numpy(),
                                   indices.numpy(),
                                   labels=y_pred_labels,
                                   average=None)

    precision_ig = precision.compute().tolist()
    precision_ig = [precision_ig[i] for i in y_pred_labels]

    assert all([a == pytest.approx(b) for a, b in zip(precision_sk, precision_ig)])


def test_multilabel_example():
    precision = Precision(is_multilabel=True, average=True, threshold_function=torch.round)

    # N x C case
    y_pred = torch.round(torch.rand(4, 4))
    y = torch.ones(4, 4).type(torch.LongTensor)

    precision.update((y_pred, y))
    assert precision.compute() == pytest.approx(precision_score(y.numpy(), y_pred.numpy(), average='samples'))

    # N x C x L case
    y_pred = torch.round(torch.rand(4, 5, 3))
    y = torch.ones(4, 5, 3).type(torch.LongTensor)

    precision.reset()
    precision.update((y_pred, y))
    num_classes = y_pred.size(1)
    y_pred = torch.transpose(y_pred, 1, 0).contiguous().view(num_classes, -1).transpose(1, 0)
    y = torch.transpose(y, 1, 0).contiguous().view(num_classes, -1).transpose(1, 0)
    assert precision.compute() == pytest.approx(precision_score(y.numpy(), y_pred.numpy(), average='samples'))

    # N x C x H x W
    y_pred = torch.round(torch.rand(4, 5, 3, 3))
    y = torch.ones(4, 5, 3, 3).type(torch.LongTensor)

    precision.reset()
    precision.update((y_pred, y))
    num_classes = y_pred.size(1)
    y_pred = torch.transpose(y_pred, 1, 0).contiguous().view(num_classes, -1).transpose(1, 0)
    y = torch.transpose(y, 1, 0).contiguous().view(num_classes, -1).transpose(1, 0)
    assert precision.compute() == pytest.approx(precision_score(y.numpy(), y_pred.numpy(), average='samples'))


def test_incorrect_multilabel_output():
    precision = Precision(is_multilabel=True, average=True, threshold_function=lambda x: x + 2)

    y_pred = torch.rand(4, 4)
    y = torch.ones(4, 4).type(torch.LongTensor)

    with pytest.raises(ValueError):
        precision.update((y_pred, y))

    precision = Precision(is_multilabel=True, average=True)

    y_pred = torch.round(torch.rand(4, 4))
    y = torch.LongTensor(16).random_(0, 10).view(4, 4)

    with pytest.raises(ValueError):
        precision.update((y_pred, y))


def test_multilabel_average_parameter():
    with pytest.warns(UserWarning):
        precision = Precision(is_multilabel=True, average=False)


def test_multilabel_incorrect_threshold():
    with pytest.raises(ValueError):
        precision = Precision(is_multilabel=True, threshold_function=2)


def test_multilabel_incorrect_shape():
    precision = Precision(is_multilabel=True, average=True)

    y_pred = torch.round(torch.rand(4, 1))
    y = torch.ones(4, 1).type(torch.LongTensor)

    with pytest.raises(ValueError):
        precision.update((y_pred, y))


def test_multilabel_compute_all_wrong():
    precision = Precision(is_multilabel=True, average=True)

    y = torch.ones(4, 4).type(torch.LongTensor)
    y_pred = torch.zeros(4, 4)

    precision.update((y_pred, y))
    assert precision.compute() == pytest.approx(precision_score(y.numpy(), y_pred.numpy(), average='samples'))


def test_mutlilabel_batch_update():
    precision = Precision(is_multilabel=True, average=True)

    y = torch.ones(2, 3).type(torch.LongTensor)
    y_pred = torch.rand(2, 3)

    precision.update((y_pred, y))
    precision.update((y_pred, y))
    precision.update((y_pred, y))

    y = torch.cat([y, y, y], dim=0)
    y_pred = torch.round(torch.cat([y_pred, y_pred, y_pred], dim=0))

    assert precision.compute() == pytest.approx(precision_score(y.numpy(), y_pred.numpy(), average='samples'))
