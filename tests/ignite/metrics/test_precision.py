from ignite.exceptions import NotComputableError
from ignite.metrics import Precision
import pytest
import torch
from sklearn.metrics import precision_score


torch.manual_seed(12)


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
    assert precision._type == 'multiclass'
    assert results[0] == 0.0
    assert results[1] == 1.0
    assert results[2] == 0.0
    assert results[3] == 0.0

    precision = Precision()
    y_pred = torch.eye(2)
    y = torch.ones(2).type(torch.LongTensor)
    with pytest.warns(UserWarning):
        precision.update((y_pred, y))
    y = torch.zeros(2).type(torch.LongTensor)
    precision.update((y_pred, y))

    results = list(precision.compute())
    assert precision._type == 'binary_multiclass'
    assert results[0] == 0.5
    assert results[1] == 0.5


def test_compute_average():
    precision = Precision(average=True)

    y_pred = torch.eye(4)
    y = torch.ones(4).type(torch.LongTensor)
    precision.update((y_pred, y))
    assert precision._type == 'multiclass'
    assert isinstance(precision.compute(), float)
    assert precision.compute() == 0.25


def test_compute_all_wrong():
    precision = Precision()

    y_pred = torch.FloatTensor([[1.0, 0.0], [1.0, 0.0]])
    y = torch.ones(2).type(torch.LongTensor)
    with pytest.warns(UserWarning):
        precision.update((y_pred, y))

    results = list(precision.compute())

    assert precision._type == 'binary_multiclass'
    assert results[0] == 0.0
    assert results[1] == 0.0


def test_binary_shapes():
    precision = Precision(average=True)

    y = torch.randint(0, 2, size=(10,)).type(torch.LongTensor)
    y_pred = torch.rand(10, 1)
    precision.update((y_pred, y))
    np_y = y.numpy()
    np_y_pred = (y_pred.numpy().ravel() > 0.5).astype('int')
    assert precision._type == 'binary'
    assert precision.compute() == pytest.approx(precision_score(np_y, np_y_pred))

    y = torch.randint(0, 2, size=(10, 1)).type(torch.LongTensor)
    y_pred = torch.rand(10)
    precision.reset()
    precision.update((y_pred, y))
    np_y = y.numpy()
    np_y_pred = (y_pred.numpy().ravel() > 0.5).astype('int')
    assert precision._type == 'binary'
    assert precision.compute() == pytest.approx(precision_score(np_y, np_y_pred))

    precision = Precision()

    y = torch.randint(0, 2, size=(10,)).type(torch.LongTensor)
    y_pred = torch.rand(10)
    precision.update((y_pred, y))
    np_y = y.numpy()
    np_y_pred = (y_pred.numpy().ravel() > 0.5).astype('int')
    assert precision._type == 'binary'
    assert precision.compute().numpy() == pytest.approx(precision_score(np_y, np_y_pred, average=None))

    y = torch.randint(0, 2, size=(10, 1)).type(torch.LongTensor)
    y_pred = torch.rand(10, 1)
    precision.reset()
    precision.update((y_pred, y))
    np_y = y.numpy()
    np_y_pred = (y_pred.numpy().ravel() > 0.5).astype('int')
    assert precision._type == 'binary'
    assert precision.compute().numpy() == pytest.approx(precision_score(np_y, np_y_pred, average=None))

    precision = Precision(average=False)
    y_pred = torch.softmax(torch.rand(10, 2), dim=1)
    y = torch.randint(0, 2, size=(10,)).type(torch.LongTensor)
    indices = torch.max(y_pred, dim=1)[1]
    with pytest.warns(UserWarning):
        precision.update((y_pred, y))
    assert precision._type == 'binary_multiclass'
    assert precision.compute().numpy() == pytest.approx(precision_score(y.numpy(), indices.numpy(), average=None))

    precision = Precision(average=True)
    y_pred = torch.softmax(torch.rand(10, 2), dim=1)
    y = torch.randint(0, 2, size=(10,)).type(torch.LongTensor)
    indices = torch.max(y_pred, dim=1)[1]
    with pytest.warns(UserWarning):
        precision.update((y_pred, y))
    assert precision._type == 'binary_multiclass'
    assert precision.compute() == pytest.approx(precision_score(y.numpy(), indices.numpy(), average='binary'))


def test_ner_example():
    precision = Precision()

    y_pred = torch.softmax(torch.rand(2, 3, 8), dim=1)
    y = torch.randint(0, 3, size=(2, 8)).type(torch.LongTensor)
    indices = torch.max(y_pred, dim=1)[1]
    y_pred_labels = list(set(indices.view(-1).tolist()))

    precision_sk = precision_score(y.view(-1).data.numpy(),
                                   indices.view(-1).data.numpy(),
                                   labels=y_pred_labels,
                                   average=None)
    precision.update((y_pred, y))
    precision_ig = precision.compute().tolist()
    precision_ig = [precision_ig[i] for i in y_pred_labels]

    assert precision._type == 'multiclass'
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

    precision_sk = precision_score(y.data.numpy(),
                                   indices.data.numpy(),
                                   labels=y_pred_labels,
                                   average=None)

    precision_ig = precision.compute().tolist()
    precision_ig = [precision_ig[i] for i in y_pred_labels]

    assert precision._type == 'multiclass'
    assert all([a == pytest.approx(b) for a, b in zip(precision_sk, precision_ig)])


def test_incorrect_binary_y():
    precision = Precision()

    y_pred = torch.rand(4, 1)
    y = 2 * torch.ones(4, 1).type(torch.LongTensor)

    with pytest.raises(ValueError):
        precision.update((y_pred, y))


def test_incorrect_type():
    precision = Precision()

    y_pred = torch.softmax(torch.rand(4, 4), dim=1)
    y = torch.ones(4).type(torch.LongTensor)
    precision.update((y_pred, y))

    y_pred = torch.rand(4, 1)
    y = torch.ones(4).type(torch.LongTensor)

    with pytest.raises(TypeError):
        precision.update((y_pred, y))
