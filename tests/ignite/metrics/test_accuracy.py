from ignite.exceptions import NotComputableError
from ignite.metrics import Accuracy
import pytest
import torch
from sklearn.metrics import accuracy_score


torch.manual_seed(12)


def test_no_update():
    acc = Accuracy()
    with pytest.raises(NotComputableError):
        acc.compute()


def test_binary_compute():
    acc = Accuracy()

    y_pred = torch.rand(10, 1)
    y = torch.randint(0, 2, size=(10,)).type(torch.LongTensor)
    acc.update((y_pred, y))
    np_y = y.numpy().ravel()
    np_y_pred = (y_pred.numpy().ravel() > 0.5).astype('int')
    assert isinstance(acc.compute(), float)
    assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

    acc.reset()
    y_pred = torch.rand(10)
    y = torch.randint(0, 2, size=(10,)).type(torch.LongTensor)
    y_pred = y_pred.unsqueeze(1)
    acc.update((y_pred, y))
    np_y = y.numpy().ravel()
    np_y_pred = (y_pred.numpy().ravel() > 0.5).astype('int')
    assert isinstance(acc.compute(), float)
    assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())


def test_compute_batch_images():
    acc = Accuracy()

    y_pred = torch.rand(10, 2, 2)
    y = torch.randint(0, 2, size=(10, 2, 2)).type(torch.LongTensor)
    y_pred = y_pred.unsqueeze(1)
    acc.update((y_pred, y))
    np_y = y.numpy().ravel()
    np_y_pred = (y_pred.numpy().ravel() > 0.5).astype('int')
    assert isinstance(acc.compute(), float)
    assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

    acc.reset()
    y_pred = torch.rand(10, 1, 2, 2)
    y = torch.randint(0, 2, size=(10, 2, 2)).type(torch.LongTensor)
    acc.update((y_pred, y))
    np_y = y.numpy().ravel()
    np_y_pred = (y_pred.numpy().ravel() > 0.5).astype('int')
    assert isinstance(acc.compute(), float)
    assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

    acc.reset()
    y_pred = torch.rand(10, 1, 2, 2)
    y = torch.randint(0, 2, size=(10, 1, 2, 2)).type(torch.LongTensor)
    acc.update((y_pred, y))
    np_y = y.numpy().ravel()
    np_y_pred = (y_pred.numpy().ravel() > 0.5).astype('int')
    assert isinstance(acc.compute(), float)
    assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())


def test_categorical_compute():
    acc = Accuracy()

    y_pred = torch.softmax(torch.rand(10, 4), dim=1)
    y = torch.randint(0, 4, size=(10,)).type(torch.LongTensor)
    indices = torch.max(y_pred, dim=1)[1]
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert accuracy_score(y.view(-1).numpy(), indices.view(-1).numpy()) == pytest.approx(acc.compute())

    acc.reset()
    y_pred = torch.softmax(torch.rand(4, 10), dim=1)
    y = torch.randint(0, 10, size=(4,)).type(torch.LongTensor)
    indices = torch.max(y_pred, dim=1)[1]
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert accuracy_score(y.view(-1).numpy(), indices.view(-1).numpy()) == pytest.approx(acc.compute())


def test_categorical_compute_batch_images():
    acc = Accuracy()

    y_pred = torch.softmax(torch.rand(4, 3, 64, 48), dim=1)
    y = torch.randint(0, 4, size=(4, 64, 48)).type(torch.LongTensor)
    indices = torch.max(y_pred, dim=1)[1]
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert accuracy_score(y.view(-1).numpy(), indices.view(-1).numpy()) == pytest.approx(acc.compute())


def test_ner_example():
    acc = Accuracy()

    y_pred = torch.softmax(torch.rand(2, 3, 8), dim=1)
    y = torch.randint(0, 3, size=(2, 8)).type(torch.LongTensor)
    indices = torch.max(y_pred, dim=1)[1]
    acc.update((y_pred, y))
    assert accuracy_score(y.view(-1).numpy(), indices.view(-1).numpy()) == pytest.approx(acc.compute())


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


def test_incorrect_binary_y():
    acc = Accuracy()

    y_pred = torch.rand(4, 1)
    y = 2 * torch.ones(4, 1).type(torch.LongTensor)

    with pytest.raises(ValueError):
        acc.update((y_pred, y))


def test_incorrect_type():
    acc = Accuracy()

    y_pred = torch.softmax(torch.rand(4, 4), dim=1)
    y = torch.ones(4).type(torch.LongTensor)
    acc.update((y_pred, y))

    y_pred = torch.rand(4, 1)
    y = torch.ones(4).type(torch.LongTensor)

    with pytest.raises(TypeError):
        acc.update((y_pred, y))
