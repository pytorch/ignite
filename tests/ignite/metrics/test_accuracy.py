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
    indices = torch.max(torch.cat([1.0 - y_pred, y_pred], dim=1), dim=1)[1]
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert accuracy_score(y.numpy(), indices.numpy()) == pytest.approx(acc.compute())

    acc.reset()
    y_pred = torch.rand(10)
    y = torch.randint(0, 2, size=(10,)).type(torch.LongTensor)
    y_pred = y_pred.unsqueeze(1)
    indices = torch.max(torch.cat([1.0 - y_pred, y_pred], dim=1), dim=1)[1]
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert accuracy_score(y.numpy(), indices.numpy()) == pytest.approx(acc.compute())


def test_compute_batch_images():
    acc = Accuracy()

    y_pred = torch.rand(10, 2, 2)
    y = torch.randint(0, 2, size=(10, 2, 2)).type(torch.LongTensor)
    y_pred = y_pred.unsqueeze(1)
    indices = torch.max(torch.cat([1.0 - y_pred, y_pred], dim=1), dim=1)[1]
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert accuracy_score(y.view(-1).numpy(), indices.view(-1).numpy()) == pytest.approx(acc.compute())

    acc.reset()
    y_pred = torch.rand(10, 1, 2, 2)
    y = torch.randint(0, 2, size=(10, 2, 2)).type(torch.LongTensor)
    indices = torch.max(torch.cat([1.0 - y_pred, y_pred], dim=1), dim=1)[1]
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert accuracy_score(y.view(-1).numpy(), indices.view(-1).numpy()) == pytest.approx(acc.compute())

    acc.reset()
    y_pred = torch.rand(10, 1, 2, 2)
    y = torch.randint(0, 2, size=(10, 1, 2, 2)).type(torch.LongTensor)
    indices = torch.max(torch.cat([1.0 - y_pred, y_pred], dim=1), dim=1)[1]
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert accuracy_score(y.view(-1).numpy(), indices.view(-1).numpy()) == pytest.approx(acc.compute())


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


def test_multilabel_compute():
    acc = Accuracy(is_multilabel=True, threshold_function=torch.round)

    # N x C case
    y_pred = torch.rand(4, 10)
    y = torch.randint(0, 2, size=(4, 10)).type(torch.LongTensor)

    acc.update((y_pred, y))
    assert acc.compute() == pytest.approx(accuracy_score(y.numpy(), torch.round(y_pred).numpy()))

    # N x C x L case
    y_pred = torch.rand(2, 3, 8)
    y = torch.randint(0, 2, size=(2, 3, 8)).type(torch.LongTensor)

    acc.reset()
    acc.update((y_pred, y))

    num_classes = y_pred.size(1)
    np_y_pred = torch.round(y_pred).numpy().transpose((0, 2, 1)).reshape(-1, num_classes)
    np_y = y.numpy().transpose((0, 2, 1)).reshape(-1, num_classes)
    assert acc.compute() == pytest.approx(accuracy_score(np_y, np_y_pred))

    # N x C x H x W
    y_pred = torch.rand(4, 5, 3, 3)
    y = torch.randint(0, 2, size=(4, 5, 3, 3)).type(torch.LongTensor)

    acc.reset()
    acc.update((y_pred, y))
    num_classes = y_pred.size(1)
    np_y_pred = torch.round(y_pred).numpy().transpose((0, 2, 3, 1)).reshape(-1, num_classes)
    np_y = y.numpy().transpose((0, 2, 3, 1)).reshape(-1, num_classes)
    assert acc.compute() == pytest.approx(accuracy_score(np_y, np_y_pred))


def test_multilabel_incorrect_shape():
    acc = Accuracy(is_multilabel=True)

    y_pred = torch.rand(4, 1)
    y = torch.ones(4, 1).type(torch.LongTensor)

    with pytest.raises(ValueError):
        acc.update((y_pred, y))


def test_multilabel_incorrect_output():
    acc = Accuracy(is_multilabel=True, threshold_function=lambda x: x + 2)

    y_pred = torch.rand(4, 4)
    y = torch.ones(4, 4).type(torch.LongTensor)

    with pytest.raises(ValueError):
        acc.update((y_pred, y))

    acc.reset()
    y_pred = torch.round(torch.rand(4, 4))
    y = torch.LongTensor(16).random_(0, 10).view(4, 4)

    with pytest.raises(ValueError):
        acc.update((y_pred, y))


def test_multilabel_incorrect_threshold():
    with pytest.raises(ValueError):
        Accuracy(is_multilabel=True, threshold_function=2)
