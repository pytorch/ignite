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


def test_multilabel_compute():
    acc = Accuracy(is_multilabel=True)

    # N x C case
    y_pred = torch.round(torch.rand(4, 4))
    y = torch.ones(4, 4).type(torch.LongTensor)

    acc.update((y_pred, y))
    assert acc.compute() == pytest.approx(accuracy_score(y.data.numpy(), y_pred.data.numpy()))

    # N x C x L case
    y_pred = torch.round(torch.rand(4, 5, 3))
    y = torch.ones(4, 5, 3).type(torch.LongTensor)

    acc.reset()
    acc.update((y_pred, y))
    num_classes = y_pred.size(1)
    y_pred = y_pred.transpose(2, 1).contiguous().view(-1, num_classes)
    y = y.transpose(2, 1).contiguous().view(-1, num_classes)
    assert acc.compute() == pytest.approx(accuracy_score(y.data.numpy(), y_pred.data.numpy()))

    # N x C x H x W
    y_pred = torch.round(torch.rand(4, 5, 3, 3))
    y = torch.ones(4, 5, 3, 3).type(torch.LongTensor)

    acc.reset()
    acc.update((y_pred, y))
    num_classes = y_pred.size(1)
    y_pred = y_pred.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
    y = y.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
    assert acc.compute() == pytest.approx(accuracy_score(y.data.numpy(), y_pred.data.numpy()))


def test_multilabel_incorrect_shape():
    acc = Accuracy(is_multilabel=True)

    y_pred = torch.round(torch.rand(4, 1))
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
