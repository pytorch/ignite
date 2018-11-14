from ignite.exceptions import NotComputableError
from ignite.metrics import MultilabelAccuracy
import pytest
import torch
from sklearn.metrics import accuracy_score


def test_zero_div():
    acc = MultilabelAccuracy()
    with pytest.raises(NotComputableError):
        acc.compute()


def test_compute():
    acc = MultilabelAccuracy()

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


def test_incorrect_shape():
    acc = MultilabelAccuracy()

    y_pred = torch.round(torch.rand(4, 1))
    y = torch.ones(4, 1).type(torch.LongTensor)

    with pytest.raises(ValueError):
        acc.update((y_pred, y))


def test_incorrect_output():
    acc = MultilabelAccuracy(threshold_function=lambda x: x + 2)

    y_pred = torch.rand(4, 4)
    y = torch.ones(4, 4).type(torch.LongTensor)

    with pytest.raises(ValueError):
        acc.update((y_pred, y))

    acc.reset()
    y_pred = torch.round(torch.rand(4, 4))
    y = torch.LongTensor(16).random_(0, 10).view(4, 4)

    with pytest.raises(ValueError):
        acc.update((y_pred, y))
