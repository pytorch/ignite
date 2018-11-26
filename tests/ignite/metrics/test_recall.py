from ignite.exceptions import NotComputableError
from ignite.metrics import Recall
import pytest
import torch
from sklearn.metrics import recall_score


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
    recall = Recall(average=True)

    y_pred = torch.FloatTensor([0.9, 0.2])
    y = torch.LongTensor([1, 0])
    y_pred = torch.round(y_pred)
    recall.update((y_pred, y))
    assert recall.compute() == pytest.approx(recall_score(y.data.numpy(), y_pred.data.numpy(), average='macro'))
    assert recall.compute() == 1.0

    recall.reset()
    y_pred = torch.FloatTensor([[0.1, 0.9], [0.8, 0.2]])
    y = torch.LongTensor([1, 0])
    indices = torch.max(y_pred, dim=1)[1]
    recall.update((y_pred, y))
    assert recall.compute() == pytest.approx(recall_score(y.data.numpy(), indices.data.numpy(), average='macro'))
    assert recall.compute() == 1.0


def test_binary_shapes():
    recall = Recall(average=True)

    y = torch.LongTensor([1, 0])
    y_pred = torch.FloatTensor([0.9, 0.2])
    y_pred = torch.round(y_pred)
    recall.update((y_pred, y))
    assert recall.compute() == pytest.approx(recall_score(y.data.numpy(), y_pred.data.numpy(), average='macro'))
    assert recall.compute() == 1.0

    y = torch.LongTensor([[1], [0]])
    y_pred = torch.FloatTensor([[0.9], [0.2]])
    y_pred = torch.round(y_pred)
    recall.reset()
    recall.update((y_pred, y))
    assert recall.compute() == pytest.approx(recall_score(y.data.numpy(), y_pred.data.numpy(), average='macro'))
    assert recall.compute() == 1.0


def test_ner_example():
    recall = Recall()

    y = torch.Tensor([[0, 1, 1, 1, 1, 1, 1, 1],
                      [2, 2, 2, 2, 2, 2, 2, 2]]).type(torch.LongTensor)
    y_pred = torch.softmax(torch.rand(2, 3, 8), dim=1)
    indices = torch.max(y_pred, dim=1)[1]
    y_pred_labels = list(set(indices.view(-1).tolist()))

    recall_sk = recall_score(y.view(-1).data.numpy(),
                             indices.view(-1).data.numpy(),
                             labels=y_pred_labels,
                             average=None)
    recall.update((y_pred, y))
    recall_ig = recall.compute().tolist()
    recall_ig = [recall_ig[i] for i in y_pred_labels]

    assert all([a == pytest.approx(b) for a, b in zip(recall_sk, recall_ig)])


def test_incorrect_shape():
    recall = Recall()

    y_pred = torch.zeros(2, 3, 2, 2)
    y = torch.zeros(2, 3)

    with pytest.raises(ValueError):
        recall.update((y_pred, y))

    y_pred = torch.zeros(2, 3, 2, 2)
    y = torch.zeros(2, 3, 4, 4)

    with pytest.raises(ValueError):
        recall.update((y_pred, y))


def test_sklearn_compute():
    recall = Recall(average=False)

    y = torch.Tensor(range(5)).type(torch.LongTensor)
    y_pred = torch.softmax(torch.rand(5, 5), dim=1)

    indices = torch.max(y_pred, dim=1)[1]
    recall.update((y_pred, y))

    y_pred_labels = list(set(indices.tolist()))

    recall_sk = recall_score(y.data.numpy(),
                             indices.data.numpy(),
                             labels=y_pred_labels,
                             average=None)

    recall_ig = recall.compute().tolist()
    recall_ig = [recall_ig[i] for i in y_pred_labels]

    assert all([a == pytest.approx(b) for a, b in zip(recall_sk, recall_ig)])


def test_multilabel_example():
    recall = Recall(is_multilabel=True, average=True)

    # N x C case
    y_pred = torch.round(torch.rand(4, 4))
    y = torch.ones(4, 4).type(torch.LongTensor)

    recall.update((y_pred, y))
    assert recall.compute() == pytest.approx(recall_score(y.data.numpy(), y_pred.data.numpy(), average='samples'))

    # N x C x L case
    y_pred = torch.round(torch.rand(4, 5, 3))
    y = torch.ones(4, 5, 3).type(torch.LongTensor)

    recall.reset()
    recall.update((y_pred, y))
    num_classes = y_pred.size(1)
    y_pred = y_pred.transpose(2, 1).contiguous().view(-1, num_classes)
    y = y.transpose(2, 1).contiguous().view(-1, num_classes)
    assert recall.compute() == pytest.approx(recall_score(y.data.numpy(), y_pred.data.numpy(), average='samples'))

    # N x C x H x W
    y_pred = torch.round(torch.rand(4, 5, 3, 3))
    y = torch.ones(4, 5, 3, 3).type(torch.LongTensor)

    recall.reset()
    recall.update((y_pred, y))
    num_classes = y_pred.size(1)
    y_pred = y_pred.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
    y = y.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
    assert recall.compute() == pytest.approx(recall_score(y.data.numpy(), y_pred.data.numpy(), average='samples'))


def test_incorrect_multilabel_output():
    recall = Recall(is_multilabel=True, average=True, threshold_function=lambda x: x + 2)

    y_pred = torch.rand(4, 4)
    y = torch.ones(4, 4).type(torch.LongTensor)

    with pytest.raises(ValueError):
        recall.update((y_pred, y))

    recall.reset()
    y_pred = torch.round(torch.rand(4, 4))
    y = torch.LongTensor(16).random_(0, 10).view(4, 4)

    with pytest.raises(ValueError):
        recall.update((y_pred, y))


def test_multilabel_average_parameter():
    with pytest.warns(UserWarning):
        recall = Recall(is_multilabel=True, average=False)


def test_multilabel_incorrect_threshold():
    with pytest.raises(ValueError):
        recall = Recall(is_multilabel=True, threshold_function=2)
