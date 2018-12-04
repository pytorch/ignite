from ignite.exceptions import NotComputableError
from ignite.metrics import Recall
import pytest
import torch
from sklearn.metrics import recall_score


torch.manual_seed(12)


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

    y_pred = torch.rand(10)
    y = torch.randint(0, 2, size=(10,)).type(torch.LongTensor)
    recall.update((y_pred, y))
    np_y = y.numpy()
    np_y_pred = (y_pred.numpy().ravel() > 0.5).astype('int')
    assert recall.compute() == pytest.approx(recall_score(np_y, np_y_pred))

    recall = Recall(average=True)
    y_pred = torch.softmax(torch.rand(10, 2), dim=1)
    y = torch.randint(0, 2, size=(10,)).type(torch.LongTensor)
    indices = torch.max(y_pred, dim=1)[1]
    recall.update((y_pred, y))
    assert recall.compute() == pytest.approx(recall_score(y.numpy(), indices.numpy(), average='macro'))


def test_binary_shapes():
    recall = Recall(average=True)

    y = torch.randint(0, 2, size=(10,)).type(torch.LongTensor)
    y_pred = torch.rand(10, 1)
    recall.update((y_pred, y))
    np_y = y.numpy()
    np_y_pred = (y_pred.numpy().ravel() > 0.5).astype('int')
    assert recall.compute() == pytest.approx(recall_score(np_y, np_y_pred))

    y = torch.randint(0, 2, size=(10, 1)).type(torch.LongTensor)
    y_pred = torch.rand(10)
    recall.reset()
    recall.update((y_pred, y))
    np_y = y.numpy()
    np_y_pred = (y_pred.numpy().ravel() > 0.5).astype('int')
    assert recall.compute() == pytest.approx(recall_score(np_y, np_y_pred))

    recall = Recall()

    y = torch.randint(0, 2, size=(10,)).type(torch.LongTensor)
    y_pred = torch.rand(10)
    recall.update((y_pred, y))
    np_y = y.numpy()
    np_y_pred = (y_pred.numpy().ravel() > 0.5).astype('int')
    assert recall.compute().numpy() == pytest.approx(recall_score(np_y, np_y_pred, average=None))

    y = torch.randint(0, 2, size=(10, 1)).type(torch.LongTensor)
    y_pred = torch.rand(10, 1)
    recall.reset()
    recall.update((y_pred, y))
    np_y = y.numpy()
    np_y_pred = (y_pred.numpy().ravel() > 0.5).astype('int')
    assert recall.compute().numpy() == pytest.approx(recall_score(np_y, np_y_pred, average=None))


def test_ner_example():
    recall = Recall()

    y_pred = torch.softmax(torch.rand(2, 3, 8), dim=1)
    y = torch.randint(0, 3, size=(2, 8)).type(torch.LongTensor)
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


def test_incorrect_binary_y():
    recall = Recall()

    y_pred = torch.rand(4, 1)
    y = 2 * torch.ones(4, 1).type(torch.LongTensor)

    with pytest.raises(ValueError):
        recall.update((y_pred, y))


def test_incorrect_type():
    recall = Recall()

    y_pred = torch.softmax(torch.rand(4, 4), dim=1)
    y = torch.ones(4).type(torch.LongTensor)
    recall.update((y_pred, y))

    y_pred = torch.rand(4, 1)
    y = torch.ones(4).type(torch.LongTensor)

    with pytest.raises(TypeError):
        recall.update((y_pred, y))
