import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve

from ignite import distributed as idist
from ignite.engine import Engine
from ignite.metrics import MeanAveragePrecision
from ignite.utils import to_onehot


def test_wrong_input():
    with pytest.raises(ValueError, match="rec_thresholds should be a one-dimensional tensor or a sequence of floats"):
        MeanAveragePrecision(rec_thresholds=torch.zeros((2, 2)))

    with pytest.raises(TypeError, match="rec_thresholds should be a sequence of floats or a tensor"):
        MeanAveragePrecision(rec_thresholds={0, 0.2, 0.4, 0.6, 0.8})

    with pytest.raises(ValueError, match="Wrong `class_mean` parameter"):
        MeanAveragePrecision(class_mean="samples")

    with pytest.raises(ValueError, match="rec_thresholds values should be between 0 and 1"):
        MeanAveragePrecision(rec_thresholds=(0.0, 0.5, 1.0, 1.5))

    metric = MeanAveragePrecision()
    with pytest.raises(RuntimeError, match="Metric could not be computed without any update method call"):
        metric.compute()


def test_wrong_classification_input():
    metric = MeanAveragePrecision()

    with pytest.raises(TypeError, match="`y_pred` should be a float tensor"):
        metric.update((torch.tensor([0, 1, 0]), torch.tensor([1, 0, 1])))

    metric = MeanAveragePrecision()
    with pytest.warns(RuntimeWarning, match="`y` should be of dtype long when entry type is multiclass"):
        metric.update((torch.tensor([[0.5, 0.4, 0.1]]), torch.tensor([2.0])))

    with pytest.raises(ValueError, match="y_pred contains fewer classes than y"):
        metric.update((torch.tensor([[0.5, 0.4, 0.1]]), torch.tensor([3])))


def test__prepare_output():
    metric = MeanAveragePrecision()

    metric._type = "binary"
    scores, y = metric._prepare_output((torch.rand((5, 4, 3, 2)), torch.randint(0, 2, (5, 4, 3, 2)).bool()))
    assert scores.shape == y.shape == (1, 120)

    metric._type = "multiclass"
    scores, y = metric._prepare_output((torch.rand((5, 4, 3, 2)), torch.randint(0, 4, (5, 3, 2))))
    assert scores.shape == (4, 30) and y.shape == (30,)

    metric._type = "multilabel"
    scores, y = metric._prepare_output((torch.rand((5, 4, 3, 2)), torch.randint(0, 2, (5, 4, 3, 2)).bool()))
    assert scores.shape == y.shape == (4, 30)


def test_update():
    metric = MeanAveragePrecision()
    assert len(metric._y_pred) == len(metric._y_true) == 0
    metric.update((torch.rand((5, 4)), torch.randint(0, 2, (5, 4)).bool()))
    assert len(metric._y_pred) == len(metric._y_true) == 1


def test__compute_recall_and_precision():
    m = MeanAveragePrecision()

    scores = torch.rand((50,))
    y_true = torch.randint(0, 2, (50,)).bool()
    precision, recall, _ = precision_recall_curve(y_true.numpy(), scores.numpy())
    P = y_true.sum(dim=-1)
    ignite_recall, ignite_precision = m._compute_recall_and_precision(y_true, scores, P)
    assert (ignite_recall.squeeze().flip(0).numpy() == recall[:-1]).all()
    assert (ignite_precision.squeeze().flip(0).numpy() == precision[:-1]).all()

    # When there's no actual positive. Numpy expectedly raises warning.
    scores = torch.rand((50,))
    y_true = torch.zeros((50,)).bool()
    precision, recall, _ = precision_recall_curve(y_true.numpy(), scores.numpy())
    P = torch.tensor(0)
    ignite_recall, ignite_precision = m._compute_recall_and_precision(y_true, scores, P)
    assert (ignite_recall.flip(0).numpy() == recall[:-1]).all()
    assert (ignite_precision.flip(0).numpy() == precision[:-1]).all()


def test__compute_average_precision():
    m = MeanAveragePrecision()

    # Binary data
    scores = np.random.rand(50)
    y_true = np.random.randint(0, 2, 50)
    ap = average_precision_score(y_true, scores)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ignite_ap = m._compute_average_precision(
        torch.from_numpy(recall[:-1]).flip(-1), torch.from_numpy(precision[:-1]).flip(-1)
    )
    assert np.allclose(ignite_ap.item(), ap)

    # Multilabel data
    scores = np.random.rand(50, 5)
    y_true = np.random.randint(0, 2, (50, 5))
    ap = average_precision_score(y_true, scores, average=None)
    ignite_ap = []
    for cls in range(scores.shape[1]):
        precision, recall, _ = precision_recall_curve(y_true[:, cls], scores[:, cls])
        ignite_ap.append(
            m._compute_average_precision(
                torch.from_numpy(recall[:-1]).flip(-1), torch.from_numpy(precision[:-1]).flip(-1)
            ).item()
        )
    ignite_ap = np.array(ignite_ap)
    assert np.allclose(ignite_ap, ap)


def test_compute_binary_data():
    m = MeanAveragePrecision()
    scores = torch.rand((130,))
    y_true = torch.randint(0, 2, (130,))

    m.update((scores[:50], y_true[:50]))
    m.update((scores[50:], y_true[50:]))
    ignite_map = m.compute()

    map = average_precision_score(y_true.numpy(), scores.numpy())

    assert np.allclose(ignite_map, map)


@pytest.mark.parametrize("class_mean", [None, "macro", "micro", "weighted"])
def test_compute_nonbinary_data(class_mean):
    scores = torch.rand((130, 5, 2, 2))
    sklearn_scores = scores.transpose(1, -1).reshape(-1, 5).numpy()

    # Multiclass
    m = MeanAveragePrecision(class_mean=class_mean)
    y_true = torch.randint(0, 5, (130, 2, 2))
    m.update((scores[:50], y_true[:50]))
    m.update((scores[50:], y_true[50:]))
    ignite_map = m.compute().numpy()

    y_true = to_onehot(y_true, 5).transpose(1, -1).reshape(-1, 5).numpy()
    sklearn_map = average_precision_score(y_true, sklearn_scores, average=class_mean)

    assert np.allclose(sklearn_map, ignite_map)

    # Multilabel
    m = MeanAveragePrecision(is_multilabel=True, class_mean=class_mean)
    y_true = torch.randint(0, 2, (130, 5, 2, 2)).bool()
    m.update((scores[:50], y_true[:50]))
    m.update((scores[50:], y_true[50:]))
    ignite_map = m.compute().numpy()

    y_true = y_true.transpose(1, -1).reshape(-1, 5).numpy()
    sklearn_map = average_precision_score(y_true, sklearn_scores, average=class_mean)

    assert np.allclose(sklearn_map, ignite_map)


@pytest.mark.parametrize("data_type", ["binary", "multiclass", "multilabel"])
@pytest.mark.parametrize("n_epochs", [1, 2])
def test_distrib_integration(distributed, data_type, n_epochs):
    rank = idist.get_rank()
    device = idist.device()
    torch.manual_seed(12 + rank)

    n_iters = 60
    batch_size = 16
    n_classes = 7

    metric_devices = [torch.device("cpu")]
    if device.type != "xla":
        metric_devices.append(device)

    for metric_device in metric_devices:

        y_true_size = (
            (n_iters * batch_size, 3, 2) if data_type != "multilabel" else (n_iters * batch_size, n_classes, 3, 2)
        )
        y_true = torch.randint(0, n_classes if data_type == "multiclass" else 2, size=y_true_size, device=device)
        y_preds_size = (
            (n_iters * batch_size, n_classes, 3, 2) if data_type != "binary" else (n_iters * batch_size, 3, 2)
        )
        y_preds = torch.rand(y_preds_size, device=device)

        def update(_, i):
            return (
                y_preds[i * batch_size : (i + 1) * batch_size, ...],
                y_true[i * batch_size : (i + 1) * batch_size, ...],
            )

        engine = Engine(update)
        mAP = MeanAveragePrecision(is_multilabel=data_type == "multilabel", device=metric_device)
        mAP.attach(engine, "mAP")

        engine.run(range(n_iters), max_epochs=n_epochs)

        y_preds = idist.all_gather(y_preds)
        y_true = idist.all_gather(y_true)

        assert "mAP" in engine.state.metrics

        if data_type == "multiclass":
            y_true = to_onehot(y_true, n_classes)

        if data_type == "binary":
            y_true = y_true.view(-1)
            y_preds = y_preds.view(-1)
        else:
            y_true = y_true.transpose(1, -1).reshape(-1, n_classes)
            y_preds = y_preds.transpose(1, -1).reshape(-1, n_classes)

        sklearn_mAP = average_precision_score(y_true.cpu().numpy(), y_preds.cpu().numpy())
        assert np.allclose(sklearn_mAP, engine.state.metrics["mAP"])
