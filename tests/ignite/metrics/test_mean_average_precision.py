from typing import Tuple

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.utils.extmath import stable_cumsum

from ignite import distributed as idist
from ignite.engine import Engine
from ignite.metrics import MeanAveragePrecision
from ignite.utils import manual_seed, to_onehot

manual_seed(41)


def test_wrong_input():
    with pytest.raises(ValueError, match="rec_thresholds should be a one-dimensional tensor or a sequence of floats"):
        MeanAveragePrecision(rec_thresholds=torch.zeros((2, 2)))

    with pytest.raises(TypeError, match="rec_thresholds should be a sequence of floats or a tensor"):
        MeanAveragePrecision(rec_thresholds={0, 0.2, 0.4, 0.6, 0.8})

    with pytest.raises(ValueError, match="Wrong `average_operand` parameter"):
        MeanAveragePrecision(average_operand=1)

    with pytest.raises(ValueError, match="Wrong `class_mean` parameter"):
        MeanAveragePrecision(class_mean="samples")

    with pytest.raises(ValueError, match="rec_thresholds values should be between 0 and 1"):
        MeanAveragePrecision(rec_thresholds=(0.0, 0.5, 1.0, 1.5))

    with pytest.raises(ValueError, match="class_mean 'with_other_dims' is not compatible with classification"):
        MeanAveragePrecision(class_mean="with_other_dims")


def test_wrong_classification_input():
    metric = MeanAveragePrecision()
    assert metric._task == "classification"

    with pytest.raises(TypeError, match="`y_pred` should be a float tensor"):
        metric.update((torch.tensor([0, 1, 0]), torch.tensor([1, 0, 1])))

    metric = MeanAveragePrecision()
    with pytest.warns(RuntimeWarning, match="`y` should be of dtype long when entry type is multiclass"):
        metric.update((torch.tensor([[0.5, 0.4, 0.1]]), torch.tensor([2.0])))

    with pytest.raises(ValueError, match="y_pred contains fewer classes than y"):
        metric.update((torch.tensor([[0.5, 0.4, 0.1]]), torch.tensor([3])))


class Dummy_mAP(MeanAveragePrecision):
    def do_matching(self, pred: Tuple, target: Tuple):
        return *pred, *target


def test_wrong_do_matching():
    metric = MeanAveragePrecision()
    with pytest.raises(NotImplementedError, match="Please subclass MeanAveragePrecision and implement"):
        metric.do_matching(None, None)

    metric = Dummy_mAP()

    with pytest.raises(ValueError, match="Returned TP, FP and scores dictionaries from do_matching should have"):
        metric.update(
            (
                ({1: torch.tensor([True])}, {1: torch.tensor([False])}),
                ({1: 1}, {1: torch.tensor([0.8]), 2: torch.tensor([0.9])}),
            )
        )

    with pytest.raises(TypeError, match="Tensors in TP and FP dictionaries should be boolean or uint8"):
        metric.update((({1: torch.tensor([1])}, {1: torch.tensor([False])}), ({1: 1}, {1: torch.tensor([0.8])})))

    with pytest.raises(
        ValueError, match="Sample dimension of tensors in TP, FP and scores should have equal size per class"
    ):
        metric.update(
            (({1: torch.tensor([True])}, {1: torch.tensor([False, False])}), ({1: 1}, {1: torch.tensor([0.8])}))
        )

    metric.update((({1: torch.tensor([True])}, {1: torch.tensor([False])}), ({1: 1}, {1: torch.tensor([0.8])})))
    with pytest.raises(ValueError, match="Tensors in returned FP from do_matching should not change in shape except"):
        metric.update(
            (
                ({1: torch.tensor([False, True])}, {1: torch.tensor([[True, False], [False, False]])}),
                ({1: 1}, {1: torch.tensor([0.8, 0.9])}),
            )
        )


def test__classification_prepare_output():
    metric = MeanAveragePrecision()

    metric._type = "binary"
    scores, y = metric._classification_prepare_output(
        torch.rand((5, 4, 3, 2)), torch.randint(0, 2, (5, 4, 3, 2)).bool()
    )
    assert scores.shape == y.shape == (1, 120)

    metric._type = "multiclass"
    scores, y = metric._classification_prepare_output(torch.rand((5, 4, 3, 2)), torch.randint(0, 4, (5, 3, 2)))
    assert scores.shape == (4, 30) and y.shape == (30,)

    metric._type = "multilabel"
    scores, y = metric._classification_prepare_output(
        torch.rand((5, 4, 3, 2)), torch.randint(0, 2, (5, 4, 3, 2)).bool()
    )
    assert scores.shape == y.shape == (4, 30)


def test_update():
    metric = MeanAveragePrecision()
    assert len(metric._scores) == len(metric._P) == 0
    metric.update((torch.rand((5, 4)), torch.randint(0, 2, (5, 4)).bool()))
    assert len(metric._scores) == len(metric._P) == 1

    metric = Dummy_mAP()
    assert len(metric._tp) == len(metric._fp) == len(metric._scores) == len(metric._P) == metric._num_classes == 0

    metric.update((({1: torch.tensor([True])}, {1: torch.tensor([False])}), ({1: 1, 2: 1}, {1: torch.tensor([0.8])})))
    assert len(metric._tp[1]) == len(metric._fp[1]) == len(metric._scores[1]) == 1
    assert len(metric._P) == 2 and metric._P[2] == 1
    assert metric._num_classes == 3

    metric.update((({}, {}), ({2: 2}, {})))
    assert metric._P[2] == 3


def sklearn_precision_recall_curve_allowing_multiple_recalls_at_single_threshold(y_true, y_score):
    y_true = y_true == 1

    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_true = y_true[desc_score_indices]
    weight = 1.0

    tps = stable_cumsum(y_true * weight)
    fps = stable_cumsum((1 - y_true) * weight)
    ps = tps + fps
    precision = np.zeros_like(tps)
    np.divide(tps, ps, out=precision, where=(ps != 0))
    if tps[-1] == 0:
        recall = np.ones_like(tps)
    else:
        recall = tps / tps[-1]

    sl = slice(None, None, -1)
    return np.hstack((precision[sl], 1)), np.hstack((recall[sl], 0)), None


@pytest.mark.parametrize(
    "allow_multiple_recalls_at_single_threshold, sklearn_pr_rec_curve",
    [
        (False, precision_recall_curve),
        (True, sklearn_precision_recall_curve_allowing_multiple_recalls_at_single_threshold),
    ],
)
def test__measure_recall_and_precision(allow_multiple_recalls_at_single_threshold, sklearn_pr_rec_curve):
    # Classification
    m = MeanAveragePrecision(allow_multiple_recalls_at_single_threshold=allow_multiple_recalls_at_single_threshold)

    scores = torch.rand((50,))
    y_true = torch.randint(0, 2, (50,)).bool()
    precision, recall, _ = sklearn_pr_rec_curve(y_true.numpy(), scores.numpy())
    if allow_multiple_recalls_at_single_threshold:
        y_true = y_true.unsqueeze(0)
        scores = scores.unsqueeze(0)
    FP = ~y_true if allow_multiple_recalls_at_single_threshold else None
    P = y_true.sum(dim=-1)
    ignite_recall, ignite_precision = m._measure_recall_and_precision(y_true, FP, scores, P)
    assert (ignite_recall.squeeze().flip(0).numpy() == recall[:-1]).all()
    assert (ignite_precision.squeeze().flip(0).numpy() == precision[:-1]).all()

    # Classification, when there's no actual positive. Numpy expectedly raises warning.
    scores = torch.rand((50,))
    y_true = torch.zeros((50,)).bool()
    precision, recall, _ = sklearn_pr_rec_curve(y_true.numpy(), scores.numpy())
    if allow_multiple_recalls_at_single_threshold:
        y_true = y_true.unsqueeze(0)
        scores = scores.unsqueeze(0)
    FP = ~y_true if allow_multiple_recalls_at_single_threshold else None
    P = torch.tensor([0]) if allow_multiple_recalls_at_single_threshold else torch.tensor(0)
    ignite_recall, ignite_precision = m._measure_recall_and_precision(y_true, FP, scores, P)
    assert (ignite_recall.flip(0).numpy() == recall[:-1]).all()
    assert (ignite_precision.flip(0).numpy() == precision[:-1]).all()

    # Detection, in the case detector detects all gt objects but also produces some wrong predictions.
    scores = torch.rand((50,))
    y_true = torch.randint(0, 2, (50,))
    m = Dummy_mAP(allow_multiple_recalls_at_single_threshold=allow_multiple_recalls_at_single_threshold)

    ignite_recall, ignite_precision = m._measure_recall_and_precision(
        y_true.bool(), ~(y_true.bool()), scores, y_true.sum()
    )
    sklearn_precision, sklearn_recall, _ = sklearn_pr_rec_curve(y_true.numpy(), scores.numpy())
    assert (ignite_recall.flip(0).numpy() == sklearn_recall[:-1]).all()
    assert (ignite_precision.flip(0).numpy() == sklearn_precision[:-1]).all()

    # Detection like above but with two additional mean dimensions.
    scores = torch.rand((50,))
    y_true = torch.zeros((6, 8, 50))
    sklearn_precisions, sklearn_recalls = [], []
    for i in range(6):
        for j in range(8):
            y_true[i, j, np.random.choice(50, size=15, replace=False)] = 1
            precision, recall, _ = sklearn_pr_rec_curve(y_true[i, j].numpy(), scores.numpy())
            sklearn_precisions.append(precision[:-1])
            sklearn_recalls.append(recall[:-1])
    sklearn_precisions = np.array(sklearn_precisions).reshape(6, 8, -1)
    sklearn_recalls = np.array(sklearn_recalls).reshape(6, 8, -1)
    ignite_recall, ignite_precision = m._measure_recall_and_precision(
        y_true.bool(), ~(y_true.bool()), scores, torch.tensor(15)
    )
    assert (ignite_recall.flip(-1).numpy() == sklearn_recalls).all()
    assert (ignite_precision.flip(-1).numpy() == sklearn_precisions).all()


def test__measure_average_precision():
    m = MeanAveragePrecision()

    # Binary data
    scores = np.random.rand(50)
    y_true = np.random.randint(0, 2, 50)
    ap = average_precision_score(y_true, scores)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ignite_ap = m._measure_average_precision(
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
            m._measure_average_precision(
                torch.from_numpy(recall[:-1]).flip(-1), torch.from_numpy(precision[:-1]).flip(-1)
            ).item()
        )
    ignite_ap = np.array(ignite_ap)
    assert np.allclose(ignite_ap, ap)


def test_compute_classification_binary_data():
    m = MeanAveragePrecision()
    scores = torch.rand((130,))
    y_true = torch.randint(0, 2, (130,))

    m.update((scores[:50], y_true[:50]))
    m.update((scores[50:], y_true[50:]))
    ignite_map = m.compute()

    map = average_precision_score(y_true.numpy(), scores.numpy())

    assert np.allclose(ignite_map, map)


@pytest.mark.parametrize("class_mean", [None, "macro", "micro", "weighted"])
def test_compute_classification_nonbinary_data(class_mean):
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
    m = MeanAveragePrecision(classification_is_multilabel=True, class_mean=class_mean)
    y_true = torch.randint(0, 2, (130, 5, 2, 2)).bool()
    m.update((scores[:50], y_true[:50]))
    m.update((scores[50:], y_true[50:]))
    ignite_map = m.compute().numpy()

    y_true = y_true.transpose(1, -1).reshape(-1, 5).numpy()
    sklearn_map = average_precision_score(y_true, sklearn_scores, average=class_mean)

    assert np.allclose(sklearn_map, ignite_map)


@pytest.mark.parametrize("class_mean", ["macro", None, "micro", "weighted", "with_other_dims"])
def test_compute_detection(class_mean):
    m = Dummy_mAP(class_mean=class_mean)

    # The case in which, detector detects all gt objects but also produces some wrong predictions. Also classes
    # have the same number of predictions.

    y_true = torch.randint(0, 2, (40, 5))
    scores = torch.rand((40, 5))

    for s in [slice(20), slice(20, 40)]:
        tp = {c: y_true[s, c].bool() for c in range(5)}
        fp = {c: ~(y_true[s, c].bool()) for c in range(5)}
        p = dict(enumerate(y_true[s].sum(dim=0).tolist()))
        score = {c: scores[s, c] for c in range(5)}
        m.update(((tp, fp), (p, score)))

    ignite_map = m.compute().numpy()

    sklearn_class_mean = class_mean if class_mean != "with_other_dims" else "macro"
    sklearn_map = average_precision_score(y_true.numpy(), scores.numpy(), average=sklearn_class_mean)
    assert np.allclose(sklearn_map, ignite_map)

    # Like above but with two additional mean dimensions.
    m.reset()
    y_true = torch.zeros((5, 6, 8, 50))
    scores = torch.rand((50, 5))
    P_counts = np.random.choice(50, size=5)
    sklearn_aps = []
    for c in range(5):
        for i in range(6):
            for j in range(8):
                y_true[c, i, j, np.random.choice(50, size=P_counts[c], replace=False)] = 1
        if class_mean != "micro":
            sklearn_aps.append(
                average_precision_score(
                    y_true[c].view(6 * 8, 50).T.numpy(), scores[:, c].repeat(6 * 8, 1).T.numpy(), average=None
                )
            )
    if class_mean == "micro":
        sklearn_aps = average_precision_score(
            torch.cat(y_true.unbind(0), dim=-1).view(6 * 8, 5 * 50).T.numpy(),
            scores.T.reshape(5 * 50).repeat(6 * 8, 1).T.numpy(),
            average=None,
        )
    sklearn_aps = np.array(sklearn_aps)
    if class_mean in (None, "micro"):
        sklearn_map = sklearn_aps.mean(axis=-1)
    elif class_mean == "macro":
        sklearn_map = sklearn_aps.mean(axis=-1)[P_counts != 0].mean()
    elif class_mean == "with_other_dims":
        sklearn_map = sklearn_aps[P_counts != 0].mean()
    else:
        sklearn_map = np.dot(P_counts, sklearn_aps.mean(axis=-1)) / P_counts.sum()

    for s in [slice(0, 20), slice(20, 50)]:
        tp = {c: y_true[c, :, :, s].bool() for c in range(5)}
        fp = {c: ~(y_true[c, :, :, s].bool()) for c in range(5)}
        p = dict(enumerate(y_true[:, 0, 0, s].sum(dim=-1).tolist()))
        score = {c: scores[s, c] for c in range(5)}
        m.update(((tp, fp), (p, score)))
    ignite_map = m.compute().numpy()
    assert np.allclose(ignite_map, sklearn_map)


@pytest.mark.parametrize("data_type", ["binary", "multiclass", "multilabel"])
def test_distrib_integration_classification(distributed, data_type):
    rank = idist.get_rank()
    world_size = idist.get_world_size()
    device = idist.device()

    def _test(metric_device):
        def update(_, i):
            return (
                y_preds[(2 * rank + i) * 10 : (2 * rank + i + 1) * 10],
                y_true[(2 * rank + i) * 10 : (2 * rank + i + 1) * 10],
            )

        engine = Engine(update)
        mAP = MeanAveragePrecision(classification_is_multilabel=data_type == "multilabel", device=metric_device)
        mAP.attach(engine, "mAP")

        y_true_size = (10 * 2 * world_size, 3, 2) if data_type != "multilabel" else (10 * 2 * world_size, 4, 3, 2)
        y_true = torch.randint(0, 4 if data_type == "multiclass" else 2, size=y_true_size).to(device)
        y_preds_size = (10 * 2 * world_size, 4, 3, 2) if data_type != "binary" else (10 * 2 * world_size, 3, 2)
        y_preds = torch.rand(y_preds_size).to(device)

        engine.run(range(2), max_epochs=1)
        assert "mAP" in engine.state.metrics

        if data_type == "multiclass":
            y_true = to_onehot(y_true, 4)

        if data_type == "binary":
            y_true = y_true.view(-1)
            y_preds = y_preds.view(-1)
        else:
            y_true = y_true.transpose(1, -1).reshape(-1, 4)
            y_preds = y_preds.transpose(1, -1).reshape(-1, 4)

        sklearn_mAP = average_precision_score(y_true.numpy(), y_preds.numpy())
        assert np.allclose(sklearn_mAP, engine.state.metrics["mAP"])

    metric_devices = [torch.device("cpu")]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for metric_device in metric_devices:
        _test(metric_device)


@pytest.mark.parametrize("class_mean", [None, "macro", "micro", "weighted", "with_other_dims"])
def test_distrib_integration_detection(distributed, class_mean):
    rank = idist.get_rank()
    device = idist.device()
    world_size = idist.get_world_size()

    def _test(metric_device):
        def update(_, i):
            y_true_batch = y_true[..., (2 * rank + i) * 10 : (2 * rank + i + 1) * 10]
            scores_batch = scores[..., (2 * rank + i) * 10 : (2 * rank + i + 1) * 10]
            return (
                ({c: y_true_batch[c].bool() for c in range(4)}, {c: ~(y_true_batch[c].bool()) for c in range(4)}),
                (
                    dict(
                        enumerate(
                            (y_true_batch[:, 0, 0] if y_true_batch.ndim == 4 else y_true_batch).sum(dim=-1).tolist()
                        )
                    ),
                    {c: scores_batch[c] for c in range(4)},
                ),
            )

        engine = Engine(update)
        # The case in which, detector detects all gt objects but also produces some wrong predictions. Also classes
        # have the same number of predictions.
        mAP = Dummy_mAP(device=metric_device, class_mean=class_mean)
        mAP.attach(engine, "mAP")

        y_true = torch.randint(0, 2, size=(4, 10 * 2 * world_size)).to(device)
        scores = torch.rand((4, 10 * 2 * world_size)).to(device)
        engine.run(range(2), max_epochs=1)
        assert "mAP" in engine.state.metrics
        sklearn_class_mean = class_mean if class_mean != "with_other_dims" else "macro"
        sklearn_map = average_precision_score(y_true.T.numpy(), scores.T.numpy(), average=sklearn_class_mean)
        assert np.allclose(sklearn_map, engine.state.metrics["mAP"])

        # Like above but with two additional mean dimensions.
        y_true = torch.zeros((4, 6, 8, 10 * 2 * world_size))

        P_counts = np.random.choice(10 * 2 * world_size, size=4)
        sklearn_aps = []
        for c in range(4):
            for i in range(6):
                for j in range(8):
                    y_true[c, i, j, np.random.choice(10 * 2 * world_size, size=P_counts[c], replace=False)] = 1
            if class_mean != "micro":
                sklearn_aps.append(
                    average_precision_score(
                        y_true[c].view(6 * 8, 10 * 2 * world_size).T.numpy(),
                        scores[c].repeat(6 * 8, 1).T.numpy(),
                        average=None,
                    )
                )
        if class_mean == "micro":
            sklearn_aps = average_precision_score(
                torch.cat(y_true.unbind(0), dim=-1).view(6 * 8, 4 * 10 * 2 * world_size).T.numpy(),
                scores.reshape(4 * 10 * 2 * world_size).repeat(6 * 8, 1).T.numpy(),
                average=None,
            )
        sklearn_aps = np.array(sklearn_aps)
        if class_mean in (None, "micro"):
            sklearn_map = sklearn_aps.mean(axis=-1)
        elif class_mean == "macro":
            sklearn_map = sklearn_aps.mean(axis=-1)[P_counts != 0].mean()
        elif class_mean == "with_other_dims":
            sklearn_map = sklearn_aps[P_counts != 0].mean()
        else:
            sklearn_map = np.dot(P_counts, sklearn_aps.mean(axis=-1)) / P_counts.sum()

        engine.run(range(2), max_epochs=1)

        assert np.allclose(sklearn_map, engine.state.metrics["mAP"])

    metric_devices = [torch.device("cpu")]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for metric_device in metric_devices:
        _test(metric_device)


# class MatchFirstDetectionFirst_mAP(MeanAveragePrecision):
#     def do_matching(self, pred: Tuple[Sequence[int], Sequence[float]] , target: Sequence[int]):
#         P = dict(Counter(target))
#         tp = defaultdict(lambda: [])
#         scores = defaultdict(lambda: [])

#         target = torch.tensor(target)
#         matched = torch.zeros((len(target),)).bool()
#         for label, score in zip(*pred):
#             try:
#                 matched[torch.logical_and(target == label, ~matched).tolist().index(True)] = True
#                 tp[label].append(True)
#             except ValueError:
#                 tp[label].append(False)
#             scores[label].append(score)

#         tp = {label: torch.tensor(_tp) for label, _tp in tp.items()}
#         fp = {label: ~_tp for label, _tp in tp.items()}
#         scores = {label: torch.tensor(_scores) for label, _scores in scores.items()}
#         return tp, fp, P, scores
