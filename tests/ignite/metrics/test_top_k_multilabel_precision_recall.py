import warnings

import pytest
import torch
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import precision_score, recall_score

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics import TopKMultilabelPrecision, TopKMultilabelRecall

METRICS = [
    pytest.param(TopKMultilabelPrecision, precision_score, id="precision"),
    pytest.param(TopKMultilabelRecall, recall_score, id="recall"),
]


def ignite_average_to_scikit_average(average):
    # All inputs here are multilabel.
    if average in [None, "micro", "samples", "weighted", "macro"]:
        return average
    if average is False:
        return None
    if average is True:
        return "macro"
    raise ValueError(f"Wrong average parameter `{average}`")


def reference_top_k_multilabel(score_fn, y_pred, y, k, average):
    """Reference value computed by binarizing the top-k predictions then deferring to scikit-learn.

    Uses the same reshape as the metric so that, per sample, y and the predicted top-k mask
    stay aligned.
    """
    num_labels = y_pred.size(1)
    yp = torch.transpose(y_pred, 1, -1).reshape(-1, num_labels)
    yy = torch.transpose(y, 1, -1).reshape(-1, num_labels)

    kk = min(k, num_labels)
    topk_indices = torch.topk(yp, kk, dim=1).indices
    mask = torch.zeros_like(yp)
    mask.scatter_(1, topk_indices, 1.0)

    np_y_pred = mask.cpu().numpy().astype(int)
    np_y = yy.cpu().numpy().astype(int)
    sk_average = ignite_average_to_scikit_average(average)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        return score_fn(np_y, np_y_pred, average=sk_average, zero_division=0)


@pytest.fixture(params=range(6))
def test_data_multilabel(request):
    # y_pred holds float prediction scores, y is binary; shapes (N, C), (N, C, L) and (N, C, H, W).
    return [
        (torch.rand(10, 5), torch.randint(0, 2, size=(10, 5)), 1),
        (torch.rand(10, 4), torch.randint(0, 2, size=(10, 4)), 1),
        (torch.rand(50, 5), torch.randint(0, 2, size=(50, 5)), 16),
        (torch.rand(50, 4, 10), torch.randint(0, 2, size=(50, 4, 10)), 16),
        (torch.rand(10, 5, 18, 16), torch.randint(0, 2, size=(10, 5, 18, 16)), 1),
        (torch.rand(50, 4, 20, 23), torch.randint(0, 2, size=(50, 4, 20, 23)), 16),
    ][request.param]


@pytest.mark.parametrize("metric_cls, _", METRICS)
def test_no_update(metric_cls, _):
    metric = metric_cls(k=2)
    with pytest.raises(
        NotComputableError, match=rf"{metric_cls.__name__} must have at least one example before it can be computed"
    ):
        metric.compute()


@pytest.mark.parametrize("metric_cls, _", METRICS)
def test_invalid_k(metric_cls, _):
    with pytest.raises(ValueError, match=r"Argument k should be a positive integer, given 0."):
        metric_cls(k=0)
    with pytest.raises(ValueError, match=r"Argument k should be a positive integer, given -1."):
        metric_cls(k=-1)


def test_invalid_average():
    with pytest.raises(ValueError, match=r"Argument average should be None or a boolean or one of values"):
        TopKMultilabelPrecision(k=2, average="something")


@pytest.mark.parametrize("metric_cls, _", METRICS)
def test_wrong_inputs(metric_cls, _):
    metric = metric_cls(k=2)

    with pytest.raises(ValueError, match=r"num_categories > 1"):
        # 1D shapes are not valid multilabel inputs
        metric.update((torch.rand(10), torch.randint(0, 2, size=(10,))))

    with pytest.raises(ValueError, match=r"compatible shapes"):
        # mismatched shapes
        metric.update((torch.rand(10, 5), torch.randint(0, 2, size=(10, 4))))

    with pytest.raises(ValueError, match=r"num_categories > 1"):
        # num_labels must be > 1
        metric.update((torch.rand(10, 1), torch.randint(0, 2, size=(10, 1))))

    with pytest.raises(ValueError, match=r"y must be comprised of 0's and 1's"):
        # y must be binary
        metric.update((torch.rand(10, 5), torch.randint(0, 5, size=(10, 5))))

    with pytest.raises(TypeError, match=r"`y_pred` should be a float tensor with prediction scores"):
        # y_pred must contain float scores, not integer labels
        metric.update((torch.randint(0, 2, size=(10, 5)), torch.randint(0, 2, size=(10, 5))))

    with pytest.raises(ValueError, match=r"Input data number of labels has changed"):
        # number of labels must stay constant across updates
        metric.update((torch.rand(10, 5), torch.randint(0, 2, size=(10, 5))))
        metric.update((torch.rand(10, 4), torch.randint(0, 2, size=(10, 4))))


@pytest.mark.parametrize("metric_cls, _", METRICS)
def test_incorrect_type(metric_cls, _):
    metric = metric_cls(k=2)
    metric.update((torch.rand(10, 5), torch.randint(0, 2, size=(10, 5))))
    assert metric._updated is True

    with pytest.raises(ValueError, match=r"Input data number of labels has changed"):
        metric.update((torch.rand(10, 6), torch.randint(0, 2, size=(10, 6))))


@pytest.mark.parametrize("metric_cls, score_fn", METRICS)
@pytest.mark.parametrize("average", [None, False, "macro", "micro", "weighted", "samples"])
@pytest.mark.parametrize("k", [1, 2, 5, 10])
def test_multilabel_input(metric_cls, score_fn, average, k, available_device, test_data_multilabel):
    metric = metric_cls(k=k, average=average, device=available_device)
    assert metric._device == torch.device(available_device)
    assert metric._updated is False

    y_pred, y, batch_size = test_data_multilabel
    metric.reset()

    if batch_size > 1:
        n_iters = y.shape[0] // batch_size + 1
        for i in range(n_iters):
            idx = i * batch_size
            metric.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))
    else:
        metric.update((y_pred, y))

    assert metric._type == "multilabel"
    assert metric._updated is True

    res = metric.compute()
    res = res.cpu().numpy() if isinstance(res, torch.Tensor) else res
    ref = reference_top_k_multilabel(score_fn, y_pred, y, k, average)
    assert ref == pytest.approx(res)


def test_hand_computed_label_wise_topk():
    # Hand-verified example: for each sample the top-k highest-scoring
    # labels become positive predictions, then precision/recall is reported per label.
    y_pred = torch.tensor(
        [
            [0.9, 0.8, 0.1],
            [0.2, 0.7, 0.6],
            [0.7, 0.1, 0.6],
            [0.1, 0.9, 0.5],
        ]
    )
    y = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
        ]
    )

    def precision(average):
        m = TopKMultilabelPrecision(k=2, average=average)
        m.update((y_pred, y))
        return m.compute()

    def recall(average):
        m = TopKMultilabelRecall(k=2, average=average)
        m.update((y_pred, y))
        return m.compute()

    assert precision(False).cpu().numpy() == pytest.approx([1.0, 2 / 3, 1 / 3])
    assert precision("micro") == pytest.approx(5 / 8)
    assert precision(True) == pytest.approx((1.0 + 2 / 3 + 1 / 3) / 3)
    assert precision("weighted") == pytest.approx((2 * 1.0 + 2 * (2 / 3) + 1 * (1 / 3)) / 5)
    assert precision("samples") == pytest.approx((0.5 + 1.0 + 0.5 + 0.5) / 4)

    assert recall(False).cpu().numpy() == pytest.approx([1.0, 1.0, 1.0])
    assert recall("micro") == pytest.approx(1.0)
    assert recall(True) == pytest.approx(1.0)
    assert recall("weighted") == pytest.approx(1.0)
    assert recall("samples") == pytest.approx(1.0)


@pytest.mark.parametrize("metric_cls, score_fn", METRICS)
@pytest.mark.parametrize("average", [False, "macro", "micro", "weighted", "samples"])
def test_ties_in_scores(metric_cls, score_fn, average):
    # All-equal scores: top-k selection is arbitrary among ties, but the metric and the reference
    # call the same torch.topk, so they must agree (and the metric must not crash).
    y_pred = torch.full((8, 5), 0.5)
    y = torch.randint(0, 2, size=(8, 5))
    metric = metric_cls(k=2, average=average)
    metric.update((y_pred, y))

    res = metric.compute()
    res = res.cpu().numpy() if isinstance(res, torch.Tensor) else res
    ref = reference_top_k_multilabel(score_fn, y_pred, y, 2, average)
    assert ref == pytest.approx(res)


@pytest.mark.parametrize("average", [False, "macro", "micro", "weighted", "samples"])
def test_zero_positive_label(average):
    # A label with no positive ground-truth samples: recall denominator is 0
    y_pred = torch.rand(6, 4)
    y = torch.randint(0, 2, size=(6, 4))
    y[:, 1] = 0  # label 1 has no positives

    metric = TopKMultilabelRecall(k=2, average=average)
    metric.update((y_pred, y))
    res = metric.compute()

    if average is False:
        assert res[1].item() == pytest.approx(0.0)

    res = res.cpu().numpy() if isinstance(res, torch.Tensor) else res
    ref = reference_top_k_multilabel(recall_score, y_pred, y, 2, average)
    assert ref == pytest.approx(res)


@pytest.mark.parametrize("metric_cls, score_fn", METRICS)
@pytest.mark.parametrize("average", [False, "macro", "micro", "weighted", "samples"])
def test_label_imbalance(metric_cls, score_fn, average):
    # One common label and one rare label; per-label handling must still match the reference.
    n = 200
    y_pred = torch.rand(n, 4)
    y = torch.zeros(n, 4, dtype=torch.long)
    y[:, 0] = 1  # common label: almost always positive
    y[0, 0] = 0
    y[:, 3] = 0
    y[0, 3] = 1  # rare label: a single positive

    metric = metric_cls(k=2, average=average)
    metric.update((y_pred, y))
    res = metric.compute()
    res = res.cpu().numpy() if isinstance(res, torch.Tensor) else res
    ref = reference_top_k_multilabel(score_fn, y_pred, y, 2, average)
    assert ref == pytest.approx(res)


@pytest.mark.usefixtures("distributed")
class TestDistributed:
    @pytest.mark.parametrize("metric_cls, score_fn", METRICS)
    @pytest.mark.parametrize("average", [False, "macro", "weighted", "micro"])
    @pytest.mark.parametrize("n_epochs", [1, 2])
    def test_integration_multilabel(self, metric_cls, score_fn, average, n_epochs):
        rank = idist.get_rank()
        torch.manual_seed(12 + rank)

        n_iters = 60
        batch_size = 16
        n_labels = 7
        k = 3

        metric_devices = ["cpu"]
        device = idist.device()
        if device.type != "xla":
            metric_devices.append(idist.device())

        for metric_device in metric_devices:
            y_true = torch.randint(0, 2, size=(n_iters * batch_size, n_labels, 6, 8)).to(device)
            y_preds = torch.rand(n_iters * batch_size, n_labels, 6, 8).to(device)

            def update(engine, i):
                return (
                    y_preds[i * batch_size : (i + 1) * batch_size, ...],
                    y_true[i * batch_size : (i + 1) * batch_size, ...],
                )

            engine = Engine(update)

            metric = metric_cls(k=k, average=average, device=metric_device)
            metric.attach(engine, "metric")
            assert metric._updated is False

            data = list(range(n_iters))
            engine.run(data=data, max_epochs=n_epochs)

            y_preds = idist.all_gather(y_preds)
            y_true = idist.all_gather(y_true)

            assert "metric" in engine.state.metrics
            assert metric._updated is True
            res = engine.state.metrics["metric"]
            if isinstance(res, torch.Tensor):
                res = res.cpu().numpy()

            assert metric._type == "multilabel"
            ref = reference_top_k_multilabel(score_fn, y_preds.cpu(), y_true.cpu(), k, average)
            assert ref == pytest.approx(res)

    @pytest.mark.parametrize("metric_cls, _", METRICS)
    @pytest.mark.parametrize("average", [False, "macro", "weighted", "micro", "samples"])
    def test_accumulator_device(self, metric_cls, _, average):
        metric_devices = [torch.device("cpu")]
        device = idist.device()
        if device.type != "xla":
            metric_devices.append(idist.device())

        for metric_device in metric_devices:
            metric = metric_cls(k=2, average=average, device=metric_device)
            assert metric._device == metric_device
            assert metric._updated is False

            y_pred = torch.rand(10, 4, 20, 23)
            y = torch.randint(0, 2, size=(10, 4, 20, 23)).long()
            metric.update((y_pred, y))

            assert metric._updated is True
            assert metric._numerator.device == metric_device, (
                f"{type(metric._numerator.device)}:{metric._numerator.device} vs {type(metric_device)}:{metric_device}"
            )

            if average != "samples":
                assert metric._denominator.device == metric_device, (
                    f"{type(metric._denominator.device)}:{metric._denominator.device} vs "
                    f"{type(metric_device)}:{metric_device}"
                )

            if average == "weighted":
                assert metric._weight.device == metric_device, (
                    f"{type(metric._weight.device)}:{metric._weight.device} vs {type(metric_device)}:{metric_device}"
                )
