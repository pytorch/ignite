import os

import numpy as np
import pytest
import torch
from pytest import approx
from sklearn.metrics import f1_score, precision_score, recall_score

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.metrics import Accuracy, Metric, MetricsLambda, Precision, Recall


class ListGatherMetric(Metric):
    def __init__(self, index):
        super(ListGatherMetric, self).__init__()
        self.index = index

    def reset(self):
        self.list_ = None

    def update(self, output):
        self.list_ = output

    def compute(self):
        return self.list_[self.index]


def test_metrics_lambda():
    m0 = ListGatherMetric(0)
    m1 = ListGatherMetric(1)
    m2 = ListGatherMetric(2)

    def process_function(engine, data):
        return data

    engine = Engine(process_function)

    def plus(this, other):
        return this + other

    m0_plus_m1 = MetricsLambda(plus, m0, other=m1)
    m2_plus_2 = MetricsLambda(plus, m2, 2)
    m0_plus_m1.attach(engine, "m0_plus_m1")
    m2_plus_2.attach(engine, "m2_plus_2")

    engine.run([[1, 10, 100]])
    assert engine.state.metrics["m0_plus_m1"] == 11
    assert engine.state.metrics["m2_plus_2"] == 102
    engine.run([[2, 20, 200]])
    assert engine.state.metrics["m0_plus_m1"] == 22
    assert engine.state.metrics["m2_plus_2"] == 202

    # metrics are partially attached
    assert not m0.is_attached(engine)
    assert not m1.is_attached(engine)
    assert not m2.is_attached(engine)

    # a dependency is detached
    m0.detach(engine)
    # so the lambda metric is too
    assert not m0_plus_m1.is_attached(engine)
    # the lambda is attached again
    m0_plus_m1.attach(engine, "m0_plus_m1")
    assert m0_plus_m1.is_attached(engine)
    # metrics are always partially attached
    assert not m0.is_attached(engine)
    m0_plus_m1.detach(engine)
    assert not m0_plus_m1.is_attached(engine)
    # detached (and no longer partially attached)
    assert not m0.is_attached(engine)


def test_metrics_lambda_reset():
    m0 = ListGatherMetric(0)
    m1 = ListGatherMetric(1)
    m2 = ListGatherMetric(2)
    m0.update([1, 10, 100])
    m1.update([1, 10, 100])
    m2.update([1, 10, 100])

    def fn(x, y, z, t):
        return 1

    m = MetricsLambda(fn, m0, m1, z=m2, t=0)

    # initiating a new instance of MetricsLambda must reset
    # its argument metrics
    assert m0.list_ is None
    assert m1.list_ is None
    assert m2.list_ is None

    m0.update([1, 10, 100])
    m1.update([1, 10, 100])
    m2.update([1, 10, 100])
    m.reset()
    assert m0.list_ is None
    assert m1.list_ is None
    assert m2.list_ is None


def test_metrics_lambda_update_and_attach_together():
    y_pred = torch.randint(0, 2, size=(15, 10, 4)).float()
    y = torch.randint(0, 2, size=(15, 10, 4)).long()

    def update_fn(engine, batch):
        y_pred, y = batch
        return y_pred, y

    engine = Engine(update_fn)

    precision = Precision(average=False)
    recall = Recall(average=False)

    def Fbeta(r, p, beta):
        return torch.mean((1 + beta**2) * p * r / (beta**2 * p + r)).item()

    F1 = MetricsLambda(Fbeta, recall, precision, 1)

    F1.attach(engine, "f1")
    with pytest.raises(ValueError, match=r"MetricsLambda is already attached to an engine"):
        F1.update((y_pred, y))

    y_pred = torch.randint(0, 2, size=(15, 10, 4)).float()
    y = torch.randint(0, 2, size=(15, 10, 4)).long()

    F1 = MetricsLambda(Fbeta, recall, precision, 1)
    F1.update((y_pred, y))

    engine = Engine(update_fn)

    with pytest.raises(ValueError, match=r"The underlying metrics are already updated"):
        F1.attach(engine, "f1")

    F1.reset()
    F1.attach(engine, "f1")


def test_metrics_lambda_update():
    """
    Test if the underlying metrics are updated
    """
    y_pred = torch.randint(0, 2, size=(15, 10, 4)).float()
    y = torch.randint(0, 2, size=(15, 10, 4)).long()

    precision = Precision(average=False)
    recall = Recall(average=False)

    def Fbeta(r, p, beta):
        return torch.mean((1 + beta**2) * p * r / (beta**2 * p + r)).item()

    F1 = MetricsLambda(Fbeta, recall, precision, 1)

    F1.update((y_pred, y))

    assert precision._updated
    assert recall._updated

    F1.reset()

    assert not precision._updated
    assert not recall._updated

    """
    Test multiple updates and if the inputs of
    the underlying metrics are updated multiple times
    """
    y_pred1 = torch.randint(0, 2, size=(15,))
    y1 = torch.randint(0, 2, size=(15,))

    y_pred2 = torch.randint(0, 2, size=(15,))
    y2 = torch.randint(0, 2, size=(15,))

    F1.update((y_pred1, y1))
    F1.update((y_pred2, y2))

    # Compute true_positives and positives for precision
    correct1 = y1 * y_pred1
    all_positives1 = y_pred1.sum(dim=0)
    if correct1.sum() == 0:
        true_positives1 = torch.zeros_like(all_positives1)
    else:
        true_positives1 = correct1.sum(dim=0)

    correct2 = y2 * y_pred2
    all_positives2 = y_pred2.sum(dim=0)
    if correct2.sum() == 0:
        true_positives2 = torch.zeros_like(all_positives2)
    else:
        true_positives2 = correct2.sum(dim=0)

    true_positives = true_positives1 + true_positives2
    positives = all_positives1 + all_positives2

    assert precision._type == "binary"
    assert precision._numerator == true_positives
    assert precision._denominator == positives

    # Computing positivies for recall is different
    positives1 = y1.sum(dim=0)
    positives2 = y2.sum(dim=0)
    positives = positives1 + positives2

    assert recall._type == "binary"
    assert recall._numerator == true_positives
    assert recall._denominator == positives

    """
    Test compute
    """
    F1.reset()
    F1.update((y_pred1, y1))
    F1_metrics_lambda = F1.compute()
    F1_sklearn = f1_score(y1.numpy(), y_pred1.numpy())
    assert pytest.approx(F1_metrics_lambda) == F1_sklearn


@pytest.mark.parametrize("attach_pr_re", [True, False])
def test_integration(attach_pr_re):
    torch.manual_seed(1)

    n_iters = 10
    batch_size = 10
    n_classes = 10

    y_true = torch.arange(0, n_iters * batch_size) % n_classes
    y_pred = 0.2 * torch.rand(n_iters * batch_size, n_classes)
    for i in range(n_iters * batch_size):
        if torch.rand(1) > 0.4:
            y_pred[i, y_true[i]] = 1.0
        else:
            j = torch.randint(0, n_classes, size=(1,))
            y_pred[i, j] = 0.7

    y_true_batch_values = iter(y_true.reshape(n_iters, batch_size))
    y_pred_batch_values = iter(y_pred.reshape(n_iters, batch_size, n_classes))

    def update_fn(engine, batch):
        y_true_batch = next(y_true_batch_values)
        y_pred_batch = next(y_pred_batch_values)
        return y_pred_batch, y_true_batch

    evaluator = Engine(update_fn)

    precision = Precision(average=False)
    recall = Recall(average=False)

    def Fbeta(r, p, beta):
        return torch.mean((1 + beta**2) * p * r / (beta**2 * p + r)).item()

    F1 = MetricsLambda(Fbeta, recall, precision, 1)

    if attach_pr_re:
        precision.attach(evaluator, "precision")
        recall.attach(evaluator, "recall")
    F1.attach(evaluator, "f1")

    data = list(range(n_iters))
    state = evaluator.run(data, max_epochs=1)

    precision_true = precision_score(y_true, y_pred.argmax(dim=-1), average=None)
    recall_true = recall_score(y_true, y_pred.argmax(dim=-1), average=None)
    f1_true = f1_score(y_true, y_pred.argmax(dim=-1), average="macro")

    assert f1_true == approx(state.metrics["f1"]), f"{f1_true} vs {state.metrics['f1']}"
    if attach_pr_re:
        precision = state.metrics["precision"].numpy()
        recall = state.metrics["recall"].numpy()

        assert precision_true == approx(precision), f"{precision_true} vs {precision}"
        assert recall_true == approx(recall), f"{recall_true} vs {recall}"

    metric_state = F1.state_dict()
    F1.reset()
    F1.load_state_dict(metric_state)
    f1_value = F1.compute()
    assert f1_value == state.metrics["f1"]


def test_load_state_dict():
    acc = Accuracy()
    error = 1.0 - acc

    acc.update(
        (
            torch.randint(0, 2, size=(8,)),
            torch.randint(0, 2, size=(8,)),
        )
    )

    e = error.compute()
    a = acc.compute()
    assert 1.0 - a == e

    metric_state = error.state_dict()
    error.reset()
    error.load_state_dict(metric_state)
    e2 = error.compute()
    assert e2 == e


def test_state_metrics():
    y_pred = torch.randint(0, 2, size=(15, 10, 4)).float()
    y = torch.randint(0, 2, size=(15, 10, 4)).long()

    def update_fn(engine, batch):
        y_pred, y = batch
        return y_pred, y

    evaluator = Engine(update_fn)

    precision = Precision(average=False)
    recall = Recall(average=False)
    F1 = precision * recall * 2 / (precision + recall + 1e-20)
    F1 = MetricsLambda(lambda t: torch.mean(t).item(), F1)

    precision.attach(evaluator, "precision")
    recall.attach(evaluator, "recall")
    F1.attach(evaluator, "f1")

    def data(y_pred, y):
        for i in range(y_pred.shape[0]):
            yield (y_pred[i], y[i])

    d = data(y_pred, y)
    state = evaluator.run(d, max_epochs=1, epoch_length=y_pred.shape[0])

    assert set(state.metrics.keys()) == set(["precision", "recall", "f1"])


def test_state_metrics_ingredients_not_attached():
    y_pred = torch.randint(0, 2, size=(15, 10, 4)).float()
    y = torch.randint(0, 2, size=(15, 10, 4)).long()

    def update_fn(engine, batch):
        y_pred, y = batch
        return y_pred, y

    evaluator = Engine(update_fn)

    precision = Precision(average=False)
    recall = Recall(average=False)
    F1 = precision * recall * 2 / (precision + recall + 1e-20)
    F1 = MetricsLambda(lambda t: torch.mean(t).item(), F1)

    F1.attach(evaluator, "F1")

    def data(y_pred, y):
        for i in range(y_pred.shape[0]):
            yield (y_pred[i], y[i])

    d = data(y_pred, y)
    state = evaluator.run(d, max_epochs=1, epoch_length=y_pred.shape[0])

    assert set(state.metrics.keys()) == set(["F1"])


def test_recursive_attachment():
    def _test(composed_metric, metric_name, compute_true_value_fn):
        metrics = {
            metric_name: composed_metric,
        }

        y_pred = torch.randint(0, 2, size=(15, 10, 4)).float()
        y = torch.randint(0, 2, size=(15, 10, 4)).long()

        def update_fn(engine, batch):
            y_pred, y = batch
            return y_pred, y

        validator = Engine(update_fn)

        for name, metric in metrics.items():
            metric.attach(validator, name)

        def data(y_pred, y):
            for i in range(y_pred.shape[0]):
                yield (y_pred[i], y[i])

        d = data(y_pred, y)
        state = validator.run(d, max_epochs=1, epoch_length=y_pred.shape[0])

        assert set(state.metrics.keys()) == set([metric_name])
        np_y_pred = y_pred.numpy().ravel()
        np_y = y.numpy().ravel()
        assert state.metrics[metric_name] == approx(compute_true_value_fn(np_y_pred, np_y))

    precision_1 = Precision()
    precision_2 = Precision()
    summed_precision = precision_1 + precision_2

    def compute_true_summed_precision(y_pred, y):
        p1 = precision_score(y, y_pred)
        p2 = precision_score(y, y_pred)
        return p1 + p2

    _test(summed_precision, "summed precision", compute_true_value_fn=compute_true_summed_precision)

    precision_1 = Precision()
    precision_2 = Precision()
    mean_precision = (precision_1 + precision_2) / 2

    def compute_true_mean_precision(y_pred, y):
        p1 = precision_score(y, y_pred)
        p2 = precision_score(y, y_pred)
        return (p1 + p2) * 0.5

    _test(mean_precision, "mean precision", compute_true_value_fn=compute_true_mean_precision)

    precision_1 = Precision()
    precision_2 = Precision()
    some_metric = 2.0 + 0.2 * (precision_1 * precision_2 + precision_1 - precision_2) ** 0.5

    def compute_true_somemetric(y_pred, y):
        p1 = precision_score(y, y_pred)
        p2 = precision_score(y, y_pred)
        return 2.0 + 0.2 * (p1 * p2 + p1 - p2) ** 0.5

    _test(some_metric, "some metric", compute_true_somemetric)


def _test_distrib_integration(device):
    rank = idist.get_rank()

    n_iters = 10
    batch_size = 10
    n_classes = 10

    def _test(metric_device):
        y_true = torch.arange(0, n_iters * batch_size, dtype=torch.int64).to(device) % n_classes
        y_pred = 0.2 * torch.rand(n_iters * batch_size, n_classes).to(device)
        for i in range(n_iters * batch_size):
            if np.random.rand() > 0.4:
                y_pred[i, y_true[i]] = 1.0
            else:
                j = np.random.randint(0, n_classes)
                y_pred[i, j] = 0.7

        def update_fn(engine, i):
            y_true_batch = y_true[i * batch_size : (i + 1) * batch_size, ...]
            y_pred_batch = y_pred[i * batch_size : (i + 1) * batch_size, ...]
            return y_pred_batch, y_true_batch

        evaluator = Engine(update_fn)

        precision = Precision(average=False, device=metric_device)
        recall = Recall(average=False, device=metric_device)

        def Fbeta(r, p, beta):
            return torch.mean((1 + beta**2) * p * r / (beta**2 * p + r)).item()

        F1 = MetricsLambda(Fbeta, recall, precision, 1)
        F1.attach(evaluator, "f1")

        another_f1 = (1.0 + precision * recall * 2 / (precision + recall + 1e-20)).mean().item()
        another_f1.attach(evaluator, "ff1")

        data = list(range(n_iters))
        state = evaluator.run(data, max_epochs=1)

        y_pred = idist.all_gather(y_pred)
        y_true = idist.all_gather(y_true)

        assert "f1" in state.metrics
        assert "ff1" in state.metrics
        f1_true = f1_score(y_true.view(-1).cpu(), y_pred.view(-1, n_classes).argmax(dim=-1).cpu(), average="macro")
        assert f1_true == approx(state.metrics["f1"])
        assert 1.0 + f1_true == approx(state.metrics["ff1"])

    for i in range(3):
        torch.manual_seed(12 + rank + i)
        _test("cpu")
        if device.type != "xla":
            _test(idist.device())


def _test_distrib_metrics_on_diff_devices(device):
    n_classes = 10
    n_iters = 12
    batch_size = 16
    rank = idist.get_rank()
    torch.manual_seed(12 + rank)

    y_true = torch.randint(0, n_classes, size=(n_iters * batch_size,)).to(device)
    y_preds = torch.rand(n_iters * batch_size, n_classes).to(device)

    def update(engine, i):
        return (
            y_preds[i * batch_size : (i + 1) * batch_size, :],
            y_true[i * batch_size : (i + 1) * batch_size],
        )

    evaluator = Engine(update)

    precision = Precision(average=False, device="cpu")
    recall = Recall(average=False, device=device)

    def Fbeta(r, p, beta):
        return torch.mean((1 + beta**2) * p * r / (beta**2 * p + r)).item()

    F1 = MetricsLambda(Fbeta, recall, precision, 1)
    F1.attach(evaluator, "f1")

    another_f1 = (1.0 + precision * recall * 2 / (precision + recall + 1e-20)).mean().item()
    another_f1.attach(evaluator, "ff1")

    data = list(range(n_iters))
    state = evaluator.run(data, max_epochs=1)

    y_preds = idist.all_gather(y_preds)
    y_true = idist.all_gather(y_true)

    assert "f1" in state.metrics
    assert "ff1" in state.metrics
    f1_true = f1_score(y_true.view(-1).cpu(), y_preds.view(-1, n_classes).argmax(dim=-1).cpu(), average="macro")
    assert f1_true == approx(state.metrics["f1"])
    assert 1.0 + f1_true == approx(state.metrics["ff1"])


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_nccl_gpu(distributed_context_single_node_nccl):
    device = idist.device()
    _test_distrib_integration(device)
    _test_distrib_metrics_on_diff_devices(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_gloo_cpu_or_gpu(distributed_context_single_node_gloo):
    device = idist.device()
    _test_distrib_integration(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_distrib_integration, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_metrics_on_diff_devices, (device,), np=nproc, do_init=True)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gloo_cpu_or_gpu(distributed_context_multi_node_gloo):
    device = idist.device()
    _test_distrib_integration(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_nccl_gpu(distributed_context_multi_node_nccl):
    device = idist.device()
    _test_distrib_integration(device)
    _test_distrib_metrics_on_diff_devices(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():
    device = idist.device()
    _test_distrib_integration(device)


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_distrib_integration(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)
