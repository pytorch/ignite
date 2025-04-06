import numbers
import os
from typing import Dict, List
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from packaging.version import Version
from pytest import approx, raises
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

import ignite.distributed as idist
from ignite.engine import Engine, Events, State
from ignite.metrics import Accuracy, ConfusionMatrix, Precision, Recall
from ignite.metrics.metric import (
    BatchFiltered,
    BatchWise,
    EpochWise,
    Metric,
    reinit__is_reduced,
    RunningBatchWise,
    RunningEpochWise,
    SingleEpochRunningBatchWise,
    sync_all_reduce,
)
from ignite.utils import _tree_map


class DummyMetric1(Metric):
    def __init__(self, true_output, output_transform=lambda x: x):
        super(DummyMetric1, self).__init__(output_transform=output_transform)
        self.true_output = true_output

    def reset(self):
        pass

    def compute(self):
        pass

    def update(self, output):
        assert output == self.true_output


def test_no_transform():
    y_pred = torch.tensor([[2.0], [-2.0]])
    y = torch.zeros(2)

    metric = DummyMetric1(true_output=(y_pred, y))
    state = State(output=(y_pred, y))
    engine = MagicMock(state=state)
    metric.iteration_completed(engine)


def test_transform():
    y_pred = torch.tensor([[2.0], [-2.0]])
    y = torch.zeros(2)

    def transform(output):
        pred_dict, target_dict = output
        return pred_dict["y"], target_dict["y"]

    metric = DummyMetric1(true_output=(y_pred, y), output_transform=transform)
    state = State(output=({"y": y_pred}, {"y": y}))
    engine = MagicMock(state=state)
    metric.iteration_completed(engine)


def test_output_as_mapping_wrong_keys():
    metric = DummyMetric1(true_output=(0, 1))
    state = State(output=({"y1": 0, "y2": 1}))
    engine = MagicMock(state=state)

    with pytest.raises(
        ValueError, match=r"When transformed engine's output is a mapping, " r"it should contain \('y_pred', 'y'\) keys"
    ):
        metric.iteration_completed(engine)


def test_output_as_mapping_keys_is_none():
    class DummyMetric(Metric):
        required_output_keys = None

        def reset(self):
            pass

        def compute(self):
            pass

        def update(self, output):
            pass

    metric = DummyMetric()
    assert metric.required_output_keys is None
    state = State(output=({"y1": 0, "y2": 1}))
    engine = MagicMock(state=state)

    with pytest.raises(TypeError, match=r"Transformed engine output for DummyMetric metric should be a tuple/list"):
        metric.iteration_completed(engine)


def test_output_as_mapping():
    y_pred = torch.tensor([[2.0], [-2.0]])
    y = torch.zeros(2)

    metric = DummyMetric1(true_output=(y_pred, y))
    state = State(output=({"y_pred": y_pred, "y": y}))
    engine = MagicMock(state=state)
    metric.iteration_completed(engine)


def test_no_grad():
    y_pred = torch.zeros(4, requires_grad=True)
    y = torch.zeros(4, requires_grad=False)

    class DummyMetric(Metric):
        def reset(self):
            pass

        def compute(self):
            pass

        def update(self, output):
            y_pred, y = output
            mse = torch.pow(y_pred - y.view_as(y_pred), 2)
            assert y_pred.requires_grad
            assert not mse.requires_grad

    metric = DummyMetric()
    state = State(output=(y_pred, y))
    engine = MagicMock(state=state)
    metric.iteration_completed(engine)


def test_arithmetics():
    class ListGatherMetric(Metric):
        def __init__(self, index):
            self.index = index
            super(ListGatherMetric, self).__init__()

        def reset(self):
            self.list_ = []

        def update(self, output):
            self.list_ = output

        def compute(self):
            return self.list_[self.index]

    m0 = ListGatherMetric(0)
    m1 = ListGatherMetric(1)
    m2 = ListGatherMetric(2)

    # __add__
    m0_plus_m1 = m0 + m1
    m0.update([1, 10, 100])
    m1.update([1, 10, 100])
    assert m0_plus_m1.compute() == 11
    m0.update([2, 20, 200])
    m1.update([2, 20, 200])
    assert m0_plus_m1.compute() == 22

    m2_plus_2 = m2 + 2
    m2.update([1, 10, 100])
    assert m2_plus_2.compute() == 102

    m2_plus_2 = 2 + m2
    m2.update([1, 10, 100])
    assert m2_plus_2.compute() == 102

    # __sub__
    m0_minus_m1 = m0 - m1
    m0.update([1, 10, 100])
    m1.update([1, 10, 100])
    assert m0_minus_m1.compute() == -9
    m0.update([2, 20, 200])
    m1.update([2, 20, 200])
    assert m0_minus_m1.compute() == -18

    m2_minus_2 = m2 - 2
    m2.update([1, 10, 100])
    assert m2_minus_2.compute() == 98

    m2_minus_2 = 2 - m2
    m2.update([1, 10, 100])
    assert m2_minus_2.compute() == -98

    # __mul__
    m0_times_m1 = m0 * m1
    m0.update([1, 10, 100])
    m1.update([1, 10, 100])
    assert m0_times_m1.compute() == 10
    m0.update([2, 20, 200])
    m1.update([2, 20, 200])
    assert m0_times_m1.compute() == 40

    m2_times_2 = m2 * 2
    m2.update([1, 10, 100])
    assert m2_times_2.compute() == 200

    m2_times_2 = 2 * m2
    m2.update([1, 10, 100])
    assert m2_times_2.compute() == 200

    # __pow__
    m0_pow_m1 = m0**m1
    m0.update([1, 10, 100])
    m1.update([1, 10, 100])
    assert m0_pow_m1.compute() == 1
    m0.update([2, 20, 200])
    m1.update([2, 20, 200])
    assert m0_pow_m1.compute() == 2**20

    m2_pow_2 = m2**2
    m2.update([1, 10, 100])
    assert m2_pow_2.compute() == 10000

    m2_pow_2 = 0.99**m2
    m2.update([1, 10, 100])
    assert m2_pow_2.compute() == 0.3660323412732292

    # __mod__
    m0_mod_m1 = m0 % m1
    m0.update([1, 10, 100])
    m1.update([1, 10, 100])
    assert m0_mod_m1.compute() == 1
    m0.update([2, 20, 200])
    m1.update([2, 20, 200])
    assert m0_mod_m1.compute() == 2

    m2_mod_2 = m2 % 2
    m2.update([1, 10, 100])
    assert m2_mod_2.compute() == 0

    # __truediv__
    m0_truediv_m1 = m0 / m1
    m0.update([1, 10, 100])
    m1.update([1, 10, 100])
    assert m0_truediv_m1.compute() == approx(0.1)
    m0.update([2, 20, 200])
    m1.update([2, 20, 200])
    assert m0_truediv_m1.compute() == approx(0.1)

    m2_truediv_2 = m2 / 2
    m2.update([1, 10, 100])
    assert m2_truediv_2.compute() == approx(50.0)

    m2_truediv_2 = 200 / m2
    m2.update([1, 10, 100])
    assert m2_truediv_2.compute() == approx(2.0)

    m0_truediv_m1 = m0.__truediv__(m1)
    m0.update([1, 10, 100])
    m1.update([1, 10, 100])
    assert m0_truediv_m1.compute() == approx(0.1)
    m0.update([2, 20, 200])
    m1.update([2, 20, 200])
    assert m0_truediv_m1.compute() == approx(0.1)

    m2_truediv_2 = m2.__truediv__(2)
    m2.update([1, 10, 100])
    assert m2_truediv_2.compute() == approx(50.0)

    m2_truediv_2 = m2.__rtruediv__(200)
    m2.update([1, 10, 100])
    assert m2_truediv_2.compute() == approx(2.0)

    # __floordiv__
    m0_floordiv_m1 = m0 // m1
    m0.update([1, 10, 100])
    m1.update([1, 10, 100])
    assert m0_floordiv_m1.compute() == 0
    m0.update([2, 20, 200])
    m1.update([2, 20, 200])
    assert m0_floordiv_m1.compute() == 0

    m2_floordiv_2 = m2 // 2
    m2.update([1, 10, 100])
    assert m2_floordiv_2.compute() == 50


def test_attach():
    class CountMetric(Metric):
        def __init__(self, value):
            self.reset_count = 0
            super(CountMetric, self).__init__()
            self.reset_count = 0
            self.compute_count = 0
            self.update_count = 0
            self.value = value

        def reset(self):
            self.reset_count += 1

        def compute(self):
            self.compute_count += 1
            return self.value

        def update(self, output):
            self.update_count += 1

    def process_function(*args, **kwargs):
        return 1

    engine = Engine(process_function)
    m1 = CountMetric(123)
    m2 = CountMetric(456)
    m1.attach(engine, "m1")
    m2.attach(engine, "m2_1")
    m2.attach(engine, "m2_2")
    engine.run(range(10), 5)

    assert engine.state.metrics["m1"] == 123
    assert engine.state.metrics["m2_1"] == 456
    assert engine.state.metrics["m2_2"] == 456

    assert m1.reset_count == 5
    assert m1.compute_count == 5
    assert m1.update_count == 50

    assert m2.reset_count == 5
    assert m2.compute_count == 10
    assert m2.update_count == 50

    assert m1.is_attached(engine)
    assert m2.is_attached(engine)


def test_detach():
    class DummyMetric(Metric):
        required_output_keys = None

        def reset(self):
            pass

        def compute(self):
            pass

        def update(self, output):
            pass

    def process_function(*args, **kwargs):
        return 1

    engine = Engine(process_function)
    m1 = DummyMetric()
    m2 = DummyMetric()
    m1.attach(engine, "m1")
    m2.attach(engine, "m2_1")
    m2.attach(engine, "m2_2")
    m1.detach(engine)
    m2.detach(engine)
    engine.run(range(10), 5)

    assert "m1" not in engine.state.metrics
    assert "m2_1" not in engine.state.metrics
    assert "m2_2" not in engine.state.metrics

    assert not m1.is_attached(engine)
    assert not m2.is_attached(engine)


def test_integration():
    np.random.seed(1)

    n_iters = 10
    batch_size = 10
    n_classes = 10

    y_true = np.arange(0, n_iters * batch_size, dtype="int64") % n_classes
    y_pred = 0.2 * np.random.rand(n_iters * batch_size, n_classes)
    for i in range(n_iters * batch_size):
        if np.random.rand() > 0.4:
            y_pred[i, y_true[i]] = 1.0
        else:
            j = np.random.randint(0, n_classes)
            y_pred[i, j] = 0.7

    y_true_batch_values = iter(y_true.reshape(n_iters, batch_size))
    y_pred_batch_values = iter(y_pred.reshape(n_iters, batch_size, n_classes))

    def update_fn(engine, batch):
        y_true_batch = next(y_true_batch_values)
        y_pred_batch = next(y_pred_batch_values)
        return torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

    evaluator = Engine(update_fn)

    precision = Precision(average=False)
    recall = Recall(average=False)
    F1 = precision * recall * 2 / (precision + recall)

    precision.attach(evaluator, "precision")
    recall.attach(evaluator, "recall")
    F1.attach(evaluator, "f1")

    data = list(range(n_iters))
    state = evaluator.run(data, max_epochs=1)

    precision_true = precision_score(y_true, np.argmax(y_pred, axis=-1), average=None)
    recall_true = recall_score(y_true, np.argmax(y_pred, axis=-1), average=None)
    f1_true = f1_score(y_true, np.argmax(y_pred, axis=-1), average=None)

    precision = state.metrics["precision"].numpy()
    recall = state.metrics["recall"].numpy()
    f1 = state.metrics["f1"].numpy()

    assert precision_true == approx(precision), f"{precision_true} vs {precision}"
    assert recall_true == approx(recall), f"{recall_true} vs {recall}"
    assert f1_true == approx(f1), f"{f1_true} vs {f1}"


def test_abstract_class():
    with raises(TypeError):
        Metric()


def test_pytorch_operators():
    def _test(composed_metric, metric_name, compute_true_value_fn):
        metrics = {
            metric_name: composed_metric,
        }

        y_pred = torch.rand(15, 10, 5).float()
        y = torch.randint(0, 5, size=(15, 10)).long()

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
        np_y_pred = np.argmax(y_pred.numpy(), axis=-1).ravel()
        np_y = y.numpy().ravel()
        assert state.metrics[metric_name] == approx(compute_true_value_fn(np_y_pred, np_y))

    precision_1 = Precision(average=False)
    precision_2 = Precision(average=False)
    norm_summed_precision = (precision_1 + precision_2).norm(p=10)

    def compute_true_norm_summed_precision(y_pred, y):
        p1 = precision_score(y, y_pred, average=None)
        p2 = precision_score(y, y_pred, average=None)
        return np.linalg.norm(p1 + p2, ord=10)

    _test(norm_summed_precision, "mean summed precision", compute_true_value_fn=compute_true_norm_summed_precision)

    precision = Precision(average=False)
    recall = Recall(average=False)
    sum_precision_recall = (precision + recall).sum()

    def compute_sum_precision_recall(y_pred, y):
        p = precision_score(y, y_pred, average=None)
        r = recall_score(y, y_pred, average=None)
        return np.sum(p + r)

    _test(sum_precision_recall, "sum precision recall", compute_true_value_fn=compute_sum_precision_recall)

    precision = Precision(average=False)
    recall = Recall(average=False)
    f1 = (precision * recall * 2 / (precision + recall + 1e-20)).mean()

    def compute_f1(y_pred, y):
        f1 = f1_score(y, y_pred, average="macro")
        return f1

    _test(f1, "f1", compute_true_value_fn=compute_f1)


def test_indexing_metric():
    def _test(ignite_metric, sklearn_metic, sklearn_args, index, num_classes=5):
        y_pred = torch.rand(15, 10, num_classes).float()
        y = torch.randint(0, num_classes, size=(15, 10)).long()

        def update_fn(engine, batch):
            y_pred, y = batch
            return y_pred, y

        metrics = {"metric": ignite_metric[index], "metric_wo_index": ignite_metric}

        validator = Engine(update_fn)

        for name, metric in metrics.items():
            metric.attach(validator, name)

        def data(y_pred, y):
            for i in range(y_pred.shape[0]):
                yield (y_pred[i], y[i])

        d = data(y_pred, y)
        state = validator.run(d, max_epochs=1, epoch_length=y_pred.shape[0])

        sklearn_output = sklearn_metic(
            y.view(-1).numpy(), y_pred.view(-1, num_classes).argmax(dim=1).numpy(), **sklearn_args
        )

        assert (state.metrics["metric_wo_index"][index] == state.metrics["metric"]).all()
        assert np.allclose(state.metrics["metric"].numpy(), sklearn_output)

    num_classes = 5

    labels = list(range(0, num_classes, 2))
    _test(Precision(), precision_score, {"labels": labels, "average": None}, index=labels)
    labels = list(range(num_classes - 1, 0, -2))
    _test(Precision(), precision_score, {"labels": labels, "average": None}, index=labels)
    labels = [1]
    _test(Precision(), precision_score, {"labels": labels, "average": None}, index=labels)

    labels = list(range(0, num_classes, 2))
    _test(Recall(), recall_score, {"labels": labels, "average": None}, index=labels)
    labels = list(range(num_classes - 1, 0, -2))
    _test(Recall(), recall_score, {"labels": labels, "average": None}, index=labels)
    labels = [1]
    _test(Recall(), recall_score, {"labels": labels, "average": None}, index=labels)

    # np.ix_ is used to allow for a 2D slice of a matrix. This is required to get accurate result from
    # ConfusionMatrix. ConfusionMatrix must be sliced the same row-wise and column-wise.
    labels = list(range(0, num_classes, 2))
    _test(ConfusionMatrix(num_classes), confusion_matrix, {"labels": labels}, index=np.ix_(labels, labels))
    labels = list(range(num_classes - 1, 0, -2))
    _test(ConfusionMatrix(num_classes), confusion_matrix, {"labels": labels}, index=np.ix_(labels, labels))
    labels = [1]
    _test(ConfusionMatrix(num_classes), confusion_matrix, {"labels": labels}, index=np.ix_(labels, labels))


class DummyMetric2(Metric):
    @reinit__is_reduced
    def reset(self):
        pass

    def compute(self):
        pass

    @reinit__is_reduced
    def update(self, output):
        pass


def _test_compute_with_sync_all_reduce_doesnt_change_attributes(device):
    class DummyMetric3(Metric):
        @reinit__is_reduced
        def reset(self):
            self.a = torch.tensor(0.0, device=self._device)
            self.b = 0.0

        def update(self, output):
            self.a += torch.tensor(1.0)
            self.b += 1.0

        @sync_all_reduce("a", "b")
        def compute(self):
            return self.a.item(), self.b

    metric_device = device if torch.device(device).type != "xla" else "cpu"
    metric = DummyMetric3(device=metric_device)
    metric.update(None)
    assert metric.a.item() == metric.b == 1.0
    metric.compute()
    assert metric.a.item() == metric.b == 1.0


def _test_invalid_sync_all_reduce(device):
    class InvalidMetric(Metric):
        @reinit__is_reduced
        def reset(self):
            self.a = torch.tensor([0.0, 1.0, 2.0, 3.0], requires_grad=False)
            self.c = 0.0
            self.n = 0
            self.m = -1
            self.d = "a string"

        def compute(self):
            pass

        def update(self):
            pass

        @sync_all_reduce("a:sum")
        def invalid_reduction_op_1(self):
            pass

        @sync_all_reduce("c:MaX")
        def invalid_reduction_op_2(self):
            pass

        @sync_all_reduce("n:MINN")
        def invalid_reduction_op_3(self):
            pass

        @sync_all_reduce("m:PROduCT")
        def invalid_reduction_op_4(self):
            pass

        @sync_all_reduce("missingattr")
        def invalid_reduction_op_5(self):
            pass

        @sync_all_reduce("d")
        def invalid_reduction_op_6(self):
            pass

    metric_device = device if torch.device(device).type != "xla" else "cpu"
    m = InvalidMetric(device=metric_device)
    m.reset()

    if idist.get_world_size() > 1:
        with pytest.raises(ValueError, match=r"Reduction operation is not valid"):
            m.invalid_reduction_op_1()

        with pytest.raises(ValueError, match=r"Reduction operation is not valid"):
            m.invalid_reduction_op_2()

        with pytest.raises(ValueError, match=r"Reduction operation is not valid"):
            m.invalid_reduction_op_3()

        with pytest.raises(ValueError, match=r"Reduction operation is not valid"):
            m.invalid_reduction_op_4()

        with pytest.raises(ValueError, match=r"has no attribute named `missingattr`."):
            m.invalid_reduction_op_5()

        with pytest.raises(
            TypeError, match=r"Attribute provided to sync_all_reduce should be a number or tensor but `d`"
        ):
            m.invalid_reduction_op_6()


def _test_distrib_sync_all_reduce_decorator(device):
    class DummyMetric(Metric):
        @reinit__is_reduced
        def reset(self):
            # SUM op
            self.a = torch.tensor([0.0, 1.0, 2.0, 3.0], device=self._device, requires_grad=False)
            self.a_nocomp = self.a.clone().to("cpu")
            self.b = torch.tensor(1.0, dtype=torch.float64, device=self._device, requires_grad=False)
            self.b_nocomp = self.b.clone().to("cpu")
            self.c = 0.0
            self.c_nocomp = self.c
            self.n = 0
            self.n_nocomp = self.n

            # MAX op
            self.m = -1

            # MIN op
            self.k = 10000

            # initialize number of updates to test (MAX, MIN) ops
            self.num_updates = 0

            # PRODUCT op
            self.prod = torch.tensor([2.0, 3.0], device=self._device, requires_grad=False)
            self.prod_nocomp = self.prod.clone().to("cpu")

        @sync_all_reduce("a", "b", "c", "n:SUM", "m:MAX", "k:MIN", "prod:PRODUCT")
        def compute(self):
            assert (self.a.cpu() == (self.a_nocomp + 10) * idist.get_world_size()).all()
            assert (self.b.cpu() == (self.b_nocomp - 5) * idist.get_world_size()).all()
            assert self.c == pytest.approx((self.c_nocomp + 1.23456) * idist.get_world_size())
            assert self.n == (self.n_nocomp + 1) * idist.get_world_size()
            assert self.m == self.num_updates * (idist.get_world_size() - 1) - 1
            assert self.k == 10000 - self.num_updates * (idist.get_world_size() - 1)
            temp_prod_nocomp = 5 * self.prod_nocomp  # new variable for the recomputing
            temp_prod_nocomp = temp_prod_nocomp.pow(idist.get_world_size())
            assert (self.prod.cpu() == temp_prod_nocomp).all()

        @reinit__is_reduced
        def update(self, output):
            # SUM op
            self.n += 1
            self.c += 1.23456
            self.a += 10.0
            self.b -= 5.0

            # MAX op
            self.m += idist.get_rank()

            # MIN op
            self.k -= idist.get_rank()

            # numper of updates for (MAX, MIN) ops
            self.num_updates += 1

            # PRODUCT op
            self.prod *= 5

    metric_device = device if torch.device(device).type != "xla" else "cpu"
    m = DummyMetric(device=metric_device)
    m.update(None)
    m.compute()
    # check if attributes are restored to their original values after previous `compute`
    m.compute()


def _test_creating_on_xla_fails(device):
    with pytest.raises(ValueError, match=r"Cannot create metric on an XLA device. Use device='cpu' instead."):
        DummyMetric2(device=device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
@pytest.mark.skipif(Version(torch.__version__) < Version("1.7.0"), reason="Skip if < 1.7.0")
def test_distrib_nccl_gpu(distributed_context_single_node_nccl):
    device = idist.device()
    _test_distrib_sync_all_reduce_decorator(device)
    _test_invalid_sync_all_reduce(device)
    _test_compute_with_sync_all_reduce_doesnt_change_attributes(device)

    test_state_dict()
    test_load_state_dict()


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(Version(torch.__version__) < Version("1.7.0"), reason="Skip if < 1.7.0")
def test_distrib_gloo_cpu_or_gpu(distributed_context_single_node_gloo):
    device = idist.device()
    _test_distrib_sync_all_reduce_decorator(device)
    _test_invalid_sync_all_reduce(device)
    _test_compute_with_sync_all_reduce_doesnt_change_attributes(device)
    test_state_dict()
    test_load_state_dict()


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_distrib_sync_all_reduce_decorator, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_invalid_sync_all_reduce, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_compute_with_sync_all_reduce_doesnt_change_attributes, (device,), np=nproc, do_init=True)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gloo_cpu_or_gpu(distributed_context_multi_node_gloo):
    device = idist.device()
    _test_distrib_sync_all_reduce_decorator(device)
    _test_invalid_sync_all_reduce(device)
    _test_compute_with_sync_all_reduce_doesnt_change_attributes(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_nccl_gpu(distributed_context_multi_node_nccl):
    device = idist.device()
    _test_distrib_sync_all_reduce_decorator(device)
    _test_invalid_sync_all_reduce(device)
    _test_compute_with_sync_all_reduce_doesnt_change_attributes(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():
    device = idist.device()
    _test_distrib_sync_all_reduce_decorator(device)
    _test_creating_on_xla_fails(device)
    _test_invalid_sync_all_reduce(device)
    _test_compute_with_sync_all_reduce_doesnt_change_attributes(device)


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_distrib_sync_all_reduce_decorator(device)
    _test_creating_on_xla_fails(device)
    _test_invalid_sync_all_reduce(device)
    _test_compute_with_sync_all_reduce_doesnt_change_attributes(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)


def test_completed():
    class DummyMetric(Metric):
        def reset(self):
            pass

        def compute(self):
            pass

        def update(self, output):
            pass

    m = DummyMetric()

    # tensor
    engine = MagicMock(state=State(metrics={}))
    m.compute = MagicMock(return_value=torch.tensor(1.0))
    m.completed(engine, "metric")
    assert engine.state.metrics == {"metric": 1.0}
    assert isinstance(engine.state.metrics["metric"], numbers.Number)

    # mapping
    engine = MagicMock(state=State(metrics={}))
    metrics = {"foo": 1, "bar": torch.tensor(2.0), "baz": {"qux": "quux"}}
    m.compute = MagicMock(return_value=metrics)
    with pytest.raises(ValueError, match=r"Argument name 'foo' is conflicting with mapping keys"):
        m.completed(engine, "foo")
    m.completed(engine, "metric")
    metrics["metric"] = metrics
    assert engine.state.metrics == metrics

    # other
    engine = MagicMock(state=State(metrics={}))
    m.compute = MagicMock(return_value="foo")
    m.completed(engine, "metric")
    assert engine.state.metrics == {"metric": "foo"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_completed_on_cuda():
    # Checks https://github.com/pytorch/ignite/issues/1635#issuecomment-863026919

    class DummyMetric(Metric):
        def reset(self):
            pass

        def compute(self):
            return torch.tensor([1.0, 2.0, 3.0], device="cuda")

        def update(self, output):
            pass

    m = DummyMetric()

    # tensor
    engine = MagicMock(state=State(metrics={}))
    m.completed(engine, "metric")
    assert "metric" in engine.state.metrics
    assert isinstance(engine.state.metrics["metric"], torch.Tensor)
    assert engine.state.metrics["metric"].device.type == "cpu"


def test_usage_exception():
    engine = Engine(lambda e, b: b)
    m = DummyMetric2()
    with pytest.raises(TypeError, match=r"Unhandled usage type"):
        m.attach(engine, "dummy", usage=1)
    with pytest.raises(
        ValueError,
        match=r"usage should be '\(Running\)EpochWise.usage_name' or '\(\(SingleEpoch\)Running\)BatchWise.usage_name'",
    ):
        m.attach(engine, "dummy", usage="fake")


class DummyAccumulateInListMetric(Metric):
    def __init__(self):
        super(DummyAccumulateInListMetric, self).__init__()
        self.value = []

    def reset(self):
        self.value = []

    def compute(self):
        return self.value

    def update(self, output):
        self.value.append(output)


@pytest.mark.parametrize("usage", ["epoch_wise", EpochWise.usage_name, EpochWise()])
def test_epochwise_usage(usage):
    engine = Engine(lambda e, b: b)

    m = DummyAccumulateInListMetric()

    m.attach(engine, "ewm", usage=usage)

    @engine.on(Events.EPOCH_COMPLETED)
    def _():
        ewm = engine.state.metrics["ewm"]
        assert len(ewm) == 3
        assert ewm == [0, 1, 2]

    engine.run([0, 1, 2], max_epochs=10)
    m.detach(engine, usage=usage)


class DummyAccumulateMetric(Metric):
    def __init__(self):
        super(DummyAccumulateMetric, self).__init__()
        self.value = 0

    def reset(self):
        self.value = 0

    def compute(self):
        return self.value

    def update(self, output):
        self.value += output


@pytest.mark.parametrize("usage", ["running_epoch_wise", RunningEpochWise.usage_name, RunningEpochWise()])
def test_running_epochwise_usage(usage):
    engine = Engine(lambda e, b: e.state.metrics["ewm"])

    engine.state.metrics["ewm"] = 0

    @engine.on(Events.EPOCH_STARTED)
    def _():
        engine.state.metrics["ewm"] += 1

    m = DummyAccumulateMetric()
    m.attach(engine, "rewm", usage=usage)

    @engine.on(Events.EPOCH_COMPLETED)
    def _():
        assert engine.state.metrics["rewm"] == sum(range(engine.state.epoch + 1))

    engine.run([0, 1, 2], max_epochs=10)

    m.detach(engine, usage=usage)


@pytest.mark.parametrize("usage", ["batch_wise", BatchWise.usage_name, BatchWise()])
def test_batchwise_usage(usage):
    engine = Engine(lambda e, b: b)

    m = DummyAccumulateInListMetric()

    m.attach(engine, "bwm", usage=usage)

    @engine.on(Events.ITERATION_COMPLETED)
    def _():
        bwm = engine.state.metrics["bwm"]
        assert len(bwm) == 1
        assert bwm[0] == (engine.state.iteration - 1) % 3

    engine.run([0, 1, 2], max_epochs=10)
    m.detach(engine, usage=usage)


@pytest.mark.parametrize("usage", ["running_batch_wise", RunningBatchWise.usage_name, RunningBatchWise()])
def test_running_batchwise_usage(usage):
    engine = Engine(lambda e, b: b)

    m = DummyAccumulateMetric()
    m.attach(engine, "rbwm", usage=usage)

    @engine.on(Events.EPOCH_COMPLETED)
    def _():
        assert engine.state.metrics["rbwm"] == 6 * engine.state.epoch

    engine.run([0, 1, 2, 3], max_epochs=10)

    m.detach(engine, usage=usage)


@pytest.mark.parametrize(
    "usage", ["single_epoch_running_batch_wise", SingleEpochRunningBatchWise.usage_name, SingleEpochRunningBatchWise()]
)
def test_single_epoch_running_batchwise_usage(usage):
    engine = Engine(lambda e, b: b)

    m = DummyAccumulateMetric()
    m.attach(engine, "rbwm", usage=usage)

    @engine.on(Events.EPOCH_COMPLETED)
    def _():
        assert engine.state.metrics["rbwm"] == 6

    engine.run([0, 1, 2, 3], max_epochs=10)

    m.detach(engine, usage=usage)


def test_batchfiltered_usage():
    class MyMetric(Metric):
        def __init__(self):
            super(MyMetric, self).__init__()
            self.value = []

        def reset(self):
            self.value = []

        def compute(self):
            return self.value

        def update(self, output):
            self.value.append(output)

    engine = Engine(lambda e, b: b)

    m = MyMetric()

    usage = BatchFiltered(every=2)

    m.attach(engine, "bfm", usage=usage)

    @engine.on(Events.EPOCH_COMPLETED)
    def _():
        bfm = engine.state.metrics["bfm"]
        assert len(bfm) == 2
        assert bfm[0] == 1

    engine.run([0, 1, 2, 3], max_epochs=10)


def test_override_required_output_keys():
    # https://discuss.pytorch.org/t/how-access-inputs-in-custom-ignite-metric/91221/5
    import torch.nn as nn

    from ignite.engine import create_supervised_evaluator

    counter = [0]

    class CustomMetric(Metric):
        required_output_keys = ("y_pred", "y", "x")

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def update(self, output):
            y_pred, y, x = output
            assert y_pred.shape == (4, 3)
            assert y.shape == (4,)
            assert x.shape == (4, 10)
            assert x.equal(data[counter[0]][0])
            assert y.equal(data[counter[0]][1])
            counter[0] += 1

        def reset(self):
            pass

        def compute(self):
            pass

    model = nn.Linear(10, 3)

    metrics = {"Precision": Precision(), "CustomMetric": CustomMetric()}

    evaluator = create_supervised_evaluator(
        model, metrics=metrics, output_transform=lambda x, y, y_pred: {"x": x, "y": y, "y_pred": y_pred}
    )

    data = [
        (torch.rand(4, 10), torch.randint(0, 3, size=(4,))),
        (torch.rand(4, 10), torch.randint(0, 3, size=(4,))),
        (torch.rand(4, 10), torch.randint(0, 3, size=(4,))),
    ]
    evaluator.run(data)


@pytest.mark.parametrize("shapes", [[(10,), ()], [(5, 32, 32), (5, 32, 32)]])
def test_list_of_tensors_and_numbers(shapes):
    def check_fn(output):
        assert len(output) == 2
        assert isinstance(output[0], torch.Tensor)
        assert isinstance(output[1], torch.Tensor)
        assert output[0].shape == (1,) + shapes[0]
        assert output[1].shape == (1,) + shapes[1]

    def get_data(gt_as_scalar=False):
        return [
            (
                [torch.rand(shapes[0]) for _ in range(3 + i)],  # predictions
                [
                    torch.rand(shapes[1]).item() if gt_as_scalar else torch.rand(shapes[1]) for _ in range(3 + i)
                ],  # ground truth
            )
            for i in range(5)
        ]

    class MyMetric(Metric):
        def __init__(self, check_fn):
            super(MyMetric, self).__init__()
            self.check_fn = check_fn

        def reset(self):
            pass

        def compute(self):
            pass

        def update(self, output):
            self.check_fn(output)

    engine = Engine(lambda e, b: b)
    m = MyMetric(check_fn)
    m.attach(engine, "m")

    data = get_data()
    engine.run(data)

    if len(shapes[1]) == 0:
        data = get_data(gt_as_scalar=True)
        engine.run(data)


def test_list_of_tensors_and_numbers_unsupported_output():
    class MyMetric(Metric):
        def reset(self):
            pass

        def compute(self):
            pass

        def update(self, output):
            pass

    engine = Engine(lambda e, b: ([0, 1, 2], [0, 1, 2], [0, 1, 2]))
    m = MyMetric()
    m.attach(engine, "m")

    with pytest.raises(ValueError, match=r"Output should have 2 items of the same length"):
        engine.run([0] * 10)

    engine = Engine(lambda e, b: ([0, 1, 2], [0, 1, 2, 4]))
    m = MyMetric()
    m.attach(engine, "m")

    with pytest.raises(ValueError, match=r"Output should have 2 items of the same length"):
        engine.run([0] * 10)


class DummyMetric4(Metric):
    _state_dict_all_req_keys = (
        "dnumber",
        "fnumber",
        "tensor",
        "tensor2",
        "metric",
        "metric_dict",
        "metric_list",
        "initially_none",
    )

    @staticmethod
    def gen_expected_state(value):
        expected_state = {
            "dnumber": value + 1,
            "fnumber": value + 2.234,
            "tensor": torch.tensor(value + 2.5),
            "tensor2": torch.tensor(value + 3.5),
            "metric": {
                "_num_correct": torch.tensor(value + 3),
                "_num_examples": value + 4,
            },
            "metric_dict": {
                "m1": {
                    "_num_correct": torch.tensor(value + 5),
                    "_num_examples": value + 6,
                },
                "m2": {
                    "_numerator": torch.tensor([value + 7, value + 8]),
                    "_denominator": torch.tensor([value + 9, value + 10]),
                    "_weight": value,
                    "_updated": True,
                },
                "n": value + 12,
            },
            "metric_list": [
                {
                    "_numerator": torch.tensor([value + 11, value + 12]),
                    "_denominator": torch.tensor([value + 13, value + 14]),
                    "_weight": value,
                    "_updated": True,
                },
                {
                    "_numerator": torch.tensor([value + 15, value + 16]),
                    "_denominator": torch.tensor([value + 17, value + 18]),
                    "_weight": value,
                    "_updated": True,
                },
                value + 234,
            ],
            "initially_none": None,
        }
        return expected_state

    def __init__(self, value):
        super().reset()

        self.expected_state = DummyMetric4.gen_expected_state(value)

        self.dnumber = self.expected_state["dnumber"]
        self.fnumber = self.expected_state["fnumber"]
        self.tensor = self.expected_state["tensor"]
        self.tensor2 = self.expected_state["tensor2"]

        self.metric = Accuracy()
        self.metric._num_correct = self.expected_state["metric"]["_num_correct"]
        self.metric._num_examples = self.expected_state["metric"]["_num_examples"]

        self.metric_dict: Dict[str, Metric] = {
            "m1": Accuracy(),
            "m2": Precision(),
            "n": self.expected_state["metric_dict"]["n"],
        }
        self.metric_dict["m1"]._num_correct = self.expected_state["metric_dict"]["m1"]["_num_correct"]
        self.metric_dict["m1"]._num_examples = self.expected_state["metric_dict"]["m1"]["_num_examples"]
        self.metric_dict["m2"]._numerator = self.expected_state["metric_dict"]["m2"]["_numerator"]
        self.metric_dict["m2"]._denominator = self.expected_state["metric_dict"]["m2"]["_denominator"]
        self.metric_dict["m2"]._weight = self.expected_state["metric_dict"]["m2"]["_weight"]
        self.metric_dict["m2"]._updated = self.expected_state["metric_dict"]["m2"]["_updated"]

        self.metric_list: List[Metric] = [
            Recall(),
            Precision(),
            self.expected_state["metric_list"][2],
        ]
        self.metric_list[0]._numerator = self.expected_state["metric_list"][0]["_numerator"]
        self.metric_list[0]._denominator = self.expected_state["metric_list"][0]["_denominator"]
        self.metric_list[0]._weight = self.expected_state["metric_list"][0]["_weight"]
        self.metric_list[0]._updated = self.expected_state["metric_list"][0]["_updated"]

        self.metric_list[1]._numerator = self.expected_state["metric_list"][1]["_numerator"]
        self.metric_list[1]._denominator = self.expected_state["metric_list"][1]["_denominator"]
        self.metric_list[1]._weight = self.expected_state["metric_list"][1]["_weight"]
        self.metric_list[1]._updated = self.expected_state["metric_list"][1]["_updated"]

        self.initially_none = None

    def reset(self):
        self.dnumber = -1
        self.fnumber = -2.0
        self.tensor = torch.tensor([-3])
        self.tensor2 = 0
        self.metric.reset()
        for m in self.metric_dict.values():
            if isinstance(m, Metric):
                m.reset()
        for m in self.metric_list:
            if isinstance(m, Metric):
                m.reset()
        self.initially_none = None

    def update(self, output):
        pass

    def compute(self):
        pass


def test_wrong_state_dict():
    class WrongMetric(Metric):
        _state_dict_all_req_keys = ("object",)

        def __init__(self, value):
            super().__init__()
            self.object = value

        def reset(self):
            pass

        def update(self, output):
            pass

        def compute(self):
            pass

    metric = WrongMetric(object())
    with pytest.raises(TypeError, match="Found attribute of unsupported type. Currently, supported types include"):
        metric.state_dict()

    delattr(metric, "object")
    with pytest.raises(ValueError, match="Found a value in _state_dict_all_req_keys that is not among"):
        metric.state_dict()


def test_wrong_load_state_dict():
    metric = DummyMetric4(1)

    with pytest.raises(TypeError, match="Argument state_dict should be a dictionary"):
        metric.load_state_dict(123)

    with pytest.raises(ValueError, match="Incorrect state_dict object. Argument state_dict should be a dictionary"):
        metric.load_state_dict({"abc": 123})

    with pytest.raises(ValueError, match="Expected a list of state_dicts of size equal world_size"):
        metric.load_state_dict({Metric._Metric__state_dict_key_per_rank: []})


# @pytest.mark.distributed
# @pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
# @pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
# def test_distrib_state_dict_metric_in_metric(distributed_context_single_node_nccl):
#     class _TestMetric(Metric):
#         _state_dict_all_req_keys = ("metric", )
#         def __init__(self):
#             self.metric = Accuracy()

#         def reset(self):
#             self.metric.reset()

#         def update(self, output):
#             self.metric.update(output)

#         def compute(self):
#             return self.metric.compute()

#     m = _TestMetric()
#     m.update((
#         torch.rand(4, 10),
#         torch.randint(0, 10, size=(4, )),
#     ))

#     rank = idist.get_rank()

#     import time
#     time.sleep(rank * 0.1)

#     print("m: ", m.state_dict())
#     assert False


def test_state_dict():
    metric = DummyMetric4(1)
    state = metric.state_dict()

    assert isinstance(state, dict) and len(state) == 1 and Metric._Metric__state_dict_key_per_rank in state

    rank = idist.get_rank()
    ws = idist.get_world_size()

    list_state_dicts = state[Metric._Metric__state_dict_key_per_rank]
    assert len(list_state_dicts) == ws

    state = list_state_dicts[rank]
    expected_state = metric.expected_state
    assert state.keys() == expected_state.keys()

    # Flatten expected state and output state and compare values
    output_flatten = []
    expected_flatten = []

    def get_func(flatten):
        def wrapper(x, key):
            if isinstance(x, Metric):
                flatten.extend([(key, getattr(x, k)) for k in x._state_dict_all_req_keys])
            else:
                flatten.append((key, x))

        return wrapper

    _tree_map(get_func(expected_flatten), expected_state)
    _tree_map(get_func(output_flatten), state)

    assert len(output_flatten) == len(expected_flatten) and len(expected_flatten) > 0, (
        expected_flatten,
        output_flatten,
    )

    for key_output, key_expected in zip(output_flatten, expected_flatten):
        key1, output = key_output
        key2, expected = key_expected
        assert key1 == key2, (key1, key2)
        if isinstance(output, torch.Tensor):
            assert isinstance(expected, torch.Tensor)
            assert (output == expected).all(), (output, expected)
        else:
            assert output == expected, (output, expected)


def test_load_state_dict():
    metric = DummyMetric4(1)
    state = metric.state_dict()

    metric.reset()
    metric.initially_none = 1
    metric.load_state_dict(state)

    rank = idist.get_rank()
    world_size = idist.get_world_size()
    assert len(state[Metric._Metric__state_dict_key_per_rank]) == world_size
    expected_state = state[Metric._Metric__state_dict_key_per_rank][rank]

    # Flatten expected state and output state and compare values
    output_flatten = []
    expected_flatten = []

    def get_func(flatten):
        def wrapper(x, **kwargs):
            if isinstance(x, Metric):
                flatten.extend([getattr(x, k) for k in x._state_dict_all_req_keys])
            else:
                flatten.append(x)

        return wrapper

    _tree_map(get_func(expected_flatten), expected_state)
    _tree_map(get_func(output_flatten), {key: getattr(metric, key) for key in metric._state_dict_all_req_keys})

    assert len(output_flatten) == len(expected_flatten) and len(expected_flatten) > 0, (
        expected_flatten,
        output_flatten,
    )

    for output, expected in zip(output_flatten, expected_flatten):
        if isinstance(output, torch.Tensor):
            assert isinstance(expected, torch.Tensor)
            assert (output == expected).all(), (output, expected)
        else:
            assert output == expected, (output, expected)


class DummyMetric5(Metric):
    def __init__(self, true_output, output_transform=lambda x: x, skip_unrolling=False):
        super(DummyMetric5, self).__init__(output_transform=output_transform, skip_unrolling=skip_unrolling)
        self.true_output = true_output

    def reset(self):
        pass

    def compute(self):
        pass

    def update(self, output):
        assert output == self.true_output


def test_skip_unrolling():
    # y_pred and y are ouputs recieved from a multi_output model
    a_pred = torch.rand(8, 1)
    b_pred = torch.rand(8, 1)
    y_pred = [a_pred, b_pred]
    a_true = torch.rand(8, 1)
    b_true = torch.rand(8, 1)
    y_true = [a_true, b_true]

    metric = DummyMetric5(true_output=(y_pred, y_true), skip_unrolling=True)
    state = State(output=(y_pred, y_true))
    engine = MagicMock(state=state)
    metric.iteration_completed(engine)


class DummyMetric6(Metric):
    def reset(self):
        pass

    def compute(self):
        pass

    def update(self, output):
        pass

    def __call__(self, value):
        pass


def test_access_to_metric_dunder_attributes():
    metric = DummyMetric6()
    import inspect

    # `inspect.signature` accesses `__signature__` attribute of the metric.
    assert "value" in inspect.signature(metric).parameters.keys()


def test_output_transform_type_check():
    y_pred = torch.tensor([[2.0], [-2.0]])
    y = torch.zeros(2)
    with pytest.raises(TypeError, match="Argument output_transform should be callable"):
        DummyMetric1(true_output=(y_pred, y), output_transform=1)
