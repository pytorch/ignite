from ignite.engine import Engine
from ignite.metrics import Metric, MetricsLambda, Precision, Recall
from pytest import approx
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from numpy.testing import assert_allclose
import torch


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
    m0_plus_m1 = MetricsLambda(lambda x, y: x + y, m0, m1)
    m2_plus_2 = MetricsLambda(lambda x, y: x + y, m2, 2)
    m0_plus_m1.attach(engine, 'm0_plus_m1')
    m2_plus_2.attach(engine, 'm2_plus_2')

    engine.run([[1, 10, 100]])
    assert engine.state.metrics['m0_plus_m1'] == 11
    assert engine.state.metrics['m2_plus_2'] == 102
    engine.run([[2, 20, 200]])
    assert engine.state.metrics['m0_plus_m1'] == 22
    assert engine.state.metrics['m2_plus_2'] == 202


def test_metrics_lambda_reset():
    m0 = ListGatherMetric(0)
    m1 = ListGatherMetric(1)
    m2 = ListGatherMetric(2)
    m0.update([1, 10, 100])
    m1.update([1, 10, 100])
    m2.update([1, 10, 100])

    m = MetricsLambda(lambda x, y, z, t: 1, m0, m1, m2, 0)

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


def test_integration():
    np.random.seed(1)

    n_iters = 10
    batch_size = 10
    n_classes = 10

    y_true = np.arange(0, n_iters * batch_size) % n_classes
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

    def Fbeta(r, p, beta):
        return torch.mean((1 + beta ** 2) * p * r / (beta ** 2 * p + r)).item()

    F1 = MetricsLambda(Fbeta, recall, precision, 1)

    precision.attach(evaluator, "precision")
    recall.attach(evaluator, "recall")
    F1.attach(evaluator, "f1")

    data = list(range(n_iters))
    state = evaluator.run(data, max_epochs=1)

    precision_true = precision_score(y_true, np.argmax(y_pred, axis=-1), average=None)
    recall_true = recall_score(y_true, np.argmax(y_pred, axis=-1), average=None)
    f1_true = f1_score(y_true, np.argmax(y_pred, axis=-1), average='macro')

    precision = state.metrics['precision'].numpy()
    recall = state.metrics['recall'].numpy()

    assert precision_true == approx(precision), "{} vs {}".format(precision_true, precision)
    assert recall_true == approx(recall), "{} vs {}".format(recall_true, recall)
    assert f1_true == approx(state.metrics['f1']), "{} vs {}".format(f1_true, state.metrics['f1'])


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
    state = evaluator.run(d, max_epochs=1)

    assert set(state.metrics.keys()) == set(["precision", "recall", "f1"])
