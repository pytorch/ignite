import torch
from ignite.engine import Engine
from ignite.metrics import Metric, MetricsLambda, Precision, Recall
import ignite.metrics.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from pytest import approx


def test_helper_ops():

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
        state = validator.run(d, max_epochs=1)

        assert set(state.metrics.keys()) == set([metric_name, ])
        np_y_pred = np.argmax(y_pred.numpy(), axis=-1).ravel()
        np_y = y.numpy().ravel()
        assert state.metrics[metric_name] == approx(compute_true_value_fn(np_y_pred, np_y))

    precision_1 = Precision(average=False)
    precision_2 = Precision(average=False)
    norm_summed_precision1 = F.norm(precision_1 + precision_2, p=10).item()
    norm_summed_precision2 = (precision_1 + precision_2).norm(p=10).item()

    def compute_true_norm_summed_precision(y_pred, y):
        p1 = precision_score(y, y_pred, average=None)
        p2 = precision_score(y, y_pred, average=None)
        return np.linalg.norm(p1 + p2, ord=10)

    _test(norm_summed_precision1, "mean summed precision", compute_true_value_fn=compute_true_norm_summed_precision)
    _test(norm_summed_precision2, "mean summed precision", compute_true_value_fn=compute_true_norm_summed_precision)

    precision = Precision(average=False)
    recall = Recall(average=False)
    sum_precision_recall1 = F.sum(precision + recall).item()
    sum_precision_recall2 = (precision + recall).sum().item()

    def compute_sum_precision_recall(y_pred, y):
        p = precision_score(y, y_pred, average=None)
        r = recall_score(y, y_pred, average=None)
        return np.sum(p + r)

    _test(sum_precision_recall1, "sum precision recall", compute_true_value_fn=compute_sum_precision_recall)
    _test(sum_precision_recall2, "sum precision recall", compute_true_value_fn=compute_sum_precision_recall)

    precision = Precision(average=False)
    recall = Recall(average=False)
    f1_1 = (precision * recall * 2 / (precision + recall + 1e-20)).mean().item()
    f1_2 = F.mean(precision * recall * 2 / (precision + recall + 1e-20)).item()

    def compute_f1(y_pred, y):
        f1 = f1_score(y, y_pred, average='macro')
        return f1

    _test(f1_1, "f1", compute_true_value_fn=compute_f1)
    _test(f1_2, "f1", compute_true_value_fn=compute_f1)
