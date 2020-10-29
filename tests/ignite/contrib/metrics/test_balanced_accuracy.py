import numpy as np
import pytest
import torch
from sklearn.metrics import balanced_accuracy_score

from ignite.contrib.metrics.balanced_accuracy import BalancedAccuracy
from ignite.engine import Engine


def test_balanced_accuracy():
    np.random.seed(1)
    size = 100
    np_y_pred = np.random.randint(2, size=size)
    np_y = np.zeros(size, dtype=np.long)
    np_y[size // 2 :] = 1
    np.random.shuffle(np_y)

    np_balanced_accuracy = balanced_accuracy_score(np_y, np_y_pred)

    balanced_accuracy_metric = BalancedAccuracy()
    y_pred = torch.from_numpy(np_y_pred)
    y = torch.from_numpy(np_y)

    balanced_accuracy_metric.update((y_pred, y))
    balanced_accuracy = balanced_accuracy_metric.compute()

    assert balanced_accuracy == np_balanced_accuracy


def test_balanced_accuracy_with_output_transform():
    np.random.seed(1)
    size = 100
    np_y_pred = np.random.randint(2, size=size)
    np_y = np.zeros(size, dtype=np.long)
    np_y[size // 2 :] = 1
    np.random.shuffle(np_y)

    np_balanced_accuracy = balanced_accuracy_score(np_y, np_y_pred)

    batch_size = 10

    def update_fn(engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        y_true_batch = np_y[idx : idx + batch_size]
        y_pred_batch = np_y_pred[idx : idx + batch_size]
        return idx, torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

    engine = Engine(update_fn)

    balanced_accuracy_metric = BalancedAccuracy(output_transform=lambda x: (x[1], x[2]))
    balanced_accuracy_metric.attach(engine, "balanced_accuracy")

    data = list(range(size // batch_size))
    balanced_accuracy = engine.run(data, max_epochs=1).metrics["balanced_accuracy"]

    assert balanced_accuracy == np_balanced_accuracy
