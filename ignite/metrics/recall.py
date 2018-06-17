from __future__ import division

import torch

from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError
from ignite._utils import to_onehot


class Recall(Metric):
    """
    Calculates recall.

    - `update` must receive output of the form `(y_pred, y)`.

    If `average` is True, returns the unweighted average across all classes.
    Otherwise, returns a tensor with the recall for each class.
    """
    def __init__(self, average=False, output_transform=lambda x: x):
        super(Recall, self).__init__(output_transform)
        self._average = average

    def reset(self):
        self._actual = None
        self._true_positives = None

    def update(self, output):
        y_pred, y = output
        num_classes = y_pred.size(1)
        indices = torch.max(y_pred, 1)[1]
        correct = torch.eq(indices, y)
        actual_onehot = to_onehot(y, num_classes)
        actual = actual_onehot.sum(dim=0)
        if correct.sum() == 0:
            true_positives = torch.zeros_like(actual)
        else:
            correct_onehot = to_onehot(indices[correct], num_classes)
            true_positives = correct_onehot.sum(dim=0)
        if self._actual is None:
            self._actual = actual
            self._true_positives = true_positives
        else:
            self._actual += actual
            self._true_positives += true_positives

    def compute(self):
        if self._actual is None:
            raise NotComputableError('Recall must have at least one example before it can be computed')
        result = self._true_positives / self._actual
        result[result != result] = 0.0
        if self._average:
            return result.mean().item()
        else:
            return result
