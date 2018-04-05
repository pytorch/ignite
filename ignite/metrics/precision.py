from __future__ import division

import torch

from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError
from ignite._utils import to_onehot


class Precision(Metric):
    """
    Calculates precision.

    `update` must receive output of the form (y_pred, y).
    """
    def reset(self):
        self._all_positives = None
        self._true_positives = None

    def update(self, output):
        y_pred, y = output
        num_classes = y_pred.size(1)
        indices = torch.max(y_pred, 1)[1]
        correct = torch.eq(indices, y)
        pred_onehot = to_onehot(indices, num_classes)
        all_positives = pred_onehot.sum(dim=0)
        if correct.sum() == 0:
            true_positives = torch.zeros(num_classes)
        else:
            correct_onehot = to_onehot(indices[correct], num_classes)
            true_positives = correct_onehot.sum(dim=0)
        if self._all_positives is None:
            self._all_positives = all_positives
            self._true_positives = true_positives
        else:
            self._all_positives += all_positives
            self._true_positives += true_positives

    def compute(self):
        if self._all_positives is None:
            raise NotComputableError('Precision must have at least one example before it can be computed')
        return self._true_positives / self._all_positives
