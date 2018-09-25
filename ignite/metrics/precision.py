from __future__ import division

import torch

from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError
from ignite._utils import to_onehot


class Precision(Metric):
    """
    Calculates precision.

    - `update` must receive output of the form `(y_pred, y)`.

    If `average` is True, returns the unweighted average across all classes.
    Otherwise, returns a tensor with the precision for each class.
    """
    def __init__(self, average=False, output_transform=lambda x: x):
        super(Precision, self).__init__(output_transform)
        self._average = average

    def reset(self):
        self._all_positives = None
        self._true_positives = None

    def update(self, output):
        y_pred, y = output

        if y.ndimension() == 2 and y.shape[1] == 1:
            y = y.squeeze(dim=-1)

        if y_pred.ndimension() == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(dim=-1)

        y_dim = list(y.shape)[1:]
        pred_dim = list(y_pred.shape)[:-len(y_dim)]

        if len(pred_dim) or y_pred.ndimension() == 1:
            num_classes = 2
            indices = torch.round(y_pred).type(y.type())
            correct = torch.eq(indices, y).view(-1)
        else:
            num_classes = y_pred.size(1)
            indices = torch.max(y_pred, 1)[1]
            correct = torch.eq(indices, y)

        pred_onehot = to_onehot(indices, num_classes)
        all_positives = pred_onehot.sum(dim=0)
        if correct.sum() == 0:
            true_positives = torch.zeros_like(all_positives)
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
        result = self._true_positives / self._all_positives
        result[result != result] = 0.0
        if self._average:
            return result.mean().item()
        else:
            return result
