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

    def __init__(self, threshold=0.5, average=False, output_transform=lambda x: x):
        super(Recall, self).__init__(output_transform)
        self._threshold = threshold
        self._average = average

    def reset(self):
        self._actual = None
        self._true_positives = None

    def update(self, output):
        y_pred, y = output

        dtype = y_pred.type()

        if y.ndimension() > 1 and y.shape[1] == 1:
            y = y.squeeze(dim=1)

        if y_pred.ndimension() > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(dim=1)

        y_shape = y.shape
        y_pred_shape = y_pred.shape

        if y.ndimension() + 1 == y_pred.ndimension():
            y_pred_shape = (y_pred_shape[0],) + y_pred_shape[2:]

        assert y_shape == y_pred_shape

        num_classes = y_pred.size(1) if y_pred.ndimension() == y.ndimension() + 1 else 1
        y_pred = y_pred >= self._threshold

        if y_pred.ndimension() == y.ndimension() + 1:
            if y.ndimension() == 1:
                y = y.unsqueeze(1)
            # Works for y.ndimension() equals 1, 2
            y = torch.cat([to_onehot(t, num_classes) for t in y], dim=0) if num_classes > 1 else y

        # Named Entity Recognition - N x C x L
        if y.ndimension() == 3:
            y = y.transpose(2, 1).contiguous().view(-1, y.size(1))

        if y_pred.ndimension() == 3:
            y_pred = y_pred.transpose(2, 1).contiguous().view(-1, y_pred.size(1))

        y = y.type(dtype)
        y_pred = y_pred.type(dtype)

        # y and y_pred are in shape of [-1, C], element-wised product outputs correct
        correct = y * y_pred
        actual = y.sum(dim=0)

        if correct.sum() == 0:
            true_positives = torch.zeros_like(actual)
        else:
            true_positives = correct.sum(dim=0)
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
