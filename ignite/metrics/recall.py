from __future__ import division
import warnings

import torch

from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError
from ignite._utils import to_onehot


class Recall(Metric):
    """
    Calculates recall.
    - `update` must receive output of the form `(y_pred, y)`.
    - `y` and `y_pred` must have same shape of (batch_size, num_classes, ...) for multilabel.

    If `average` is True, returns the unweighted average across all classes.
    Otherwise, returns a tensor with the recall for each class.

    For multilabel cases `average` is True and returns the unweighted average across all samples.
    """

    def __init__(self, average=False, threshold_function=lambda x: torch.round(x), output_transform=lambda x: x):
        super(Recall, self).__init__(output_transform)
        self._average = average
        self._threshold = threshold_function
        self._is_multi = False

    def reset(self):
        self._actual = None
        self._true_positives = None

    def update(self, output):
        y_pred, y = output
        dtype = y_pred.type()
        axis = 0

        if not (y.ndimension() == y_pred.ndimension() or y.ndimension() + 1 == y_pred.ndimension()):
            raise ValueError("y must have shape of (batch_size, ...) and y_pred "
                             "must have shape of (batch_size, num_classes, ...) or (batch_size, ...).")

        if y.ndimension() == 1 or y.shape[1] == 1:
            # Binary Case, flattens y
            y = y.squeeze(dim=1).view(-1) if (y.ndimension() > 1) else y.view(-1)

        if y_pred.ndimension() == 1 or y_pred.shape[1] == 1:
            # Binary Case, flattens y_pred
            y_pred = y_pred.squeeze(dim=1).view(-1) if (y_pred.ndimension() > 1) else y_pred.view(-1)

        y_shape = y.shape
        y_pred_shape = y_pred.shape

        if (y_shape == y_pred_shape) and y.ndimension() > 1:
            # Multilabel Case, as y is flatted in binary case. average has to be True and calculated across samples.
            self._is_multi = True
            if not(self._average):
                warnings.warn('average should be True for multilabel cases.', UserWarning)
                self._average = True
            axis = 1

            if y_pred.ndimension() == 3:
                # Converts y and y_pred to (-1, num_classes) from N x C x L
                y_pred = y_pred.transpose(2, 1).contiguous().view(-1, y_pred.size(1))
                y = y.transpose(2, 1).contiguous().view(-1, y_pred.size(1))

            if y_pred.ndimension() == 4:
                # Converts y and y_pred to (-1, num_classes) from N x C x H x W
                y_pred = y_pred.permute(0, 2, 3, 1).contiguous().view(-1, y_pred.size(1))
                y = y.permute(0, 2, 3, 1).contiguous().view(-1, y_pred.size(1))

        if y.ndimension() + 1 == y_pred.ndimension():
            y_pred_shape = (y_pred_shape[0],) + y_pred_shape[2:]

        if not (y_shape == y_pred_shape):
            raise ValueError("y and y_pred must have compatible shapes.")

        if y_pred.ndimension() == y.ndimension():
            y_pred = self._threshold(y_pred)

            if not torch.equal(y_pred, y_pred ** 2):
                raise ValueError("threshold_function must convert y_pred to 0's and 1's only.")

            if not torch.equal(y, y ** 2):
                raise ValueError("For binary and multilabel cases, y must contain 0's and 1's only.")

        else:
            y = to_onehot(y.view(-1), num_classes=y_pred.size(1))
            indices = torch.max(y_pred, dim=1)[1].view(-1)
            y_pred = to_onehot(indices, num_classes=y_pred.size(1))

        y_pred = y_pred.type(dtype)
        y = y.type(dtype)

        correct = y * y_pred
        actual = y.sum(dim=axis)

        if correct.sum() == 0:
            true_positives = torch.zeros_like(actual)
        else:
            true_positives = correct.sum(dim=axis)
        if self._actual is None:
            self._actual = actual
            self._true_positives = true_positives
        else:
            # In multilabel case, concatenate all positive and true positive tensors to calculate precision per sample
            # In binary or multiclass case, all positive and true positives are summed over classes
            self._actual = torch.cat([self._actual, actual]) if self._is_multi else self._actual + actual
            self._true_positives = torch.cat([self._true_positives, true_positives]) \
                if self._is_multi else self._true_positives + true_positives

    def compute(self):
        if self._actual is None:
            raise NotComputableError('Recall must have at least one example before it can be computed')
        result = self._true_positives / self._actual
        result[result != result] = 0.0
        if self._average:
            return result.mean().item()
        else:
            return result
