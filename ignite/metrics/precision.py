from __future__ import division
import warnings

import torch

from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError
from ignite._utils import to_onehot


class Precision(Metric):
    """
    Calculates precision.
    - `is_multilabel`, True for multilabel cases and False for binary or multiclass cases.
    - `threshold_function` is only needed for multilabel cases. Default is `torch.round(x)`. It is
       used to convert `y_pred` to 0's and 1's.
    - `update` must receive output of the form `(y_pred, y)`.
    - For binary or multiclass cases, `y_pred` must be in the following shape (batch_size, num_categories, ...)
      or (batch_size, ...) and `y` must be in the following shape (batch_size, ...).
    - For multilabel cases, `y` and `y_pred` must have same shape of (batch_size, num_categories, ...).
    For binary or multiclass cases, if `average` is True, returns the unweighted average across all classes.
    Otherwise, returns a tensor with the precision for each class.
    For multilabel cases `average` is True and returns the unweighted average across all samples.
    """
    def __init__(self, output_transform=lambda x: x, is_multilabel=False, average=False, threshold_function=None):
        self._average = average
        if is_multilabel:
            if threshold_function is None:
                self._threshold = torch.round
            else:
                if callable(threshold_function):
                    self._threshold = threshold_function
                else:
                    raise ValueError("threshold_function must be a callable function.")
            if not self._average:
                warnings.warn('average should be True for multilabel cases. Precision._average updated'
                              ' to True. Average is calculated across samples, instead of classes.', UserWarning)
                self._average = True
            self.update = self._update_multilabel
        else:
            self.update = self._update_multiclass
        super(Precision, self).__init__(output_transform=output_transform)

    def reset(self):
        self._all_positives = None
        self._true_positives = None

    def update(self, output):
        pass

    def compute(self):
        if self._all_positives is None:
            raise NotComputableError('Precision must have at least one example before it can be computed')

        result = self._true_positives / self._all_positives
        result[result != result] = 0.0
        if self._average:
            return result.mean().item()
        else:
            return result

    def _update_multilabel(self, output):
        y_pred, y = output
        dtype = y_pred.type()
        axis = 1

        if not (y.shape == y_pred.shape and y.ndimension() > 1 and y.shape[1] != 1):
            raise ValueError("y and y_pred must have same shape of (batch_size, num_categories, ...).")

        if y_pred.ndimension() == 3:
            # Converts y and y_pred to (-1, num_classes) from N x C x L
            y_pred = y_pred.transpose(2, 1).contiguous().view(-1, y_pred.size(1))
            y = y.transpose(2, 1).contiguous().view(-1, y_pred.size(1))

        elif y_pred.ndimension() == 4:
            # Converts y and y_pred to (-1, num_classes) from N x C x H x W
            y_pred = y_pred.permute(0, 2, 3, 1).contiguous().view(-1, y_pred.size(1))
            y = y.permute(0, 2, 3, 1).contiguous().view(-1, y_pred.size(1))

        else:
            pass

        y_pred = self._threshold(y_pred)

        if not torch.equal(y_pred, y_pred ** 2):
            raise ValueError("threshold_function must convert y_pred to 0's and 1's only.")

        if not torch.equal(y, y ** 2):
            raise ValueError("For binary and multilabel cases, y must contain 0's and 1's only.")

        y_pred = y_pred.type(dtype)
        y = y.type(dtype)

        correct = y * y_pred
        all_positives = y_pred.sum(dim=axis)

        if correct.sum() == 0:
            true_positives = torch.zeros_like(all_positives)
        else:
            true_positives = correct.sum(dim=axis)
        if self._all_positives is None:
            self._all_positives = all_positives
            self._true_positives = true_positives
        else:
            self._all_positives = torch.cat([self._all_positives, all_positives])
            self._true_positives = torch.cat([self._true_positives, true_positives])

    def _update_multiclass(self, output):
        y_pred, y = output
        dtype = y_pred.type()
        axis = 0

        if not (y.ndimension() == y_pred.ndimension() or y.ndimension() + 1 == y_pred.ndimension()):
            raise ValueError("y must have shape of (batch_size, ...) and y_pred "
                             "must have shape of (batch_size, num_classes, ...) or (batch_size, ...).")

        if y.ndimension() == 1 or y.shape[1] == 1:
            # Binary Case, flattens y and num_classes is equal to 1.
            y = y.squeeze(dim=1).view(-1) if (y.ndimension() > 1) else y.view(-1)

        if y_pred.ndimension() == 1 or y_pred.shape[1] == 1:
            # Binary Case, flattens y and num_classes is equal to 1.
            y_pred = y_pred.squeeze(dim=1).view(-1) if (y_pred.ndimension() > 1) else y_pred.view(-1)

        y_shape = y.shape
        y_pred_shape = y_pred.shape

        if y.ndimension() + 1 == y_pred.ndimension():
            y_pred_shape = (y_pred_shape[0],) + y_pred_shape[2:]

        if not (y_shape == y_pred_shape):
            raise ValueError("y and y_pred must have compatible shapes.")

        if y_pred.ndimension() == y.ndimension():
            # Binary Case
            y_pred = y_pred.unsqueeze(dim=1)
            y_pred = torch.cat([1.0 - y_pred, y_pred], dim=1)

        y = to_onehot(y.view(-1), num_classes=y_pred.size(1))
        indices = torch.max(y_pred, dim=1)[1].view(-1)
        y_pred = to_onehot(indices, num_classes=y_pred.size(1))

        y_pred = y_pred.type(dtype)
        y = y.type(dtype)

        correct = y * y_pred
        all_positives = y_pred.sum(dim=axis)

        if correct.sum() == 0:
            true_positives = torch.zeros_like(all_positives)
        else:
            true_positives = correct.sum(dim=axis)
        if self._all_positives is None:
            self._all_positives = all_positives
            self._true_positives = true_positives
        else:
            self._all_positives += all_positives
            self._true_positives += true_positives
