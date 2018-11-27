from __future__ import division

import torch

from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError


class Accuracy(Metric):
    """
    Calculates the accuracy.
    - `is_multilabel`, True for multilabel cases and False for binary or multiclass cases.
    - | `threshold_function` is only needed for multilabel cases. Default is `torch.round(x)`. It is used to convert
      | `y_pred` to 0's and 1's.
    - `update` must receive output of the form `(y_pred, y)`.
    - | For binary or multiclass cases, `y_pred` must be in the following shape (batch_size, num_categories, ...) or
      | (batch_size, ...) and `y` must be in the following shape (batch_size, ...).
    - For multilabel cases, `y` and `y_pred` must have same shape of (batch_size, num_categories, ...).
    """
    def __init__(self, output_transform=lambda x: x, is_multilabel=False, threshold_function=None):
        if is_multilabel:
            if threshold_function is None:
                self._threshold = torch.round
            else:
                if callable(threshold_function):
                    self._threshold = threshold_function
                else:
                    raise ValueError("threshold_function must be a callable function.")
            self.update = self._update_multilabel
        else:
            self.update = self._update_multiclass
        super(Accuracy, self).__init__(output_transform)

    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    def update(self, output):
        pass

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Accuracy must have at least one example before it can be computed')
        return self._num_correct / self._num_examples

    def _update_multilabel(self, output):
        y_pred, y = output

        if not (y.shape == y_pred.shape and y.ndimension() > 1 and y.shape[1] != 1):
            raise ValueError("y and y_pred must have same shape of (batch_size, num_categories, ...).")

        if y_pred.ndimension() > 2:
            num_classes = y_pred.size(1)
            y_pred = torch.transpose(y_pred, 1, 0).contiguous().view(num_classes, -1).transpose(1, 0)
            y = torch.transpose(y, 1, 0).contiguous().view(num_classes, -1).transpose(1, 0)

        y_pred = self._threshold(y_pred).type(y.type())

        if not torch.equal(y, y ** 2):
            raise ValueError("For binary and multilabel cases, y must contain 0's and 1's only.")

        if not torch.equal(y_pred, y_pred ** 2):
            raise ValueError("threshold_function must convert y_pred to 0's and 1's only.")

        correct = [torch.equal(true, pred) for true, pred in zip(y, y_pred)]

        self._num_correct += sum(correct)
        self._num_examples += len(correct)

    def _update_multiclass(self, output):
        y_pred, y = output

        if not (y.ndimension() == y_pred.ndimension() or y.ndimension() + 1 == y_pred.ndimension()):
            raise ValueError("y must have shape of (batch_size, ...) and y_pred must have "
                             "shape of (batch_size, num_categories, ...) or (batch_size, ...).")

        if y.ndimension() > 1 and y.shape[1] == 1:
            y = y.squeeze(dim=1)

        if y_pred.ndimension() > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(dim=1)

        y_shape = y.shape
        y_pred_shape = y_pred.shape

        if y.ndimension() + 1 == y_pred.ndimension():
            y_pred_shape = (y_pred_shape[0], ) + y_pred_shape[2:]

        if not (y_shape == y_pred_shape):
            raise ValueError("y and y_pred must have compatible shapes.")

        if y_pred.ndimension() == y.ndimension():
            # Maps Binary Case to Categorical Case with 2 classes
            y_pred = y_pred.unsqueeze(dim=1)
            y_pred = torch.cat([1.0 - y_pred, y_pred], dim=1)

        indices = torch.max(y_pred, dim=1)[1]
        correct = torch.eq(indices, y).view(-1)

        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0]
