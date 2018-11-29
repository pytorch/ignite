from __future__ import division

import torch

from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError
from ignite._utils import to_onehot


class Recall(Metric):
    """
    - | `threshold_function` is only needed for binary cases. Default is `torch.round(x)`. It is used to convert
      | `y_pred` to 0's and 1's.
    - `update` must receive output of the form `(y_pred, y)`.
    - | For binary or multiclass cases, `y_pred` must be in the following shape (batch_size, num_categories, ...) or
      | (batch_size, ...) and `y` must be in the following shape (batch_size, ...).
    For binary or multiclass cases, if `average` is True, returns the unweighted average across all classes.
    Otherwise, returns a tensor with the recall for each class.
    """
    def __init__(self, output_transform=lambda x: x, average=False, threshold_function=None):
        self._average = average
        if threshold_function is not None:
            if callable(threshold_function):
                self._threshold = threshold_function
            else:
                raise ValueError("threshold_function must be a callable function.")
        else:
            self._threshold = torch.round
        self._updated = False
        super(Recall, self).__init__(output_transform=output_transform)

    def reset(self):
        self._actual = None
        self._true_positives = None

    def update(self, output):
        y_pred, y = self._check_shape(output)
        self._check_type((y_pred, y))

        dtype = y_pred.type()

        if self._type == 'binary':
            y_pred = self._threshold(y_pred)
            if not torch.equal(y, y **2):
                raise ValueError("For binary cases, y must contain 0's and 1's only.")
            if not torch.equal(y_pred, y_pred**2):
                raise ValueError("For binary cases, y_pred must contain 0's and 1's only.")
        else:
            y = to_onehot(y.view(-1), num_classes=y_pred.size(1))
            indices = torch.max(y_pred, dim=1)[1].view(-1)
            y_pred = to_onehot(indices, num_classes=y_pred.size(1))
        
        y_pred = y_pred.type(dtype)
        y = y.type(dtype)

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

    def _check_shape(self, output):
        y_pred, y = output

        if not (y.ndimension() == y_pred.ndimension() or y.ndimension() + 1 == y_pred.ndimension()):
            raise ValueError("y must have shape of (batch_size, ...) and y_pred must have "
                             "shape of (batch_size, num_categories, ...) or (batch_size, ...).")

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

        return y_pred, y

    def _check_type(self, output):
        y_pred, y = output

        if y.ndimension() + 1 == y_pred.ndimension():
            update_type = 'multiclass'
        elif y_pred.shape == y.shape and y.ndimension() == 1:
            update_type = 'binary'
        else:
            raise TypeError('Invalid shapes of y (shape={}) and y_pred (shape={}), check documentation'
                            ' for expected shapes of y and y_pred.'.format(y.shape, y_pred.shape))
        if not self._updated:
            self._type = update_type
            self._updated = True
        else:
            if self._type != update_type:
                raise TypeError('update_type has changed from {} to {}.'.format(self._type, update_type))
