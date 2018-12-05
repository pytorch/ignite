from __future__ import division
import warnings

import torch

from ignite.metrics.accuracy import _BaseClassification
from ignite.exceptions import NotComputableError
from ignite._utils import to_onehot


class _BasePrecisionRecall(_BaseClassification):
    def __init__(self, output_transform=lambda x: x, average=False):
        self._average = average
        super(_BasePrecisionRecall, self).__init__(output_transform=output_transform)

    def _check_type(self, output):
        y_pred, y = output

        if y.ndimension() + 1 == y_pred.ndimension():
            if y_pred.shape[1] == 2:
                update_type = 'binary_multiclass'
                if self._type is None:
                    warnings.warn('Given num_classes=2, only {}'
                                  ' for positive class, 1, will be computed.'.format(self.__class__.__name__))
            else:
                update_type = 'multiclass'
        elif y.ndimension() == y_pred.ndimension():
            update_type = 'binary'
            if not torch.equal(y, y**2):
                raise ValueError('For binary cases, y must be comprised of 0\'s and 1\'s.')
        else:
            raise TypeError('Invalid shapes of y (shape={}) and y_pred (shape={}), check documentation'
                            ' for expected shapes of y and y_pred.'.format(y.shape, y_pred.shape))
        if self._type is None:
            self._type = update_type
        else:
            if self._type != update_type:
                raise TypeError('update_type has changed from {} to {}.'.format(self._type, update_type))

    def reset(self):
        self._true_positives = None
        self._positives = None

    def compute(self):
        if self._positives is None:
            raise NotComputableError('{} must have at least one example before'
                                     ' it can be computed'.format(self.__class__.__name__))

        result = self._true_positives / self._positives
        result[result != result] = 0.0
        if self._average:
            if 'binary' in self._type:
                return result[1].item()
            else:
                return result.mean().item()
        else:
            return result


class Precision(_BasePrecisionRecall):
    """
    Calculates precision.
    - `update` must receive output of the form `(y_pred, y)`.
    - | For binary or multiclass cases, `y_pred` must be in the following shape (batch_size, num_categories, ...) or
      | (batch_size, ...) and `y` must be in the following shape (batch_size, ...).
    - In binary cases if a specific threshold probability is required, use output_transform.
    For binary cases, if `average` is True, returns precision of positive class, assumed to be 1.
    For binary or multiclass cases, if `average` is True, returns the unweighted average across all classes.
    Otherwise, returns a tensor with the precision for each class.
    """

    def update(self, output):
        y_pred, y = self._check_shape(output)
        self._check_type((y_pred, y))

        dtype = y_pred.type()

        if y_pred.ndimension() == y.ndimension():
            y_pred = y_pred.unsqueeze(dim=1)
            y_pred = torch.cat([1.0 - y_pred, y_pred], dim=1)

        num_classes = y_pred.size(1)
        y = to_onehot(y.view(-1), num_classes=num_classes)
        indices = torch.max(y_pred, dim=1)[1].view(-1)
        y_pred = to_onehot(indices, num_classes=num_classes)

        y_pred = y_pred.type(dtype)
        y = y.type(dtype)

        correct = y * y_pred
        all_positives = y_pred.sum(dim=0)

        if correct.sum() == 0:
            true_positives = torch.zeros_like(all_positives)
        else:
            true_positives = correct.sum(dim=0)
        if self._true_positives is None:
            self._true_positives = true_positives
            self._positives = all_positives
        else:
            self._true_positives += true_positives
            self._positives += all_positives


class Recall(_BasePrecisionRecall):
    """
    Calculates recall.
    - `update` must receive output of the form `(y_pred, y)`.
    - | For binary or multiclass cases, `y_pred` must be in the following shape (batch_size, num_categories, ...) or
      | (batch_size, ...) and `y` must be in the following shape (batch_size, ...).
    - In binary cases if a specific threshold probability is required, use output_transform.
    For binary cases, if `average` is True, returns precision of positive class, assumed to be 1.
    For multiclass cases, if `average` is True, returns the unweighted average across all classes.
    Otherwise, returns a tensor with the recall for each class.
    """

    def update(self, output):
        y_pred, y = self._check_shape(output)
        self._check_type((y_pred, y))

        dtype = y_pred.type()

        if y_pred.ndimension() == y.ndimension():
            y_pred = y_pred.unsqueeze(dim=1)
            y_pred = torch.cat([1.0 - y_pred, y_pred], dim=1)

        num_classes = y_pred.size(1)
        y = to_onehot(y.view(-1), num_classes=num_classes)
        indices = torch.max(y_pred, dim=1)[1].view(-1)
        y_pred = to_onehot(indices, num_classes=num_classes)

        y_pred = y_pred.type(dtype)
        y = y.type(dtype)

        correct = y * y_pred
        actual_positives = y.sum(dim=0)

        if correct.sum() == 0:
            true_positives = torch.zeros_like(actual_positives)
        else:
            true_positives = correct.sum(dim=0)
        if self._true_positives is None:
            self._true_positives = true_positives
            self._positives = actual_positives
        else:
            self._true_positives += true_positives
            self._positives += actual_positives
