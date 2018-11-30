from __future__ import division

import torch

from ignite.metrics.accuracy import _BaseClassification
from ignite.exceptions import NotComputableError
from ignite._utils import to_onehot


class _BasePrecisionRecallSupport(_BaseClassification):
    def __init__(self, output_transform=lambda x: x, average=False, threshold_function=None):
        self._average = average
        super(_BasePrecisionRecallSupport, self).__init__(output_transform=output_transform,
                                                          threshold_function=threshold_function)

    def reset(self):
        self._true_positives = None
        self._positives = None

    def _calculate_correct(self, output):
        y_pred, y = self._check_shape(output)
        self._check_type((y_pred, y))

        dtype = y_pred.type()

        if self._type == 'binary':
            y_pred = y_pred.view(-1)
            y = y.view(-1)
            y_pred = self._threshold(y_pred)
            if not torch.equal(y, y ** 2):
                raise ValueError("For binary cases, y must contain 0's and 1's only.")
            if not torch.equal(y_pred, y_pred ** 2):
                raise ValueError("For binary cases, y_pred must contain 0's and 1's only.")
        else:
            y = to_onehot(y.view(-1), num_classes=y_pred.size(1))
            indices = torch.max(y_pred, dim=1)[1].view(-1)
            y_pred = to_onehot(indices, num_classes=y_pred.size(1))

        y_pred = y_pred.type(dtype)
        y = y.type(dtype)
        correct = y * y_pred

        return correct, y_pred, y

    def _sum_positives(self, correct, positives):
        if correct.sum() == 0:
            true_positives = torch.zeros_like(positives)
        else:
            true_positives = correct.sum(dim=0)
        if self._true_positives is None:
            self._true_positives = true_positives
            self._positives = positives
        else:
            self._true_positives += true_positives
            self._positives += positives

    def compute(self):
        if self._positives is None:
            raise NotComputableError('{} must have at least one example before'
                                     ' it can be computed'.format(self.__class__.__name__))

        result = self._true_positives / self._positives
        result[result != result] = 0.0
        if self._average:
            return result.mean().item()
        else:
            return result


class Precision(_BasePrecisionRecallSupport):
    """
    Calculates precision.
    - | `threshold_function` is only needed for binary cases. Default is `torch.round(x)`. It is used to convert
      | `y_pred` to 0's and 1's.
    - `update` must receive output of the form `(y_pred, y)`.
    - | For binary or multiclass cases, `y_pred` must be in the following shape (batch_size, num_categories, ...) or
      | (batch_size, ...) and `y` must be in the following shape (batch_size, ...).
    For binary or multiclass cases, if `average` is True, returns the unweighted average across all classes.
    Otherwise, returns a tensor with the precision for each class.
    """
    def __init__(self, output_transform=lambda x: x, average=False, threshold_function=None):
        self._precision_vs_recall = True
        super(Precision, self).__init__(output_transform=output_transform, average=average,
                                        threshold_function=threshold_function)

    def update(self, output):
        correct, y_pred, _ = self._calculate_correct(output)
        all_positives = y_pred.sum(dim=0)
        self._sum_positives(correct, all_positives)
