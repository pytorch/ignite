from __future__ import division
import warnings

import torch

from ignite.metrics.accuracy import _BaseClassification
from ignite.exceptions import NotComputableError
from ignite._utils import to_onehot


class _BasePrecisionRecall(_BaseClassification):

    def __init__(self, output_transform=lambda x: x, average=False, is_multilabel=False):
        self.eps = 1e-30
        self._average = average
        if is_multilabel:
            if not self._average:
                warnings.warn("average should be True for multilabel cases. {}._average "
                              "updated to True. Average is calculated across samples, "
                              "instead of classes.".format(self.__class__.__name__), UserWarning)
                self._average = True
            self.reset = self._reset_multilabel
            self._collect_positives = self._collect_positives_multilabel
        else:
            self.reset = self._reset_multiclass
            self._collect_positives = self._collect_positives_multiclass
        super(_BasePrecisionRecall, self).__init__(output_transform=output_transform, is_multilabel=is_multilabel)

    def _reset_multilabel(self):
        self._true_positives = None
        self._positives = None

    def _reset_multiclass(self):
        self._true_positives = 0
        self._positives = 0

    def _collect_positives_multilabel(self, true_positives, positives):
        if self._positives is None:
            self._true_positives = true_positives
            self._positives = positives
        else:
            self._true_positives = torch.cat([self._true_positives, true_positives])
            self._positives = torch.cat([self._positives, positives])

    def _collect_positives_multiclass(self, true_positives, positives):
        self._true_positives += true_positives
        self._positives += positives

    def compute(self):
        if not isinstance(self._positives, torch.Tensor):
            raise NotComputableError("{} must have at least one example before"
                                     " it can be computed".format(self.__class__.__name__))

        result = self._true_positives.float() / (self._positives.float() + self.eps)

        if self._average:
            return result.mean().item()
        else:
            return result

    def reset(self):
        pass


class Precision(_BasePrecisionRecall):
    """
    Calculates precision for binary and multiclass data
    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...) or (batch_size, ...)
    - `y` must be in the following shape (batch_size, ...)
    In binary case, when `y` has 0 or 1 values, the elements of `y_pred` must be between 0 and 1. Precision is
    computed over positive class, assumed to be 1.
    Args:
        average (bool, optional): if True, precision is computed as the unweighted average (across all classes
            in multiclass case), otherwise, returns a tensor with the precision (for each class in multiclass case).
    """

    def update(self, output):
        y_pred, y = self._check_shape(output)
        self._check_type((y_pred, y))

        dtype = y_pred.type()

        if self._type == "binary":
            y_pred = y_pred.view(-1)
            y = y.view(-1)
            axis = 0
        elif self._type == "multiclass":
            num_classes = y_pred.size(1)
            y = to_onehot(y.view(-1), num_classes=num_classes)
            indices = torch.max(y_pred, dim=1)[1].view(-1)
            y_pred = to_onehot(indices, num_classes=num_classes)
            axis = 0
        elif self._type == "multilabel":
            if y_pred.ndimension() > 2:
                num_classes = y_pred.size(1)
                y_pred = torch.transpose(y_pred, 1, 0).contiguous().view(num_classes, -1).transpose(1, 0)
                y = torch.transpose(y, 1, 0).contiguous().view(num_classes, -1).transpose(1, 0)
            axis = 1

        y_pred = y_pred.type(dtype)
        y = y.type(dtype)
        correct = y * y_pred
        all_positives = y_pred.sum(dim=axis)

        if correct.sum() == 0:
            true_positives = torch.zeros_like(all_positives)
        else:
            true_positives = correct.sum(dim=axis)

        self._collect_positives(true_positives, all_positives)
