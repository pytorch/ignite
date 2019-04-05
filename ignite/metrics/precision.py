from __future__ import division

import torch

from ignite.metrics.accuracy import _BaseClassification
from ignite.exceptions import NotComputableError
from ignite.utils import to_onehot


class _BasePrecisionRecall(_BaseClassification):

    def __init__(self, output_transform=lambda x: x, average=False, is_multilabel=False):
        self._average = average
        self._true_positives = None
        self._positives = None
        self.eps = 1e-20
        super(_BasePrecisionRecall, self).__init__(output_transform=output_transform, is_multilabel=is_multilabel)

    def reset(self):
        self._true_positives = torch.DoubleTensor(0) if (self._is_multilabel and not self._average) else 0
        self._positives = torch.DoubleTensor(0) if (self._is_multilabel and not self._average) else 0
        super(_BasePrecisionRecall, self).reset()

    def compute(self):
        if not (isinstance(self._positives, torch.Tensor) or self._positives > 0):
            raise NotComputableError("{} must have at least one example before"
                                     " it can be computed.".format(self.__class__.__name__))

        result = self._true_positives / (self._positives + self.eps)

        if self._average:
            return result.mean().item()
        else:
            return result


class Precision(_BasePrecisionRecall):
    """
    Calculates precision for binary and multiclass data.

    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...) or (batch_size, ...).
    - `y` must be in the following shape (batch_size, ...).

    In binary and multilabel cases, the elements of `y` and `y_pred` should have 0 or 1 values. Thresholding of
    predictions can be done as below:

    .. code-block:: python

        def thresholded_output_transform(output):
            y_pred, y = output
            y_pred = torch.round(y_pred)
            return y_pred, y

        precision = Precision(output_transform=thresholded_output_transform)

    In multilabel cases, average parameter should be True. However, if user would like to compute F1 metric, for
    example, average parameter should be False. This can be done as shown below:

    .. code-block:: python

        precision = Precision(average=False, is_multilabel=True)
        recall = Recall(average=False, is_multilabel=True)
        F1 = precision * recall * 2 / (precision + recall + 1e-20)
        F1 = MetricsLambda(lambda t: torch.mean(t).item(), F1)

    .. warning::

        In multilabel cases, if average is False, current implementation stores all input data (output and target) in
        as tensors before computing a metric. This can potentially lead to a memory error if the input data is larger
        than available RAM.

    Args:
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        average (bool, optional): if True, precision is computed as the unweighted average (across all classes
            in multiclass case), otherwise, returns a tensor with the precision (for each class in multiclass case).
        is_multilabel (bool, optional) flag to use in multilabel case. By default, value is False. If True, average
            parameter should be True and the average is computed across samples, instead of classes.
    """

    def __init__(self, output_transform=lambda x: x, average=False, is_multilabel=False):
        super(Precision, self).__init__(output_transform=output_transform,
                                        average=average, is_multilabel=is_multilabel)

    def update(self, output):
        y_pred, y = self._check_shape(output)
        self._check_type((y_pred, y))

        if self._type == "binary":
            y_pred = y_pred.view(-1)
            y = y.view(-1)
        elif self._type == "multiclass":
            num_classes = y_pred.size(1)
            if y.max() + 1 > num_classes:
                raise ValueError("y_pred contains less classes than y. Number of predicted classes is {}"
                                 " and element in y has invalid class = {}.".format(num_classes, y.max().item() + 1))
            y = to_onehot(y.view(-1), num_classes=num_classes)
            indices = torch.argmax(y_pred, dim=1).view(-1)
            y_pred = to_onehot(indices, num_classes=num_classes)
        elif self._type == "multilabel":
            # if y, y_pred shape is (N, C, ...) -> (C, N x ...)
            num_classes = y_pred.size(1)
            y_pred = torch.transpose(y_pred, 1, 0).reshape(num_classes, -1)
            y = torch.transpose(y, 1, 0).reshape(num_classes, -1)

        y = y.type_as(y_pred)
        correct = y * y_pred
        all_positives = y_pred.sum(dim=0).type(torch.DoubleTensor)  # Convert from int cuda/cpu to double cpu

        if correct.sum() == 0:
            true_positives = torch.zeros_like(all_positives)
        else:
            true_positives = correct.sum(dim=0)
        # Convert from int cuda/cpu to double cpu
        # We need double precision for the division true_positives / all_positives
        true_positives = true_positives.type(torch.DoubleTensor)

        if self._type == "multilabel":
            if not self._average:
                self._true_positives = torch.cat([self._true_positives, true_positives], dim=0)
                self._positives = torch.cat([self._positives, all_positives], dim=0)
            else:
                self._true_positives += torch.sum(true_positives / (all_positives + self.eps))
                self._positives += len(all_positives)
        else:
            self._true_positives += true_positives
            self._positives += all_positives
