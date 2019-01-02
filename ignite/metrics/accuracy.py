from __future__ import division

import torch

from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError


class _BaseClassification(Metric):

    def __init__(self, output_transform=lambda x: x):
        self._type = None
        super(_BaseClassification, self).__init__(output_transform=output_transform)

    def _check_shape(self, output):
        y_pred, y = output

        if y.ndimension() > 1 and y.shape[1] == 1:
            # (N, 1, ...) -> (N, ...)
            y = y.squeeze(dim=1)

        if y_pred.ndimension() > 1 and y_pred.shape[1] == 1:
            # (N, 1, ...) -> (N, ...)
            y_pred = y_pred.squeeze(dim=1)

        if not (y.ndimension() == y_pred.ndimension() or y.ndimension() + 1 == y_pred.ndimension()):
            raise ValueError("y must have shape of (batch_size, ...) and y_pred must have "
                             "shape of (batch_size, num_categories, ...) or (batch_size, ...), "
                             "but given {} vs {}".format(y.shape, y_pred.shape))

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
            update_type = "multiclass"
        elif y.ndimension() == y_pred.ndimension():
            update_type = "binary"
            if not torch.equal(y, y ** 2):
                raise ValueError("For binary cases, y must be comprised of 0's and 1's.")
            # TODO: Uncomment the following after 0.1.2 release
            # if not torch.equal(y_pred, y_pred ** 2):
            #     raise ValueError("For binary cases, y_pred must be comprised of 0's and 1's.")
        else:
            raise RuntimeError("Invalid shapes of y (shape={}) and y_pred (shape={}), check documentation"
                               " for expected shapes of y and y_pred.".format(y.shape, y_pred.shape))
        if self._type is None:
            self._type = update_type
        else:
            if self._type != update_type:
                raise RuntimeError("update_type has changed from {} to {}.".format(self._type, update_type))


class Accuracy(_BaseClassification):
    """
    Calculates the accuracy for binary and multiclass data

    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...) or (batch_size, ...)
    - `y` must be in the following shape (batch_size, ...)

    In binary case, when `y` has 0 or 1 values, the elements of `y_pred` must be between 0 and 1.
    """
    # TODO: Include the following into the docstring after 0.1.2 release
    # .. code-block:: python
    #
    #     def thresholded_output_transform(output):
    #         y_pred, y = output
    #         y_pred = torch.round(y_pred)
    #         return y_pred, y
    #
    #     binary_accuracy = Accuracy(thresholded_output_transform)

    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    def update(self, output):

        y_pred, y = self._check_shape(output)
        self._check_type((y_pred, y))

        if self._type == "binary":
            indices = torch.round(y_pred).type(y.type())
        elif self._type == "multiclass":
            indices = torch.max(y_pred, dim=1)[1]

        correct = torch.eq(indices, y).view(-1)
        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Accuracy must have at least one example before it can be computed')
        return self._num_correct / self._num_examples
