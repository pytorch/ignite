from __future__ import division

import torch

from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError


class MultiLabelAccuracy(Metric):
    """
    Calculates the accuracy.

    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...) or (batch_size, ...)
    - `y` must be in the following shape (batch_size, ...)
    """
    
    def __init__(self, threshold_function=lambda x: torch.round(x), output_transform=lambda x: x):
        super(MultiLabelAccuracy, self).__init__(output_transform)
        self._threshold = threshold_function
    
    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output

        if not (y.ndimension() == y_pred.ndimension() and y.dimension() > 1):
            raise ValueError("y must have shape of (batch_size, num_classes, ...) " +
                             "and y_pred must have shape of (batch_size, num_classes, ...).")

        if y_pred.ndimension() == 3:
            y_pred = y_pred.transpose(2, 1).contiguous().view(-1, y_pred.size(1))
            y = y.transpose(2, 1).contiguous().view(-1, y.size(1))
        
        if y_pred.ndimension() == 4:
            y_pred = y_pred.permute(0, 2, 3, 1).contiguous().view(-1, y_pred.size(1))
            y = y.permute(0, 2, 3, 1).contiguous().view(-1, y.size(1))
        
        indices = torch.round(y_pred).type(y.type())
        correct = [torch.equal(y_i, y_pred_i) for y_i, y_pred_i in zip(y, y_pred)]

        self._num_correct += sum(correct)
        self._num_examples += len(correct)

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Accuracy must have at least one example before it can be computed')
        return self._num_correct / self._num_examples
