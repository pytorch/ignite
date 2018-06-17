from __future__ import division

import torch

from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError


class TopKCategoricalAccuracy(Metric):
    """
    Calculates the top-k categorical accuracy.

    - `update` must receive output of the form `(y_pred, y)`.
    """
    def __init__(self, k=5, output_transform=lambda x: x):
        super(TopKCategoricalAccuracy, self).__init__(output_transform)
        self._k = k

    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output
        sorted_indices = torch.topk(y_pred, self._k, dim=1)[1]
        expanded_y = y.view(-1, 1).expand(-1, self._k)
        correct = torch.sum(torch.eq(sorted_indices, expanded_y), dim=1)
        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('TopKCategoricalAccuracy must have at least one example before it can be computed')
        return self._num_correct / self._num_examples
