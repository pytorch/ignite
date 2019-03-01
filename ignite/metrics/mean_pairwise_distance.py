from __future__ import division

import torch
from torch.nn.functional import pairwise_distance

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric


class MeanPairwiseDistance(Metric):
    """
    Calculates the mean pairwise distance.

    - `update` must receive output of the form `(y_pred, y)`.
    """
    def __init__(self, p=2, eps=1e-6, output_transform=lambda x: x):
        super(MeanPairwiseDistance, self).__init__(output_transform)
        self._p = p
        self._eps = eps

    def reset(self):
        self._sum_of_distances = 0.0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output
        distances = pairwise_distance(y_pred, y, p=self._p, eps=self._eps)
        self._sum_of_distances += torch.sum(distances).item()
        self._num_examples += y.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('MeanAbsoluteError must have at least one example before it can be computed.')
        return self._sum_of_distances / self._num_examples
