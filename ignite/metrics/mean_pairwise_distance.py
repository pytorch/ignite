from __future__ import division

import torch
from torch.nn.functional import pairwise_distance

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class MeanPairwiseDistance(Metric):
    """
    Calculates the mean pairwise distance: average of pairwise distances computed on provided batches.

    - `update` must receive output of the form `(y_pred, y)`.
    """
    def __init__(self, p=2, eps=1e-6, output_transform=lambda x: x, device=None):
        super(MeanPairwiseDistance, self).__init__(output_transform, device=device)
        self._p = p
        self._eps = eps

    @reinit__is_reduced
    def reset(self):
        self._sum_of_distances = 0.0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output
        distances = pairwise_distance(y_pred, y, p=self._p, eps=self._eps)
        self._sum_of_distances += torch.sum(distances).item()
        self._num_examples += y.shape[0]

    @sync_all_reduce("_sum_of_distances", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('MeanAbsoluteError must have at least one example before it can be computed.')
        return self._sum_of_distances / self._num_examples
