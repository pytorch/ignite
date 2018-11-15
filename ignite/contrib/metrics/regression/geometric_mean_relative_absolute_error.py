from __future__ import division
import torch
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric
import numpy as np


class GeometricMeanRelativeAbsoluteError(Metric):
    r"""
    Calculates the Geometric Mean Relative Absolute Error.
    It has been proposed in `Botchkarev 2018`__.
    :math:`\text{GMRAE} = \sqrt[n]{\prod_{j=1}^{\n}\frac{|A_j - P_j|}{A_j - \bar{A}}}`
    Where, :math:`A_j` is the ground truth and :math:`P_j` is the predicted value.
    - `update` must receive output of the form `(y_pred, y)`.
    - `y` and `y_pred` must be of same shape.
    __ https://arxiv.org/abs/1809.03006
    """

    def reset(self):
        self._product_of_errors = 1.0
        self._mean_of_actuals = 0.0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output
        errors = torch.abs(y.view_as(y_pred) - y_pred)
        # previous sum is required to calculate new average of ground truth
        prev_sum_of_gt = self._mean_of_actuals * self._num_examples
        self._num_examples += y.shape[0]
        # once num_examples is updated, new average is (prev_sum + y.sum())/num_examples
        self._mean_of_actuals = (prev_sum_of_gt + y.sum().item()) / self._num_examples
        relative_error = errors / torch.abs(y - self._mean_of_actuals)
        self._product_of_errors = self._product_of_errors * torch.prod(relative_error).item()

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('GeometricMeanRelativeAbsoluteError must have at'
                                     'least one example before it can be computed.')
        return np.power(self._product_of_errors, 1.0 / self._num_examples)
