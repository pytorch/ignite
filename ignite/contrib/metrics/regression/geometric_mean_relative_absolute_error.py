from __future__ import division
import torch
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric


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
        self._product_of_errors = self._product_of_errors * torch.cumprod(errors).item()
        prev_mean_of_actuals = self._mean_of_actuals * self._num_examples
        self._num_examples += y.shape[0]
        self._mean_of_actuals = (prev_mean_of_actuals + torch.mean(y.view_as(y_pred))) / self._num_examples

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('GeometricMeanRelativeAbsoluteError must have at'
                                     'least one example before it can be computed.')
        return torch.pow(self._product_of_errors / self._mean_of_actuals, 1.0 / self._num_examples)
