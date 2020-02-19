import torch

from ignite.contrib.metrics.regression._base import _BaseRegression


class GeometricMeanRelativeAbsoluteError(_BaseRegression):
    r"""
    Calculates the Geometric Mean Relative Absolute Error:

    :math:`\text{GMRAE} = \exp(\frac{1}{n}\sum_{j=1}^n \ln\frac{|A_j - P_j|}{|A_j - \bar{A}|})`

    where :math:`A_j` is the ground truth and :math:`P_j` is the predicted value.

    More details can be found in `Botchkarev 2018`__.

    - `update` must receive output of the form `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}`.
    - `y` and `y_pred` must be of same shape `(N, )` or `(N, 1)`.


    __ https://arxiv.org/abs/1809.03006

    """

    def reset(self):
        self._sum_y = 0.0
        self._num_examples = 0
        self._sum_of_errors = 0.0

    def _update(self, output):
        y_pred, y = output
        self._sum_y += y.sum()
        self._num_examples += y.shape[0]
        y_mean = self._sum_y / self._num_examples
        numerator = torch.abs(y.view_as(y_pred) - y_pred)
        denominator = torch.abs(y.view_as(y_pred) - y_mean)
        self._sum_of_errors += torch.log(numerator / denominator).sum()

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                "GeometricMeanRelativeAbsoluteError must have at least " "one example before it can be computed."
            )
        return torch.exp(torch.mean(self._sum_of_errors / self._num_examples)).item()
