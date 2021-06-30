from typing import Tuple

import torch

from ignite.contrib.metrics.regression._base import _BaseRegression
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce


class GeometricMeanRelativeAbsoluteError(_BaseRegression):
    r"""Calculates the Geometric Mean Relative Absolute Error.

    .. math::
        \text{GMRAE} = \exp(\frac{1}{n}\sum_{j=1}^n \ln\frac{|A_j - P_j|}{|A_j - \bar{A}|})

    where :math:`A_j` is the ground truth and :math:`P_j` is the predicted value.

    More details can be found in `Botchkarev 2018`__.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y` and `y_pred` must be of same shape `(N, )` or `(N, 1)`.

    __ https://arxiv.org/abs/1809.03006

    Parameters are inherited from ``Metric.__init__``.

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            By default, metrics require the output as ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.
    """

    @reinit__is_reduced
    def reset(self) -> None:
        # self._sum_y = 0.0  # type: Union[float, torch.Tensor]
        self._num_examples = 0
        self._sum_of_errors = torch.tensor(0.0, device=self._device)

    def _update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()
        errors = torch.log(torch.abs(y - y_pred) / torch.abs(y - y.mean()))
        self._sum_of_errors += torch.sum(errors).to(self._device)
        self._num_examples += y.shape[0]

        # np.exp(np.log(np.abs(np_y - np_y_pred) / np.abs(np_y - np_y.mean())).mean())

    @sync_all_reduce("_sum_of_errors", "_num_examples")
    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError(
                "GeometricMeanRelativeAbsoluteError must have at least one example before it can be computed."
            )
        return torch.exp(self._sum_of_errors / self._num_examples).item()
