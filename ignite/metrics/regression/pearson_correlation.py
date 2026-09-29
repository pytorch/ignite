from collections.abc import Callable

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce

from ignite.metrics.regression._base import _BaseRegression


class PearsonCorrelation(_BaseRegression):
    r"""Calculates the
    `Pearson correlation coefficient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_.

    .. math::
        r = \frac{\sum_{j=1}^n (P_j-\bar{P})(A_j-\bar{A})}
        {\max (\sqrt{\sum_{j=1}^n (P_j-\bar{P})^2 \sum_{j=1}^n (A_j-\bar{A})^2}, \epsilon)},
        \quad \bar{P}=\frac{1}{n}\sum_{j=1}^n P_j, \quad \bar{A}=\frac{1}{n}\sum_{j=1}^n A_j

    where :math:`A_j` is the ground truth and :math:`P_j` is the predicted value.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y` and `y_pred` must be of same shape `(N, )` or `(N, 1)`.

    Parameters are inherited from ``Metric.__init__``.

    Args:
        eps: a small value to avoid division by zero. Default: 1e-8
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            By default, metrics require the output as ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.

    Examples:
        To use with ``Engine`` and ``process_function``, simply attach the metric instance to the engine.
        The output of the engine's ``process_function`` needs to be in format of
        ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y, ...}``.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            metric = PearsonCorrelation()
            metric.attach(default_evaluator, 'corr')
            y_true = torch.tensor([0., 1., 2., 3., 4., 5.])
            y_pred = torch.tensor([0.5, 1.3, 1.9, 2.8, 4.1, 6.0])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics['corr'])

        .. testoutput::

            0.9768688678741455
    """

    def __init__(
        self,
        eps: float = 1e-8,
        output_transform: Callable = lambda x: x,
        device: str | torch.device = torch.device("cpu"),
    ):
        super().__init__(output_transform, device)

        self.eps = eps

    _state_dict_all_req_keys = (
        "_mean_y_preds",
        "_mean_ys",
        "_var_y_preds",
        "_var_ys",
        "_cov",
        "_num_examples",
    )

    @reinit__is_reduced
    def reset(self) -> None:
        self._mean_y_preds = torch.tensor(0.0, dtype=torch.float64, device=self._device)
        self._mean_ys = torch.tensor(0.0, dtype=torch.float64, device=self._device)
        self._var_y_preds = torch.tensor(0.0, dtype=torch.float64, device=self._device)
        self._var_ys = torch.tensor(0.0, dtype=torch.float64, device=self._device)
        self._cov = torch.tensor(0.0, dtype=torch.float64, device=self._device)
        self._num_examples = 0

    def _update(self, output: tuple[torch.Tensor, torch.Tensor]) -> None:
        y_pred, y = output[0].detach().to(torch.float64), output[1].detach().to(torch.float64)
        
        m = y.shape[0]
        n = self._num_examples
        
        y_pred_mean = y_pred.mean()
        y_mean = y.mean()
        
        y_pred_sq = ((y_pred - y_pred_mean) ** 2).sum()
        y_sq = ((y - y_mean) ** 2).sum()
        prod = ((y_pred - y_pred_mean) * (y - y_mean)).sum()
        
        if n == 0:
            self._mean_y_preds += y_pred_mean
            self._mean_ys += y_mean
            self._var_y_preds += y_pred_sq
            self._var_ys += y_sq
            self._cov += prod
        else:
            delta_y_preds = y_pred_mean - self._mean_y_preds
            delta_ys = y_mean - self._mean_ys
            
            self._mean_y_preds += delta_y_preds * m / (n + m)
            self._mean_ys += delta_ys * m / (n + m)
            
            self._var_y_preds += y_pred_sq + (delta_y_preds ** 2) * n * m / (n + m)
            self._var_ys += y_sq + (delta_ys ** 2) * n * m / (n + m)
            self._cov += prod + delta_y_preds * delta_ys * n * m / (n + m)
            
        self._num_examples += m

    @sync_all_reduce(
        "_mean_y_preds",
        "_mean_ys",
        "_var_y_preds",
        "_var_ys",
        "_cov",
        "_num_examples",
    )
    def compute(self) -> float:
        n = self._num_examples
        if n == 0:
            raise NotComputableError("PearsonCorrelation must have at least one example before it can be computed.")

        y_pred_var = torch.clamp(self._var_y_preds / n, min=0.0)
        y_var = torch.clamp(self._var_ys / n, min=0.0)
        cov = self._cov / n

        r = cov / torch.clamp(torch.sqrt(y_pred_var * y_var), min=self.eps)
        return float(r.item())
