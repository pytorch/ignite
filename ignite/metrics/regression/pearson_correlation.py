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
        "_mean_x",
        "_mean_y",
        "_var_x",
        "_var_y",
        "_cov",
        "_num_examples",
    )

    @reinit__is_reduced
    def reset(self) -> None:
        self._mean_x = torch.tensor(0.0, dtype=self._double_dtype, device=self._device)
        self._mean_y = torch.tensor(0.0, dtype=self._double_dtype, device=self._device)
        self._var_x = torch.tensor(0.0, dtype=self._double_dtype, device=self._device)
        self._var_y = torch.tensor(0.0, dtype=self._double_dtype, device=self._device)
        self._cov = torch.tensor(0.0, dtype=self._double_dtype, device=self._device)
        self._num_examples = 0

    def _update(self, output: tuple[torch.Tensor, torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()

        n_B = y.shape[0]
        if n_B == 0:
            return

        mean_x_B = y_pred.mean(dtype=self._double_dtype)
        mean_y_B = y.mean(dtype=self._double_dtype)

        y_pred_d = y_pred.to(self._double_dtype)
        y_d = y.to(self._double_dtype)

        var_x_B = ((y_pred_d - mean_x_B) ** 2).sum()
        var_y_B = ((y_d - mean_y_B) ** 2).sum()
        cov_B = ((y_pred_d - mean_x_B) * (y_d - mean_y_B)).sum()

        if self._num_examples == 0:
            self._mean_x = mean_x_B
            self._mean_y = mean_y_B
            self._var_x = var_x_B
            self._var_y = var_y_B
            self._cov = cov_B
            self._num_examples = n_B
        else:
            n_A = self._num_examples
            n_AB = n_A + n_B
            delta_x = mean_x_B - self._mean_x
            delta_y = mean_y_B - self._mean_y

            self._var_x += var_x_B + delta_x ** 2 * (n_A * n_B) / n_AB
            self._var_y += var_y_B + delta_y ** 2 * (n_A * n_B) / n_AB
            self._cov += cov_B + delta_x * delta_y * (n_A * n_B) / n_AB

            self._mean_x += delta_x * (n_B / n_AB)
            self._mean_y += delta_y * (n_B / n_AB)
            self._num_examples = n_AB

    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError("PearsonCorrelation must have at least one example before it can be computed.")

        import ignite.distributed as idist

        if idist.get_world_size() > 1:
            state = torch.stack([
                self._mean_x,
                self._mean_y,
                self._var_x,
                self._var_y,
                self._cov,
                torch.tensor(self._num_examples, dtype=self._double_dtype, device=self._device)
            ]).unsqueeze(0)

            gathered_state = idist.all_gather(state)

            total_mean_x = gathered_state[0, 0].clone()
            total_mean_y = gathered_state[0, 1].clone()
            total_var_x = gathered_state[0, 2].clone()
            total_var_y = gathered_state[0, 3].clone()
            total_cov = gathered_state[0, 4].clone()
            total_n = gathered_state[0, 5].item()

            for i in range(1, idist.get_world_size()):
                n_B = gathered_state[i, 5].item()
                if n_B == 0:
                    continue

                n_A = total_n
                total_n = n_A + n_B
                delta_x = gathered_state[i, 0] - total_mean_x
                delta_y = gathered_state[i, 1] - total_mean_y

                total_var_x += gathered_state[i, 2] + delta_x**2 * (n_A * n_B) / total_n
                total_var_y += gathered_state[i, 3] + delta_y**2 * (n_A * n_B) / total_n
                total_cov += gathered_state[i, 4] + delta_x * delta_y * (n_A * n_B) / total_n

                total_mean_x += delta_x * (n_B / total_n)
                total_mean_y += delta_y * (n_B / total_n)
        else:
            total_n = self._num_examples
            total_var_x = self._var_x
            total_var_y = self._var_y
            total_cov = self._cov

        # var_x and var_y are sum of squared differences, so variance is var_x / total_n
        y_pred_var = torch.clamp(total_var_x / total_n, min=0.0)
        y_var = torch.clamp(total_var_y / total_n, min=0.0)
        cov_val = total_cov / total_n

        r = cov_val / torch.clamp(torch.sqrt(y_pred_var * y_var), min=self.eps)
        return float(r.item())
