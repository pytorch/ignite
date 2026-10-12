from collections.abc import Callable

import torch

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import reinit__is_reduced

from ignite.metrics.regression._base import _BaseRegression


class PearsonCorrelation(_BaseRegression):
    r"""Calculates the
    `Pearson correlation coefficient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_.

    .. math::
        r = \frac{\sum_{j=1}^n (P_j-\bar{P})(A_j-\bar{A})}
        {\max (\sqrt{\sum_{j=1}^n (P_j-\bar{P})^2 \sum_{j=1}^n (A_j-\bar{A})^2}, \epsilon)},
        \quad \bar{P}=\frac{1}{n}\sum_{j=1}^n P_j, \quad \bar{A}=\frac{1}{n}\sum_{j=1}^n A_j

    where :math:`A_j` is the ground truth and :math:`P_j` is the predicted value.

    Internally uses `Welford's online algorithm <https://en.wikipedia.org/wiki/
    Algorithms_for_calculating_variance#Welford's_online_algorithm>`_ for numerically
    stable computation, avoiding catastrophic cancellation that can occur with the
    naive sum-of-squares formula in float32.

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

            0.9768687504744322

    .. versionchanged:: 0.6.0
        Uses Welford's online algorithm for improved numerical stability.
    """

    def __init__(
        self,
        eps: float = 1e-8,
        output_transform: Callable = lambda x: x,
        device: str | torch.device = torch.device("cpu"),
    ):
        super().__init__(output_transform, device)

        self.eps = eps

    # Welford state kept between updates. `x` denotes `y_pred`, `y` denotes the
    # ground truth. Names follow the convention used in Welford (1962) and on
    # the Wikipedia page linked in the class docstring:
    #   _num_examples  -- running count of samples seen (n)
    #   _mean_x, _mean_y -- running means of x and y
    #   _m2_x, _m2_y   -- running sums of squared deviations from the mean:
    #                       M2_x = Σ (x_i - mean_x)^2
    #                     (dividing by n at the end gives the variance)
    #   _cxy           -- running sum of paired deviations:
    #                       C_xy = Σ (x_i - mean_x)(y_i - mean_y)
    #                     (dividing by n at the end gives the covariance)
    _state_dict_all_req_keys = (
        "_num_examples",
        "_mean_x",
        "_mean_y",
        "_m2_x",
        "_m2_y",
        "_cxy",
    )

    @reinit__is_reduced
    def reset(self) -> None:
        self._num_examples = 0
        self._mean_x = torch.tensor(0.0, dtype=torch.float64, device=self._device)
        self._mean_y = torch.tensor(0.0, dtype=torch.float64, device=self._device)
        self._m2_x = torch.tensor(0.0, dtype=torch.float64, device=self._device)
        self._m2_y = torch.tensor(0.0, dtype=torch.float64, device=self._device)
        self._cxy = torch.tensor(0.0, dtype=torch.float64, device=self._device)

    def _update(self, output: tuple[torch.Tensor, torch.Tensor]) -> None:
        # Parallel Welford: compute the current batch's Welford state in one
        # shot, then merge it into the running state. "_a" suffix = prior
        # (accumulated) state, "_b" suffix = this batch, "_ab" = combined.
        y_pred, y = output[0].detach(), output[1].detach()
        y_pred = y_pred.to(dtype=torch.float64)
        y = y.to(dtype=torch.float64)

        n_b = y.shape[0]
        n_a = self._num_examples
        n_ab = n_a + n_b

        # ---- Batch stats (mean + M2/Cxy computed about the batch mean) ----
        mean_x_b = y_pred.mean().to(self._device)
        mean_y_b = y.mean().to(self._device)
        dx_b = y_pred - mean_x_b
        dy_b = y - mean_y_b
        m2_x_b = dx_b.square().sum().to(self._device)
        m2_y_b = dy_b.square().sum().to(self._device)
        cxy_b = (dx_b * dy_b).sum().to(self._device)

        if n_a == 0:
            # First batch: running state is just the batch state.
            self._mean_x = mean_x_b
            self._mean_y = mean_y_b
            self._m2_x = m2_x_b
            self._m2_y = m2_y_b
            self._cxy = cxy_b
        else:
            # ---- Welford merge: combine running state (a) with batch (b) ----
            # delta = mean_b - mean_a is the shift between the two group means;
            # the correction term n_a*n_b/n_ab * delta^2 accounts for the fact
            # that M2_a and M2_b are measured about *different* centres.
            delta_x = mean_x_b - self._mean_x
            delta_y = mean_y_b - self._mean_y

            self._mean_x += delta_x * n_b / n_ab
            self._mean_y += delta_y * n_b / n_ab

            self._m2_x += m2_x_b + delta_x * delta_x * n_a * n_b / n_ab
            self._m2_y += m2_y_b + delta_y * delta_y * n_a * n_b / n_ab
            self._cxy += cxy_b + delta_x * delta_y * n_a * n_b / n_ab

        self._num_examples = n_ab

    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError("PearsonCorrelation must have at least one example before it can be computed.")

        n = self._num_examples
        mean_x = self._mean_x
        mean_y = self._mean_y
        m2_x = self._m2_x
        m2_y = self._m2_y
        cxy = self._cxy

        ws = idist.get_world_size()
        if ws > 1:
            # Distributed reduce: each rank already holds its local Welford
            # state (n, mean_x, mean_y, M2_x, M2_y, C_xy). We can't sum them
            # directly — M2/Cxy on each rank are measured about *that rank's*
            # local mean. Instead we all_gather the per-rank states and fold
            # them in one by one using the same pairwise merge as _update.
            state = torch.stack([
                torch.tensor(float(n), dtype=torch.float64, device=self._device),
                mean_x, mean_y, m2_x, m2_y, cxy,
            ])
            all_states = idist.all_gather(state)
            all_states = all_states.reshape(ws, 6)

            # Seed the accumulator with rank 0's state, then merge ranks 1..ws-1.
            n = all_states[0, 0].item()
            mean_x = all_states[0, 1]
            mean_y = all_states[0, 2]
            m2_x = all_states[0, 3]
            m2_y = all_states[0, 4]
            cxy = all_states[0, 5]

            for i in range(1, ws):
                n_i = all_states[i, 0].item()
                n_combined = n + n_i
                dx = all_states[i, 1] - mean_x
                dy = all_states[i, 2] - mean_y

                mean_x = mean_x + dx * n_i / n_combined
                mean_y = mean_y + dy * n_i / n_combined
                m2_x = m2_x + all_states[i, 3] + dx * dx * n * n_i / n_combined
                m2_y = m2_y + all_states[i, 4] + dy * dy * n * n_i / n_combined
                cxy = cxy + all_states[i, 5] + dx * dy * n * n_i / n_combined
                n = n_combined

        var_x = torch.clamp(m2_x / n, min=0.0)
        var_y = torch.clamp(m2_y / n, min=0.0)
        cov = cxy / n

        r = cov / torch.clamp(torch.sqrt(var_x * var_y), min=self.eps)
        return float(r.item())
