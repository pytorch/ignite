from typing import Callable, Sequence

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["MaximumMeanDiscrepancy"]


class MaximumMeanDiscrepancy(Metric):
    r"""Calculates the mean of `maximum mean discrepancy (MMD)
    <https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html>`_.

    .. math::
       \begin{align*}
           \text{MMD}^2 (P,Q) &= \underset{\| f \| \leq 1}{\text{sup}} | \mathbb{E}_{X\sim P}[f(X)]
           - \mathbb{E}_{Y\sim Q}[f(Y)] |^2 \\
           &\approx \frac{1}{B(B-1)} \sum_{i=1}^B \sum_{\substack{j=1 \\ j\neq i}}^B k(\mathbf{x}_i,\mathbf{x}_j)
           -\frac{2}{B^2}\sum_{i=1}^B \sum_{j=1}^B k(\mathbf{x}_i,\mathbf{y}_j)
           + \frac{1}{B(B-1)} \sum_{i=1}^B \sum_{\substack{j=1 \\ j\neq i}}^B k(\mathbf{y}_i,\mathbf{y}_j)
       \end{align*}

    where :math:`B` is the batch size, and :math:`\mathbf{x}_i` and :math:`\mathbf{y}_j` are
    feature vectors sampled from :math:`P` and :math:`Q`, respectively.
    :math:`k(\mathbf{x},\mathbf{y})=\exp(-\| \mathbf{x}-\mathbf{y} \|^2/ 2\sigma^2)` is the Gaussian RBF kernel.

    This metric computes the MMD for each batch and takes the average.

    More details can be found in `Gretton et al. 2012`__.

    __ https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf

    - ``update`` must receive output of the form ``(x, y)``.
    - ``x`` and ``y`` are expected to be in the same shape :math:`(B, \ldots)`.

    Args:
        var: the bandwidth :math:`\sigma^2` of the kernel. Default: 1.0
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            By default, this metric requires the output as ``(x, y)``.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.
        skip_unrolling: specifies whether output should be unrolled before being fed to update method. Should be
            true for multi-output model, for example, if ``y_pred`` contains multi-ouput as ``(y_pred_a, y_pred_b)``
            Alternatively, ``output_transform`` can be used to handle this.

    Examples:
        To use with ``Engine`` and ``process_function``, simply attach the metric instance to the engine.
        The output of the engine's ``process_function`` needs to be in the format of
        ``(x, y)``. If not, ``output_tranform`` can be added
        to the metric to transform the output into the form expected by the metric.

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            metric = MaximumMeanDiscrepancy()
            metric.attach(default_evaluator, "mmd")
            x = torch.tensor([[-0.80324818, -0.95768364, -0.03807209],
                            [-0.11059691, -0.38230813, -0.4111988],
                            [-0.8864329, -0.02890403, -0.60119252],
                            [-0.68732452, -0.12854739, -0.72095073],
                            [-0.62604613, -0.52368328, -0.24112842]])
            y = torch.tensor([[0.0686768, 0.80502737, 0.53321717],
                            [0.83849465, 0.59099726, 0.76385441],
                            [0.68688272, 0.56833803, 0.98100778],
                            [0.55267761, 0.13084654, 0.45382906],
                            [0.0754253, 0.70317304, 0.4756805]])
            state = default_evaluator.run([[x, y]])
            print(state.metrics["mmd"])

        .. testoutput::

           1.072697639465332

    .. versionchanged:: 0.5.1
        ``skip_unrolling`` argument is added.
    """

    _state_dict_all_req_keys = ("_xx_sum", "_yy_sum", "_xy_sum", "_num_batches")

    def __init__(
        self,
        var: float = 1.0,
        output_transform: Callable = lambda x: x,
        device: torch.device = torch.device("cpu"),
        skip_unrolling: bool = False,
    ):
        self.var = var
        super().__init__(output_transform, device, skip_unrolling=skip_unrolling)

    @reinit__is_reduced
    def reset(self) -> None:
        self._xx_sum = torch.tensor(0.0, device=self._device)
        self._yy_sum = torch.tensor(0.0, device=self._device)
        self._xy_sum = torch.tensor(0.0, device=self._device)
        self._num_batches = 0

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        x, y = output[0].detach(), output[1].detach()
        if x.shape != y.shape:
            raise ValueError(f"x and y must be in the same shape, got {x.shape} != {y.shape}.")

        if x.ndim >= 3:
            x = x.flatten(start_dim=1)
            y = y.flatten(start_dim=1)
        elif x.ndim == 1:
            raise ValueError(f"x must be in the shape of (B, ...), got {x.shape}.")

        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = xx.diag().unsqueeze(0).expand_as(xx)
        ry = yy.diag().unsqueeze(0).expand_as(yy)

        dxx = rx.t() + rx - 2.0 * xx
        dyy = ry.t() + ry - 2.0 * yy
        dxy = rx.t() + ry - 2.0 * zz

        v = self.var
        XX = torch.exp(-0.5 * dxx / v)
        YY = torch.exp(-0.5 * dyy / v)
        XY = torch.exp(-0.5 * dxy / v)

        # unbiased
        n = x.shape[0]
        XX = (XX.sum() - n) / (n * (n - 1))
        YY = (YY.sum() - n) / (n * (n - 1))
        XY = XY.sum() / (n * n)

        self._xx_sum += XX.to(self._device)
        self._yy_sum += YY.to(self._device)
        self._xy_sum += XY.to(self._device)

        self._num_batches += 1

    @sync_all_reduce("_xx_sum", "_yy_sum", "_xy_sum", "_num_batches")
    def compute(self) -> float:
        if self._num_batches == 0:
            raise NotComputableError("MaximumMeanDiscrepacy must have at least one batch before it can be computed.")
        mmd2 = (self._xx_sum + self._yy_sum - 2.0 * self._xy_sum).clamp(min=0.0) / self._num_batches
        return mmd2.sqrt().item()
