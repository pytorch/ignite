from typing import Callable, Sequence, Union

import torch
from torch import Tensor

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["HSIC"]


class HSIC(Metric):
    r"""Calculates the `Hilbert-Schmidt Independence Criterion (HSIC)
    <https://papers.nips.cc/paper_files/paper/2007/hash/d5cfead94f5350c12c322b5b664544c1-Abstract.html>`_.

    .. math::
        \text{HSIC}(X,Y) = \frac{1}{B(B-3)}\left[ \text{tr}(\tilde{\mathbf{K}}\tilde{\mathbf{L}})
         + \frac{\mathbf{1}^\top \tilde{\mathbf{K}} \mathbf{11}^\top \tilde{\mathbf{L}} \mathbf{1}}{(B-1)(B-2)}
         -\frac{2}{B-2}\mathbf{1}^\top \tilde{\mathbf{K}}\tilde{\mathbf{L}} \mathbf{1}  \right]

    where :math:`B` is the batch size, and :math:`\tilde{\mathbf{K}}`
    and :math:`\tilde{\mathbf{L}}` are the Gram matrices of
    the Gaussian RBF kernel with their diagonal entries being set to zero.

    HSIC measures non-linear statistical independence between features :math:`X` and :math:`Y`.
    HSIC becomes zero if and only if :math:`X` and :math:`Y` are independent.

    This metric computes the unbiased estimator of HSIC proposed in
    `Song et al. (2012) <https://jmlr.csail.mit.edu/papers/v13/song12a.html>`_.
    The HSIC is estimated using Eq. (5) of the paper for each batch and the average is accumulated.

    Each batch must contain at least four samples.

    - ``update`` must receive output of the form ``(y_pred, y)``.

    Args:
        sigma_x: bandwidth of the kernel for :math:`X`.
            If negative, a heuristic value determined by the median of the distances between
            the samples is used. Default: -1
        sigma_y: bandwidth of the kernel for :math:`Y`.
            If negative, a heuristic value determined by the median of the distances
            between the samples is used. Default: -1
        ignore_invalid_batch: If ``True``, computation for a batch with less than four samples is skipped.
            If ``False``, ``ValueError`` is raised when received such a batch.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            By default, metrics require the output as ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.
        skip_unrolling: specifies whether output should be unrolled before being fed to update method. Should be
            true for multi-output model, for example, if ``y_pred`` contains multi-ouput as ``(y_pred_a, y_pred_b)``
            Alternatively, ``output_transform`` can be used to handle this.

    Examples:
        To use with ``Engine`` and ``process_function``, simply attach the metric instance to the engine.
        The output of the engine's ``process_function`` needs to be in the format of
        ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y, ...}``. If not, ``output_tranform`` can be added
        to the metric to transform the output into the form expected by the metric.

        ``y_pred`` and ``y`` should have the same shape.

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            metric = HSIC()
            metric.attach(default_evaluator, "hsic")
            X = torch.tensor([[0., 1., 2., 3., 4.],
                            [5., 6., 7., 8., 9.],
                            [10., 11., 12., 13., 14.],
                            [15., 16., 17., 18., 19.],
                            [20., 21., 22., 23., 24.],
                            [25., 26., 27., 28., 29.],
                            [30., 31., 32., 33., 34.],
                            [35., 36., 37., 38., 39.],
                            [40., 41., 42., 43., 44.],
                            [45., 46., 47., 48., 49.]])
            Y = torch.sin(X * torch.pi * 2 / 50)
            state = default_evaluator.run([[X, Y]])
            print(state.metrics["hsic"])

        .. testoutput::

            0.09226646274328232

    .. versionadded:: 0.5.2
    """

    def __init__(
        self,
        sigma_x: float = -1,
        sigma_y: float = -1,
        ignore_invalid_batch: bool = True,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
        skip_unrolling: bool = False,
    ):
        super().__init__(output_transform, device, skip_unrolling=skip_unrolling)

        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.ignore_invalid_batch = ignore_invalid_batch

    _state_dict_all_req_keys = ("_sum_of_hsic", "_num_batches")

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_of_hsic = torch.tensor(0.0, device=self._device)
        self._num_batches = 0

    @reinit__is_reduced
    def update(self, output: Sequence[Tensor]) -> None:
        X = output[0].detach().flatten(start_dim=1)
        Y = output[1].detach().flatten(start_dim=1)
        b = X.shape[0]

        if b <= 3:
            if self.ignore_invalid_batch:
                return
            else:
                raise ValueError(f"A batch must contain more than four samples, got only {b} samples.")

        mask = 1.0 - torch.eye(b, device=X.device)

        xx = X @ X.T
        rx = xx.diag().unsqueeze(0).expand_as(xx)
        dxx = rx.T + rx - xx * 2

        vx: Union[Tensor, float]
        if self.sigma_x < 0:
            # vx = torch.quantile(dxx, 0.5)
            vx = torch.quantile(dxx, 0.5)
        else:
            vx = self.sigma_x**2
        K = torch.exp(-0.5 * dxx / vx) * mask

        yy = Y @ Y.T
        ry = yy.diag().unsqueeze(0).expand_as(yy)
        dyy = ry.T + ry - yy * 2

        vy: Union[Tensor, float]
        if self.sigma_y < 0:
            vy = torch.quantile(dyy, 0.5)
        else:
            vy = self.sigma_y**2
        L = torch.exp(-0.5 * dyy / vy) * mask

        KL = K @ L
        trace = KL.trace()
        second_term = K.sum() * L.sum() / ((b - 1) * (b - 2))
        third_term = KL.sum() / (b - 2)

        hsic = trace + second_term - third_term * 2.0
        hsic /= b * (b - 3)
        hsic = torch.clamp(hsic, min=0.0)  # HSIC must not be negative
        self._sum_of_hsic += hsic.to(self._device)

        self._num_batches += 1

    @sync_all_reduce("_sum_of_hsic", "_num_batches")
    def compute(self) -> float:
        if self._num_batches == 0:
            raise NotComputableError("HSIC must have at least one batch before it can be computed.")

        return self._sum_of_hsic.item() / self._num_batches
