from typing import Sequence

import torch
import torch.nn.functional as F
from packaging.version import Version

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["KLDivergence"]

TORCH_VERSION_GE_160 = Version(torch.__version__) >= Version("1.6.0")


class KLDivergence(Metric):
    r"""Calculates the mean of `Kullback-Leibler (KL) divergence
    <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_.

    .. math:: D_\text{KL}(\mathbf{p}_i \| \mathbf{q}_i) = \sum_{c=1}^C p_{i,c} \log \frac{p_{i,c}}{q_{i,c}}

    where :math:`\mathbf{p}_i` and :math:`\mathbf{q}_i` are the ground truth and prediction probability tensors.

    - ``update`` must receive output of the form ``(y_pred, y)``.
    - ``y_pred`` and ``y`` are expected to be the unnormalized logits for each class. :math:`(B, C)` (classification)
      or :math:`(B, C, ...)` (e.g., image segmentation) shapes are allowed.

    Args:
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

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            metric = KLDivergence()
            metric.attach(default_evaluator, 'kl-div')
            y_true = torch.tensor([
                [ 0.0000, -2.3026, -2.3026],
                [ 1.3863,  1.6094,  1.6094],
                [ 0.0000,  0.6931,  1.0986]
            ])
            y_pred = torch.tensor([
                [ 0.0000,  0.6931,  1.0986],
                [ 1.3863,  1.6094,  1.6094],
                [ 0.0000, -2.3026, -2.3026]
            ])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics['kl-div'])

        .. testoutput::

           0.7220296859741211

    .. versionchanged:: 0.5.1
        ``skip_unrolling`` argument is added.
    """

    _state_dict_all_req_keys = ("_sum_of_kl", "_num_examples")

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_of_kl = torch.tensor(0.0, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()
        if y_pred.shape != y.shape:
            raise ValueError(f"y_pred and y must be in the same shape, got {y_pred.shape} != {y.shape}.")

        if y_pred.ndim >= 3:
            num_classes = y_pred.shape[1]
            # (B, C, ...) -> (B, ..., C) -> (B*..., C)
            # regarding as B*... predictions
            y_pred = y_pred.movedim(1, -1).reshape(-1, num_classes)
            y = y.movedim(1, -1).reshape(-1, num_classes)
        elif y_pred.ndim == 1:
            raise ValueError(f"y_pred must be in the shape of (B, C) or (B, C, ...), got {y_pred.shape}.")

        self._num_examples += y_pred.shape[0]
        self._update(y_pred, y)

    def _update(self, y_pred: torch.Tensor, y: torch.Tensor) -> None:
        y_pred = F.log_softmax(y_pred, dim=1)

        if TORCH_VERSION_GE_160:
            # log_target option can be used from 1.6.0
            y = F.log_softmax(y, dim=1)
            kl_sum = F.kl_div(y_pred, y, log_target=True, reduction="sum")
        else:
            # y is expected to be a probability tensor
            y = F.softmax(y, dim=1)
            kl_sum = F.kl_div(y_pred, y, reduction="sum")

        self._sum_of_kl += kl_sum.to(self._device)

    @sync_all_reduce("_sum_of_kl", "_num_examples")
    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError("KLDivergence must have at least one example before it can be computed.")
        return self._sum_of_kl.item() / self._num_examples
