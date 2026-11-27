from collections.abc import Callable, Sequence

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["ExpectedCalibrationError"]


class ExpectedCalibrationError(Metric):
    r"""Calculates the `Expected Calibration Error (ECE)
    <https://arxiv.org/abs/1706.04599>`_ for multiclass classification.

    .. math::
        \text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|

    where :math:`M` is the number of bins, :math:`B_m` is the set of samples whose predicted confidence
    falls into bin :math:`m`, :math:`n` is the total number of samples, :math:`\text{acc}(B_m)` is
    the accuracy of samples in bin :math:`m`, and :math:`\text{conf}(B_m)` is the mean confidence
    of samples in bin :math:`m`.

    - ``update`` must receive output of the form ``(y_pred, y)``.
    - ``y_pred`` is expected to be the softmax probabilities for each class with shape :math:`(B, C)`.
    - ``y`` is expected to be the ground truth class indices with shape :math:`(B,)`.

    Args:
        n_bins: number of equally-spaced confidence bins. Default: 15.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            By default, metrics require the output as ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.
        skip_unrolling: specifies whether output should be unrolled before being fed to update method. Should be
            true for multi-output model, for example, if ``y_pred`` contains multi-output as ``(y_pred_a, y_pred_b)``
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

            metric = ExpectedCalibrationError(n_bins=10)
            metric.attach(default_evaluator, 'ece')
            y_true = torch.tensor([0, 1, 2, 0])
            y_pred = torch.tensor([
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.6, 0.3, 0.1],
            ])
            state = default_evaluator.run([[y_pred, y_true]])
            print(f"{state.metrics['ece']:.4f}")

        .. testoutput::

            0.2250

    .. versionadded:: 0.5.2
    """

    _state_dict_all_req_keys = ("_confidences", "_corrects")

    def __init__(
        self,
        n_bins: int = 15,
        output_transform: Callable = lambda x: x,
        device: str | torch.device = torch.device("cpu"),
        skip_unrolling: bool = False,
    ):
        if n_bins < 1:
            raise ValueError(f"n_bins must be a positive integer, got {n_bins}.")
        self.n_bins = n_bins
        super().__init__(output_transform=output_transform, device=device, skip_unrolling=skip_unrolling)

    @reinit__is_reduced
    def reset(self) -> None:
        self._confidences = torch.tensor([], dtype=torch.float, device=self._device)
        self._corrects = torch.tensor([], dtype=torch.long, device=self._device)

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()

        if y_pred.ndim != 2:
            raise ValueError(f"y_pred must be of shape (batch_size, num_classes), got {y_pred.shape}.")
        if y.ndim != 1:
            raise ValueError(f"y must be of shape (batch_size,), got {y.shape}.")

        confidences, predicted = torch.max(y_pred, dim=1)
        corrects = predicted.eq(y).long()

        self._confidences = torch.cat([self._confidences, confidences.to(self._device)])
        self._corrects = torch.cat([self._corrects, corrects.to(self._device)])

    @sync_all_reduce("_confidences", "_corrects")
    def compute(self) -> float:
        if self._confidences.shape[0] == 0:
            raise NotComputableError(
                "ExpectedCalibrationError must have at least one example before it can be computed."
            )

        n = self._confidences.shape[0]
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=self._device)

        ece = torch.tensor(0.0, device=self._device)
        for i in range(self.n_bins):
            lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
            if i == self.n_bins - 1:
                # ponytail: include right boundary in last bin
                in_bin = (self._confidences >= lower) & (self._confidences <= upper)
            else:
                in_bin = (self._confidences >= lower) & (self._confidences < upper)

            bin_size = in_bin.sum().item()
            if bin_size > 0:
                bin_acc = self._corrects[in_bin].float().mean()
                bin_conf = self._confidences[in_bin].mean()
                ece += (bin_size / n) * torch.abs(bin_acc - bin_conf)

        return ece.item()
