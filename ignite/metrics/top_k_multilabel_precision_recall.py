from collections.abc import Callable, Sequence

import torch

from ignite.metrics.metric import reinit__is_reduced
from ignite.metrics.precision import _BasePrecisionRecall

__all__ = ["TopKMultilabelPrecision", "TopKMultilabelRecall"]


class _BaseTopKMultilabelPrecisionRecall(_BasePrecisionRecall):
    """Base class for top-k multilabel precision and recall metrics.

    For each sample, the ``k`` labels with the highest prediction scores are
    selected as positive predictions. Precision and recall are then computed per
    label, treating each label independently across the accumulated samples.
    """

    def __init__(
        self,
        k: int = 5,
        output_transform: Callable = lambda x: x,
        average: bool | str | None = False,
        device: str | torch.device = torch.device("cpu"),
        skip_unrolling: bool = False,
    ) -> None:
        if k < 1:
            raise ValueError(f"Argument k should be a positive integer, given {k}.")

        self._k = k
        super().__init__(
            output_transform=output_transform,
            average=average,
            is_multilabel=True,
            device=device,
            skip_unrolling=skip_unrolling,
        )

    def _check_type(self, output: Sequence[torch.Tensor]) -> None:
        self._check_shape(output)
        y_pred, y = output

        if not (y.shape == y_pred.shape and y.ndimension() > 1 and y.shape[1] > 1):
            raise ValueError(
                "y and y_pred must have same shape of (batch_size, num_labels, ...) "
                f"with num_labels > 1, but given {y.shape} vs {y_pred.shape}."
            )

        if not torch.equal(y, y**2):
            raise ValueError("For multilabel cases, y must be comprised of 0's and 1's.")

        if y_pred.dtype in (torch.int, torch.long):
            raise TypeError(f"`y_pred` should be a float tensor with prediction scores, given {y_pred.dtype}.")

        num_labels = y_pred.shape[1]
        if self._type is None:
            self._type = "multilabel"
            self._num_classes = num_labels
        elif self._type != "multilabel":
            raise RuntimeError(f"Input data type has changed from {self._type} to multilabel.")
        elif self._num_classes != num_labels:
            raise ValueError(
                f"Input data number of labels has changed from {self._num_classes} to {num_labels}."
            )

    def _prepare_output(self, output: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        y_pred, y = output[0].detach(), output[1].detach()

        num_labels = y_pred.size(1)
        y_pred = torch.transpose(y_pred, 1, -1).reshape(-1, num_labels)
        y = torch.transpose(y, 1, -1).reshape(-1, num_labels)

        k = min(self._k, num_labels)
        topk_indices = torch.topk(y_pred, k, dim=1).indices
        y_pred_topk = torch.zeros_like(y_pred)
        y_pred_topk.scatter_(1, topk_indices, 1.0)

        y_pred_topk = y_pred_topk.to(dtype=self._double_dtype, device=self._device)
        y = y.to(dtype=self._double_dtype, device=self._device)
        correct = y * y_pred_topk

        return y_pred_topk, y, correct


class TopKMultilabelPrecision(_BaseTopKMultilabelPrecisionRecall):
    r"""Calculates top-k precision for multilabel data.

    For each sample, the ``k`` labels with the highest scores are treated as
    positive predictions. Per-label precision is then computed as

    .. math:: \text{Precision}_k = \frac{TP_k}{TP_k + FP_k}

    where :math:`TP_k` and :math:`FP_k` are counted per label across samples.

    - ``update`` must receive output of the form ``(y_pred, y)``.
    - ``y_pred`` must be prediction scores of shape ``(batch_size, num_labels, ...)``.
    - ``y`` must be a binary tensor of the same shape as ``y_pred``.

    Args:
        k: number of top-scoring labels to consider as positive predictions per sample.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric.
        average: reduction option. Same semantics as :class:`~ignite.metrics.precision.Precision`
            for multilabel inputs.
        device: specifies which device updates are accumulated on.
        skip_unrolling: specifies whether output should be unrolled before being fed to update method.
    """

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        self._check_shape(output)
        self._check_type(output)
        y_pred, y, correct = self._prepare_output(output)

        if self._average == "samples":
            all_positives = y_pred.sum(dim=1)
            true_positives = correct.sum(dim=1)
            self._numerator += torch.sum(true_positives / (all_positives + self.eps))
            self._denominator += y.size(0)
        elif self._average == "micro":
            self._denominator += y_pred.sum()
            self._numerator += correct.sum()
        else:
            self._denominator += y_pred.sum(dim=0)
            self._numerator += correct.sum(dim=0)

            if self._average == "weighted":
                self._weight += y.sum(dim=0)

        self._updated = True


class TopKMultilabelRecall(_BaseTopKMultilabelPrecisionRecall):
    r"""Calculates top-k recall for multilabel data.

    For each sample, the ``k`` labels with the highest scores are treated as
    positive predictions. Per-label recall is then computed as

    .. math:: \text{Recall}_k = \frac{TP_k}{TP_k + FN_k}

    where :math:`TP_k` and :math:`FN_k` are counted per label across samples.

    - ``update`` must receive output of the form ``(y_pred, y)``.
    - ``y_pred`` must be prediction scores of shape ``(batch_size, num_labels, ...)``.
    - ``y`` must be a binary tensor of the same shape as ``y_pred``.

    Args:
        k: number of top-scoring labels to consider as positive predictions per sample.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric.
        average: reduction option. Same semantics as :class:`~ignite.metrics.recall.Recall`
            for multilabel inputs.
        device: specifies which device updates are accumulated on.
        skip_unrolling: specifies whether output should be unrolled before being fed to update method.
    """

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        self._check_shape(output)
        self._check_type(output)
        _, y, correct = self._prepare_output(output)

        if self._average == "samples":
            actual_positives = y.sum(dim=1)
            true_positives = correct.sum(dim=1)
            self._numerator += torch.sum(true_positives / (actual_positives + self.eps))
            self._denominator += y.size(0)
        elif self._average == "micro":
            self._denominator += y.sum()
            self._numerator += correct.sum()
        else:
            self._denominator += y.sum(dim=0)
            self._numerator += correct.sum(dim=0)

            if self._average == "weighted":
                self._weight += y.sum(dim=0)

        self._updated = True
