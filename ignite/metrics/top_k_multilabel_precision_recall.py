from collections.abc import Callable, Sequence

import torch

from ignite.metrics.metric import reinit__is_reduced
from ignite.metrics.precision import _BasePrecisionRecall

__all__ = ["TopKMultilabelPrecision", "TopKMultilabelRecall"]


class _BaseTopKMultilabelPrecisionRecall(_BasePrecisionRecall):
    """Base class for top-k multilabel precision and recall metrics.

    For each sample, the k labels with the highest prediction scores are
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
        # Override the parent's method to accommodate for this metric inputting floats instead of binary labels
        y_pred, y = output

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
            raise ValueError(f"Input data number of labels has changed from {self._num_classes} to {num_labels}.")

    def _prepare_output(self, output: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        y_pred, y = output[0].detach(), output[1].detach()

        num_labels = y_pred.size(1)
        y_pred = torch.transpose(y_pred, 1, -1).reshape(-1, num_labels)
        y = torch.transpose(y, 1, -1).reshape(-1, num_labels)

        # Select the top-k highest-scoring labels per sample as positive predictions.
        # k is clamped to num_labels: when k >= num_labels every label is selected.
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

    For each sample, the ``k`` labels with the highest prediction scores are treated as positive
    predictions. Per-label precision is then computed as

    .. math:: \text{Precision}_k = \frac{TP_k}{TP_k + FP_k}

    where :math:`TP_k` is the number of true positives and :math:`FP_k` the number of false positives,
    counted per label across all accumulated samples.

    - ``update`` must receive output of the form ``(y_pred, y)``.
    - ``y_pred`` must contain prediction scores of shape ``(batch_size, num_labels, ...)``.
    - ``y`` must be a binary tensor (0's and 1's) of the same shape as ``y_pred``.

    Args:
        k: number of top-scoring labels to consider as positive predictions per sample. If ``k`` is greater
            than the number of labels, it is clamped to the number of labels (i.e. every label is selected).
            By default, 5.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        average: available options are

            False
              default option. Per label precision is returned.

            None
              like the ``False`` option.

            'micro'
              precision is computed counting stats of all labels altogether.

              .. math::
                  \text{Micro Precision} = \frac{\sum_{k=1}^C TP_k}{\sum_{k=1}^C TP_k+FP_k}

              where :math:`C` is the number of labels.

            'samples'
              precision is computed on a per sample basis and then averaged across samples.

              .. math::
                  \text{Sample-averaged Precision} = \frac{\sum_{n=1}^N \frac{TP_n}{TP_n+FP_n}}{N}

              where :math:`N` is the number of samples.

            'weighted'
              like macro precision but considers label imbalance. It computes precision for each label
              then returns the average of them weighted by the support of labels (number of actual positive
              samples in each label).

              .. math::
                  \text{Weighted Precision} = \frac{\sum_{k=1}^C P_k \cdot \text{Precision}_k}{\sum_{k=1}^C P_k}

              where :math:`P_k` is the number of positive samples belonging to label :math:`k`.

            macro
              computes macro precision which is the unweighted average of the per-label precision.

              .. math::
                  \text{Macro Precision} = \frac{\sum_{k=1}^C \text{Precision}_k}{C}

            True
              like macro option.
        device: specifies which device updates are accumulated on. Setting the metric's
            device to be the same as your ``update`` arguments ensures the ``update`` method is non-blocking. By
            default, CPU.
        skip_unrolling: specifies whether output should be unrolled before being fed to update method. Should be
            true for multi-output model, for example, if ``y_pred`` contains multi-output as ``(y_pred_a, y_pred_b)``
            Alternatively, ``output_transform`` can be used to handle this.

    Examples:

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        The shapes must be ``(batch_size, num_labels, ...)``. ``y_pred`` holds prediction scores and ``y`` is binary.

        .. testcode::

            metric = TopKMultilabelPrecision(k=2)
            micro_metric = TopKMultilabelPrecision(k=2, average='micro')
            macro_metric = TopKMultilabelPrecision(k=2, average=True)
            weighted_metric = TopKMultilabelPrecision(k=2, average='weighted')
            samples_metric = TopKMultilabelPrecision(k=2, average='samples')

            metric.attach(default_evaluator, "precision")
            micro_metric.attach(default_evaluator, "micro precision")
            macro_metric.attach(default_evaluator, "macro precision")
            weighted_metric.attach(default_evaluator, "weighted precision")
            samples_metric.attach(default_evaluator, "samples precision")

            y_true = torch.tensor([
                [0, 1, 0, 1],
                [1, 0, 0, 1],
                [0, 0, 1, 0],
                [1, 1, 0, 0],
                [0, 1, 1, 0],
            ])
            y_pred = torch.tensor([
                [0.1, 0.9, 0.2, 0.8],
                [0.7, 0.3, 0.6, 0.4],
                [0.2, 0.5, 0.6, 0.1],
                [0.9, 0.85, 0.05, 0.1],
                [0.3, 0.7, 0.75, 0.2],
            ])
            state = default_evaluator.run([[y_pred, y_true]])
            print(f"Precision: {state.metrics['precision']}")
            print(f"Micro Precision: {state.metrics['micro precision']}")
            print(f"Macro Precision: {state.metrics['macro precision']}")
            print(f"Weighted Precision: {state.metrics['weighted precision']}")
            print(f"Samples Precision: {state.metrics['samples precision']}")

        .. testoutput::

            Precision: tensor([1.0000, 0.7500, 0.6667, 1.0000], dtype=torch.float64)
            Micro Precision: 0.8
            Macro Precision: 0.8541666666666666
            Weighted Precision: 0.8425925925925926
            Samples Precision: 0.8

    .. versionadded:: 0.6.0
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
        else:  # _average in [False, None, 'macro', 'weighted']
            self._denominator += y_pred.sum(dim=0)
            self._numerator += correct.sum(dim=0)

            if self._average == "weighted":
                self._weight += y.sum(dim=0)

        self._updated = True


class TopKMultilabelRecall(_BaseTopKMultilabelPrecisionRecall):
    r"""Calculates top-k recall for multilabel data.

    For each sample, the ``k`` labels with the highest prediction scores are treated as positive
    predictions. Per-label recall is then computed as

    .. math:: \text{Recall}_k = \frac{TP_k}{TP_k + FN_k}

    where :math:`TP_k` is the number of true positives and :math:`FN_k` the number of false negatives,
    counted per label across all accumulated samples.

    - ``update`` must receive output of the form ``(y_pred, y)``.
    - ``y_pred`` must contain prediction scores of shape ``(batch_size, num_labels, ...)``.
    - ``y`` must be a binary tensor (0's and 1's) of the same shape as ``y_pred``.

    Args:
        k: number of top-scoring labels to consider as positive predictions per sample. If ``k`` is greater
            than the number of labels, it is clamped to the number of labels (i.e. every label is selected).
            By default, 5.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        average: available options are

            False
              default option. Per label recall is returned.

            None
              like the ``False`` option.

            'micro'
              recall is computed counting stats of all labels altogether.

              .. math::
                  \text{Micro Recall} = \frac{\sum_{k=1}^C TP_k}{\sum_{k=1}^C TP_k+FN_k}

              where :math:`C` is the number of labels.

            'samples'
              recall is computed on a per sample basis and then averaged across samples.

              .. math::
                  \text{Sample-averaged Recall} = \frac{\sum_{n=1}^N \frac{TP_n}{TP_n+FN_n}}{N}

              where :math:`N` is the number of samples.

            'weighted'
              like macro recall but considers label imbalance. It computes recall for each label
              then returns the average of them weighted by the support of labels (number of actual positive
              samples in each label).

              .. math::
                  \text{Weighted Recall} = \frac{\sum_{k=1}^C P_k \cdot \text{Recall}_k}{\sum_{k=1}^C P_k}

              where :math:`P_k` is the number of positive samples belonging to label :math:`k`.

            macro
              computes macro recall which is the unweighted average of the per-label recall.

              .. math::
                  \text{Macro Recall} = \frac{\sum_{k=1}^C \text{Recall}_k}{C}

            True
              like macro option.
        device: specifies which device updates are accumulated on. Setting the metric's
            device to be the same as your ``update`` arguments ensures the ``update`` method is non-blocking. By
            default, CPU.
        skip_unrolling: specifies whether output should be unrolled before being fed to update method. Should be
            true for multi-output model, for example, if ``y_pred`` contains multi-output as ``(y_pred_a, y_pred_b)``
            Alternatively, ``output_transform`` can be used to handle this.

    Examples:

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        The shapes must be ``(batch_size, num_labels, ...)``. ``y_pred`` holds prediction scores and ``y`` is binary.

        .. testcode::

            metric = TopKMultilabelRecall(k=2)
            micro_metric = TopKMultilabelRecall(k=2, average='micro')
            macro_metric = TopKMultilabelRecall(k=2, average=True)
            weighted_metric = TopKMultilabelRecall(k=2, average='weighted')
            samples_metric = TopKMultilabelRecall(k=2, average='samples')

            metric.attach(default_evaluator, "recall")
            micro_metric.attach(default_evaluator, "micro recall")
            macro_metric.attach(default_evaluator, "macro recall")
            weighted_metric.attach(default_evaluator, "weighted recall")
            samples_metric.attach(default_evaluator, "samples recall")

            y_true = torch.tensor([
                [0, 1, 0, 1],
                [1, 0, 0, 1],
                [0, 0, 1, 0],
                [1, 1, 0, 0],
                [0, 1, 1, 0],
            ])
            y_pred = torch.tensor([
                [0.1, 0.9, 0.2, 0.8],
                [0.7, 0.3, 0.6, 0.4],
                [0.2, 0.5, 0.6, 0.1],
                [0.9, 0.85, 0.05, 0.1],
                [0.3, 0.7, 0.75, 0.2],
            ])
            state = default_evaluator.run([[y_pred, y_true]])
            print(f"Recall: {state.metrics['recall']}")
            print(f"Micro Recall: {state.metrics['micro recall']}")
            print(f"Macro Recall: {state.metrics['macro recall']}")
            print(f"Weighted Recall: {state.metrics['weighted recall']}")
            print(f"Samples Recall: {state.metrics['samples recall']}")

        .. testoutput::

            Recall: tensor([1.0000, 1.0000, 1.0000, 0.5000], dtype=torch.float64)
            Micro Recall: 0.8888888888888888
            Macro Recall: 0.875
            Weighted Recall: 0.8888888888888888
            Samples Recall: 0.9

    .. versionadded:: 0.6.0
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
        else:  # _average in [False, None, 'macro', 'weighted']
            self._denominator += y.sum(dim=0)
            self._numerator += correct.sum(dim=0)

            if self._average == "weighted":
                self._weight += y.sum(dim=0)

        self._updated = True
