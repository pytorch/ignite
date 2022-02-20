from typing import Callable, Sequence, Union

import torch

from ignite.metrics.metric import reinit__is_reduced
from ignite.metrics.precision import _BasePrecisionRecall
from ignite.utils import to_onehot

__all__ = ["Recall"]


class Recall(_BasePrecisionRecall):
    r"""Calculates recall for binary and multiclass data.

    .. math:: \text{Recall} = \frac{ TP }{ TP + FN }

    where :math:`\text{TP}` is true positives and :math:`\text{FN}` is false negatives.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...) or (batch_size, ...).
    - `y` must be in the following shape (batch_size, ...).

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        average: if True, precision is computed as the unweighted average (across all classes
            in multiclass case), otherwise, returns a tensor with the precision (for each class in multiclass case).
        is_multilabel: flag to use in multilabel case. By default, value is False. If True, average
            parameter should be True and the average is computed across samples, instead of classes.
        device: specifies which device updates are accumulated on. Setting the metric's
            device to be the same as your ``update`` arguments ensures the ``update`` method is non-blocking. By
            default, CPU.

    Examples:

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        Binary case

        .. testcode:: 1

            metric = Recall(average=False)
            metric.attach(default_evaluator, "recall")
            y_true = torch.Tensor([1, 0, 1, 1, 0, 1])
            y_pred = torch.Tensor([1, 0, 1, 0, 1, 1])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics["recall"])

        .. testoutput:: 1

            0.75

        Multiclass case

        .. testcode:: 2

            metric = Recall(average=False)
            metric.attach(default_evaluator, "recall")
            y_true = torch.Tensor([2, 0, 2, 1, 0, 1]).long()
            y_pred = torch.Tensor([
                [0.0266, 0.1719, 0.3055],
                [0.6886, 0.3978, 0.8176],
                [0.9230, 0.0197, 0.8395],
                [0.1785, 0.2670, 0.6084],
                [0.8448, 0.7177, 0.7288],
                [0.7748, 0.9542, 0.8573],
            ])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics["recall"])

        .. testoutput:: 2

            tensor([0.5000, 0.5000, 0.5000], dtype=torch.float64)

        Precision can be computed as the unweighted average across all classes:

        .. testcode:: 3

            metric = Recall(average=True)
            metric.attach(default_evaluator, "recall")
            y_true = torch.Tensor([2, 0, 2, 1, 0, 1]).long()
            y_pred = torch.Tensor([
                [0.0266, 0.1719, 0.3055],
                [0.6886, 0.3978, 0.8176],
                [0.9230, 0.0197, 0.8395],
                [0.1785, 0.2670, 0.6084],
                [0.8448, 0.7177, 0.7288],
                [0.7748, 0.9542, 0.8573],
            ])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics["recall"])

        .. testoutput:: 3

            0.5

        Multilabel case, the shapes must be (batch_size, num_categories, ...)

        .. testcode:: 4

            metric = Recall(is_multilabel=True)
            metric.attach(default_evaluator, "recall")
            y_true = torch.Tensor([
                [0, 0, 1],
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 1],
            ]).unsqueeze(0)
            y_pred = torch.Tensor([
                [1, 1, 0],
                [1, 0, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
            ]).unsqueeze(0)
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics["recall"])

        .. testoutput:: 4

            tensor([1., 1., 0.], dtype=torch.float64)

        In binary and multilabel cases, the elements of `y` and `y_pred` should have 0 or 1 values. Thresholding of
        predictions can be done as below:

        .. testcode:: 5

            def thresholded_output_transform(output):
                y_pred, y = output
                y_pred = torch.round(y_pred)
                return y_pred, y

            metric = Recall(average=False, output_transform=thresholded_output_transform)
            metric.attach(default_evaluator, "recall")
            y_true = torch.Tensor([1, 0, 1, 1, 0, 1])
            y_pred = torch.Tensor([0.6, 0.2, 0.9, 0.4, 0.7, 0.65])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics["recall"])

        .. testoutput:: 5

            0.75

        In multilabel cases, average parameter should be True. However, if user would like to compute F1 metric, for
        example, average parameter should be False. This can be done as shown below:

        .. code-block:: python

            precision = Precision(average=False)
            recall = Recall(average=False)
            F1 = precision * recall * 2 / (precision + recall + 1e-20)
            F1 = MetricsLambda(lambda t: torch.mean(t).item(), F1)

    .. warning::

        In multilabel cases, if average is False, current implementation stores all input data (output and target) in
        as tensors before computing a metric. This can potentially lead to a memory error if the input data is larger
        than available RAM.
    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        average: bool = False,
        is_multilabel: bool = False,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(Recall, self).__init__(
            output_transform=output_transform, average=average, is_multilabel=is_multilabel, device=device
        )

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        self._check_shape(output)
        self._check_type(output)
        y_pred, y = output[0].detach(), output[1].detach()

        if self._type == "binary":
            y_pred = y_pred.view(-1)
            y = y.view(-1)
        elif self._type == "multiclass":
            num_classes = y_pred.size(1)
            if y.max() + 1 > num_classes:
                raise ValueError(
                    f"y_pred contains less classes than y. Number of predicted classes is {num_classes}"
                    f" and element in y has invalid class = {y.max().item() + 1}."
                )
            y = to_onehot(y.view(-1), num_classes=num_classes)
            indices = torch.argmax(y_pred, dim=1).view(-1)
            y_pred = to_onehot(indices, num_classes=num_classes)
        elif self._type == "multilabel":
            # if y, y_pred shape is (N, C, ...) -> (C, N x ...)
            num_classes = y_pred.size(1)
            y_pred = torch.transpose(y_pred, 1, 0).reshape(num_classes, -1)
            y = torch.transpose(y, 1, 0).reshape(num_classes, -1)

        # Convert from int cuda/cpu to double on self._device
        y_pred = y_pred.to(dtype=torch.float64, device=self._device)
        y = y.to(dtype=torch.float64, device=self._device)
        correct = y * y_pred
        actual_positives = y.sum(dim=0)

        if correct.sum() == 0:
            true_positives = torch.zeros_like(actual_positives)
        else:
            true_positives = correct.sum(dim=0)

        if self._type == "multilabel":
            if not self._average:
                self._true_positives = torch.cat([self._true_positives, true_positives], dim=0)  # type: torch.Tensor
                self._positives = torch.cat([self._positives, actual_positives], dim=0)  # type: torch.Tensor
            else:
                self._true_positives += torch.sum(true_positives / (actual_positives + self.eps))
                self._positives += len(actual_positives)
        else:
            self._true_positives += true_positives
            self._positives += actual_positives

        self._updated = True
