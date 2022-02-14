from typing import Callable, cast, Sequence, Union

import torch

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics.accuracy import _BaseClassification
from ignite.metrics.metric import reinit__is_reduced
from ignite.utils import to_onehot

__all__ = ["Precision"]


class _BasePrecisionRecall(_BaseClassification):
    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        average: bool = False,
        is_multilabel: bool = False,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):

        self._average = average
        self.eps = 1e-20
        self._updated = False
        super(_BasePrecisionRecall, self).__init__(
            output_transform=output_transform, is_multilabel=is_multilabel, device=device
        )

    @reinit__is_reduced
    def reset(self) -> None:
        self._true_positives = 0  # type: Union[int, torch.Tensor]
        self._positives = 0  # type: Union[int, torch.Tensor]
        self._updated = False

        if self._is_multilabel:
            init_value = 0.0 if self._average else []
            self._true_positives = torch.tensor(init_value, dtype=torch.float64, device=self._device)
            self._positives = torch.tensor(init_value, dtype=torch.float64, device=self._device)

        super(_BasePrecisionRecall, self).reset()

    def compute(self) -> Union[torch.Tensor, float]:
        if not self._updated:
            raise NotComputableError(
                f"{self.__class__.__name__} must have at least one example before it can be computed."
            )
        if not self._is_reduced:
            if not (self._type == "multilabel" and not self._average):
                self._true_positives = idist.all_reduce(self._true_positives)  # type: ignore[assignment]
                self._positives = idist.all_reduce(self._positives)  # type: ignore[assignment]
            else:
                self._true_positives = cast(torch.Tensor, idist.all_gather(self._true_positives))
                self._positives = cast(torch.Tensor, idist.all_gather(self._positives))
            self._is_reduced = True  # type: bool

        result = self._true_positives / (self._positives + self.eps)

        if self._average:
            return cast(torch.Tensor, result).mean().item()
        else:
            return result


class Precision(_BasePrecisionRecall):
    r"""Calculates precision for binary and multiclass data.

    .. math:: \text{Precision} = \frac{ TP }{ TP + FP }

    where :math:`\text{TP}` is true positives and :math:`\text{FP}` is false positives.

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

        .. include:: defaults.rst
            :start-after: :orphan:

        Binary case

        .. testcode:: 1

            metric = Precision(average=False)
            metric.attach(default_evaluator, "precision")
            y_true = torch.Tensor([1, 0, 1, 1, 0, 1])
            y_pred = torch.Tensor([1, 0, 1, 0, 1, 1])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics["precision"])

        .. testoutput:: 1

            0.75

        Multiclass case

        .. testcode:: 2

            metric = Precision(average=False)
            metric.attach(default_evaluator, "precision")
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
            print(state.metrics["precision"])

        .. testoutput:: 2

            tensor([0.5000, 1.0000, 0.3333], dtype=torch.float64)

        Precision can be computed as the unweighted average across all classes:

        .. testcode:: 3

            metric = Precision(average=True)
            metric.attach(default_evaluator, "precision")
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
            print(state.metrics["precision"])

        .. testoutput:: 3

            0.6111...

        Multilabel case, the shapes must be (batch_size, num_categories, ...)

        .. testcode:: 4

            metric = Precision(is_multilabel=True)
            metric.attach(default_evaluator, "precision")
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
            print(state.metrics["precision"])

        .. testoutput:: 4

            tensor([0.2000, 0.5000, 0.0000], dtype=torch.float64)

        In binary and multilabel cases, the elements of `y` and `y_pred` should have 0 or 1 values. Thresholding of
        predictions can be done as below:

        .. testcode:: 5

            def thresholded_output_transform(output):
                y_pred, y = output
                y_pred = torch.round(y_pred)
                return y_pred, y

            metric = Precision(average=False, output_transform=thresholded_output_transform)
            metric.attach(default_evaluator, "precision")
            y_true = torch.Tensor([1, 0, 1, 1, 0, 1])
            y_pred = torch.Tensor([0.6, 0.2, 0.9, 0.4, 0.7, 0.65])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics["precision"])

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
        super(Precision, self).__init__(
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
        all_positives = y_pred.sum(dim=0)

        if correct.sum() == 0:
            true_positives = torch.zeros_like(all_positives)
        else:
            true_positives = correct.sum(dim=0)

        if self._type == "multilabel":
            if not self._average:
                self._true_positives = torch.cat([self._true_positives, true_positives], dim=0)  # type: torch.Tensor
                self._positives = torch.cat([self._positives, all_positives], dim=0)  # type: torch.Tensor
            else:
                self._true_positives += torch.sum(true_positives / (all_positives + self.eps))
                self._positives += len(all_positives)
        else:
            self._true_positives += true_positives
            self._positives += all_positives

        self._updated = True
