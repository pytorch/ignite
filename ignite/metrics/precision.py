from typing import Callable, Sequence, Union

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
        average: Union[bool, str] = False,
        is_multilabel: bool = False,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):

        self._average = average
        self.eps = 1e-20
        self._updated = False
        super(_BasePrecisionRecall, self).__init__(
            output_transform=output_transform, is_multilabel=is_multilabel, device=device
        )

    def _check_type(self, output: Sequence[torch.Tensor]) -> None:
        super()._check_type(output)

        if self._average == "micro" and self._type in ["binary", "multiclass"]:
            raise ValueError(
                "Precision and Recall with average='micro' and binary or multiclass "
                "input data are equivalent with Accuracy, so use this metric."
            )
        if self._type in ["binary", "multiclass"] and self._average == "samples":
            raise ValueError("Argument average='samples' is incompatible with binary and multiclass input data.")

    @reinit__is_reduced
    def reset(self) -> None:
        if self._average == "samples":
            self._sum_samples_metric = 0  # type: Union[int, torch.Tensor]
            self._samples_cnt = 0  # type: int
        else:
            self._true_positives = 0  # type: Union[int, torch.Tensor]
            self._positives = 0  # type: Union[int, torch.Tensor]

        if self._average == "weighted":
            self._actual_positives = 0  # type: Union[int, torch.Tensor]
        self._updated = False

        super(_BasePrecisionRecall, self).reset()

    def compute(self) -> Union[torch.Tensor, float]:
        if not self._updated:
            raise NotComputableError(
                f"{self.__class__.__name__} must have at least one example before it can be computed."
            )
        if not self._is_reduced:
            if self._average == "samples":
                self._sum_samples_metric = idist.all_reduce(self._sum_samples_metric)  # type: ignore[assignment]
                self._samples_cnt = idist.all_reduce(self._samples_cnt)  # type: ignore[assignment]
            else:
                self._true_positives = idist.all_reduce(self._true_positives)  # type: ignore[assignment]
                self._positives = idist.all_reduce(self._positives)  # type: ignore[assignment]
            if self._average == "weighted":
                self._actual_positives = idist.all_reduce(self._actual_positives)  # type: ignore[assignment]
            self._is_reduced = True  # type: bool

        if self._average == "samples":
            return (self._sum_samples_metric / self._samples_cnt).item()  # type: ignore

        result = self._true_positives / (self._positives + self.eps)
        if self._average == "weighted":
            denominator = self._actual_positives.sum() + self.eps  # type: ignore
            return ((result @ self._actual_positives) / denominator).item()  # type: ignore
        elif self._average == "micro":
            return result.item()  # type: ignore
        else:
            return result


class Precision(_BasePrecisionRecall):
    r"""Calculates precision for binary, multiclass and multilabel data.

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
        average: available options are

            False
              default option. For multicalss and multilabel
              inputs, per class and per label metric is returned. By calling `mean()` on the
              metric instance, the `macro` setting (which is unweighted average across
              classes or labels) is returned.

            'micro'
              for multilabel input, every label of each sample is considered itself
              a sample then precision is computed. For binary and multiclass
              inputs, this is equivalent with `Accuracy`, so use that metric.

            'samples'
              for multilabel input, at first, precision is computed
              on a per sample basis and then average across samples is
              returned. Incompatible with binary and multiclass inputs.

            'weighted'
              for binary and multiclass input, it computes metric for each class then
              returns average of them weighted by support of classes (number of actual samples
              in each class). For multilabel input, it computes precision for each label then
              returns average of them weighted by support of labels (number of actual positive
              samples in each label).
        is_multilabel: flag to use in multilabel case. By default, value is False.
        device: specifies which device updates are accumulated on. Setting the metric's
            device to be the same as your ``update`` arguments ensures the ``update`` method is non-blocking. By
            default, CPU.

    Examples:

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        Binary case. In binary and multilabel cases, the elements of
        `y` and `y_pred` should have 0 or 1 values.

        .. testcode:: 1

            metric = Precision()
            weighted_metric = Precision(average='weighted')
            metric.attach(default_evaluator, "precision")
            weighted_metric.attach(default_evaluator, "weighted precision")
            y_true = torch.Tensor([1, 0, 1, 1, 0, 1]).long()
            y_pred = torch.Tensor([1, 0, 1, 0, 1, 1])
            state = default_evaluator.run([[y_pred, y_true]])
            print(f"Precision: {state.metrics['precision']}")
            print(f"Weighted Precision: {state.metrics['weighted precision']}")

        .. testoutput:: 1

            Precision: 0.75
            Weighted Precision: 0.6666666666666666

        Multiclass case

        .. testcode:: 2

            metric = Precision()
            macro_metric = metric.mean()
            weighted_metric = Precision(average='weighted')

            metric.attach(default_evaluator, "precision")
            macro_metric.attach(default_evaluator, "macro precision")
            weighted_metric.attach(default_evaluator, "weighted precision")

            y_true = torch.Tensor([2, 0, 2, 1, 0]).long()
            y_pred = torch.Tensor([
                [0.0266, 0.1719, 0.3055],
                [0.6886, 0.3978, 0.8176],
                [0.9230, 0.0197, 0.8395],
                [0.1785, 0.2670, 0.6084],
                [0.8448, 0.7177, 0.7288]
            ])
            state = default_evaluator.run([[y_pred, y_true]])
            print(f"Precision: {state.metrics['precision']}")
            print(f"Macro Precision: {state.metrics['macro precision']}")
            print(f"Weighted Precision: {state.metrics['weighted precision']}")

        .. testoutput:: 2

            Precision: tensor([0.5000, 0.0000, 0.3333], dtype=torch.float64)
            Macro Precision: 0.27777777777777773
            Weighted Precision: 0.3333333333333333

        Multilabel case, the shapes must be (batch_size, num_labels, ...)

        .. testcode:: 3

            metric = Precision(is_multilabel=True)
            micro_metric = Precision(is_multilabel=True, average='micro')
            macro_metric = metric.mean()
            weighted_metric = Precision(is_multilabel=True, average='weighted')
            samples_metric = Precision(is_multilabel=True, average='samples')

            metric.attach(default_evaluator, "precision")
            micro_metric.attach(default_evaluator, "micro precision")
            macro_metric.attach(default_evaluator, "macro precision")
            weighted_metric.attach(default_evaluator, "weighted precision")
            samples_metric.attach(default_evaluator, "samples precision")

            y_true = torch.Tensor([
                [0, 0, 1],
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 1],
            ])
            y_pred = torch.Tensor([
                [1, 1, 0],
                [1, 0, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
            ])
            state = default_evaluator.run([[y_pred, y_true]])
            print(f"Precision: {state.metrics['precision']}")
            print(f"Micro Precision: {state.metrics['micro precision']}")
            print(f"Macro Precision: {state.metrics['macro precision']}")
            print(f"Weighted Precision: {state.metrics['weighted precision']}")
            print(f"Samples Precision: {state.metrics['samples precision']}")

        .. testoutput:: 3

            Precision: tensor([0.2000, 0.5000, 0.0000], dtype=torch.float64)
            Micro Precision: 0.2222222222222222
            Macro Precision: 0.2333333333333333
            Weighted Precision: 0.175
            Samples Precision: 0.2

        Thresholding of predictions can be done as below:

        .. testcode:: 4

            def thresholded_output_transform(output):
                y_pred, y = output
                y_pred = torch.round(y_pred)
                return y_pred, y

            metric = Precision(output_transform=thresholded_output_transform)
            metric.attach(default_evaluator, "precision")
            y_true = torch.Tensor([1, 0, 1, 1, 0, 1])
            y_pred = torch.Tensor([0.6, 0.2, 0.9, 0.4, 0.7, 0.65])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics["precision"])

        .. testoutput:: 4

            0.75


    .. versionchanged:: 0.5.0
            `average` parameter's semantic changed and three options were added to it.
    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        average: Union[bool, str] = False,
        is_multilabel: bool = False,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):

        if average not in [False, "micro", "weighted", "samples"]:
            raise ValueError("Argument average should be one of values " "False, 'micro', 'weighted' and 'samples'.")
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

            if self._average == "weighted":
                y = to_onehot(y, num_classes=2)
                y_pred = to_onehot(y_pred.long(), num_classes=2)
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

            num_labels = y_pred.size(1)
            y_pred = torch.transpose(y_pred, 1, -1).reshape(-1, num_labels)
            y = torch.transpose(y, 1, -1).reshape(-1, num_labels)

        # Convert from int cuda/cpu to double on self._device
        y_pred = y_pred.to(dtype=torch.float64, device=self._device)
        y = y.to(dtype=torch.float64, device=self._device)
        correct = y * y_pred

        if self._average == "samples":

            all_positives = y_pred.sum(dim=1)
            true_positives = correct.sum(dim=1)
            self._sum_samples_metric += torch.sum(true_positives / (all_positives + self.eps))
            self._samples_cnt += y.size(0)
        elif self._average == "micro":

            self._positives += y_pred.sum()
            self._true_positives += correct.sum()
        else:  # _average in [False, 'weighted']

            self._positives += y_pred.sum(dim=0)
            self._true_positives += correct.sum(dim=0)

            if self._average == "weighted":
                self._actual_positives += y.sum(dim=0)

        self._updated = True
