import warnings
from typing import Callable, cast, Optional, Sequence, Union

import torch

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics.accuracy import _BaseClassification
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
from ignite.utils import to_onehot

__all__ = ["Precision"]


class _BasePrecisionRecall(_BaseClassification):
    _state_dict_all_req_keys = ("_numerator", "_denominator", "_weight")

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        average: Optional[Union[bool, str]] = False,
        is_multilabel: bool = False,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        if not (average is None or isinstance(average, bool) or average in ["macro", "micro", "weighted", "samples"]):
            raise ValueError(
                "Argument average should be None or a boolean or one of values"
                " 'macro', 'micro', 'weighted' and 'samples'."
            )

        if average is True:
            self._average: Optional[Union[bool, str]] = "macro"
        else:
            self._average = average
        self.eps = 1e-20
        self._updated = False
        super(_BasePrecisionRecall, self).__init__(
            output_transform=output_transform, is_multilabel=is_multilabel, device=device
        )

    def _check_type(self, output: Sequence[torch.Tensor]) -> None:
        super()._check_type(output)

        if self._type in ["binary", "multiclass"] and self._average == "samples":
            raise ValueError("Argument average='samples' is incompatible with binary and multiclass input data.")

        y_pred, y = output
        if self._type == "multiclass" and y.dtype != torch.long:
            warnings.warn("`y` should be of dtype long when entry type is multiclass", RuntimeWarning)
        if (
            self._type == "binary"
            and self._average is not False
            and (y.dtype != torch.long or y_pred.dtype != torch.long)
        ):
            warnings.warn(
                "`y` and `y_pred` should be of dtype long when entry type is binary and average!=False", RuntimeWarning
            )

    def _prepare_output(self, output: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        y_pred, y = output[0].detach(), output[1].detach()

        if self._type == "binary" or self._type == "multiclass":
            num_classes = 2 if self._type == "binary" else y_pred.size(1)
            if self._type == "multiclass" and y.max() + 1 > num_classes:
                raise ValueError(
                    f"y_pred contains fewer classes than y. Number of classes in the prediction is {num_classes}"
                    f" and an element in y has invalid class = {y.max().item() + 1}."
                )
            y = y.view(-1)
            if self._type == "binary" and self._average is False:
                y_pred = y_pred.view(-1)
            else:
                y = to_onehot(y.long(), num_classes=num_classes)
                indices = torch.argmax(y_pred, dim=1) if self._type == "multiclass" else y_pred.long()
                y_pred = to_onehot(indices.view(-1), num_classes=num_classes)
        elif self._type == "multilabel":
            # if y, y_pred shape is (N, C, ...) -> (N * ..., C)
            num_labels = y_pred.size(1)
            y_pred = torch.transpose(y_pred, 1, -1).reshape(-1, num_labels)
            y = torch.transpose(y, 1, -1).reshape(-1, num_labels)

        # Convert from int cuda/cpu to double on self._device
        y_pred = y_pred.to(dtype=torch.float64, device=self._device)
        y = y.to(dtype=torch.float64, device=self._device)
        correct = y * y_pred

        return y_pred, y, correct

    @reinit__is_reduced
    def reset(self) -> None:
        """
        `numerator`, `denominator` and `weight` are three variables chosen to be abstract
        representatives of the ones that are measured for cases with different `average` parameters.
        `weight` is only used when `average='weighted'`. Actual value of these three variables is
        as follows.

        average='samples':
          numerator (torch.Tensor): sum of metric value for samples
          denominator (int): number of samples

        average='weighted':
          numerator (torch.Tensor): number of true positives per class/label
          denominator (torch.Tensor): number of predicted(for precision) or actual(for recall) positives per
            class/label.
          weight (torch.Tensor): number of actual positives per class

        average='micro':
          numerator (torch.Tensor): sum of number of true positives for classes/labels
          denominator (torch.Tensor): sum of number of predicted(for precision) or actual(for recall) positives for
            classes/labels.

        average='macro' or boolean or None:
          numerator (torch.Tensor): number of true positives per class/label
          denominator (torch.Tensor): number of predicted(for precision) or actual(for recall) positives per
            class/label.
        """

        self._numerator: Union[int, torch.Tensor] = 0
        self._denominator: Union[int, torch.Tensor] = 0
        self._weight: Union[int, torch.Tensor] = 0
        self._updated = False

        super(_BasePrecisionRecall, self).reset()

    @sync_all_reduce("_numerator", "_denominator")
    def compute(self) -> Union[torch.Tensor, float]:
        r"""
        Return value of the metric for `average` options `'weighted'` and `'macro'` is computed as follows.

        .. math::
            \text{Precision/Recall} = \frac{ numerator }{ denominator } \cdot weight

        wherein `weight` is the internal variable `_weight` for `'weighted'` option and :math:`1/C`
        for the `macro` one. :math:`C` is the number of classes/labels.

        Return value of the metric for `average` options `'micro'`, `'samples'`, `False` and None is as follows.

        .. math::
            \text{Precision/Recall} = \frac{ numerator }{ denominator }
        """

        if not self._updated:
            raise NotComputableError(
                f"{self.__class__.__name__} must have at least one example before it can be computed."
            )

        fraction = self._numerator / (self._denominator + (self.eps if self._average != "samples" else 0))

        if self._average == "weighted":
            _weight = idist.all_reduce(self._weight.clone())  # type: ignore[union-attr]
            sum_of_weights = cast(torch.Tensor, _weight).sum() + self.eps
            return ((fraction @ _weight) / sum_of_weights).item()  # type: ignore
        elif self._average == "micro" or self._average == "samples":
            return cast(torch.Tensor, fraction).item()
        elif self._average == "macro":
            return cast(torch.Tensor, fraction).mean().item()
        else:
            return fraction


class Precision(_BasePrecisionRecall):
    r"""Calculates precision for binary, multiclass and multilabel data.

    .. math:: \text{Precision} = \frac{ TP }{ TP + FP }

    where :math:`\text{TP}` is true positives and :math:`\text{FP}` is false positives.

    - ``update`` must receive output of the form ``(y_pred, y)``.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...) or (batch_size, ...).
    - `y` must be in the following shape (batch_size, ...).

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        average: available options are

            False
              default option. For multicalss and multilabel inputs, per class and per label
              metric is returned respectively.

            None
              like `False` option except that per class metric is returned for binary data as well.
              For compatibility with Scikit-Learn api.

            'micro'
              Metric is computed counting stats of classes/labels altogether.

              .. math::
                  \text{Micro Precision} = \frac{\sum_{k=1}^C TP_k}{\sum_{k=1}^C TP_k+FP_k}

              where :math:`C` is the number of classes/labels (2 in binary case). :math:`k` in :math:`TP_k`
              and :math:`FP_k` means that the measures are computed for class/label :math:`k` (in a one-vs-rest
              sense in multiclass case).

              For binary and multiclass inputs, this is equivalent with accuracy,
              so use :class:`~ignite.metrics.accuracy.Accuracy`.

            'samples'
              for multilabel input, at first, precision is computed on a
              per sample basis and then average across samples is returned.

              .. math::
                  \text{Sample-averaged Precision} = \frac{\sum_{n=1}^N \frac{TP_n}{TP_n+FP_n}}{N}

              where :math:`N` is the number of samples. :math:`n` in :math:`TP_n` and :math:`FP_n`
              means that the measures are computed for sample :math:`n`, across labels.

              Incompatible with binary and multiclass inputs.

            'weighted'
              like macro precision but considers class/label imbalance. for binary and multiclass
              input, it computes metric for each class then returns average of them weighted by
              support of classes (number of actual samples in each class). For multilabel input,
              it computes precision for each label then returns average of them weighted by support
              of labels (number of actual positive samples in each label).

              .. math::
                  Precision_k = \frac{TP_k}{TP_k+FP_k}

              .. math::
                  \text{Weighted Precision} = \frac{\sum_{k=1}^C P_k * Precision_k}{N}

              where :math:`C` is the number of classes (2 in binary case). :math:`P_k` is the number
              of samples belonged to class :math:`k` in binary and multiclass case, and the number of
              positive samples belonged to label :math:`k` in multilabel case.

            macro
              computes macro precision which is unweighted average of metric computed across
              classes/labels.

              .. math::
                  \text{Macro Precision} = \frac{\sum_{k=1}^C Precision_k}{C}

              where :math:`C` is the number of classes (2 in binary case).

            True
              like macro option. For backward compatibility.
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
            two_class_metric = Precision(average=None) # Returns precision for both classes
            metric.attach(default_evaluator, "precision")
            weighted_metric.attach(default_evaluator, "weighted precision")
            two_class_metric.attach(default_evaluator, "both classes precision")
            y_true = torch.tensor([1, 0, 1, 1, 0, 1])
            y_pred = torch.tensor([1, 0, 1, 0, 1, 1])
            state = default_evaluator.run([[y_pred, y_true]])
            print(f"Precision: {state.metrics['precision']}")
            print(f"Weighted Precision: {state.metrics['weighted precision']}")
            print(f"Precision for class 0 and class 1: {state.metrics['both classes precision']}")

        .. testoutput:: 1

            Precision: 0.75
            Weighted Precision: 0.6666666666666666
            Precision for class 0 and class 1: tensor([0.5000, 0.7500], dtype=torch.float64)

        Multiclass case

        .. testcode:: 2

            metric = Precision()
            macro_metric = Precision(average=True)
            weighted_metric = Precision(average='weighted')

            metric.attach(default_evaluator, "precision")
            macro_metric.attach(default_evaluator, "macro precision")
            weighted_metric.attach(default_evaluator, "weighted precision")

            y_true = torch.tensor([2, 0, 2, 1, 0])
            y_pred = torch.tensor([
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
            macro_metric = Precision(is_multilabel=True, average=True)
            weighted_metric = Precision(is_multilabel=True, average='weighted')
            samples_metric = Precision(is_multilabel=True, average='samples')

            metric.attach(default_evaluator, "precision")
            micro_metric.attach(default_evaluator, "micro precision")
            macro_metric.attach(default_evaluator, "macro precision")
            weighted_metric.attach(default_evaluator, "weighted precision")
            samples_metric.attach(default_evaluator, "samples precision")

            y_true = torch.tensor([
                [0, 0, 1],
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 1],
            ])
            y_pred = torch.tensor([
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
            y_true = torch.tensor([1, 0, 1, 1, 0, 1])
            y_pred = torch.tensor([0.6, 0.2, 0.9, 0.4, 0.7, 0.65])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics["precision"])

        .. testoutput:: 4

            0.75

    .. versionchanged:: 0.4.10
            Some new options were added to `average` parameter.
    """

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        r"""
        Update the metric state using prediction and target.

        Args:
            output: a binary tuple of tensors (y_pred, y) whose shapes follow the table below. N stands for the batch
                dimension, `...` for possible additional dimensions and C for class dimension.

                .. list-table::
                    :widths: 20 10 10 10
                    :header-rows: 1

                    * - Output member\\Data type
                      - Binary
                      - Multiclass
                      - Multilabel
                    * - y_pred
                      - (N, ...)
                      - (N, C, ...)
                      - (N, C, ...)
                    * - y
                      - (N, ...)
                      - (N, ...)
                      - (N, C, ...)

                For binary and multilabel data, both y and y_pred should consist of 0's and 1's, but for multiclass
                data, y_pred and y should consist of probabilities and integers respectively.
        """
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
