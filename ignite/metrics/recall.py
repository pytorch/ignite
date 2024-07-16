from typing import Sequence

import torch

from ignite.metrics.metric import reinit__is_reduced
from ignite.metrics.precision import _BasePrecisionRecall

__all__ = ["Recall"]


class Recall(_BasePrecisionRecall):
    r"""Calculates recall for binary, multiclass and multilabel data.

    .. math:: \text{Recall} = \frac{ TP }{ TP + FN }

    where :math:`\text{TP}` is true positives and :math:`\text{FN}` is false negatives.

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
                  \text{Micro Recall} = \frac{\sum_{k=1}^C TP_k}{\sum_{k=1}^C TP_k+FN_k}

              where :math:`C` is the number of classes/labels (2 in binary case). :math:`k` in
              :math:`TP_k` and :math:`FN_k`means that the measures are computed for class/label :math:`k` (in
              a one-vs-rest sense in multiclass case).

              For binary and multiclass inputs, this is equivalent with accuracy,
              so use :class:`~ignite.metrics.accuracy.Accuracy`.

            'samples'
              for multilabel input, at first, recall is computed on a
              per sample basis and then average across samples is returned.

              .. math::
                  \text{Sample-averaged Recall} = \frac{\sum_{n=1}^N \frac{TP_n}{TP_n+FN_n}}{N}

              where :math:`N` is the number of samples. :math:`n` in :math:`TP_n` and :math:`FN_n`
              means that the measures are computed for sample :math:`n`, across labels.

              Incompatible with binary and multiclass inputs.

            'weighted'
              like macro recall but considers class/label imbalance. For binary and multiclass
              input, it computes metric for each class then returns average of them weighted by
              support of classes (number of actual samples in each class). For multilabel input,
              it computes recall for each label then returns average of them weighted by support
              of labels (number of actual positive samples in each label).

              .. math::
                  Recall_k = \frac{TP_k}{TP_k+FN_k}

              .. math::
                  \text{Weighted Recall} = \frac{\sum_{k=1}^C P_k * Recall_k}{N}

              where :math:`C` is the number of classes (2 in binary case). :math:`P_k` is the number
              of samples belonged to class :math:`k` in binary and multiclass case, and the number of
              positive samples belonged to label :math:`k` in multilabel case.

              Note that for binary and multiclass data, weighted recall is equivalent
              with accuracy, so use :class:`~ignite.metrics.accuracy.Accuracy`.

            macro
              computes macro recall which is unweighted average of metric computed across
              classes or labels.

              .. math::
                  \text{Macro Recall} = \frac{\sum_{k=1}^C Recall_k}{C}

              where :math:`C` is the number of classes (2 in binary case).

            True
              like macro option. For backward compatibility.
        is_multilabel: flag to use in multilabel case. By default, value is False.
        device: specifies which device updates are accumulated on. Setting the metric's
            device to be the same as your ``update`` arguments ensures the ``update`` method is non-blocking. By
            default, CPU.
        skip_unrolling: specifies whether output should be unrolled before being fed to update method. Should be
            true for multi-output model, for example, if ``y_pred`` contains multi-ouput as ``(y_pred_a, y_pred_b)``
            Alternatively, ``output_transform`` can be used to handle this.

    Examples:

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        Binary case. In binary and multilabel cases, the elements of
        `y` and `y_pred` should have 0 or 1 values.

        .. testcode:: 1

            metric = Recall()
            two_class_metric = Recall(average=None) # Returns recall for both classes
            metric.attach(default_evaluator, "recall")
            two_class_metric.attach(default_evaluator, "both classes recall")
            y_true = torch.tensor([1, 0, 1, 1, 0, 1])
            y_pred = torch.tensor([1, 0, 1, 0, 1, 1])
            state = default_evaluator.run([[y_pred, y_true]])
            print(f"Recall: {state.metrics['recall']}")
            print(f"Recall for class 0 and class 1: {state.metrics['both classes recall']}")

        .. testoutput:: 1

            Recall: 0.75
            Recall for class 0 and class 1: tensor([0.5000, 0.7500], dtype=torch.float64)

        Multiclass case

        .. testcode:: 2

            metric = Recall()
            macro_metric = Recall(average=True)

            metric.attach(default_evaluator, "recall")
            macro_metric.attach(default_evaluator, "macro recall")

            y_true = torch.tensor([2, 0, 2, 1, 0])
            y_pred = torch.tensor([
                [0.0266, 0.1719, 0.3055],
                [0.6886, 0.3978, 0.8176],
                [0.9230, 0.0197, 0.8395],
                [0.1785, 0.2670, 0.6084],
                [0.8448, 0.7177, 0.7288]
            ])
            state = default_evaluator.run([[y_pred, y_true]])
            print(f"Recall: {state.metrics['recall']}")
            print(f"Macro Recall: {state.metrics['macro recall']}")

        .. testoutput:: 2

            Recall: tensor([0.5000, 0.0000, 0.5000], dtype=torch.float64)
            Macro Recall: 0.3333333333333333

        Multilabel case, the shapes must be (batch_size, num_categories, ...)

        .. testcode:: 3

            metric = Recall(is_multilabel=True)
            micro_metric = Recall(is_multilabel=True, average='micro')
            macro_metric = Recall(is_multilabel=True, average=True)
            samples_metric = Recall(is_multilabel=True, average='samples')

            metric.attach(default_evaluator, "recall")
            micro_metric.attach(default_evaluator, "micro recall")
            macro_metric.attach(default_evaluator, "macro recall")
            samples_metric.attach(default_evaluator, "samples recall")

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
            print(f"Recall: {state.metrics['recall']}")
            print(f"Micro Recall: {state.metrics['micro recall']}")
            print(f"Macro Recall: {state.metrics['macro recall']}")
            print(f"Samples Recall: {state.metrics['samples recall']}")

        .. testoutput:: 3

            Recall: tensor([1., 1., 0.], dtype=torch.float64)
            Micro Recall: 0.5
            Macro Recall: 0.6666666666666666
            Samples Recall: 0.3

        Thresholding of predictions can be done as below:

        .. testcode:: 4

            def thresholded_output_transform(output):
                y_pred, y = output
                y_pred = torch.round(y_pred)
                return y_pred, y

            metric = Recall(output_transform=thresholded_output_transform)
            metric.attach(default_evaluator, "recall")
            y_true = torch.tensor([1, 0, 1, 1, 0, 1])
            y_pred = torch.tensor([0.6, 0.2, 0.9, 0.4, 0.7, 0.65])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics['recall'])

        .. testoutput:: 4

            0.75

    .. versionchanged:: 0.4.10
            Some new options were added to `average` parameter.

    .. versionchanged:: 0.5.1
        ``skip_unrolling`` argument is added.
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
        else:  # _average in [False, 'macro', 'weighted']
            self._denominator += y.sum(dim=0)
            self._numerator += correct.sum(dim=0)

            if self._average == "weighted":
                self._weight += y.sum(dim=0)

        self._updated = True
