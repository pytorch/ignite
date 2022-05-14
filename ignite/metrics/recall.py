from typing import Callable, Sequence, Union

import torch

from ignite.metrics.metric import reinit__is_reduced
from ignite.metrics.precision import _BasePrecisionRecall
from ignite.utils import to_onehot

__all__ = ["Recall"]


class Recall(_BasePrecisionRecall):
    r"""Calculates recall for binary, multiclass and multilabel data.

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
        average: available options are
            `False`: default option. For multicalss and multilabel
                    inputs, per class and per label metric is returned. By calling `mean()` on the
                    metric instance, the `macro` setting (which is unweighted average across
                    classes or labels) is returned.
            `micro`: for multilabel input, every label of each sample is considered itself
                    a sample then recall is computed. For binary and multiclass inputs, this is
                    equivalent with `Accuracy`, so use that metric.
            `samples`: for multilabel input, at first, recall is computed on a per sample
                    basis and then average across samples is returned. Incompatible with
                    binary and multiclass inputs.
            `Recall` does not have `weighted` option as there is in :class:`~ignite.metrics.Precision`,
            because for binary and multiclass input, weighted recall, micro recall and `Accuracy`
            are equivalent and for multilabel input, weighted recall is equivalent with the micro one.
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

            metric = Recall()
            metric.attach(default_evaluator, "recall")
            y_true = torch.tensor([1, 0, 1, 1, 0, 1])
            y_pred = torch.tensor([1, 0, 1, 0, 1, 1])
            state = default_evaluator.run([[y_pred, y_true]])
            print(f"Recall: {state.metrics["recall"]}")

        .. testoutput:: 1

            Recall: 0.75

        Multiclass case

        .. testcode:: 2

            metric = Recall()
            macro_metric = metric.mean()

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
            print(f"Recall: {state.metrics["recall"]}")
            print(f"Macro Recall: {state.metrics["macro recall"]}")

        .. testoutput:: 2

            Recall: tensor([0.5000, 0.0, 0.5000], dtype=torch.float64)
            Macro Recall: 0.3333333333333333

        Multilabel case, the shapes must be (batch_size, num_categories, ...)

        .. testcode:: 3

            metric = Recall(is_multilabel=True)
            micro_metric = Recall(is_multilabel=True, average='micro')
            macro_metric = metric.mean()
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
            print(f"Recall: {state.metrics["recall"]}")
            print(f"Micro Recall: {state.metrics["micro recall"]}")
            print(f"Macro Recall: {state.metrics["macro recall"]}")
            print(f"Samples Recall: {state.metrics["samples recall"]}")

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
            print(state.metrics["recall"])

        .. testoutput:: 4

            0.75


    .. versionchanged:: 0.5.0
            `average` parameter's semantic changed and two options were added to it.
    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        average: bool = False,
        is_multilabel: bool = False,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):

        if average not in [False, 'micro', 'samples']:
            raise ValueError(
                "Argument average should be one of values "
                "False, 'micro' and 'samples'."
            )
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

            num_labels = y_pred.size(1)
            y_pred = torch.transpose(y_pred, 1, -1).reshape(-1, num_labels)
            y = torch.transpose(y, 1, -1).reshape(-1, num_labels)

        # Convert from int cuda/cpu to double on self._device
        y_pred = y_pred.to(dtype=torch.float64, device=self._device)
        y = y.to(dtype=torch.float64, device=self._device)
        correct = y * y_pred

        if self._average == 'samples':

            actual_positives = y.sum(dim=1)
            true_positives = correct.sum(dim=1)
            self._sum_samples_metric += torch.sum(true_positives / (actual_positives + self.eps))
            self._samples_cnt += y.size(0)
        elif self._average == 'micro':

            self._positives += y.sum()
            self._true_positives += correct.sum()
        else: # _average == False

            self._positives += y.sum(dim=0)
            self._true_positives += correct.sum(dim=0)

        self._updated = True
