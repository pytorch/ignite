from typing import Callable, Optional, Sequence, Tuple, Union

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["Accuracy"]


class _BaseClassification(Metric):
    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        is_multilabel: bool = False,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        self._is_multilabel = is_multilabel
        self._type = None  # type: Optional[str]
        self._num_classes = None  # type: Optional[int]
        super(_BaseClassification, self).__init__(output_transform=output_transform, device=device)

    def reset(self) -> None:
        self._type = None
        self._num_classes = None

    def _check_shape(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output

        if not (y.ndimension() == y_pred.ndimension() or y.ndimension() + 1 == y_pred.ndimension()):
            raise ValueError(
                "y must have shape of (batch_size, ...) and y_pred must have "
                "shape of (batch_size, num_categories, ...) or (batch_size, ...), "
                "but given {} vs {}.".format(y.shape, y_pred.shape)
            )

        y_shape = y.shape
        y_pred_shape = y_pred.shape  # type: Tuple[int, ...]

        if y.ndimension() + 1 == y_pred.ndimension():
            y_pred_shape = (y_pred_shape[0],) + y_pred_shape[2:]

        if not (y_shape == y_pred_shape):
            raise ValueError("y and y_pred must have compatible shapes.")

        if self._is_multilabel and not (y.shape == y_pred.shape and y.ndimension() > 1 and y.shape[1] > 1):
            raise ValueError(
                "y and y_pred must have same shape of (batch_size, num_categories, ...) and num_categories > 1."
            )

    def _check_binary_multilabel_cases(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output

        if not torch.equal(y, y ** 2):
            raise ValueError("For binary cases, y must be comprised of 0's and 1's.")

        if not torch.equal(y_pred, y_pred ** 2):
            raise ValueError("For binary cases, y_pred must be comprised of 0's and 1's.")

    def _check_type(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output

        if y.ndimension() + 1 == y_pred.ndimension():
            num_classes = y_pred.shape[1]
            if num_classes == 1:
                update_type = "binary"
                self._check_binary_multilabel_cases((y_pred, y))
            else:
                update_type = "multiclass"
        elif y.ndimension() == y_pred.ndimension():
            self._check_binary_multilabel_cases((y_pred, y))

            if self._is_multilabel:
                update_type = "multilabel"
                num_classes = y_pred.shape[1]
            else:
                update_type = "binary"
                num_classes = 1
        else:
            raise RuntimeError(
                "Invalid shapes of y (shape={}) and y_pred (shape={}), check documentation."
                " for expected shapes of y and y_pred.".format(y.shape, y_pred.shape)
            )
        if self._type is None:
            self._type = update_type
            self._num_classes = num_classes
        else:
            if self._type != update_type:
                raise RuntimeError("Input data type has changed from {} to {}.".format(self._type, update_type))
            if self._num_classes != num_classes:
                raise ValueError(
                    "Input data number of classes has changed from {} to {}".format(self._num_classes, num_classes)
                )


class Accuracy(_BaseClassification):
    """
    Calculates the accuracy for binary, multiclass and multilabel data.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...) or (batch_size, ...).
    - `y` must be in the following shape (batch_size, ...).
    - `y` and `y_pred` must be in the following shape of (batch_size, num_categories, ...) and
      num_categories must be greater than 1 for multilabel cases.

    In binary and multilabel cases, the elements of `y` and `y_pred` should have 0 or 1 values. Thresholding of
    predictions can be done as below:

    .. code-block:: python

        def thresholded_output_transform(output):
            y_pred, y = output
            y_pred = torch.round(y_pred)
            return y_pred, y

        binary_accuracy = Accuracy(thresholded_output_transform)


    Args:
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        is_multilabel (bool, optional): flag to use in multilabel case. By default, False.
        device (str or torch.device): specifies which device updates are accumulated on. Setting the metric's
            device to be the same as your ``update`` arguments ensures the ``update`` method is non-blocking. By
            default, CPU.

    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        is_multilabel: bool = False,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(Accuracy, self).__init__(output_transform=output_transform, is_multilabel=is_multilabel, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self._num_correct = torch.tensor(0, device=self._device)
        self._num_examples = 0
        super(Accuracy, self).reset()

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        self._check_shape(output)
        self._check_type(output)
        y_pred, y = output[0].detach(), output[1].detach()

        if self._type == "binary":
            correct = torch.eq(y_pred.view(-1).to(y), y.view(-1))
        elif self._type == "multiclass":
            indices = torch.argmax(y_pred, dim=1)
            correct = torch.eq(indices, y).view(-1)
        elif self._type == "multilabel":
            # if y, y_pred shape is (N, C, ...) -> (N x ..., C)
            num_classes = y_pred.size(1)
            last_dim = y_pred.ndimension()
            y_pred = torch.transpose(y_pred, 1, last_dim - 1).reshape(-1, num_classes)
            y = torch.transpose(y, 1, last_dim - 1).reshape(-1, num_classes)
            correct = torch.all(y == y_pred.type_as(y), dim=-1)

        self._num_correct += torch.sum(correct).to(self._device)
        self._num_examples += correct.shape[0]

    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError("Accuracy must have at least one example before it can be computed.")
        return self._num_correct.item() / self._num_examples
