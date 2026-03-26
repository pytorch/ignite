import torch
from collections.abc import Callable, Sequence

from ignite.exceptions import NotComputableError
from ignite.metrics.accuracy import _BaseClassification
from ignite.metrics.metric import sync_all_reduce
from ignite.metrics.fairness.base import SubgroupDifference

__all__ = ["DemographicParityDifference", "SelectionRate"]


class SelectionRate(_BaseClassification):
    """Calculates the selection rate (rate of positive predictions).

    - ``update`` must receive output of the form ``(y_pred, y)``.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...) or (batch_size, ...).
    - `y` must be in the following shape (batch_size, ...).

    Args:
        output_transform: a callable that is used to transform the engine output.
        is_multilabel: if True, multilabel selection rate is calculated. By default, False.
        device: specifies the computation device.
        skip_unrolling: specifies whether output should be unrolled before being fed to update method.

    Examples:

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            metric = SelectionRate()
            metric.attach(default_evaluator, 'selection_rate')
            y_pred = torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.9, 0.1], [0.9, 0.1]])
            y_true = torch.tensor([1, 1, 0, 0])  # ignored
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics['selection_rate'])

        .. testoutput::

            tensor([0.5000, 0.5000])

    .. versionadded:: 0.6.0
    """

    _state_dict_all_req_keys = ("_num_positives", "_num_examples")

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        is_multilabel: bool = False,
        device: str | torch.device = torch.device("cpu"),
        skip_unrolling: bool = False,
    ):
        self._num_positives: torch.Tensor = torch.tensor(0.0, device=device)
        super().__init__(
            output_transform=output_transform, is_multilabel=is_multilabel, device=device, skip_unrolling=skip_unrolling
        )

    def reset(self) -> None:
        self._num_positives = torch.tensor(0.0, device=self._device)
        self._num_examples = 0
        super().reset()

    def update(self, output: Sequence[torch.Tensor]) -> None:
        self._check_shape(output)
        self._check_type(output)
        y_pred = output[0].detach()

        if self._type == "binary":
            positives = torch.bincount(y_pred.view(-1).to(torch.long), minlength=2).float()
            total = y_pred.numel()
        elif self._type == "multiclass":
            if self._num_classes is None:
                raise RuntimeError("num_classes must be set for multiclass data.")
            predicted_classes = torch.argmax(y_pred, dim=1)
            positives = torch.bincount(predicted_classes.view(-1), minlength=self._num_classes).float()
            total = predicted_classes.numel()
        elif self._type == "multilabel":
            if self._num_classes is None:
                raise RuntimeError("num_classes must be set for multilabel data.")
            num_classes: int = self._num_classes
            positives = y_pred.movedim(1, -1).reshape(-1, num_classes).sum(dim=0).float()
            total = int(y_pred.numel() / num_classes)
        else:
            raise ValueError(f"Unexpected type: {self._type}")

        if self._num_positives.ndim == 0:
            self._num_positives = positives.to(self._device)
        else:
            self._num_positives = self._num_positives + positives.to(self._device)
        self._num_examples += total

    @sync_all_reduce("_num_examples", "_num_positives")
    def compute(self) -> torch.Tensor:
        """Computes the selection rate.

        Returns:
            The selection rate for each category/label.
        """
        if self._num_examples == 0 or self._num_positives.ndim == 0:
            raise NotComputableError("SelectionRate must have at least one example before it can be computed.")
        return self._num_positives / self._num_examples


class DemographicParityDifference(SubgroupDifference):
    r"""Calculates the Demographic Parity Difference.

    This metric computes the selection rate (the rate of positive predictions) for each unique
    subgroup in the dataset and returns the maximum difference in selection rates between any
    two subgroups.

    A lower value indicates that the model predicts the positive outcome at roughly the same rate
    across all subgroups, a standard definition of fairness. This metric is referred to as
    *Group Fairness / Statistical Parity* in the fairness literature.

    - ``update`` must receive output of the form ``(y_pred, y, group_labels)`` or
      ``{'y_pred': y_pred, 'y': y, 'group_labels': group_labels}``.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...) or (batch_size, ...).
    - `y` must be in the following shape (batch_size, ...).
    - `group_labels` must be a 1D tensor of shape (batch_size,) containing discrete labels.

    Args:
        groups: a sequence of unique group identifiers.
        is_multilabel: if True, multilabel selection rate is calculated. By default, False.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.

    Examples:
        To use with ``Engine`` and ``process_function``, simply attach the metric instance to the engine.
        The output of the engine's ``process_function`` needs to be in the format of
        ``(y_pred, y, group_labels)``.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            metric = DemographicParityDifference(groups=[0, 1])
            metric.attach(default_evaluator, 'demographic_parity_diff')

            # Predictions for 4 items:
            # Items 1 and 2 are predicted as class 1 (index 1 has highest prob)
            # Items 3 and 4 are predicted as class 0 (index 0 has highest prob)
            y_pred = torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.9, 0.1], [0.9, 0.1]])

            # Targets (Not actually used for parity, but required by API)
            y_true = torch.tensor([1, 1, 0, 0])

            # Subgroups
            group_labels = torch.tensor([0, 0, 1, 1])

            # Subgroup 0: 2 positive predictions / 2 total = 1.0 selection rate
            # Subgroup 1: 0 positive predictions / 2 total = 0.0 selection rate

            state = default_evaluator.run([[y_pred, y_true, group_labels]])
            print(state.metrics['demographic_parity_diff'])

        .. testoutput::

            1.0

    .. versionadded:: 0.6.0

    References:
        - Verma & Rubin, `Fairness Definitions Explained
          <https://fairware.cs.umass.edu/papers/Verma.pdf>`_, 2018.
    """

    def __init__(
        self,
        groups: Sequence[int],
        is_multilabel: bool = False,
        output_transform: Callable = lambda x: x,
        device: torch.device | str = torch.device("cpu"),
    ) -> None:
        sr = SelectionRate(is_multilabel=is_multilabel, device=device)
        super().__init__(base_metric=sr, groups=groups, output_transform=output_transform, device=device)
