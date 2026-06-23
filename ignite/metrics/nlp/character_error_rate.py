from typing import Any, Callable, Sequence, Union

import torch
from torch.types import Number

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["CharacterErrorRate"]


def _edit_distance(ref: Sequence[Any], pred: Sequence[Any]) -> int:
    """Computes the Levenshtein distance between two sequences."""
    n, m = len(ref), len(pred)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev_diag = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            dp[j] = prev_diag if ref[i - 1] == pred[j - 1] else min(dp[j - 1], dp[j], prev_diag) + 1
            prev_diag = temp
    return dp[m]


class CharacterErrorRate(Metric):
    r"""Calculates the Character Error Rate (CER).

    CER is defined as the total number of errors (substitutions, deletions, and insertions)
    at the character level divided by the total number of characters in the reference sequence.

    .. math::
        \text{CER} = \frac{S + D + I}{N} = \frac{S + D + I}{S + D + C}

    where :math:`S` is the number of substitutions, :math:`D` is the number of deletions,
    :math:`I` is the number of insertions, :math:`C` is the number of correct characters,
    and :math:`N` is the total number of characters in the reference (:math:`N = S + D + C`).

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y_pred` must be either a ``str`` or a list of ``str`` (predicted sequences).
    - `y` must be either a ``str`` or a list of ``str`` (reference sequences).
    - When both inputs are plain ``str``, they are treated as a single-element batch.

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric.
        device: specifies which device updates are accumulated on. By default, CPU.
        skip_unrolling: specifies whether output should be unrolled before being fed to update method.

    Examples:
        .. code-block:: python

            from ignite.metrics.nlp import CharacterErrorRate

            cer = CharacterErrorRate()

            y_pred = ["the cat sat on the mat", "hello world"]
            y = ["the cat sat on mat", "hello world"]

            cer.update((y_pred, y))
            print(cer.compute())  # Output: 0.1724... (5 insertions / 29 reference chars)

    .. versionadded:: 0.5.2
    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
        skip_unrolling: bool = False,
    ):
        super().__init__(output_transform=output_transform, device=device, skip_unrolling=skip_unrolling)

    @reinit__is_reduced
    def reset(self) -> None:
        self._num_errors = torch.tensor(0.0, device=self._device)
        self._num_refs = torch.tensor(0.0, device=self._device)
        super().reset()

    @reinit__is_reduced
    def update(self, output: Sequence[str]) -> None:
        y_pred, y = output[0], output[1]
        # Handle single-string input — treat as a one-element batch
        if isinstance(y_pred, str) and isinstance(y, str):
            y_pred = [y_pred]
            y = [y]
        if not isinstance(y_pred, (str, list)) or not isinstance(y, (str, list)):
            raise TypeError(
                f"All inputs should be either str or list[str], "
                f"got y_pred: {type(y_pred)} and y: {type(y)}"
            )
        if type(y_pred) is not type(y):
            raise TypeError(
                f"y_pred and y must be the same type, "
                f"got y_pred: {type(y_pred)} and y: {type(y)}"
            )
        if isinstance(y_pred, list) and not all(isinstance(p, str) for p in y_pred):
            raise TypeError("All elements of y_pred must be strings.")
        if isinstance(y, list) and not all(isinstance(r, str) for r in y):
            raise TypeError("All elements of y must be strings.")
        if len(y_pred) != len(y):
            raise ValueError(
                f"y_pred and y must have the same length. Got y_pred of length {len(y_pred)} and y of length {len(y)}."
            )
        errors = 0.0
        refs = 0.0
        for p, r in zip(y_pred, y):
            # Strings are already character sequences — no need for explicit list()
            errors += _edit_distance(r, p)
            refs += len(r)
        self._num_errors += torch.tensor(errors, device=self._device)
        self._num_refs += torch.tensor(refs, device=self._device)

    @sync_all_reduce("_num_errors", "_num_refs")
    def compute(self) -> Number:
        if self._num_refs == 0:
            raise NotComputableError("CharacterErrorRate must have at least one valid reference sequence to be computed.")
        return (self._num_errors / self._num_refs).item()
