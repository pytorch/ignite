from typing import Any, Callable, Sequence

import torch
from torch.types import Number

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["WordErrorRate"]


def _edit_distance(ref: Sequence[Any], pred: Sequence[Any]) -> int:
    """Computes the Levenshtein distance between two sequences."""
    n = len(ref)
    m = len(pred)

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
            if ref[i - 1] == pred[j - 1]:
                dp[j] = prev_diag
            else:
                dp[j] = min(dp[j - 1], dp[j], prev_diag) + 1
            prev_diag = temp

    return dp[m]


class _BaseErrorRate(Metric):
    """
    Base class for error rate metrics based on Levenshtein distance (edit distance).
    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        device: str | torch.device = torch.device("cpu"),
        skip_unrolling: bool = False,
    ):
        super().__init__(output_transform=output_transform, device=device, skip_unrolling=skip_unrolling)

    @reinit__is_reduced
    def reset(self) -> None:
        self._num_errors = torch.tensor(0.0, device=self._device)
        self._num_refs = torch.tensor(0.0, device=self._device)
        super().reset()

    def _tokenize(self, text: str) -> Sequence[Any]:
        raise NotImplementedError

    @reinit__is_reduced
    def update(self, output: Sequence[str]) -> None:
        y_pred, y = output[0], output[1]

        if isinstance(y_pred, str) and isinstance(y, str):
            y_pred = [y_pred]
            y = [y]

        if len(y_pred) != len(y):
            raise ValueError(
                f"y_pred and y must have the same length. Got y_pred of length {len(y_pred)} and y of length {len(y)}."
            )

        errors = 0.0
        refs = 0.0
        for p, r in zip(y_pred, y):
            p_tokens = self._tokenize(p)
            r_tokens = self._tokenize(r)

            errors += _edit_distance(r_tokens, p_tokens)
            refs += len(r_tokens)

        self._num_errors += torch.tensor(errors, device=self._device)
        self._num_refs += torch.tensor(refs, device=self._device)

    @sync_all_reduce("_num_errors", "_num_refs")
    def compute(self) -> Number:
        if self._num_refs == 0:
            raise NotComputableError("Error rate must have at least one valid reference sequence to be computed.")
        return (self._num_errors / self._num_refs).item()


class WordErrorRate(_BaseErrorRate):
    r"""Calculates the Word Error Rate (WER).

    WER is defined as the total number of errors (substitutions, deletions, and insertions)
    at the word level divided by the total number of words in the reference sequence.

    .. math::
        \text{WER} = \frac{S + D + I}{N} = \frac{S + D + I}{S + D + C}

    where :math:`S` is the number of substitutions, :math:`D` is the number of deletions,
    :math:`I` is the number of insertions, :math:`C` is the number of correct words,
    and :math:`N` is the total number of words in the reference (:math:`N = S + D + C`).

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y_pred` must be a list of strings (predicted sentences).
    - `y` must be a list of strings (reference sentences).

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        device: specifies which device updates are accumulated on. Setting the metric's
            device to be the same as your ``update`` arguments ensures the ``update`` method is non-blocking. By
            default, CPU.
        skip_unrolling: specifies whether output should be unrolled before being fed to update method. Should be
            true for multi-output model, for example, if ``y_pred`` contains multi-ouput as ``(y_pred_a, y_pred_b)``
            Alternatively, ``output_transform`` can be used to handle this.

    Examples:
        .. code-block:: python

            from ignite.metrics.nlp import WordErrorRate

            wer = WordErrorRate()
            y_pred = ["the cat sat on the mat", "hello world"]
            y = ["the cat sat on mat", "hello world"]
            wer.update((y_pred, y))
            print(wer.compute()) # Output: 0.2 (1 insertion / 5 reference words)
    """

    def _tokenize(self, text: str) -> Sequence[str]:
        return text.split()
