from typing import Callable, Sequence

import torch
from torch.types import Number

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce
from ignite.metrics.nlp.word_error_rate import _BaseErrorRate, _edit_distance

__all__ = ["CharacterErrorRate"]


class CharacterErrorRate(_BaseErrorRate):
    r"""Calculates the Character Error Rate (CER).

    CER is defined as the total number of errors (substitutions, deletions, and insertions)
    at the character level divided by the total number of characters in the reference sequence.

    .. math::
        \text{CER} = \frac{S + D + I}{N} = \frac{S + D + I}{S + D + C}

    where :math:`S` is the number of substitutions, :math:`D` is the number of deletions,
    :math:`I` is the number of insertions, :math:`C` is the number of correct characters,
    and :math:`N` is the total number of characters in the reference (:math:`N = S + D + C`).

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y_pred` must be a list of strings (predicted sequences).
    - `y` must be a list of strings (reference sequences).

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

            from ignite.metrics.nlp import CharacterErrorRate

            cer = CharacterErrorRate()

            y_pred = ["the cat sat on the mat", "hello world"]
            y = ["the cat sat on mat", "hello world"]

            cer.update((y_pred, y))
            print(cer.compute())  # Output: 0.1724... (5 insertions / 29 reference chars)

    .. versionadded:: 0.5.2
    """

    def _tokenize(self, text: str) -> Sequence[str]:
        return list(text)
