from collections.abc import Callable

import torch
import torch.nn.functional as F

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["Perplexity"]


class Perplexity(Metric):
    r"""Calculates the `Perplexity <https://en.wikipedia.org/wiki/Perplexity>`_ of a language model.

    .. math::
        \text{PPL}(W) = \exp \left( -\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_1, \ldots, w_{i-1}) \right)

    where :math:`N` is the total number of tokens and :math:`P(w_i | w_1, \ldots, w_{i-1})` is the
    conditional probability of token :math:`w_i` given the preceding tokens.

    Perplexity is computed as :math:`\exp(\text{NLL})` where NLL is the mean negative log-likelihood
    over all tokens. Lower perplexity indicates a better language model.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y_pred` must be a floating-point tensor of shape ``(batch_size, vocab_size, seq_len)``
      containing the unnormalized log-probabilities (logits).
    - `y` must be a long tensor of shape ``(batch_size, seq_len)`` containing the target token indices.

    Note:
        Perplexity uses token-weighted accumulation rather than batch-average to avoid bias
        towards shorter sequences. The total NLL and total token count are accumulated across
        all batches, and the final perplexity is computed as ``exp(total_nll / total_tokens)``.

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            By default, metrics require the output as ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.

    Examples:

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. testcode::

            from ignite.metrics.nlp import Perplexity
            import torch

            ppl = Perplexity()

            # batch_size=2, vocab_size=5, seq_len=3
            y_pred = torch.randn(2, 5, 3)
            y = torch.randint(0, 5, (2, 3))

            ppl.update((y_pred, y))

            print(type(ppl.compute()))

        .. testoutput::

            <class 'float'>

    .. versionadded:: 0.5.5
    """

    _state_dict_all_req_keys = ("_sum_of_nll", "_num_tokens")

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        device: str | torch.device = torch.device("cpu"),
        ignore_index: int = -100,
    ):
        self._ignore_index = ignore_index
        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_of_nll = torch.tensor(0.0, device=self._device)
        self._num_tokens = torch.tensor(0, device=self._device)

    @reinit__is_reduced
    def update(self, output: tuple[torch.Tensor, torch.Tensor]) -> None:
        y_pred, y = output
        y_pred = y_pred.detach()
        y = y.detach()

        if y_pred.ndim < 2:
            raise ValueError(f"y_pred must be at least 2-dimensional (got shape: {y_pred.shape})")

        if y.ndim < 1:
            raise ValueError(f"y must be at least 1-dimensional (got shape: {y.shape})")

        if y_pred.ndim != y.ndim + 1:
            raise ValueError(
                f"y_pred must have exactly one more dimension than y "
                f"(got y_pred={y_pred.ndim}D and y={y.ndim}D)."
            )

        if y_pred.shape[0] != y.shape[0] or y_pred.shape[2:] != y.shape[1:]:
            raise ValueError(
                f"y_pred and y have incompatible shapes: "
                f"y_pred={y_pred.shape}, y={y.shape} "
                f"(expected y_pred[0] == y[0] and y_pred[2:] == y[1:])."
            )

        nll = F.cross_entropy(y_pred, y, reduction="sum", ignore_index=self._ignore_index)
        self._sum_of_nll += nll.to(self._device)
        self._num_tokens += (y != self._ignore_index).sum().to(self._device)

    @sync_all_reduce("_sum_of_nll", "_num_tokens")
    def compute(self) -> float:
        if self._num_tokens == 0:
            raise NotComputableError("Perplexity must have at least one example before it can be computed.")

        return torch.exp(self._sum_of_nll / self._num_tokens).item()
