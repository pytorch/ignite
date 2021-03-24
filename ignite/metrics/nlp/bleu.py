import math
from collections import Counter
from typing import Any, Callable, Sequence, Tuple, Union

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce
from ignite.metrics.nlp.utils import modified_precision

__all__ = ["Bleu"]


def _closest_ref_length(references: Sequence[Sequence[Any]], hyp_len: int) -> int:
    ref_lens = (len(reference) for reference in references)
    closest_ref_len = min(ref_lens, key=lambda ref_len: (abs(ref_len - hyp_len), ref_len))
    return closest_ref_len


class _Smoother:
    """
    Smoothing helper
    http://acl2014.org/acl2014/W14-33/pdf/W14-3346.pdf
    """

    def __init__(self, method: str):
        valid = ["no_smooth", "smooth1", "nltk_smooth2", "smooth2"]
        if method not in valid:
            raise ValueError(f"Smooth is not valid (expected: {valid}, got: {method})")
        self.smooth = method

    def __call__(self, numerators: Counter, denominators: Counter) -> Sequence[float]:
        method = getattr(self, self.smooth)
        return method(numerators, denominators)

    @staticmethod
    def smooth1(numerators: Counter, denominators: Counter) -> Sequence[float]:
        epsilon = 0.1
        denominators_ = [max(1, d) for d in denominators.values()]
        return [n / d if n != 0 else epsilon / d for n, d in zip(numerators.values(), denominators_)]

    @staticmethod
    def nltk_smooth2(numerators: Counter, denominators: Counter) -> Sequence[float]:
        denominators_ = [max(1, d) for d in denominators.values()]
        return [(n + 1) / (d + 1) for n, d in zip(numerators.values(), denominators_)]

    @staticmethod
    def smooth2(numerators: Counter, denominators: Counter) -> Sequence[float]:
        return [(n + 1) / (d + 1) for n, d in zip(numerators.values(), denominators.values())]

    @staticmethod
    def no_smooth(numerators: Counter, denominators: Counter) -> Sequence[float]:
        denominators_ = [max(1, d) for d in denominators.values()]
        return [n / d for n, d in zip(numerators.values(), denominators_)]


class Bleu(Metric):
    r"""Calculates the `BLEU score <https://en.wikipedia.org/wiki/BLEU>`_.

    .. math::
       \text{BLEU} = b_{p} \cdot \exp \left( \sum_{n=1}^{N} w_{n} \: \log p_{n} \right)

    where :math:`N` is the order of n-grams, :math:`b_{p}` is a sentence brevety penalty, :math:`w_{n}` are
    positive weights summing to one and :math:`p_{n}` are modified n-gram precisions.

    More details can be found in `Papineni et al. 2002`__.

    __ https://www.aclweb.org/anthology/P02-1040.pdf

    In addition, a review of smoothing techniques can be found in `Chen et al. 2014`__

    __ http://acl2014.org/acl2014/W14-33/pdf/W14-3346.pdf

    Remark :

        This implementation is inspired by nltk

    Args:
        ngram: order of n-grams.
        smooth: enable smoothing. Valid are ``no_smooth``, ``smooth1``, ``nltk_smooth2`` or ``smooth2``.
            Default: ``no_smooth``.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            By default, metrics require the output as ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.

    Example:

    .. code-block:: python

        from ignite.metrics.nlp import Bleu

        m = Bleu(ngram=4, smooth="smooth1")

        y_pred = "the the the the the the the"
        y = ["the cat is on the mat", "there is a cat on the mat"]

        m.update((y_pred.split(), [y.split()]))

        print(m.compute())

    .. versionadded:: 0.5.0
    """

    def __init__(
        self,
        ngram: int = 4,
        smooth: str = "no_smooth",
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        if ngram <= 0:
            raise ValueError(f"ngram order must be greater than zero (got: {ngram})")
        self.ngrams_order = ngram
        self.weights = [1 / self.ngrams_order] * self.ngrams_order
        self.smoother = _Smoother(method=smooth)
        super(Bleu, self).__init__(output_transform=output_transform, device=device)

    def _corpus_bleu(self, references: Sequence[Sequence[Any]], candidates: Sequence[Sequence[Any]],) -> float:
        p_numerators: Counter = Counter()
        p_denominators: Counter = Counter()

        if len(references) != len(candidates):
            raise ValueError(
                f"nb of candidates should be equal to nb of reference lists ({len(candidates)} != "
                f"{len(references)})"
            )

        # Iterate through each hypothesis and their corresponding references.
        for refs, hyp in zip(references, candidates):
            # For each order of ngram, calculate the numerator and
            # denominator for the corpus-level modified precision.
            for i in range(1, self.ngrams_order + 1):
                numerator, denominator = modified_precision(refs, hyp, i)
                p_numerators[i] += numerator
                p_denominators[i] += denominator

        # Returns 0 if there's no matching n-grams
        # We only need to check for p_numerators[1] == 0, since if there's
        # no unigrams, there won't be any higher order ngrams.
        if p_numerators[1] == 0:
            return 0

        # If no smoother, returns 0 if there's at least one a not matching n-grams
        if self.smoother.smooth == "no_smooth" and min(p_numerators.values()) == 0:
            return 0

        # Calculate the hypothesis lengths
        hyp_lengths = [len(hyp) for hyp in candidates]

        # Calculate the closest reference lengths.
        ref_lengths = [_closest_ref_length(refs, hyp_len) for refs, hyp_len in zip(references, hyp_lengths)]

        # Sum of hypothesis and references lengths
        hyp_len = sum(hyp_lengths)
        ref_len = sum(ref_lengths)

        # Calculate corpus-level brevity penalty.
        if hyp_len < ref_len:
            bp = math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0
        else:
            bp = 1.0

        # Smoothing
        p_n = self.smoother(p_numerators, p_denominators)

        # Compute the geometric mean
        s = [w_i * math.log(p_i) for w_i, p_i in zip(self.weights, p_n)]
        gm = bp * math.exp(math.fsum(s))
        return gm

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_of_bleu = torch.tensor(0.0, dtype=torch.double, device=self._device)
        self._num_sentences = 0

    @reinit__is_reduced
    def update(self, output: Tuple[Sequence[Any], Sequence[Sequence[Any]]]) -> None:
        y_pred, y = output
        self._sum_of_bleu += self._corpus_bleu(references=[y], candidates=[y_pred])
        self._num_sentences += 1

    @sync_all_reduce("_sum_of_bleu", "_num_sentences")
    def compute(self) -> torch.Tensor:
        if self._num_sentences == 0:
            raise NotComputableError("Bleu must have at least one example before it can be computed.")
        return self._sum_of_bleu / self._num_sentences
