import math
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

    def __call__(self, numerators: torch.Tensor, denominators: torch.Tensor) -> Sequence[float]:
        method = getattr(self, self.smooth)
        return method(numerators, denominators)

    @staticmethod
    def smooth1(numerators: torch.Tensor, denominators: torch.Tensor) -> Sequence[float]:
        epsilon = 0.1
        denominators_ = [max(1, d.item()) for d in denominators]
        return [n.item() / d if n != 0 else epsilon / d for n, d in zip(numerators, denominators_)]

    @staticmethod
    def nltk_smooth2(numerators: torch.Tensor, denominators: torch.Tensor) -> Sequence[float]:
        denominators_ = torch.tensor([max(1, d.item()) for d in denominators])
        return _Smoother._smooth2(numerators, denominators_)

    @staticmethod
    def smooth2(numerators: torch.Tensor, denominators: torch.Tensor) -> Sequence[float]:
        return _Smoother._smooth2(numerators, denominators)

    @staticmethod
    def _smooth2(numerators: torch.Tensor, denominators: torch.Tensor) -> Sequence[float]:

        return [
            (n.item() + 1) / (d.item() + 1) if i != 0 else n.item() / d.item()
            for i, (n, d) in enumerate(zip(numerators, denominators))
        ]

    @staticmethod
    def no_smooth(numerators: torch.Tensor, denominators: torch.Tensor) -> Sequence[float]:
        denominators_ = [max(1, d) for d in denominators]
        return [n.item() / d for n, d in zip(numerators, denominators_)]


class Bleu(Metric):
    r"""Calculates the `BLEU score <https://en.wikipedia.org/wiki/BLEU>`_.

    .. math::
       \text{BLEU} = b_{p} \cdot \exp \left( \sum_{n=1}^{N} w_{n} \: \log p_{n} \right)

    where :math:`N` is the order of n-grams, :math:`b_{p}` is a sentence brevety penalty, :math:`w_{n}` are
    positive weights summing to one and :math:`p_{n}` are modified n-gram precisions.

    More details can be found in `Papineni et al. 2002`__.

    __ https://www.aclweb.org/anthology/P02-1040

    In addition, a review of smoothing techniques can be found in `Chen et al. 2014`__

    __ http://acl2014.org/acl2014/W14-33/pdf/W14-3346.pdf

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y_pred` (list(list(str))) - a list of hypotheses sentences.
    - `y` (list(list(list(str))) - a corpus of lists of reference sentences w.r.t hypotheses.

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
        average: specifies which type of averaging to use (macro or micro)
            for more details refer https://www.nltk.org/_modules/nltk/translate/bleu_score.html
            Default: "macro"

    Examples:
        .. code-block:: python

            from ignite.metrics.nlp import Bleu

            m = Bleu(ngram=4, smooth="smooth1")

            y_pred = "the the the the the the the"
            y = ["the cat is on the mat", "there is a cat on the mat"]

            m.update(([y_pred.split()], [[_y.split() for _y in y]]))

            print(m.compute())

    .. versionadded:: 0.4.5
    .. versionchanged:: 0.5.0
        added ``average`` option to handle micro and macro averaging modes.
    """

    def __init__(
        self,
        ngram: int = 4,
        smooth: str = "no_smooth",
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
        average: str = "macro",
    ):
        if ngram <= 0:
            raise ValueError(f"ngram order must be greater than zero (got: {ngram})")
        self.ngrams_order = ngram
        self.weights = [1 / self.ngrams_order] * self.ngrams_order
        self.smoother = _Smoother(method=smooth)

        if average not in ["macro", "micro"]:
            raise ValueError(f'Average must be either "macro" or "micro" (got: {average})')
        self.average = average

        super(Bleu, self).__init__(output_transform=output_transform, device=device)

    def _n_gram_counter(
        self,
        references: Sequence[Sequence[Sequence[Any]]],
        candidates: Sequence[Sequence[Any]],
        p_numerators: torch.Tensor,
        p_denominators: torch.Tensor,
    ) -> Tuple[int, int]:

        if len(references) != len(candidates):
            raise ValueError(
                f"nb of candidates should be equal to nb of reference lists ({len(candidates)} != "
                f"{len(references)})"
            )

        hyp_lengths = 0
        ref_lengths = 0

        # Iterate through each hypothesis and their corresponding references.
        for refs, hyp in zip(references, candidates):
            # For each order of ngram, calculate the numerator and
            # denominator for the corpus-level modified precision.
            for i in range(1, self.ngrams_order + 1):
                numerator, denominator = modified_precision(refs, hyp, i)
                p_numerators[i] += numerator
                p_denominators[i] += denominator

            # Calculate the hypothesis lengths
            hyp_lengths += len(hyp)

            # Calculate the closest reference lengths.
            ref_lengths += _closest_ref_length(refs, len(hyp))

        return hyp_lengths, ref_lengths

    def _brevity_penalty_smoothing(
        self, p_numerators: torch.Tensor, p_denominators: torch.Tensor, hyp_length_sum: int, ref_length_sum: int,
    ) -> float:

        # Returns 0 if there's no matching n-grams
        # We only need to check for p_numerators[1] == 0, since if there's
        # no unigrams, there won't be any higher order ngrams.
        if p_numerators[1] == 0:
            return 0

        # If no smoother, returns 0 if there's at least one a not matching n-grams]
        if self.smoother.smooth == "no_smooth" and min(p_numerators[1:]).item() == 0:
            return 0

        # Calculate corpus-level brevity penalty.
        if hyp_length_sum < ref_length_sum:
            bp = math.exp(1 - ref_length_sum / hyp_length_sum) if hyp_length_sum > 0 else 0.0
        else:
            bp = 1.0

        # Smoothing
        p_n = self.smoother(p_numerators[1:], p_denominators[1:])

        # Compute the geometric mean
        s = [w_i * math.log(p_i) for w_i, p_i in zip(self.weights, p_n)]
        gm = bp * math.exp(math.fsum(s))
        return gm

    def _sentence_bleu(self, references: Sequence[Sequence[Any]], candidates: Sequence[Any],) -> float:
        return self._corpus_bleu([references], [candidates])

    def _corpus_bleu(
        self, references: Sequence[Sequence[Sequence[Any]]], candidates: Sequence[Sequence[Any]],
    ) -> float:

        p_numerators: torch.Tensor = torch.zeros(self.ngrams_order + 1)
        p_denominators: torch.Tensor = torch.zeros(self.ngrams_order + 1)

        hyp_length_sum, ref_length_sum = self._n_gram_counter(
            references=references, candidates=candidates, p_numerators=p_numerators, p_denominators=p_denominators,
        )
        bleu_score = self._brevity_penalty_smoothing(
            p_numerators=p_numerators,
            p_denominators=p_denominators,
            hyp_length_sum=hyp_length_sum,
            ref_length_sum=ref_length_sum,
        )

        return bleu_score

    @reinit__is_reduced
    def reset(self) -> None:

        if self.average == "macro":
            self._sum_of_bleu = torch.tensor(0.0, dtype=torch.double, device=self._device)
            self._num_sentences = 0

        if self.average == "micro":
            self.p_numerators = torch.zeros(self.ngrams_order + 1)
            self.p_denominators = torch.zeros(self.ngrams_order + 1)
            self.hyp_length_sum = 0
            self.ref_length_sum = 0

    @reinit__is_reduced
    def update(self, output: Tuple[Sequence[Sequence[Any]], Sequence[Sequence[Sequence[Any]]]]) -> None:
        y_pred, y = output

        if self.average == "macro":
            for refs, hyp in zip(y, y_pred):
                self._sum_of_bleu += self._sentence_bleu(references=refs, candidates=hyp)
                self._num_sentences += 1

        elif self.average == "micro":
            hyp_lengths, ref_lengths = self._n_gram_counter(
                references=y, candidates=y_pred, p_numerators=self.p_numerators, p_denominators=self.p_denominators
            )
            self.hyp_length_sum += hyp_lengths
            self.ref_length_sum += ref_lengths

    @sync_all_reduce("_sum_of_bleu", "_num_sentences")
    def _compute_macro(self) -> torch.Tensor:
        if self._num_sentences == 0:
            raise NotComputableError("Bleu must have at least one example before it can be computed.")

        return self._sum_of_bleu / self._num_sentences

    @sync_all_reduce("p_numerators", "p_denominators", "hyp_length_sum", "ref_length_sum")
    def _compute_micro(self) -> float:

        bleu_score = self._brevity_penalty_smoothing(
            p_numerators=self.p_numerators,
            p_denominators=self.p_denominators,
            hyp_length_sum=self.hyp_length_sum,
            ref_length_sum=self.ref_length_sum,
        )
        return bleu_score

    def compute(self) -> None:
        if self.average == "macro":
            return self._compute_macro()
        elif self.average == "micro":
            return self._compute_micro()
