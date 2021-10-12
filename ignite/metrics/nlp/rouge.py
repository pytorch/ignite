from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics import Metric

# These decorators helps with distributed settings
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
from ignite.metrics.nlp.utils import lcs, ngrams

__all__ = ["Rouge", "RougeN", "RougeL"]


class Score(namedtuple("Score", ["match", "candidate", "reference"])):
    r"""
    Computes precision and recall for given matches, candidate and reference lengths.
    """

    def precision(self) -> float:
        """
        Calculates precision.
        """
        return self.match / self.candidate if self.candidate > 0 else 0

    def recall(self) -> float:
        """
        Calculates recall.
        """
        return self.match / self.reference if self.reference > 0 else 0


def compute_ngram_scores(candidate: Sequence[Any], reference: Sequence[Any], n: int = 4) -> Score:
    """
    Compute the score based on ngram co-occurence of sequences of items

    Args:
        candidate: candidate sequence of items
        reference: reference sequence of items
        n: ngram order

    Returns:
        The score containing the number of ngram co-occurences

    .. versionadded:: 0.4.5
    """

    # ngrams of the candidate
    candidate_counter = ngrams(candidate, n)
    # ngrams of the references
    reference_counter = ngrams(reference, n)
    # ngram co-occurences in the candidate and the references
    match_counters = candidate_counter & reference_counter

    # the score is defined using Fraction
    return Score(
        match=sum(match_counters.values()),
        candidate=sum(candidate_counter.values()),
        reference=sum(reference_counter.values()),
    )


def compute_lcs_scores(candidate: Sequence[Any], reference: Sequence[Any]) -> Score:
    """
    Compute the score based on longest common subsequence of sequences of items

    Args:
        candidate: candidate sequence of items
        reference: reference sequence of items

    Returns:
        The score containing the length of longest common subsequence

    .. versionadded:: 0.4.5
    """

    # lcs of candidate and reference
    match = lcs(candidate, reference)

    # the score is defined using Fraction
    return Score(match=match, candidate=len(candidate), reference=len(reference))


class MultiRefReducer(metaclass=ABCMeta):
    r"""
    Reducer interface for multi-reference
    """

    @abstractmethod
    def __call__(self, scores: Sequence[Score]) -> Score:
        pass


class MultiRefAverageReducer(MultiRefReducer):
    r"""
    Reducer for averaging the scores
    """

    def __call__(self, scores: Sequence[Score]) -> Score:
        match = sum([score.match for score in scores])
        candidate = sum([score.candidate for score in scores])
        reference = sum([score.reference for score in scores])
        return Score(match=match, candidate=candidate, reference=reference)


class MultiRefBestReducer(MultiRefReducer):
    r"""
    Reducer for selecting the best score
    """

    def __call__(self, scores: Sequence[Score]) -> Score:
        return max(scores, key=lambda x: x.recall())


class _BaseRouge(Metric):
    r"""
    Rouge interface for Rouge-L and Rouge-N
    """

    def __init__(
        self,
        multiref: str = "average",
        alpha: float = 0,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        super(_BaseRouge, self).__init__(output_transform=output_transform, device=device)
        self._alpha = alpha
        if not 0 <= self._alpha <= 1:
            raise ValueError(f"alpha must be in interval [0, 1] (got : {self._alpha})")
        self._multiref = multiref
        valid_multiref = ["best", "average"]
        if self._multiref not in valid_multiref:
            raise ValueError(f"multiref : valid values are {valid_multiref} (got : {self._multiref})")
        self._mutliref_reducer = self._get_multiref_reducer()

    def _get_multiref_reducer(self) -> MultiRefReducer:
        if self._multiref == "average":
            return MultiRefAverageReducer()
        return MultiRefBestReducer()

    @reinit__is_reduced
    def reset(self) -> None:
        self._recall = 0.0
        self._precision = 0.0
        self._fmeasure = 0.0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Tuple[Sequence[Sequence[Any]], Sequence[Sequence[Sequence[Any]]]]) -> None:
        candidates, references = output
        for _candidate, _reference in zip(candidates, references):
            multiref_scores = [self._compute_score(candidate=_candidate, reference=_ref,) for _ref in _reference]
            score = self._mutliref_reducer(multiref_scores)
            precision = score.precision()
            recall = score.recall()
            self._precision += precision
            self._recall += recall
            precision_recall = precision * recall
            if precision_recall > 0:  # avoid zero division
                self._fmeasure += precision_recall / ((1 - self._alpha) * precision + self._alpha * recall)
            self._num_examples += 1

    @sync_all_reduce("_precision", "_recall", "_fmeasure", "_num_examples")
    def compute(self) -> Mapping:
        if self._num_examples == 0:
            raise NotComputableError("Rouge metric must have at least one example before be computed")

        return {
            f"{self._metric_name()}-P": float(self._precision / self._num_examples),
            f"{self._metric_name()}-R": float(self._recall / self._num_examples),
            f"{self._metric_name()}-F": float(self._fmeasure / self._num_examples),
        }

    @abstractmethod
    def _compute_score(self, candidate: Sequence[Any], reference: Sequence[Any]) -> Score:
        pass

    @abstractmethod
    def _metric_name(self) -> str:
        pass


class RougeN(_BaseRouge):
    r"""Calculates the Rouge-N score.

    The Rouge-N is based on the ngram co-occurences of candidates and references.

    More details can be found in `Lin 2004`__.

    __ https://www.aclweb.org/anthology/W04-1013.pdf

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y_pred` (list(list(str))) must be a sequence of tokens.
    - `y` (list(list(list(str))) must be a list of sequence of tokens.

    Args:
        ngram: ngram order (default: 4).
        multiref: reduces scores for multi references. Valid values are "best" and "average"
            (default: "average").
        alpha: controls the importance between recall and precision (alpha -> 0: recall is more important, alpha -> 1:
            precision is more important)
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        device: specifies which device updates are accumulated on. Setting the metric's
            device to be the same as your ``update`` arguments ensures the ``update`` method is non-blocking. By
            default, CPU.

    Examples:
        .. code-block:: python

            from ignite.metrics import RougeN

            m = RougeN(ngram=2, multiref="best")

            candidate = "the cat is not there".split()
            references = [
                "the cat is on the mat".split(),
                "there is a cat on the mat".split()
            ]

            m.update(([candidate], [references]))

            m.compute()
            # {'Rouge-2-P': 0.5, 'Rouge-2-R': 0.4, 'Rouge-2-F': 0.4}

    .. versionadded:: 0.4.5
    """

    def __init__(
        self,
        ngram: int = 4,
        multiref: str = "average",
        alpha: float = 0,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(RougeN, self).__init__(multiref=multiref, alpha=alpha, output_transform=output_transform, device=device)
        self._ngram = ngram
        if self._ngram < 1:
            raise ValueError(f"ngram order must be greater than zero (got : {self._ngram})")

    def _compute_score(self, candidate: Sequence[Any], reference: Sequence[Any]) -> Score:
        return compute_ngram_scores(candidate=candidate, reference=reference, n=self._ngram)

    def _metric_name(self) -> str:
        return f"Rouge-{self._ngram}"


class RougeL(_BaseRouge):
    r"""Calculates the Rouge-L score.

    The Rouge-L is based on the length of the longest common subsequence of candidates and references.

    More details can be found in `Lin 2004`__.

    __ https://www.aclweb.org/anthology/W04-1013.pdf

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y_pred` (list(list(str))) must be a sequence of tokens.
    - `y` (list(list(list(str))) must be a list of sequence of tokens.

    Args:
        multiref: reduces scores for multi references. Valid values are "best" and "average" (default: "average").
        alpha: controls the importance between recall and precision (alpha -> 0: recall is more important, alpha -> 1:
           precision is more important)
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        device: specifies which device updates are accumulated on. Setting the metric's
            device to be the same as your ``update`` arguments ensures the ``update`` method is non-blocking. By
            default, CPU.

    Examples:
        .. code-block:: python

            from ignite.metrics import RougeL

            m = RougeL(multiref="best")

            candidate = "the cat is not there".split()
            references = [
               "the cat is on the mat".split(),
                "there is a cat on the mat".split()
            ]

            m.update(([candidate], [references]))

           m.compute()
           # {'Rouge-L-P': 0.6, 'Rouge-L-R': 0.5, 'Rouge-L-F': 0.5}

    .. versionadded:: 0.4.5
    """

    def __init__(
        self,
        multiref: str = "average",
        alpha: float = 0,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(RougeL, self).__init__(multiref=multiref, alpha=alpha, output_transform=output_transform, device=device)

    def _compute_score(self, candidate: Sequence[Any], reference: Sequence[Any]) -> Score:
        return compute_lcs_scores(candidate=candidate, reference=reference)

    def _metric_name(self) -> str:
        return "Rouge-L"


class Rouge(Metric):
    r"""Calculates the Rouge score for multiples Rouge-N and Rouge-L metrics.

    More details can be found in `Lin 2004`__.

    __ https://www.aclweb.org/anthology/W04-1013.pdf

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y_pred` (list(list(str))) must be a sequence of tokens.
    - `y` (list(list(list(str))) must be a list of sequence of tokens.

    Args:
        variants: set of metrics computed. Valid inputs are "L" and integer 1 <= n <= 9.
        multiref: reduces scores for multi references. Valid values are "best" and "average" (default: "average").
        alpha: controls the importance between recall and precision (alpha -> 0: recall is more important, alpha -> 1:
           precision is more important)
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        device: specifies which device updates are accumulated on. Setting the metric's
            device to be the same as your ``update`` arguments ensures the ``update`` method is non-blocking. By
            default, CPU.

    Examples:
        .. code-block:: python

            from ignite.metrics import Rouge

            m = Rouge(variants=["L", 2], multiref="best")

            candidate = "the cat is not there".split()
            references = [
                "the cat is on the mat".split(),
                "there is a cat on the mat".split()
            ]

            m.update(([candidate], [references]))

            m.compute()
            # {'Rouge-L-P': 0.6, 'Rouge-L-R': 0.5, 'Rouge-L-F': 0.5, 'Rouge-2-P': 0.5, 'Rouge-2-R': 0.4,
            # 'Rouge-2-F': 0.4}

    .. versionadded:: 0.4.5
    .. versionchanged:: 0.5.0
        Changed input type to work on batch of inputs
    """

    def __init__(
        self,
        variants: Optional[Sequence[Union[str, int]]] = None,
        multiref: str = "average",
        alpha: float = 0,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        if variants is None or len(variants) == 0:
            variants = [1, 2, 4, "L"]
        self.internal_metrics: List[_BaseRouge] = []
        for m in variants:
            variant: Optional[_BaseRouge] = None
            if isinstance(m, str) and m == "L":
                variant = RougeL(multiref=multiref, alpha=alpha, output_transform=output_transform, device=device)
            elif isinstance(m, int):
                variant = RougeN(
                    ngram=m, multiref=multiref, alpha=alpha, output_transform=output_transform, device=device
                )
            else:
                raise ValueError("variant must be 'L' or integer greater to zero")
            self.internal_metrics.append(variant)
        super(Rouge, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        for m in self.internal_metrics:
            m.reset()

    @reinit__is_reduced
    def update(self, output: Tuple[Sequence[Sequence[Any]], Sequence[Sequence[Sequence[Any]]]]) -> None:
        for m in self.internal_metrics:
            m.update(output)

    def compute(self) -> Mapping:
        results = {}
        for m in self.internal_metrics:
            results.update(m.compute())
        return results
