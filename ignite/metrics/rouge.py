from collections import defaultdict
from typing import Callable, DefaultDict, List, Tuple, Union

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics import Metric

# These decorators helps with distributed settings
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce


def _ngramify(tokens: List[str], n: int) -> DefaultDict:
    ngrams: DefaultDict = defaultdict(int)
    for ngram in (tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)):
        ngrams[ngram] += 1
    return ngrams


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator > 0:
        return numerator / denominator
    else:
        return 0.0


def _fbeta_score(matches: int, recall_total: int, precision_total: int, beta_sq: float) -> float:
    recall_score = _safe_divide(matches, recall_total)
    precision_score = _safe_divide(matches, precision_total)
    denom = (beta_sq) * precision_score + recall_score
    fbeta_score = _safe_divide((1 + beta_sq) * precision_score * recall_score, denom)
    return fbeta_score


def _lcs(a: List[str], b: List[str]) -> int:
    if len(a) < len(b):
        a, b = b, a

    if len(b) == 0:
        return 0

    row = [0] * len(b)
    for ai in a:
        left = 0
        diag = 0
        for j, bj in enumerate(b):
            up = row[j]
            if ai == bj:
                value = diag + 1
            else:
                value = max(left, up)
            row[j] = value
            left = value
            diag = up
    return left


class Rouge(Metric):
    r"""Calculates the Rouge Score for two Sequence of Tokens.

    Paper: Lin, Chin-Yew. 2004. ROUGE: a Package for Automatic Evaluation of Summaries.
    In Proceedings of the Workshop on Text Summarization Branches Out (WAS 2004), Barcelona, Spain, July 25 - 26, 2004.

    Contains two metric Rouge-n and Rouge-L.
        Rouge-n: precision, recall and f-beta based on overlapping n-grams between the predicted and the reference text
        Rouge-L: Only the sentence level Rouge-L score is implemented. It finds the Longest Common Subsequence between
            the predicted and reference text and uses it to calculate precison, recall, f-beta score

    .. math::
        F_\beta = \left( 1 + \beta^2 \right) * \frac{ \text{precision} * \text{recall} }
        { \left( \beta^2 * \text{precision} \right) + \text{recall} }

    where :math:`\beta` is a float.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y_pred` must be a Sequence of Tokens.
    - `y` must be in the form of List of Sequence of Tokens of Model Texts.

    Args:
        beta: beta value for calculating f-beta score
        n: Rouge score to be calculated using n-grams of tokens or 'l' or 'L' string for calculating Rouge-L
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        device (Union[str, torch.device]) – specifies which device updates are accumulated on. Setting the metric’s
            device to be the same as your update arguments ensures the update method is non-blocking. By default, CPU.

    Example:

    .. code-block:: python

        from ignite.metrics import Rouge
        m = Rouge(beta=1,n='l')
        y_pred = "the cat was found under the bed"
        y = "the cat was under the bed"
        m.update([y_pred.split(),[y.split()]]) #Using space to separate sentences into tokens
        y_pred = "the tiny little cat was found under the big funny bed"
        m.update([y_pred.split(),[y.split()]])
        print(m.compute())

    .. versionadded:: 0.5.0
    """

    def __init__(
        self,
        beta: float = 0.0,
        n: Union[int, str] = 1,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        self._rougetotal: torch.Tensor = torch.tensor(0, device=device, dtype=torch.double)
        self._num_examples: int = 0
        self.beta: float = 0.0
        self.n: int = 1
        self.beta, self.n = self._check_parameters(beta, n)
        self.beta_sq = self.beta ** 2
        self.rouge_fn = self.rouge_l if isinstance(self.n, str) else self.rouge_n
        super(Rouge, self).__init__(output_transform=output_transform, device=device)

    def _check_parameters(self, beta: float, n: Union[int, str]) -> Tuple:
        if not isinstance(beta, float):
            raise TypeError("Beta should be a float.")
        if isinstance(n, str) and n != "l" and n != "L":
            raise ValueError('Invalid String, Only Rouge-L supported.Please use "l" or "L"')
        elif isinstance(n, int) and n < 1:
            raise ValueError("Ignite needs atleast unigram to calculate Rouge")
        return (beta, n)

    def rouge_n(self, y_pred: List[str], y: List[str]) -> float:
        matches = 0
        n = self.n
        y_pred_dict = _ngramify(y_pred, n)
        model_dict = _ngramify(y, n)
        recall_total = max(len(y) - n + 1, 0)
        for ngram in y_pred_dict:
            if model_dict[ngram]:
                matches += y_pred_dict[ngram]
        precision_total = max((len(y_pred) - n + 1), 0)
        fbeta_score = _fbeta_score(matches, recall_total, precision_total, self.beta_sq)
        return fbeta_score

    def rouge_l(self, y_pred: List[str], y: List[str]) -> float:
        matches = 0
        matches += int(_lcs(y, y_pred))
        recall_total = len(y)
        precision_total = len(y_pred)
        fbeta_score = _fbeta_score(matches, recall_total, precision_total, self.beta_sq)
        return fbeta_score

    @reinit__is_reduced
    def reset(self) -> None:
        self._rougetotal = torch.tensor(0, device=self._device, dtype=torch.double)
        self._num_examples = 0
        super(Rouge, self).reset()

    @reinit__is_reduced
    def update(self, output: List[List]) -> None:
        y_pred, y = output[0], output[1]
        self._rougetotal += self.rouge_fn(y_pred, y)
        self._num_examples += 1

    @sync_all_reduce("_num_examples", "_rougetotal")
    def compute(self) -> torch.Tensor:
        if self._num_examples == 0:
            raise NotComputableError("Rouge must have at least one example before it can be computed.")
        return self._rougetotal / self._num_examples
