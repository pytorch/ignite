import numbers
from collections import Counter
from typing import Callable, DefaultDict, List, Tuple, Union

import torch
from torch.types import Number

from ignite.exceptions import NotComputableError
from ignite.metrics import Metric

# These decorators helps with distributed settings
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce


def _ngramify(sequence, n):
    return Counter([tuple(sequence[i : i + n]) for i in range(len(sequence) - n + 1)])


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator > 0:
        return numerator / denominator
    else:
        return 0.0


def _fbeta_score(matches: int, recall_total: int, precision_total: int, alpha: float) -> float:
    print(alpha)
    recall_score = _safe_divide(matches, recall_total)
    precision_score = _safe_divide(matches, precision_total)
    denom = (1 - alpha) * precision_score + (alpha) * recall_score
    fbeta_score = _safe_divide(precision_score * recall_score, denom)
    return fbeta_score


def _lcs(X, Y):
    m = len(X)
    n = len(Y)

    L = [[None] * (n + 1) for i in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    return L[m][n]


class Rouge(Metric):
    r"""Calculates the Rouge Score for two Sequence of Tokens.

    Paper: Lin, Chin-Yew. 2004. ROUGE: a Package for Automatic Evaluation of Summaries.
    In Proceedings of the Workshop on Text Summarization Branches Out (WAS 2004), Barcelona, Spain, July 25 - 26, 2004.

    Contains two metric Rouge-n and Rouge-L.

    - Rouge-n: precision, recall and f-beta based on overlapping n-grams between the predicted and the reference text
    - Rouge-L: Only the sentence level Rouge-L score is implemented. It finds the Longest Common Subsequence between
      the predicted and reference text and uses it to calculate precison, recall, f-beta score

    .. math::
        F_\beta = \left( 1 + \beta^2 \right) * \frac{ \text{precision} * \text{recall} }
        { \left( \beta^2 * \text{precision} \right) + \text{recall} }

    where :math:`\beta` is a float.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y_pred` must be a Sequence of Tokens.
    - `y` must be in the form of List of Sequence of Tokens of Model Texts.

    Args:
        beta: beta value for calculating f-beta
        variant: Which variant of Rouge Score to be evaluated
            Valid Values - "rougeN", "rougen", "n", "N", "rougeL", "rougel", "l", "L"
        n: Rouge score to be calculated using n-grams of tokens
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        device (Union[str, torch.device]) – specifies which device updates are accumulated on. Setting the metric’s
            device to be the same as your update arguments ensures the update method is non-blocking. By default, CPU.

    Example:

    .. code-block:: python

        from ignite.metrics import Rouge
        m = Rouge(beta=1,metric="rouge-L")
        y_pred = "the cat was found under the bed"
        y = "the cat was under the bed"
        m.update([y_pred.split(),y.split()]]) #Using space to separate sentences into tokens
        y_pred = "the tiny little cat was found under the big funny bed"
        m.update([y_pred.split(),y.split()])
        print(m.compute())

    .. versionadded:: 0.5.0
    """

    def __init__(
        self,
        beta: float = 0.0,
        metric: str = "rouge-1",
        version: str = "sentence",
        aggregate: str = "single",
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        self._rougetotal: torch.Tensor = torch.tensor(0, device=device, dtype=torch.double)
        self._num_examples: int = 0
        self.beta: float = 0.0
        self.n: int = 1
        self.beta, self.n, self.rouge_fn, self.aggregate = self._check_parameters(beta, metric, version, aggregate)
        self.alpha = 1 / (1 + self.beta ** 2)
        super(Rouge, self).__init__(output_transform=output_transform, device=device)

    def _check_parameters(self, beta: float, metric: str, variant: str, aggregate: str) -> Tuple:
        if not isinstance(beta, numbers.Number):
            raise TypeError("Beta should be a float.")
        n = 1
        if metric[-1].isnumeric():
            n = int(metric[-1])
            if n < 1:
                raise ValueError("n has to be greater than 0 to calculate Rouge-n.")
            rouge_fn = self.rouge_n
        elif metric[-1] in ["l", "L"]:
            if variant == "sentence":
                rouge_fn = self.rouge_l
        else:
            raise ValueError("Please provide a valid variant of Rouge to evaluate.")
        if aggregate not in ["single", "mean", "max"]:
            raise ValueError("Aggrregate must be single, mean or max.")
        return beta, n, rouge_fn, aggregate

    def rouge_n(self, y_pred: List[str], y: List[str]) -> float:
        n = self.n
        y_pred_count = _ngramify(y_pred, n)
        y_count = _ngramify(y, n)
        recall_total = max(len(y) - n + 1, 0)
        matches = sum((y_pred_count & y_count).values())
        precision_total = max((len(y_pred) - n + 1), 0)
        fbeta_score = _fbeta_score(matches, recall_total, precision_total, self.alpha)
        return fbeta_score

    def rouge_mean(self, y_pred: List[str], y: List[List[str]]):
        acc_f_score = 0
        for model in y:
            acc_f_score += self.rouge_fn(y_pred, model)
        return acc_f_score / len(y)

    def rouge_max(self, y_pred: List[str], y: List[List[str]]):
        max_f_score = 0
        for model in y:
            max_f_score = max(max_f_score, self.rouge_fn(y_pred, model))
        return max_f_score

    def rouge_l(self, y_pred: List[str], y: List[str]) -> float:
        matches = int(_lcs(y, y_pred))
        recall_total = len(y)
        precision_total = len(y_pred)
        fbeta_score = _fbeta_score(matches, recall_total, precision_total, self.alpha)
        return fbeta_score

    @reinit__is_reduced
    def reset(self) -> None:
        self._rougetotal = torch.tensor(0, device=self._device, dtype=torch.double)
        self._num_examples = 0
        super(Rouge, self).reset()

    @reinit__is_reduced
    def update(self, output: List[List]) -> None:
        y_pred, y = output[0], output[1]
        if self.aggregate == "single":
            self._rougetotal += self.rouge_fn(y_pred, y)
        if self.aggregate == "mean":
            self._rougetotal += self.rouge_mean(y_pred, y)
        if self.aggregate == "max":
            self._rougetotal += self.rouge_max(y_pred, y)
        self._num_examples += 1

    @sync_all_reduce("_num_examples", "_rougetotal")
    def compute(self) -> torch.Tensor:
        if self._num_examples == 0:
            raise NotComputableError("Rouge must have at least one example before it can be computed.")
        return self._rougetotal / self._num_examples
