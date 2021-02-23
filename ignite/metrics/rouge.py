from collections import defaultdict
from typing import Callable, DefaultDict, List, Union

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics import Metric

# These decorators helps with distributed settings
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce


class Rouge(Metric):
    r"""Calculates the Rouge Score for two Sequence of Tokens.

    Paper: Lin, Chin-Yew. 2004. ROUGE: a Package for Automatic Evaluation of Summaries.
    In Proceedings of the Workshop on Text Summarization Branches Out (WAS 2004), Barcelona, Spain, July 25 - 26, 2004.


    .. math::
        F_\1 = 2 * \frac{\text{precision} * \text{recall}} {\text{precision} + \text{recall}}

    where :math:`\alpha` is a float between 0 and 1.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y_pred` must be a Sequence of Tokens.
    - `y` must be in the form of List of Sequence of Tokens of Model Texts.

    Args:
        alpha: alpha value for calculating f1 score
        n: Rouge score to be calculated using n-grams of tokens
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.

    Example:
        from ignite.metrics import Rouge
        import torch
        m = Rouge(alpha=1,n='l')
        y_pred = "the cat was found under the bed"
        y = "the cat was under the bed"
        m.update([y_pred.split(),[y.split()]]) #Using space to separate sentences into tokens
        y_pred = "the tiny little cat was found under the big funny bed"
        y = "the cat was under the bed"
        m.update([y_pred.split(),[y.split()]])
        print(m._rougetotal)
        print(m._num_examples)
        print(m.compute())

    .. versionadded:: 0.5.0
    """

    def __init__(
        self,
        n: Union[int, str],
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        self._rougetotal: torch.Tensor = torch.tensor(0, device=device)
        self._num_examples: int = 0
        self._check_parameters(n)
        self.n: int = n
        super(Rouge, self).__init__(output_transform=output_transform, device=device)

    def _check_parameters(self, n: Union[int, str]) -> None:
        if isinstance(n, str) and n != "l" and n != "L":
            raise ValueError('Invalid String, Only Rouge-L supported.Please use "l" or "L"')
        elif isinstance(n, int) and n < 1:
            raise ValueError("Ignite needs atleast unigram to calculate Rouge")

    def _ngramify(self, tokens: List[str], n: int) -> DefaultDict:
        ngrams = defaultdict(int)
        for ngram in (tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)):
            ngrams[ngram] += 1
        return ngrams

    def _safe_divide(self, numerator: float, denominator: float) -> float:
        if denominator > 0:
            return numerator / denominator
        else:
            return 0.0

    def _f1_score(self, matches: int, recall_total: int, precision_total: int) -> float:
        recall_score = self._safe_divide(matches, recall_total)
        precision_score = self._safe_divide(matches, precision_total)
        denom = precision_score + recall_score
        f1_score = self._safe_divide(2 * precision_score * recall_score, denom)
        return f1_score

    def _lcs(self, a: List[str], b: List[str]) -> int:
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

    def rouge_n(self, y_pred: List[str], y: List[List[str]]) -> float:
        matches = 0
        recall_total = 0
        n = self.n
        y_pred_dict = self._ngramify(y_pred, n)
        for model in y:
            model_dict = self._ngramify(model, n)
            recall_total += max(len(model) - n + 1, 0)
            for ngram in y_pred_dict:
                if model_dict[ngram]:
                    matches += y_pred_dict[ngram]
        precision_total = len(y) * max((len(y_pred) - n + 1), 0)
        f1_score = self._f1_score(matches, recall_total, precision_total)
        return f1_score

    def rouge_l(self, y_pred: List[str], y: List[List[str]]) -> float:
        matches = 0
        recall_total = 0
        for model in y:
            matches += int(self._lcs(model, y_pred))
            recall_total += len(model)
        precision_total = len(y) * len(y_pred)
        f1_score = self._f1_score(matches, recall_total, precision_total)
        return f1_score

    @reinit__is_reduced
    def reset(self) -> None:
        self._rougetotal = torch.tensor(0, device=self._device)
        self._num_examples = 0
        super(Rouge, self).reset()

    @reinit__is_reduced
    def update(self, output: List[List]) -> None:
        y_pred, y = output[0], output[1]
        if isinstance(self.n, str):
            self._rougetotal = torch.add(self._rougetotal, self.rouge_l(y_pred, y))
        else:
            self._rougetotal = torch.add(self._rougetotal, self.rouge_n(y_pred, y))
        self._num_examples += 1

    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self) -> torch.Tensor:
        if self._num_examples == 0:
            raise NotComputableError("Rouge must have at least one example before it can be computed.")
        return torch.div(self._rougetotal, self._num_examples)
