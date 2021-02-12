from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from collections import defaultdict
import torch

# These decorators helps with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class Rouge(Metric):

    def __init__(self, alpha, n, output_transform=lambda x: x, device="cpu"):
        self._rougelist = None
        self._num_examples = 0
        self.alpha = alpha
        self.n = n
        super(Rouge, self).__init__(output_transform=output_transform, device=device)

    def _ngramify(self, text, n):
        ngram_dict = defaultdict(int)
        start = 0
        end = n
        ngram = ''
        for i in range(start, end):
            ngram += text[i]
            ngram += ' '
        while end < len(text):
            ngram_dict[ngram] += 1
            ngram = ngram[len(text[start]) + 1:]
            ngram += text[end]
            ngram += ' '
            start += 1
            end += 1
        ngram_dict[ngram] += 1
        return ngram_dict

    def _safe_divide(self, numerator, denominator):
        if denominator > 0:
            return numerator / denominator
        else:
            return 0

    def _f1_score(self, matches, recall_total, precision_total, alpha=1):
        recall_score = self._safe_divide(matches, recall_total)
        precision_score = self._safe_divide(matches, precision_total)
        # print('Recall: ',recall_score)
        # print('Precision: ',precision_score)
        denom = (1.0 - alpha) * precision_score + alpha * recall_score
        if denom > 0.0:
            return (precision_score * recall_score) / denom
        else:
            return 0.0

    def rouge_n(self, peer, models):
        matches = 0
        recall_total = 0
        n = self.n
        peer_dict = self._ngramify(peer, n)
        for model in models:
            model_dict = self._ngramify(model, n)
            recall_total += max(len(model) - n + 1, 0)
            for ngram in peer_dict:
                if model_dict[ngram]:
                    matches += peer_dict[ngram]
        precision_total = len(models) * max((len(peer) - n + 1), 0)
        return self._f1_score(matches, recall_total, precision_total)

    @reinit__is_reduced
    def reset(self):
        self._rougelist = torch.tensor([], device=self._device)
        self._num_examples = 0
        super(Rouge, self).reset()

    @reinit__is_reduced
    def update(self, output):
        peer, models = output[0], output[1]
        self._rougelist = torch.cat((self._rougelist, torch.tensor(
            self.rouge_n(peer, models), device=self._device).unsqueeze(0)), dim=-1)
        self._num_examples += 1

    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Rouge must have at least one example before it can be computed.')
        return torch.mean(self._rougelist)
