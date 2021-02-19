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
        self._scorelist = []
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
        f1_score = self._safe_divide(precision_score * recall_score , denom)
        scores = dict()
        scores['recall'] = recall_score
        scores['precision'] = precision_score
        scores['f1'] = f1_score
        self._scorelist.append(scores)
        return f1_score

    def _lcs(self, a, b):
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
        print(matches, recall_total, precision_total)
        f1_score = self._f1_score(matches, recall_total, precision_total)
        return f1_score

    def rouge_l(self,peer,models):
        matches = 0
        recall_total = 0
        for model in models:
            matches += int(self._lcs(model, peer))
            recall_total += len(model)
        precision_total = len(models) * len(peer)
        print(matches, recall_total, precision_total)
        f1_score = self._f1_score(matches, recall_total, precision_total)
        return f1_score

    @reinit__is_reduced
    def reset(self):
        self._rougelist = torch.tensor([], device=self._device)
        self._num_examples = 0
        self._scorelist = []
        super(Rouge, self).reset()

    @reinit__is_reduced
    def update(self, output):
        peer, models = output[0], output[1]
        if self.n == 'l':
            self._rougelist = torch.cat((self._rougelist, torch.tensor(
                self.rouge_l(peer, models), device=self._device).unsqueeze(0)), dim=-1)
        else:
            self._rougelist = torch.cat((self._rougelist, torch.tensor(
                self.rouge_n(peer, models), device=self._device).unsqueeze(0)), dim=-1)
        self._num_examples += 1

    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Rouge must have at least one example before it can be computed.')
        return torch.mean(self._rougelist)
