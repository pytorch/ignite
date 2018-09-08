import logging
from typing import Tuple, List
from typing import Union

import torch as th
from torch.nn import Linear, ModuleList

logger = logging.getLogger(__name__)
FREQ = 250


def detach(H: Union[List, Tuple, th.Tensor]):
    if H is None:
        return

    typ = type(H)

    if typ in {list, tuple}:
        return typ(detach(h) for h in H)

    return H.detach()


def set_weight(module, W, check_size=False, clone=False):
    if clone:
        W = W.clone().detach()

    if not isinstance(W, th.nn.Parameter):
        W = th.nn.Parameter(W)

    if isinstance(module, th.nn.Linear):
        assert module.bias is None, "cannot set weight for linear with bias"

        if check_size:
            assert module.in_features == W.size(1)
            assert module.out_features == W.size(0)

        module.out_features = W.size(0)
        module.in_features = W.size(1)
        module.weight = W

    elif isinstance(module, th.nn.Embedding):
        if check_size:
            assert module.num_embeddings == W.size(0)
            assert module.embedding_dim == W.size(1)

        module.num_embeddings = W.size(0)
        module.embedding_dim = W.size(1)
        module.weight = W

    elif isinstance(module, th.nn.AdaptiveLogSoftmaxWithLoss):

        assert module.head_bias is False

        in_features = W.size(1)
        n_classes = W.size(0)
        cutoffs = module.cutoffs[:-1]

        module.in_features = in_features
        module.n_classes = n_classes
        module.cutoffs = cutoffs + [n_classes]

        module.shortlist_size = module.cutoffs[0]
        module.head_size = module.shortlist_size + module.n_clusters

        module.head.weight[:module.shortlist_size] = W[:module.shortlist_size]
        module.tail = ModuleList()

        for i in range(module.n_clusters):
            i0 = module.cutoffs[i + 1]
            i1 = module.cutoffs[i]
            osz = i0 - i1

            projection = Linear(module.in_features, osz, bias=False)
            projection.weight = W[i0:i1]

            module.tail.append(projection)

    elif hasattr(module, 'weight'):
        module.weight = W

    else:
        raise RuntimeError(f"Don't know how to set the weight for module "
                           f"{module}: "
                           f"it is not a [Linear, Embedding, Adaptive]")


class Average:
    def __init__(self):
        self._stat = 0.
        self._cnt = 0

    def reset(self):
        self.__init__()

    def update(self, output, count=1):
        self._stat += float(output)
        self._cnt += count

    def compute(self):
        return self._stat / self._cnt


class RunningAverage:
    def __init__(self, alpha):
        self._alpha = alpha
        self._stat = 0.

    def reset(self):
        self._stat = 0.

    def update(self, output):
        output = float(output)
        self._stat = (1 - self._alpha) * self._stat + self._alpha * output

    def compute(self):
        return self._stat

