""" Credit goes to https://github.com/salesforce/awd-lstm-lm """

import logging

import torch as th
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class LockedDropout(nn.Module):
    def __init__(self, dropout=None):
        super().__init__()
        self.dropout = dropout

    def forward(self, x, dropout=None):
        dropout = dropout or self.dropout

        if not self.training or not dropout:
            return x

        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = m / (1 - dropout)
        mask = mask.expand_as(x)

        return mask * x


class WeightDrop(th.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super().__init__()

        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational

        self._setup()

    def noop(*args, **kwargs):
        return

    # noinspection PyProtectedMember
    def _setup(self):
        if issubclass(type(self.module), th.nn.RNNBase):
            self.module.flatten_parameters = self.noop

        for name_w in self.weights:
            w = getattr(self.module, name_w)
            self.module._parameters.pop(name_w)
            self.module.register_parameter(name_w + "_raw", Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:

            raw_w = getattr(self.module, name_w + "_raw")

            if self.variational:
                mask = raw_w.new_ones(raw_w.size(0), 1)
                mask = F.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w

            else:
                w = F.dropout(raw_w, p=self.dropout, training=self.training)

            if not isinstance(w, th.nn.Parameter):
                w = th.nn.Parameter(w, requires_grad=w.requires_grad)

            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = (embed.weight.data.new()
                .resize_((embed.weight.size(0), 1))
                .bernoulli_(1 - dropout)
                .expand_as(embed.weight) / (1 - dropout))

        masked_embed_weight = mask * embed.weight

    else:
        masked_embed_weight = embed.weight

    if scale:
        masked_embed_weight *= scale.expand_as(masked_embed_weight)

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    X = F.embedding(
        words,
        masked_embed_weight,
        padding_idx,
        embed.max_norm,
        embed.norm_type,
        embed.scale_grad_by_freq,
        embed.sparse,
    )

    return X
