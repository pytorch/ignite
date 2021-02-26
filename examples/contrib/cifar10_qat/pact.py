# Implementation taken from https://discuss.pytorch.org/t/evaluator-returns-nan/107972/3
# Ref: https://arxiv.org/abs/1805.06085

import torch
import torch.nn as nn


class PACTClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return torch.clamp(x, 0, alpha.data)

    @staticmethod
    def backward(ctx, dy):
        x, alpha = ctx.saved_tensors

        dx = dy.clone()
        dx[x < 0] = 0
        dx[x > alpha] = 0

        dalpha = dy.clone()
        dalpha[x <= alpha] = 0

        return dx, torch.sum(dalpha)


class PACTReLU(nn.Module):
    def __init__(self, alpha=6.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x):
        return PACTClip.apply(x, self.alpha)
