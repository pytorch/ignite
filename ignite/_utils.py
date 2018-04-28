import collections

import torch
from torch._six import string_classes
from torch.autograd import Variable


def _to_hours_mins_secs(time_taken):
    mins, secs = divmod(time_taken, 60)
    hours, mins = divmod(mins, 60)
    return hours, mins, secs


def convert_tensor(tensor, requires_grad=False, device=None):
    if device:
        tensor = tensor.to(device=device)
    if requires_grad != tensor.requires_grad:
        tensor.requires_grad_(requires_grad)
    return tensor


def to_onehot(indices, num_classes):
    onehot = torch.zeros(indices.size(0), num_classes)
    if indices.is_cuda:
        onehot = onehot.to('cuda')
    return onehot.scatter_(1, indices.unsqueeze(1), 1)
