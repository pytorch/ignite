#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 15:57:57 2018

@author: weihuangxu
"""

from ignite.contrib.metrics import MeanAbsoluteRelativeError
import torch
from pytest import approx


def test_mean_absolute_relative_error():
    a = torch.tensor([2.0, -1.0, -1.0, 2.0])
    b = torch.tensor([-1.0, 2.0, -3.0, -1.0])
    c = torch.tensor([1.0, 0.0, -1.0, 0.0])
    d = torch.tensor([3.0, -1.0, -2.0, 1.0])
    ground_truth = torch.tensor([1.0, 0.5, 0.2, 1.0])

    m = MeanAbsoluteRelativeError()
    m.reset()
    m.update((a, ground_truth))

    assert m.compute() == approx(2.75)
    m.update((b, ground_truth))

    assert m.compute() == approx(4.25)
    m.update((c, ground_truth))

    assert m.compute() == approx(3.5)
    m.update((d, ground_truth))

    assert m.compute() == approx(3.625)
