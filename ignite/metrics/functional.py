import torch
import torch.nn.functional as F
from ignite.metrics.metrics_lambda import MetricsLambda
from ignite._pep562 import Pep562
import sys
import functools


PY37 = sys.version_info >= (3, 7)


def __getattr__(attr):
    if hasattr(torch, attr):
        fn = getattr(torch, attr)
    elif hasattr(F, attr):
        fn = getattr(F, attr)
    else:
        raise AttributeError('Unknown PyTorch operators')

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return MetricsLambda(fn, *args, **kwargs)
    return wrapper


if not PY37:
    Pep562(__name__)
