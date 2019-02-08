import torch
import torch.nn.functional as F
from ignite.metrics.metrics_lambda import MetricsLambda


def __getattr__(attr):
    if hasattr(torch, attr):
        fn = getattr(torch, attr)
    elif hasattr(F, attr):
        fn = getattr(F, attr)
    else:
        raise AttributeError('Unknown PyTorch operators')

    def wrapper(*args, **kwargs):
        return MetricsLambda(fn, *args, **kwargs)
    return wrapper
