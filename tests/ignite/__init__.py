import torch


def cpu_and_maybe_cuda():
    return ("cpu",) + (("cuda",) if torch.cuda.is_available() else ())
