from ignite.metrics import Precision, Recall

__all__ = [
    'Fbeta'
]


def Fbeta(beta, average=True, precision=None, recall=None, output_transform=None, device=None):
    """Calculates F-beta score

    Args:
        beta (float): weight of precision in harmonic mean
        average (bool, optional): if True, F-beta score is computed as the unweighted average (across all classes
            in multiclass case), otherwise, returns a tensor with F-beta score for each class in multiclass case.
        precision (Precision, optional): precision object metric with `average=False` to compute F-beta score
        recall (Precision, optional): recall object metric with `average=False` to compute F-beta score
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric. It is used only if precision or recall are not provided.
        device (str of torch.device, optional): device specification in case of distributed computation usage.
            In most of the cases, it can be defined as "cuda:local_rank" or "cuda"
            if already set `torch.cuda.set_device(local_rank)`. By default, if a distributed process group is
            initialized and available, device is set to `cuda`.

    Returns:
        MetricsLambda, F-beta metric
    """
    if not (beta > 0):
        raise ValueError("Beta should be a positive integer, but given {}".format(beta))

    if precision is not None and output_transform is not None:
        raise ValueError("If precision argument is provided, output_transform should be None")

    if recall is not None and output_transform is not None:
        raise ValueError("If recall argument is provided, output_transform should be None")

    if precision is None:
        precision = Precision(output_transform=(lambda x: x) if output_transform is None else output_transform,
                              average=False, device=device)
    elif precision._average:
        raise ValueError("Input precision metric should have average=False")

    if recall is None:
        recall = Recall(output_transform=(lambda x: x) if output_transform is None else output_transform,
                        average=False, device=device)
    elif recall._average:
        raise ValueError("Input recall metric should have average=False")

    fbeta = (1.0 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall + 1e-15)

    if average:
        fbeta = fbeta.mean().item()

    return fbeta
