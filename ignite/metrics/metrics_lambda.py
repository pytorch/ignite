from ignite.metrics.metric import Metric


class MetricsLambda(Metric):
    """
    Apply a function to other metrics to obtain a new metric.
    The result of the new metric is defined to be the result
    of applying the function to the result of argument metrics.

    Arguments:
        f (callable): the function that defines the computation
        args (sequence): Sequence of other metrics or something
            else that will be fed to ``f`` as arguments.

    Examples::
        >>> precision = Precision()
        >>> recall = Recall()
        >>> def Fbeta(r, p, beta):
        >>>     return (1 + beta ** 2) * p * r / (beta ** 2 * p + r)
        >>> F1 = MetricsLambda(Fbeta, recall, precison, 1)
        >>> F2 = MetricsLambda(Fbeta, recall, precison, 2)
        >>> F3 = MetricsLambda(Fbeta, recall, precison, 3)
        >>> F4 = MetricsLambda(Fbeta, recall, precison, 4)
    """
    def __init__(self, f, *args):
        self.function = f
        self.args = args

    def reset(self):
        for i in self.args:
            if isinstance(i, Metric):
                i.reset()

    def update(self, output):
        for i in self.args:
            if isinstance(i, Metric):
                i.update(output)

    def compute(self):
        materialized = [i.compute() if isinstance(i, Metric) else i for i in self.args]
        return self.function(*materialized)
