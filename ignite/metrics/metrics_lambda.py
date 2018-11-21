from ignite.metrics.metric import Metric


class MetricsLambda(Metric):
    """
    Apply a function to other metrics to obtain a new metric.
    The result of the new metric is defined to be the result
    of applying the function to the result of argument metrics.

    When update, this metric does not recursively update the metrics
    it depends on. When reset, all its dependency metrics would be
    resetted. When attach, all its dependencies would be automatically
    attached.

    Arguments:
        f (callable): the function that defines the computation
        args (sequence): Sequence of other metrics or something
            else that will be fed to ``f`` as arguments.

    Examples:
        >>> precision = Precision(average=False)
        >>> recall = Recall(average=False)
        >>> def Fbeta(r, p, beta):
        >>>     return torch.mean((1 + beta ** 2) * p * r / (beta ** 2 * p + r)).item()
        >>> F1 = MetricsLambda(Fbeta, recall, precision, 1)
        >>> F2 = MetricsLambda(Fbeta, recall, precision, 2)
        >>> F3 = MetricsLambda(Fbeta, recall, precision, 3)
        >>> F4 = MetricsLambda(Fbeta, recall, precision, 4)
    """
    def __init__(self, f, *args):
        self.function = f
        self.args = args
        super(MetricsLambda, self).__init__()

    def reset(self):
        for i in self.args:
            if isinstance(i, Metric):
                i.reset()

    def update(self, output):
        # NB: this method does not recursively update dependency metrics,
        # which might cause duplicate update issue. To update this metric,
        # users should manually update its dependencies.
        pass

    def compute(self):
        materialized = [i.compute() if isinstance(i, Metric) else i for i in self.args]
        return self.function(*materialized)

    def attach(self, engine, name):
        # recursively attach all its dependencies
        for index, metric in enumerate(self.args):
            if isinstance(metric, Metric):
                metric.attach(engine, name + '[{}]'.format(index))
        super(MetricsLambda, self).attach(engine, name)
