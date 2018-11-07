from ignite.metrics.metric import Metric


class MetricsLambda(Metric):
    """
    Apply a function to other metrics to obtain a new metric.
    The result of the new metric is defined to be the result
    of applying the function to the result of argument metrics.
    For example, ``MetricsLambda(lambda x, y: x + y, metric1, 2)``
    will give a new metrics whose value is always the value of metric1
    plus 2.

    Arguments:
        f (callable): the function that defines the computation
        args (sequence): Sequence of other metrics or something
            else that will be fed to ``f`` as arguments.
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
