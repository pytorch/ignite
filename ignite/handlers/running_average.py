from ignite.metrics import Metric
from ignite.engine import Events


class RunningAverage(object):
    """Compute running average of a metric or output

    Args:
        src (str or Metric): input source: "output" string or an instance of :class:`ignite.metrics.Metric`
        alpha (float, optional): running average decay factor, default 0.98
        output_transform (Callable, optional): a function to use to transform the output if `src` is "output".
            Otherwise it should be None

    """

    def __init__(self, src, alpha=0.98, output_transform=None):
        if not (isinstance(src, Metric) or src == 'output'):
            raise TypeError("Argument src should be a Metric or string 'output'")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("Argument alpha should be a float between 0.0 and 1.0")

        if isinstance(src, Metric):
            if output_transform is not None:
                raise ValueError("Argument output_transform should be None if src is 'output'")
            self.src = src
            self._get_src_value = self._get_metric_value
            self._iteration_completed = self._metric_iteration_completed
        else:
            self.src = None
            self._get_src_value = self._get_output_value
            self._iteration_completed = self._output_iteration_completed

        self.alpha = alpha
        self._output_transform = output_transform
        super(RunningAverage, self).__init__()

    def _reset(self, engine):
        self._value = None

    def _metric_iteration_completed(self, engine):
        self.src.started(engine)
        self.src.iteration_completed(engine)

    def _output_iteration_completed(self, engine):
        self.src = self._output_transform(engine.state.output)

    def _completed(self, engine, name):
        if self._value is None:
            self._value = self._get_src_value()
        else:
            self._value = self._value * self.alpha + (1.0 - self.alpha) * self._get_src_value()
        engine.state.metrics[name] = self._value

    def attach(self, engine, name):
        # restart average every epoch
        engine.add_event_handler(Events.EPOCH_STARTED, self._reset)
        # compute metric
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._iteration_completed)
        # apply running average
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._completed, name)

    def _get_metric_value(self):
        return self.src.compute()

    def _get_output_value(self):
        return self.src
