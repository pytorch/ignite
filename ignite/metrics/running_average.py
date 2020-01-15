from ignite.engine import Events
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce


class RunningAverage(Metric):
    """Compute running average of a metric or the output of process function.

    Args:
        src (Metric or None): input source: an instance of :class:`~ignite.metrics.Metric` or None. The latter
            corresponds to `engine.state.output` which holds the output of process function.
        alpha (float, optional): running average decay factor, default 0.98
        output_transform (callable, optional): a function to use to transform the output if `src` is None and
            corresponds the output of process function. Otherwise it should be None.
        epoch_bound (boolean, optional): whether the running average should be reset after each epoch (defaults
            to True).
        device (str of torch.device, optional): device specification in case of distributed computation usage.
            This is necessary when running average is computed on the output of process function.
            In most of the cases, it can be defined as "cuda:local_rank" or "cuda"
            if already set `torch.cuda.set_device(local_rank)`. By default, if a distributed process group is
            initialized and available, device is set to `cuda`.

    Examples:

    .. code-block:: python

        alpha = 0.98
        acc_metric = RunningAverage(Accuracy(output_transform=lambda x: [x[1], x[2]]), alpha=alpha)
        acc_metric.attach(trainer, 'running_avg_accuracy')

        avg_output = RunningAverage(output_transform=lambda x: x[0], alpha=alpha)
        avg_output.attach(trainer, 'running_avg_loss')

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_running_avg_metrics(engine):
            print("running avg accuracy:", engine.state.metrics['running_avg_accuracy'])
            print("running avg loss:", engine.state.metrics['running_avg_loss'])

    """
    _required_output_keys = None

    def __init__(self, src=None, alpha=0.98, output_transform=None, epoch_bound=True, device=None):
        if not (isinstance(src, Metric) or src is None):
            raise TypeError("Argument src should be a Metric or None.")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("Argument alpha should be a float between 0.0 and 1.0.")

        if isinstance(src, Metric):
            if output_transform is not None:
                raise ValueError("Argument output_transform should be None if src is a Metric.")
            if device is not None:
                raise ValueError("Argument device should be None if src is a Metric.")
            self.src = src
            self._get_src_value = self._get_metric_value
            self.iteration_completed = self._metric_iteration_completed
        else:
            if output_transform is None:
                raise ValueError("Argument output_transform should not be None if src corresponds "
                                 "to the output of process function.")
            self._get_src_value = self._get_output_value
            self.update = self._output_update

        self.alpha = alpha
        self.epoch_bound = epoch_bound
        super(RunningAverage, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._value = None

    @reinit__is_reduced
    def update(self, output):
        # Implement abstract method
        pass

    def compute(self):
        if self._value is None:
            self._value = self._get_src_value()
        else:
            self._value = self._value * self.alpha + (1.0 - self.alpha) * self._get_src_value()

        return self._value

    def attach(self, engine, name):
        if self.epoch_bound:
            # restart average every epoch
            engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        # compute metric
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        # apply running average
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed, name)

    def _get_metric_value(self):
        return self.src.compute()

    @sync_all_reduce("src")
    def _get_output_value(self):
        return self.src

    def _metric_iteration_completed(self, engine):
        self.src.started(engine)
        self.src.iteration_completed(engine)

    @reinit__is_reduced
    def _output_update(self, output):
        self.src = output
