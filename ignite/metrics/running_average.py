from typing import Callable, cast, Optional, Sequence, Union

import torch

import ignite.distributed as idist
from ignite.engine import Engine, Events
from ignite.metrics.metric import EpochWise, Metric, MetricUsage, reinit__is_reduced, sync_all_reduce

__all__ = ["RunningAverage"]


class RunningAverage(Metric):
    """Compute running average of a metric or the output of process function.

    Args:
        src: input source: an instance of :class:`~ignite.metrics.metric.Metric` or None. The latter
            corresponds to `engine.state.output` which holds the output of process function.
        alpha: running average decay factor, default 0.98
        output_transform: a function to use to transform the output if `src` is None and
            corresponds the output of process function. Otherwise it should be None.
        epoch_bound: whether the running average should be reset after each epoch (defaults
            to True).
        device: specifies which device updates are accumulated on. Should be
            None when ``src`` is an instance of :class:`~ignite.metrics.metric.Metric`, as the running average will
            use the ``src``'s device. Otherwise, defaults to CPU. Only applicable when the computed value
            from the metric is a tensor.

    Examples:

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode:: 1

            default_trainer = get_default_trainer()

            accuracy = Accuracy()
            metric = RunningAverage(accuracy)
            metric.attach(default_trainer, 'running_avg_accuracy')

            @default_trainer.on(Events.ITERATION_COMPLETED)
            def log_running_avg_metrics():
                print(default_trainer.state.metrics['running_avg_accuracy'])

            y_true = [torch.tensor(y) for y in [[0], [1], [0], [1], [0], [1]]]
            y_pred = [torch.tensor(y) for y in [[0], [0], [0], [1], [1], [1]]]

            state = default_trainer.run(zip(y_pred, y_true))

        .. testoutput:: 1

            1.0
            0.98
            0.98039...
            0.98079...
            0.96117...
            0.96195...

        .. testcode:: 2

            default_trainer = get_default_trainer()

            metric = RunningAverage(output_transform=lambda x: x.item())
            metric.attach(default_trainer, 'running_avg_accuracy')

            @default_trainer.on(Events.ITERATION_COMPLETED)
            def log_running_avg_metrics():
                print(default_trainer.state.metrics['running_avg_accuracy'])

            y = [torch.tensor(y) for y in [[0], [1], [0], [1], [0], [1]]]

            state = default_trainer.run(y)

        .. testoutput:: 2

            0.0
            0.020000...
            0.019600...
            0.039208...
            0.038423...
            0.057655...
    """

    required_output_keys = None

    def __init__(
        self,
        src: Optional[Metric] = None,
        alpha: float = 0.98,
        output_transform: Optional[Callable] = None,
        epoch_bound: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ):
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
            setattr(self, "iteration_completed", self._metric_iteration_completed)
            device = src._device
        else:
            if output_transform is None:
                raise ValueError(
                    "Argument output_transform should not be None if src corresponds "
                    "to the output of process function."
                )
            self._get_src_value = self._get_output_value
            setattr(self, "update", self._output_update)
            if device is None:
                device = torch.device("cpu")

        self.alpha = alpha
        self.epoch_bound = epoch_bound
        super(RunningAverage, self).__init__(output_transform=output_transform, device=device)  # type: ignore[arg-type]

    @reinit__is_reduced
    def reset(self) -> None:
        self._value: Optional[Union[float, torch.Tensor]] = None

    @reinit__is_reduced
    def update(self, output: Sequence) -> None:
        # Implement abstract method
        pass

    def compute(self) -> Union[torch.Tensor, float]:
        if self._value is None:
            self._value = self._get_src_value()
        else:
            self._value = self._value * self.alpha + (1.0 - self.alpha) * self._get_src_value()

        return self._value

    def attach(self, engine: Engine, name: str, _usage: Union[str, MetricUsage] = EpochWise()) -> None:
        if self.epoch_bound:
            # restart average every epoch
            engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        # compute metric
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        # apply running average
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed, name)

    def _get_metric_value(self) -> Union[torch.Tensor, float]:
        return self.src.compute()

    @sync_all_reduce("src")
    def _get_output_value(self) -> Union[torch.Tensor, float]:
        # we need to compute average instead of sum produced by @sync_all_reduce("src")
        output = cast(Union[torch.Tensor, float], self.src) / idist.get_world_size()
        return output

    def _metric_iteration_completed(self, engine: Engine) -> None:
        self.src.started(engine)
        self.src.iteration_completed(engine)

    @reinit__is_reduced
    def _output_update(self, output: Union[torch.Tensor, float]) -> None:
        if isinstance(output, torch.Tensor):
            output = output.detach().to(self._device, copy=True)
        self.src = output  # type: ignore[assignment]
