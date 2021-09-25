from typing import Callable, Optional, Sequence, Union, cast

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
        self._value = None  # type: Optional[Union[float, torch.Tensor]]

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
