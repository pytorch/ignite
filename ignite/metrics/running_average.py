import warnings
from typing import Any, Callable, cast, Optional, Union

import torch

import ignite.distributed as idist
from ignite.engine import Engine, Events
from ignite.metrics.metric import Metric, MetricUsage, reinit__is_reduced, RunningBatchWise, SingleEpochRunningBatchWise

__all__ = ["RunningAverage"]


class RunningAverage(Metric):
    """Compute running average of a metric or the output of process function.

    Args:
        src: input source: an instance of :class:`~ignite.metrics.metric.Metric` or None. The latter
            corresponds to `engine.state.output` which holds the output of process function.
        alpha: running average decay factor, default 0.98
        output_transform: a function to use to transform the output if `src` is None and
            corresponds the output of process function. Otherwise it should be None.
        epoch_bound: whether the running average should be reset after each epoch. It is depracated in favor of
            ``usage`` argument in :meth:`attach` method. Setting ``epoch_bound`` to ``False`` is equivalent to
            ``usage=SingleEpochRunningBatchWise()`` and setting it to ``True`` is equivalent to
            ``usage=RunningBatchWise()`` in the :meth:`attach` method. Default None.
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
    _state_dict_all_req_keys = ("_value", "src")

    def __init__(
        self,
        src: Optional[Metric] = None,
        alpha: float = 0.98,
        output_transform: Optional[Callable] = None,
        epoch_bound: Optional[bool] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if not (isinstance(src, Metric) or src is None):
            raise TypeError("Argument src should be a Metric or None.")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("Argument alpha should be a float between 0.0 and 1.0.")

        if isinstance(src, Metric):
            if output_transform is not None:
                raise ValueError("Argument output_transform should be None if src is a Metric.")

            def output_transform(x: Any) -> Any:
                return x

            if device is not None:
                raise ValueError("Argument device should be None if src is a Metric.")
            self.src: Union[Metric, None] = src
            device = src._device
        else:
            if output_transform is None:
                raise ValueError(
                    "Argument output_transform should not be None if src corresponds "
                    "to the output of process function."
                )
            self.src = None
            if device is None:
                device = torch.device("cpu")

        if epoch_bound is not None:
            warnings.warn(
                "`epoch_bound` is deprecated and will be removed in the future. Consider using `usage` argument of"
                "`attach` method instead. `epoch_bound=True` is equivalent with `usage=SingleEpochRunningBatchWise()`"
                " and `epoch_bound=False` is equivalent with `usage=RunningBatchWise()`."
            )
        self.epoch_bound = epoch_bound
        self.alpha = alpha
        super(RunningAverage, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self._value: Optional[Union[float, torch.Tensor]] = None
        if isinstance(self.src, Metric):
            self.src.reset()

    @reinit__is_reduced
    def update(self, output: Union[torch.Tensor, float]) -> None:
        if self.src is None:
            output = output.detach().to(self._device, copy=True) if isinstance(output, torch.Tensor) else output
            value = idist.all_reduce(output) / idist.get_world_size()
        else:
            value = self.src.compute()
            self.src.reset()

        if self._value is None:
            self._value = value
        else:
            self._value = self._value * self.alpha + (1.0 - self.alpha) * value

    def compute(self) -> Union[torch.Tensor, float]:
        return cast(Union[torch.Tensor, float], self._value)

    def attach(self, engine: Engine, name: str, usage: Union[str, MetricUsage] = RunningBatchWise()) -> None:
        r"""
        Attach the metric to the ``engine`` using the events determined by the ``usage``.

        Args:
            engine: the engine to get attached to.
            name: by which, the metric is inserted into ``engine.state.metrics`` dictionary.
            usage: the usage determining on which events the metric is reset, updated and computed. It should be an
                instance of the :class:`~ignite.metrics.metric.MetricUsage`\ s in the following table.

                ======================================================= ===========================================
                ``usage`` **class**                                     **Description**
                ======================================================= ===========================================
                :class:`~.metrics.metric.RunningBatchWise`              Running average of the ``src`` metric or
                                                                        ``engine.state.output`` is computed across
                                                                        batches. In the former case, on each batch,
                                                                        ``src`` is reset, updated and computed then
                                                                        its value is retrieved. Default.
                :class:`~.metrics.metric.SingleEpochRunningBatchWise`   Same as above but the running average is
                                                                        computed across batches in an epoch so it
                                                                        is reset at the end of the epoch.
                :class:`~.metrics.metric.RunningEpochWise`              Running average of the ``src`` metric or
                                                                        ``engine.state.output`` is computed across
                                                                        epochs. In the former case, ``src`` works
                                                                        as if it was attached in a
                                                                        :class:`~ignite.metrics.metric.EpochWise`
                                                                        manner and its computed value is retrieved
                                                                        at the end of the epoch. The latter case
                                                                        doesn't make much sense for this usage as
                                                                        the ``engine.state.output`` of the last
                                                                        batch is retrieved then.
                ======================================================= ===========================================

        ``RunningAverage`` retrieves ``engine.state.output`` at ``usage.ITERATION_COMPLETED`` if the ``src`` is not
        given and it's computed and updated using ``src``, by manually calling its ``compute`` method, or
        ``engine.state.output`` at ``usage.COMPLETED`` event.
        Also if ``src`` is given, it is updated at ``usage.ITERATION_COMPLETED``, but its reset event is determined by
        ``usage`` type. If ``isinstance(usage, BatchWise)`` holds true, ``src`` is reset on ``BatchWise().STARTED``,
        otherwise on ``EpochWise().STARTED`` if ``isinstance(usage, EpochWise)``.

        .. versionchanged:: 0.5.1
            Added `usage` argument
        """
        usage = self._check_usage(usage)
        if self.epoch_bound is not None:
            usage = SingleEpochRunningBatchWise() if self.epoch_bound else RunningBatchWise()

        if isinstance(self.src, Metric) and not engine.has_event_handler(
            self.src.iteration_completed, Events.ITERATION_COMPLETED
        ):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.src.iteration_completed)

        super().attach(engine, name, usage)

    def detach(self, engine: Engine, usage: Union[str, MetricUsage] = RunningBatchWise()) -> None:
        usage = self._check_usage(usage)
        if self.epoch_bound is not None:
            usage = SingleEpochRunningBatchWise() if self.epoch_bound else RunningBatchWise()

        if isinstance(self.src, Metric) and engine.has_event_handler(
            self.src.iteration_completed, Events.ITERATION_COMPLETED
        ):
            engine.remove_event_handler(self.src.iteration_completed, Events.ITERATION_COMPLETED)

        super().detach(engine, usage)
