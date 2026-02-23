from typing import Any, Optional, Union

from ignite.engine.events import CallableEventWithFilter, Events

__all__ = [
    "Usage",
    "EpochWise",
    "BatchWise",
    "BatchFiltered",
    "RunningEpochWise",
    "RunningBatchWise",
    "SingleEpochRunningBatchWise",
    "RunWise",
]


class Usage:
    """
    Base class for all usages of metrics and handlers.

    A usage defines the events when a metric/handler starts to compute, updates and completes.
    Valid events are from :class:`~ignite.engine.events.Events`.

    Args:
        started: optional event when the metric/handler starts to compute. This event will be associated to
            :meth:`~ignite.metrics.metric.Metric.started`.
        completed: optional event when the metric/handler completes. This event will be associated to
            :meth:`~ignite.metrics.metric.Metric.completed`.
        iteration_completed: optional event when the metric/handler updates. This event will be associated to
            :meth:`~ignite.metrics.metric.Metric.iteration_completed`.
    """

    usage_name: str

    def __init__(
        self,
        started: Optional[Union[Events, CallableEventWithFilter, Any]] = None,
        completed: Optional[Union[Events, CallableEventWithFilter, Any]] = None,
        iteration_completed: Optional[Union[Events, CallableEventWithFilter, Any]] = None,
    ) -> None:
        self.__started = started
        self.__completed = completed
        self.__iteration_completed = iteration_completed

    @property
    def STARTED(self) -> Optional[Union[Events, CallableEventWithFilter, Any]]:
        return self.__started

    @property
    def COMPLETED(self) -> Optional[Union[Events, CallableEventWithFilter, Any]]:
        return self.__completed

    @property
    def ITERATION_COMPLETED(self) -> Optional[Union[Events, CallableEventWithFilter, Any]]:
        return self.__iteration_completed


class EpochWise(Usage):
    """
    Epoch-wise usage. It's the default and most common usage.

    Methods are triggered on the following engine events:

    - :meth:`started` on every ``EPOCH_STARTED``
      (See :class:`~ignite.engine.events.Events`).
    - :meth:`iteration_completed` on every ``ITERATION_COMPLETED``.
    - :meth:`completed` on every ``EPOCH_COMPLETED``.

    Attributes:
        usage_name: usage name string
    """

    usage_name: str = "epoch_wise"

    def __init__(self) -> None:
        super(EpochWise, self).__init__(
            started=Events.EPOCH_STARTED,
            completed=Events.EPOCH_COMPLETED,
            iteration_completed=Events.ITERATION_COMPLETED,
        )


class RunningEpochWise(EpochWise):
    """
    Running epoch-wise usage. It's the running version of the :class:`~.base.usage.EpochWise` usage.
    A metric with such a usage most likely accompanies an :class:`~.base.usage.EpochWise` one to compute
    a running measure of it e.g. running average.

    Methods are triggered on the following engine events:

    - :meth:`started` on every ``STARTED``
      (See :class:`~ignite.engine.events.Events`).
    - :meth:`iteration_completed` on every ``EPOCH_COMPLETED``.
    - :meth:`completed` on every ``EPOCH_COMPLETED``.

    Attributes:
        usage_name: usage name string
    """

    usage_name: str = "running_epoch_wise"

    def __init__(self) -> None:
        super(EpochWise, self).__init__(
            started=Events.STARTED,
            completed=Events.EPOCH_COMPLETED,
            iteration_completed=Events.EPOCH_COMPLETED,
        )


class BatchWise(Usage):
    """
    Batch-wise usage.

    Methods are triggered on the following engine events:

    - :meth:`started` on every ``ITERATION_STARTED``
      (See :class:`~ignite.engine.events.Events`).
    - :meth:`iteration_completed` on every ``ITERATION_COMPLETED``.
    - :meth:`completed` on every ``ITERATION_COMPLETED``.

    Attributes:
        usage_name: usage name string
    """

    usage_name: str = "batch_wise"

    def __init__(self) -> None:
        super(BatchWise, self).__init__(
            started=Events.ITERATION_STARTED,
            completed=Events.ITERATION_COMPLETED,
            iteration_completed=Events.ITERATION_COMPLETED,
        )


class RunningBatchWise(BatchWise):
    """
    Running batch-wise usage. It's the running version of the :class:`~.base.usage.EpochWise` usage.
    A metric with such a usage could for example accompany a :class:`~.base.usage.BatchWise` one to compute
    a running measure of it e.g. running average.

    Methods are triggered on the following engine events:

    - :meth:`started` on every ``STARTED``
      (See :class:`~ignite.engine.events.Events`).
    - :meth:`iteration_completed` on every ``ITERATION_COMPLETED``.
    - :meth:`completed` on every ``ITERATION_COMPLETED``.

    Attributes:
        usage_name: usage name string
    """

    usage_name: str = "running_batch_wise"

    def __init__(self) -> None:
        super(BatchWise, self).__init__(
            started=Events.STARTED,
            completed=Events.ITERATION_COMPLETED,
            iteration_completed=Events.ITERATION_COMPLETED,
        )


class SingleEpochRunningBatchWise(BatchWise):
    """
    Running batch-wise usage in a single epoch. It's like :class:`~.base.usage.RunningBatchWise` usage
    with the difference that is used during a single epoch.

    Methods are triggered on the following engine events:

    - :meth:`started` on every ``EPOCH_STARTED``
      (See :class:`~ignite.engine.events.Events`).
    - :meth:`iteration_completed` on every ``ITERATION_COMPLETED``.
    - :meth:`completed` on every ``ITERATION_COMPLETED``.

    Attributes:
        usage_name: usage name string
    """

    usage_name: str = "single_epoch_running_batch_wise"

    def __init__(self) -> None:
        super(BatchWise, self).__init__(
            started=Events.EPOCH_STARTED,
            completed=Events.ITERATION_COMPLETED,
            iteration_completed=Events.ITERATION_COMPLETED,
        )


class BatchFiltered(Usage):
    """
    Batch filtered usage. This usage is similar to epoch-wise but update event is filtered.

    Methods are triggered on the following engine events:

    - :meth:`started` on every ``EPOCH_STARTED``
      (See :class:`~ignite.engine.events.Events`).
    - :meth:`iteration_completed` on filtered ``ITERATION_COMPLETED``.
    - :meth:`completed` on every ``EPOCH_COMPLETED``.

    Args:
        args: Positional arguments to setup :attr:`~ignite.engine.events.Events.ITERATION_COMPLETED`
        kwargs: Keyword arguments to setup :attr:`~ignite.engine.events.Events.ITERATION_COMPLETED`
            handled by :meth:`iteration_completed`.

    """

    def __init__(self, *args: any, **kwargs: any) -> None:
        super(BatchFiltered, self).__init__(
            started=Events.EPOCH_STARTED,
            completed=Events.EPOCH_COMPLETED,
            iteration_completed=Events.ITERATION_COMPLETED(*args, **kwargs),
        )


class RunWise(Usage):
    """
    Run-wise usage of Handlers. It's the default and most common usage of handlers.

    Methods are triggered on the following engine events:

    - :meth:`started` on every ``STARTED``
      (See :class:`~ignite.engine.events.Events`).
    - :meth:`iteration_completed` on every ``ITERATION_COMPLETED``.
    - :meth:`completed` on every ``COMPLETED``.

    Attributes:
        usage_name: usage name string
    """

    usage_name: str = "run_wise"

    def __init__(self) -> None:
        super(RunWise, self).__init__(
            started=Events.STARTED,
            completed=Events.COMPLETED,
            iteration_completed=Events.ITERATION_COMPLETED,
        )
