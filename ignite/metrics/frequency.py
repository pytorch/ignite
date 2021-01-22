from typing import Any, Callable, Union

import torch

import ignite.distributed as idist
from ignite.engine import Engine, Events, CallableEventWithFilter
from ignite.handlers.timing import Timer
from ignite.metrics.metric import Metric, MetricUsage, reinit__is_reduced, sync_all_reduce


class FrequencyWise(MetricUsage):

    def __init__(self, event: CallableEventWithFilter = Events.ITERATION_COMPLETED) -> None:
        super(FrequencyWise, self).__init__(
            started=Events.EPOCH_STARTED,
            completed=event,
            iteration_completed=Events.ITERATION_COMPLETED,
        )


class Frequency(Metric):
    """Provides metrics for the number of examples processed per second.

    Examples:

        .. code-block:: python

            # Compute number of tokens processed
            wps_metric = Frequency(output_transform=lambda x: x['ntokens'])
            wps_metric.attach(trainer, name='wps')
            # Logging with TQDM
            ProgressBar(persist=True).attach(trainer, metric_names=['wps'])
            # Progress bar will look like
            # Epoch [2/10]: [12/24]  50%|█████      , wps=400 [00:17<1:23]


        To compute examples processed per second every 50th iteration:

        .. code-block:: python

            # Compute number of tokens processed
            wps_metric = Frequency(output_transform=lambda x: x['ntokens'])
            wps_metric.attach(trainer, name='wps', usage=FrequencyWise(Events.ITERATION_COMPLETED(every=50)))
            # Logging with TQDM
            ProgressBar(persist=True).attach(trainer, metric_names=['wps'])
            # Progress bar will look like
            # Epoch [2/10]: [50/100]  50%|█████      , wps=400 [00:17<00:35]
    """

    def __init__(
        self, output_transform: Callable = lambda x: x, device: Union[str, torch.device] = torch.device("cpu")
    ) -> None:
        super(Frequency, self).__init__(output_transform=output_transform, device=device)
        self._timer = Timer()
        self._acc = 0
        self._n = 0
        self._elapsed = 0.0

    @reinit__is_reduced
    def reset(self) -> None:
        self._timer = Timer()
        self._acc = 0
        self._n = 0
        self._elapsed = 0.0

    @reinit__is_reduced
    def update(self, output: int) -> None:
        self._acc += output
        self._n = self._acc
        self._elapsed = self._timer.value()

    @sync_all_reduce("_n", "_elapsed")
    def compute(self) -> float:
        # Returns the average processed objects per second across all workers
        return int(self._n / self._elapsed * idist.get_world_size())

    # override the method attach() of Metrics to define a different default value for usage
    def attach(self, engine: Engine, name: str, usage: Union[str, MetricUsage] = FrequencyWise()) -> None:
        super(Frequency, self).attach(engine, name, usage)
