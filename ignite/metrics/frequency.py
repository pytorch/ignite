import torch

import ignite.distributed as idist
from ignite.engine import Events
from ignite.handlers.timing import Timer
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce


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
            wps_metric.attach(trainer, name='wps', event_name=Events.ITERATION_COMPLETED(every=50))
            # Logging with TQDM
            ProgressBar(persist=True).attach(trainer, metric_names=['wps'])
            # Progress bar will look like
            # Epoch [2/10]: [50/100]  50%|█████      , wps=400 [00:17<00:35]
    """

    def __init__(self, output_transform=lambda x: x, device=None):
        self._timer = None
        self._acc = None
        self._n = None
        self._elapsed = None
        super(Frequency, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._timer = Timer()
        self._acc = 0
        self._n = 0
        self._elapsed = 0.0
        super(Frequency, self).reset()

    @reinit__is_reduced
    def update(self, output):
        self._acc += output
        self._n = self._acc
        self._elapsed = torch.tensor(self._timer.value(), device=self._device)

    @sync_all_reduce("_n", "_elapsed")
    def compute(self):
        time_divisor = 1.0

        if idist.get_world_size() > 1:
            time_divisor *= idist.get_world_size()

        # Returns the average processed objects per second across all workers
        return self._n / self._elapsed.item() * time_divisor

    def completed(self, engine, name):
        engine.state.metrics[name] = int(self.compute())

    def attach(self, engine, name, event_name=Events.ITERATION_COMPLETED):
        engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        engine.add_event_handler(event_name, self.completed, name)
