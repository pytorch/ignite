import torch
import torch.distributed as dist

from ignite.engine import Events
from ignite.metrics import Metric
from ignite.handlers.timing import Timer
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class Frequency(Metric):
    """Provides metrics for the number of examples processed per second.

    Examples:

        .. code-block:: python

            # Compute number of tokens processed
            wps_metric = Frequency(output_transformer=lambda x: x['ntokens'])
            wps_metric.attach(trainer, name='wps')
            # Logging with TQDM
            ProgressBar(persist=True).attach(trainer, metric_names=['wps'])
            # Progress bar will looks like
            # Epoch [2/10]: [12/24]  50%|█████      , wps=400 [00:17<1:23]
    """

    def __init__(self, output_transform=lambda x: x, device=None):
        self._timer = None
        self._n = None
        super(Frequency, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self.timer = Timer()
        self._n = 0
        super(Frequency, self).reset()

    @reinit__is_reduced
    def update(self, output):
        self._n += output

    @sync_all_reduce("_n")
    def compute(self):
        elapsed = torch.tensor(self.timer.value(), device=self._device)

        if not (dist.is_available() and dist.is_initialized()):
            return self._n / elapsed.item()

        dist.barrier()
        # Reduces the time across all workers into `elapsed`
        dist.all_reduce(elapsed)
        # Returns the average processed objects per second across all workers
        return self._n / elapsed.item() * dist.get_world_size()

    def completed(self, engine, name):
        engine.state.metrics[name] = int(self.compute())

    def attach(self, engine, name, event_name=Events.ITERATION_COMPLETED):
        engine.add_event_handler(event_name, self.iteration_completed)
        engine.add_event_handler(event_name, self.completed, name)
