from ignite.metrics import Metric
from ignite.handlers.timing import Timer
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class FrequencyMetric(Metric):
    """Provides metrics for the number of examples processed per second.

    Examples:

        .. code-block:: python

            # Compute number of tokens processed
            wps_metric = FrequencyMetric(output_transformer=lambda x: x['ntokens'])
            average_wps_metric = RunningAverage(wps_metric, alpha=1.0)
            average_wps_metric.attach(trainer, name='wps')
            # Logging with TQDM
            ProgressBar(persist=True).attach(trainer, metric_names=['wps'])
            # Progress bar will looks like
            # Epoch [2/10]: [12/24]  50%|█████      , wps=400 [00:17<1:23]
    """

    def __init__(self, output_transform=lambda x: x, device=None):
        self._timer = None
        self._n = None
        super(FrequencyMetric, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self.timer = Timer()
        self._n = 0
        super(FrequencyMetric, self).reset()

    @reinit__is_reduced
    def update(self, output):
        self._n += output

    @sync_all_reduce("_n")
    def compute(self):
        return self._n / self.timer.value()
