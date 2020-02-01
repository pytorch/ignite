from ignite.metrics import Metric
from ignite.handlers.timing import Timer
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

class FrequencyMetric(Metric):
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
