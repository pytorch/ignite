from typing import Any, Dict

from ignite.metrics import Metric


class MetricGroup(Metric):
    def __init__(self, metrics: Dict[str, Metric]):
        self.metrics = metrics
        super(MetricGroup, self).__init__()

    def reset(self):
        for m in self.metrics.values():
            m.reset()

    def update(self, output):
        for m in self.metrics.values():
            m.update(m._output_transform(output))

    def compute(self) -> Dict[str, Any]:
        return {k: m.compute() for k, m in self.metrics.items()}
