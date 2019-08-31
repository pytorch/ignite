import warnings

import torch

from ignite.metrics import Metric
from ignite.engine import Events


class GpuMemory(Metric):
    """GPU used / max memory values as Metric.

    Examples:

        .. code-block:: python

            # Default GPU measurement
            GpuMemory().attach(trainer)  # default metric name is 'gpu memory'
            ProgressBar(persist=True).attach(trainer, metric_names=['gpu memory', ])

    """

    def __init__(self):
        try:
            import pynvml
        except ImportError:
            raise RuntimeError("This contrib module requires pynvml to be installed. "
                               "Please install it with command: \n pip install pynvml")
            # Let's check available devices
        if not torch.cuda.is_available():
            raise RuntimeError("This contrib module requires available GPU")

        from pynvml.smi import nvidia_smi
        # Let it fail if no libnvidia drivers or NMVL library found
        self.nvsmi = nvidia_smi.getInstance()
        super(GpuMemory, self).__init__()

    def reset(self):
        pass

    def update(self, output):
        pass

    def compute(self):
        data = self.nvsmi.DeviceQuery('memory.used, memory.total')
        if len(data) == 0 or ('gpu' not in data):
            warnings.warn("No GPU information available")
            return []
        return data['gpu']

    def completed(self, engine, name, local_rank):
        data = self.compute()
        if local_rank >= len(data):
            warnings.warn("No GPU information available")
            return
        if 'fb_memory_usage' not in data[local_rank]:
            warnings.warn("No GPU memory usage information available in {}".format(data[local_rank]))
            return
        report = data[local_rank]['fb_memory_usage']
        if not ('used' in report and 'total' in report):
            warnings.warn("GPU memory usage information does not provide used/total memory consumption information in "
                          "{}".format(report))
            return
        engine.state.metrics[name] = "{} / {} MiB".format(int(report['used']), int(report['total']))

    def attach(self, engine, name="gpu memory", event_name=Events.ITERATION_COMPLETED, local_rank=0):
        engine.add_event_handler(event_name, self.completed, name, local_rank)
