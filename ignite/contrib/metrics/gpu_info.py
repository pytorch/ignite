# -*- coding: utf-8 -*-
import warnings

import torch

from ignite.engine import Events
from ignite.metrics import Metric


class GpuInfo(Metric):
    """Provides GPU information: a) used memory percentage, b) gpu utilization percentage values as Metric
    on each iterations.

    .. Note ::

        In case if gpu utilization reports "N/A" on a given GPU, corresponding metric value is not set.

    Examples:

        .. code-block:: python

            # Default GPU measurements
            GpuInfo().attach(trainer, name='gpu')  # metric names are 'gpu:X mem(%)', 'gpu:X util(%)'

            # Logging with TQDM
            ProgressBar(persist=True).attach(trainer, metric_names=['gpu:0 mem(%)', 'gpu:0 util(%)'])
            # Progress bar will looks like
            # Epoch [2/10]: [12/24]  50%|█████      , gpu:0 mem(%)=79, gpu:0 util(%)=59 [00:17<1:23]

            # Logging with Tensorboard
            tb_logger.attach(trainer,
                             log_handler=OutputHandler(tag="training", metric_names='all'),
                             event_name=Events.ITERATION_COMPLETED)
    """

    def __init__(self):
        try:
            import pynvml
        except ImportError:
            raise RuntimeError(
                "This contrib module requires pynvml to be installed. "
                "Please install it with command: \n pip install pynvml"
            )
            # Let's check available devices
        if not torch.cuda.is_available():
            raise RuntimeError("This contrib module requires available GPU")

        from pynvml.smi import nvidia_smi

        # Let it fail if no libnvidia drivers or NMVL library found
        self.nvsmi = nvidia_smi.getInstance()
        super(GpuInfo, self).__init__()

    def reset(self):
        pass

    def update(self, output):
        pass

    def compute(self):
        data = self.nvsmi.DeviceQuery("memory.used, memory.total, utilization.gpu")
        if len(data) == 0 or ("gpu" not in data):
            warnings.warn("No GPU information available")
            return []
        return data["gpu"]

    def completed(self, engine, name):
        data = self.compute()
        if len(data) < 1:
            warnings.warn("No GPU information available")
            return

        for i, data_by_rank in enumerate(data):
            mem_name = "{}:{} mem(%)".format(name, i)

            if "fb_memory_usage" not in data_by_rank:
                warnings.warn("No GPU memory usage information available in {}".format(data_by_rank))
                continue
            mem_report = data_by_rank["fb_memory_usage"]
            if not ("used" in mem_report and "total" in mem_report):
                warnings.warn(
                    "GPU memory usage information does not provide used/total "
                    "memory consumption information in {}".format(mem_report)
                )
                continue

            engine.state.metrics[mem_name] = int(mem_report["used"] * 100.0 / mem_report["total"])

        for i, data_by_rank in enumerate(data):
            util_name = "{}:{} util(%)".format(name, i)
            if "utilization" not in data_by_rank:
                warnings.warn("No GPU utilization information available in {}".format(data_by_rank))
                continue
            util_report = data_by_rank["utilization"]
            if not ("gpu_util" in util_report):
                warnings.warn(
                    "GPU utilization information does not provide 'gpu_util' information in {}".format(util_report)
                )
                continue
            try:
                engine.state.metrics[util_name] = int(util_report["gpu_util"])
            except ValueError:
                # Do not set GPU utilization information
                pass

    def attach(self, engine, name="gpu", event_name=Events.ITERATION_COMPLETED):
        engine.add_event_handler(event_name, self.completed, name)
