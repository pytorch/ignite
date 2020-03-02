from threading import Thread
from time import sleep, time
from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, \
    nvmlDeviceGetName, nvmlDeviceGetUtilizationRates
from torch.utils.tensorboard import SummaryWriter


class GpuPynvmlLogger(Thread):
    """
    Logger for the GPU resources GPU and RAM utilization.

    `CpuInfo` is implemented on a separate thread as any attachment to an event would would effectively measure
    the GPU/CPU-utilization of the downtime, as all events are not fired during the `Engine().process()` where the
    GPU/CPU is in use. Triggering the logging independently will randomize the measurements during
    _up times_ (when `Engine().process()` is running) and _down times_.

    Args:
        logger_directory: directory for tensorboard logs
        logger_name: name of logger
        log_interval_seconds: logging interval in seconds. Decreasing the `log_interval_seconds < 0.1` may \
        reasonably increase (> ~5-30%) the GPU-utilization by the measurement task
        unit (['KB', 'MB', 'GB']): logging unit defaults to `'GB'`
    """
    def __init__(self, logger_directory, logger_name='GPULogger', log_interval_seconds=1, unit='GB'):
        super(GpuPynvmlLogger, self).__init__(name=logger_name, daemon=True)
        # CAUTION: Always avoid more than one `SummaryWriter` logging to the same directory
        # because this will lead to log file losses
        self.logger_directory = logger_directory
        self.log_interval_seconds = log_interval_seconds
        nvmlInit()
        self.gpu_count = nvmlDeviceGetCount()
        self.gpu_handles = {}
        for gpu_idx in range(self.gpu_count):
            hdl = nvmlDeviceGetHandleByIndex(gpu_idx)
            name = nvmlDeviceGetName(hdl).decode('ascii').replace(' ', '_')
            self.gpu_handles['GPU{}_{}'.format(gpu_idx, name)] = hdl
        self._tb_logger = SummaryWriter(logdir=self.logger_directory)
        self._memory_stats_to_log = ['total', 'used', 'free']
        self._log_gpu = True
        self._unit = unit
        self._units = {'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
        self._start_time = None

    def run(self):
        """
        Target function of the thread that logs GPU resources to tensoboard till it is closed.
        CAUTION:
            DO NOT CALL `self.run()`  on its own but CALL `self.start()` inherited from `Thread`.
            Otherwise `self.run()` will simple be executed in the `MainThread` instead of passed
            as target function to the new thread.
        :return:
        """
        self._start_time = time()
        while self._log_gpu:
            self._log_gpu_utilization()
            self._log_gpu_memory()
            sleep(self.log_interval_seconds)

    def _log_gpu_memory(self):
        # Get memory statistics for each GPU
        for gpu_name, gpu_hdl in self.gpu_handles.items():
            # Get current memory stats
            memory_sizes = nvmlDeviceGetMemoryInfo(handle=gpu_hdl)
            memory_stats = {}
            # Select memory statistics to be logged and calculate units
            for mem_stat in self._memory_stats_to_log:
                memory_stats[mem_stat] = memory_sizes.__getattribute__(mem_stat) / self._units[self._unit]
            # log memory statistics to tensorboard
            self._tb_logger.add_scalars(main_tag='{}_memory_{}'.format(gpu_name, self._unit),
                                        tag_scalar_dict=memory_stats,
                                        global_step=time() - self._start_time)

    def _log_gpu_utilization(self):
        gpu_utilizations = {}
        # Get current GPU utilizations in percent
        for gpu_name, gpu_hdl in self.gpu_handles.items():
            gpu_percentage = nvmlDeviceGetUtilizationRates(handle=gpu_hdl).gpu
            gpu_utilizations[gpu_name] = gpu_percentage
        # log CPU utilization to tensorboard
        self._tb_logger.add_scalars(main_tag='GPUs_utilization_percentage',
                                    tag_scalar_dict=gpu_utilizations,
                                    global_step=time() - self._start_time)

    def close(self):
        # Quit while-loop in `self.run()`
        self._log_gpu = False
        # Close tensorboard logger
        self._tb_logger.close()
        # Join thread
        self.join()