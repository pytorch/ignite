from threading import Thread
from time import sleep, time
from psutil import cpu_count, cpu_percent, virtual_memory
from torch.utils.tensorboard import SummaryWriter


class CpuInfo(Thread):
    """
    Logger for the CPU resources CPU and RAM utilization run on a separate thread.

    `CpuInfo` is implemented on a separate thread as any attachment to an event would would effectively measure
    the GPU/CPU-utilization of the downtime, as all events are not fired during the `Engine().process()` where the
    GPU/CPU is in use. Triggering the logging independently will randomize the measurements during
    _up times_ (when `Engine().process()` is running) and _down times_.

    Args:
        logger_directory: directory for tensorboard logs
        logger_name: name of logger
        log_interval_seconds: logging interval in seconds. This interval practically cannot be reduced below 0.1sec \
        as this is the minimum CPU measurement time of `psutil.cpu_percent` (see docu).
        per_cpu (bool): log overall CPU-utilization (`False`) by default or each CPU separately (`True`)
        unit (['KB', 'MB', 'GB']): logging unit defaults to `'GB'`
    """

    def __init__(self, logger_directory, logger_name='CPULogger', log_interval_seconds=0.5, per_cpu=False, unit='GB'):
        super().__init__(name=logger_name, daemon=True)
        # CAUTION: Always avoid more than one `SummaryWriter` logging to the same directory
        # because this will lead to log file losses
        self.logger_directory = logger_directory
        self._log_interval_seconds = log_interval_seconds
        self._per_cpu = per_cpu
        self.cpu_count = cpu_count()
        self._tb_logger = SummaryWriter(logdir=self.logger_directory)
        self._memory_stats_to_log = ['total', 'used', 'free']
        self._log_cpu = True
        self._unit = unit
        self._units = {'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
        self._start_time = None

    def run(self):
        """
        Target function of the thread that logs cpu resources to tensoboard till it is closed.
        CAUTION:
            DO NOT CALL `self.run()`  on its own but CALL `self.start()` inherited from `Thread`.Otherwise
            `self.run()` will simple be executed instead of passed as target function to the thread.
        :return:
        """
        self._start_time = time()
        while self._log_cpu:
            self._log_cpu_utilization()
            self._log_cpu_memory()
            sleep(secs=self._log_interval_seconds)

    def _log_cpu_memory(self):
        # Get current memory stats
        memory_sizes = virtual_memory()._asdict()
        memory_stats = {}
        # Select memory statistics to be logged and calculate units
        for mem_stat in self._memory_stats_to_log:
            memory_stats[mem_stat] = memory_sizes[mem_stat] / self._units[self._unit]
        # log memory statistics to tensorboard
        self._tb_logger.add_scalars(main_tag='CPU_memory_' + self._unit,
                                    tag_scalar_dict=memory_stats,
                                    global_step=time() - self._start_time)

    def _log_cpu_utilization(self):
        # Get current CPU utilization in percent
        # NOTE: the argument `interval=0.1` is shortest (usefull) interval for cpu logging of `cpu_percent`, see docu
        cpu_percentages = cpu_percent(interval=0.1, percpu=self._per_cpu)
        cpu_utilization = {}
        if self._per_cpu:
            for idx_cpu, cpu_percentage in enumerate(cpu_percentages):
                cpu_utilization['CPU{}'.format(idx_cpu)] = cpu_percentage
        else:
            cpu_utilization['overall'] = cpu_percentages
        # log CPU utilization to tensorboard
        self._tb_logger.add_scalars(main_tag='CPUs_utilization_percentage',
                                    tag_scalar_dict=cpu_utilization,
                                    global_step=time() - self._start_time)

    def close(self):
        # Quit while-loop in `self.run()`
        self._log_cpu = False
        # Close tensorboard logger
        self._tb_logger.close()
        # Join thread
        self.join()
