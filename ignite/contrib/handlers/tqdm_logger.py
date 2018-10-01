try:
    from tqdm import tqdm
except ImportError:
    raise RuntimeError("This contrib module requires tqdm to be installed")

from ignite.engine import Events


class ProgressBar:
    """
    TQDM progress bar handler to log training progress and computed metrics.

    Examples:

        Create a progress bar that shows you some metrics as they are computed,
        by simply attaching the progress bar object to your engine.

        .. code-block:: python

            pbar = ProgressBar()
            pbar.attach(trainer, ['loss'])

    Note:
        When adding attaching the progress bar to an engine, it is recommend that you replace
        every print operation in the engine's handlers triggered every iteration with
        ``pbar.log_message`` to guarantee the correct format of the stdout.
    """

    def __init__(self):
        self.pbar = None

    def _reset(self, engine):
        self.pbar = tqdm(
            total=len(engine.state.dataloader),
            leave=False,
            bar_format='{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]')

    def _close(self, engine):
        self.pbar.close()
        self.pbar = None

    def _update(self, engine, metric_names=None):
        if self.pbar is None:
            self._reset(engine)

        self.pbar.set_description('Epoch {}'.format(engine.state.epoch))

        if metric_names is not None:
            if not engine.state.metrics \
                    or not all(metric in metric_names for metric in engine.state.metrics):
                raise KeyError("metrics not found in engine.state.metrics")

            metrics = {name: '{:.2e}'.format(engine.state.metrics[name]) for name in metric_names}
            self.pbar.set_postfix(**metrics)

        self.pbar.update()

    @staticmethod
    def log_message(message):
        """
        Logs a message, preserving the progress bar correct output format

        Args:
            message (str): string you wish to log
        """
        tqdm.write(message)

    def attach(self, engine, metric_names=None):
        """
        Attaches the progress bar to an engine object

        Args:
            engine (Engine): engine object
            metric_names (list): (Optional) list of the metrics names to log as the bar progresses
        """
        if not isinstance(metric_names, list):
            raise TypeError("metric_names should be a list, got {} instead".format(type(metric_names)))

        engine.add_event_handler(Events.EPOCH_COMPLETED, self._close)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._update, metric_names)
