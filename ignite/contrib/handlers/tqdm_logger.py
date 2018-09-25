try:
    from tqdm import tqdm
except ImportError:
    raise RuntimeError("This contrib module requires tqdm to be installed")

from ignite.engine import Events


class ProgressBar:
    """
    TQDM progress bar handler to log training progress and computed metrics

    Examples:

    .. code-block:: python

        pbar = ProgressBar()
        pbar.attach(trainer, len(data_loader), ['loss'], mode='iteration', log_interval=1)

    Note:
        When adding this handler to an engine, it is recommend that you replace every print
        operation in the engine's handlers with `tqdm.write()` to guarantee the correct stdout format.
    """

    def __init__(self):
        self.pbar = None

    def _reset(self, engine, num_iterations):
        self.pbar = tqdm(
            total=num_iterations,
            leave=False,
            bar_format='{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]')

    def _close(self, engine):
        self.pbar.close()

    @staticmethod
    def _log_message(engine, metric_names, mode, log_interval):

        i = engine.state.epoch if mode == 'epoch' else engine.state.iteration

        if log_interval and i % log_interval == 0:
            if mode == 'epoch':
                message = 'Epoch {}'.format(engine.state.epoch)
            else:
                message = 'Iteration {}'.format(engine.state.iteration)

            metrics = {name: '{:.2e}'.format(engine.state.metrics[name]) for name in metric_names}
            for name, value in metrics.items():
                message += ' | {}={}'.format(name, value)

            tqdm.write(message)

    def _update(self, engine, metric_names):
        metrics = {name: '{:.2e}'.format(engine.state.metrics[name]) for name in metric_names}
        self.pbar.set_description('Epoch {}'.format(engine.state.epoch))
        self.pbar.set_postfix(**metrics)
        self.pbar.update()

    def attach(self, engine, num_iterations, metric_names, mode='epoch', log_interval=1):
        """
        Attaches the progress bar to an engine object

        Args:
            engine (Engine): trainer object
            num_iterations (int): number of iterations of one epoch
            metric_names (list): list of the metrics names to log
            mode (str): 'iteration' or 'epoch' (default=epoch)
            log_interval (int or None): interval of which the metrics information is displayed.
                                If set to None, only the progress bar is shown and not
                                the metrics. (default=1)
        """

        if log_interval is not None:
            assert log_interval >= 1, 'log_frequency must be positive'

        assert mode in {'iteration', 'epoch'}, \
            'incompatible mode {}, accepted modes {}'.format(mode, {'iteration', 'epoch'})

        log_event = Events.EPOCH_COMPLETED if mode == 'epoch' else Events.ITERATION_COMPLETED

        engine.add_event_handler(Events.EPOCH_STARTED, self._reset, num_iterations)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self._close)
        engine.add_event_handler(log_event, self._log_message, metric_names, mode, log_interval)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._update, metric_names)
