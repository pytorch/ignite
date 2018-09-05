try:
    from tqdm import tqdm
except ImportError:
    raise RuntimeError("This contrib module requires tqdm to be installed")

from ignite.engine import Events


class ProgressBar:
    """
    TQDM progress bar handler to log training progress and computed metrics

    Args:
        engine (ignite.Engine): an engine object
        loader (iterable or DataLoader): data loader object
        output_transform: transform a function that transforms engine.state.output
                into a dictionary of format {metric_name: metric_value}
        mode (str): 'iteration' or 'epoch' (default=epoch)
        log_interval (int or None): interval of which the metrics information is displayed.
                            If set to None, only the progress bar is shown and not
                            the metrics. (default=1)

    Example:
        (...)
        pbar = ProgressBar(trainer, train_loader, output_transform=lambda x: {'loss': x})
        trainer.add_event_handler(Events.ITERATION_COMPLETED, pbar)

    Note:
        Bear in mind that `output_transform` should return a dictionary, whose values are floats.
        This is due to the running average over every metric that is being computed. If you have
        metrics that are tensors or arrays, you will have to unroll each value to its own
        dictionary key.
    """

    def __init__(self, engine, loader, output_transform=lambda x: x, mode='epoch', log_interval=1):
        self.num_iterations = len(loader)
        self.metrics = {}
        self.alpha = 0.98
        self.output_transform = output_transform
        self.pbar = None
        self.mode = mode
        self.log_interval = log_interval

        if log_interval is not None:
            assert log_interval >= 1, 'log_frequency must be positive'

        assert mode in {'iteration', 'epoch'}, \
            'incompatible mode {}, accepted modes {}'.format(mode, {'iteration', 'epoch'})

        log_event = Events.EPOCH_COMPLETED if mode == 'epoch' else Events.ITERATION_COMPLETED

        engine.add_event_handler(Events.EPOCH_STARTED, self._reset)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self._close)
        engine.add_event_handler(log_event, self._log_message)

    def _calc_running_avg(self, engine):
        output = self.output_transform(engine.state.output)
        for k, v in output.items():
            old_v = self.metrics.get(k, v)
            new_v = self.alpha * old_v + (1 - self.alpha) * v
            self.metrics[k] = new_v

    def _reset(self, engine):
        self.pbar = tqdm(
            total=self.num_iterations,
            leave=False,
            bar_format='{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]')

    def _close(self, engine):
        self.pbar.close()

    def _log_message(self, engine):

        i = engine.state.epoch if self.mode == 'epoch' else engine.state.iteration

        if self.log_interval and i % self.log_interval == 0:
            if self.mode == 'epoch':
                message = 'Epoch {}'.format(engine.state.epoch)
            else:
                message = 'Iteration {}'.format(engine.state.iteration)

            for name, value in self.metrics.items():
                message += ' | {}={:.2e}'.format(name, value)

            tqdm.write(message)

    def _format_metrics(self):
        formatted_metrics = {}
        for key in self.metrics:
            formatted_metrics[key] = '{:.2e}'.format(self.metrics[key])
        return formatted_metrics

    def __call__(self, engine):
        self._calc_running_avg(engine)
        self.pbar.set_description('Epoch {}'.format(engine.state.epoch))
        self.pbar.set_postfix(**self._format_metrics())
        self.pbar.update()
