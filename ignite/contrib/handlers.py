try:
    from tqdm import tqdm
except ImportError:
    raise RuntimeError("This contrib module requires tqdm to be installed")

from ignite.engine import Events


class ProgressBar:
    """
    TQDM progress bar handler to log training progress and computed metrics

    Args:
        loader: iterable or dataloader object
        output_transform: transform a function that transforms engine.state.output
                into a dictionary of format {name: value}

    Example:
        (...)
        pbar = ProgressBar(train_loader, output_transform=lambda x: {'loss': x})
        trainer.add_handler(Events.ITERATION_COMPLETED, pbar)
    """

    def __init__(self, engine, loader, output_transform=lambda x: x):
        self.num_iterations = len(loader)
        self.metrics = {}
        self.alpha = 0.98
        self.output_transform = output_transform
        self.pbar = None

        engine.add_event_handler(Events.EPOCH_STARTED, self._reset)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self._close)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self._log_message)

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
        message = 'Epoch {}'.format(engine.state.epoch)
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
