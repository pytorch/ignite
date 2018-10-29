try:
    from tqdm import tqdm
except ImportError:
    raise RuntimeError("This contrib module requires tqdm to be installed")

from ignite.engine import Events


class ProgressBar:
    """
    TQDM progress bar handler to log training progress and computed metrics.

    Args:
        persist (bool, optional): set to ``True`` to persist the progress bar after completion (default = ``False``)

    Examples:

        Simple progress bar

        .. code-block:: python

            trainer = create_supervised_trainer(model, optimizer, loss)

            pbar = ProgressBar()
            pbar.attach(trainer)

        Attach metrics that already have been computed at `ITERATION_COMPLETED` (such as `RunningAverage`)

        .. code-block:: python

            trainer = create_supervised_trainer(model, optimizer, loss)

            RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

            pbar = ProgressBar()
            pbar.attach(trainer, ['loss'])

        Directly attach the engine's output

        .. code-block:: python

            trainer = create_supervised_trainer(model, optimizer, loss)

            pbar = ProgressBar()
            pbar.attach(trainer, output_transform=lambda x: {'loss': x})

    Note:
        When adding attaching the progress bar to an engine, it is recommend that you replace
        every print operation in the engine's handlers triggered every iteration with
        ``pbar.log_message`` to guarantee the correct format of the stdout.
    """

    def __init__(self, persist=False):
        self.pbar = None
        self.persist = persist

    def _reset(self, engine):
        self.pbar = tqdm(
            total=len(engine.state.dataloader),
            leave=self.persist,
            bar_format='{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]')

    def _close(self, engine):
        self.pbar.close()
        self.pbar = None

    def _update(self, engine, metric_names=None, output_transform=None):
        if self.pbar is None:
            self._reset(engine)

        self.pbar.set_description('Epoch [{}/{}]'.format(engine.state.epoch, engine.state.max_epochs))

        metrics = {}
        if metric_names is not None:
            if not all(metric in engine.state.metrics for metric in metric_names):
                self._close(engine)
                raise KeyError("metrics not found in engine.state.metrics")

            metrics.update({name: '{:.2e}'.format(engine.state.metrics[name]) for name in metric_names})

        if output_transform is not None:
            output_dict = output_transform(engine.state.output)

            if not isinstance(output_dict, dict):
                output_dict = {"output": output_dict}

            metrics.update({name: '{:.2e}'.format(value) for name, value in output_dict.items()})

        if metrics:
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

    def attach(self, engine, metric_names=None, output_transform=None):
        """
        Attaches the progress bar to an engine object

        Args:
            engine (Engine): engine object
            metric_names (list, optional): list of the metrics names to log as the bar progresses
            output_transform (Callable, optional): a function to select what you want to print from the engine's
                output. This function may return either a dictionary with entries in the format of ``{name: value}``,
                or a single scalar, which will be displayed with the default name `output`.
        """
        if metric_names is not None and not isinstance(metric_names, list):
            raise TypeError("metric_names should be a list, got {} instead".format(type(metric_names)))

        if output_transform is not None and not callable(output_transform):
            raise TypeError("output_transform should be a function, got {} instead"
                            .format(type(output_transform)))

        engine.add_event_handler(Events.EPOCH_COMPLETED, self._close)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._update, metric_names, output_transform)
