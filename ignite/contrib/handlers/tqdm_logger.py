# -*- coding: utf-8 -*-
try:
    from tqdm import tqdm
except ImportError:
    raise RuntimeError("This contrib module requires tqdm to be installed")

from ignite.engine import Events


class ProgressBar(object):
    """
    TQDM progress bar handler to log training progress and computed metrics.

    Args:
        persist (bool, optional): set to ``True`` to persist the progress bar after completion (default = ``False``)
        bar_format  (str, optional): Specify a custom bar string formatting. May impact performance.
            [default: '{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]'].
            Set to ``None`` to use ``tqdm`` default bar formatting: '{l_bar}{bar}{r_bar}', where
            l_bar='{desc}: {percentage:3.0f}%|' and
            r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'. For more details on the
            formatting, see `tqdm docs <https://tqdm.github.io/docs/tqdm/>`_.
        **tqdm_kwargs: kwargs passed to tqdm progress bar.
            By default, progress bar description displays "Epoch [5/10]" where 5 is the current epoch and 10 is the
            number of epochs. If tqdm_kwargs defines `desc`, e.g. "Predictions", than the description is
            "Predictions [5/10]" if number of epochs is more than one otherwise it is simply "Predictions".

    Examples:

        Simple progress bar

        .. code-block:: python

            trainer = create_supervised_trainer(model, optimizer, loss)

            pbar = ProgressBar()
            pbar.attach(trainer)

            # Progress bar will looks like
            # Epoch [2/50]: [64/128]  50%|█████      [06:17<12:34]

        Attach metrics that already have been computed at :attr:`~ignite.engine.Events.ITERATION_COMPLETED`
        (such as :class:`~ignite.metrics.RunningAverage`)

        .. code-block:: python

            trainer = create_supervised_trainer(model, optimizer, loss)

            RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

            pbar = ProgressBar()
            pbar.attach(trainer, ['loss'])

            # Progress bar will looks like
            # Epoch [2/50]: [64/128]  50%|█████      , loss=12.34e-02 [06:17<12:34]

        Directly attach the engine's output

        .. code-block:: python

            trainer = create_supervised_trainer(model, optimizer, loss)

            pbar = ProgressBar()
            pbar.attach(trainer, output_transform=lambda x: {'loss': x})

            # Progress bar will looks like
            # Epoch [2/50]: [64/128]  50%|█████      , loss=12.34e-02 [06:17<12:34]

    Note:
        When adding attaching the progress bar to an engine, it is recommend that you replace
        every print operation in the engine's handlers triggered every iteration with
        ``pbar.log_message`` to guarantee the correct format of the stdout.
    """

    def __init__(self, persist=False,
                 bar_format='{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]',
                 **tqdm_kwargs):
        self.pbar = None
        self.persist = persist
        self.bar_format = bar_format
        self.tqdm_kwargs = tqdm_kwargs

    def _reset(self, engine):
        self.pbar = tqdm(
            total=len(engine.state.dataloader),
            leave=self.persist,
            bar_format=self.bar_format,
            **self.tqdm_kwargs
        )

    def _close(self, engine):
        self.pbar.close()
        self.pbar = None

    def _update(self, engine, metric_names=None, output_transform=None):
        if self.pbar is None:
            self._reset(engine)

        desc = self.tqdm_kwargs.get("desc", "Epoch")
        if engine.state.max_epochs > 1:
            desc += " [{}/{}]".format(engine.state.epoch, engine.state.max_epochs)
        self.pbar.set_description(desc)

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
        Logs a message, preserving the progress bar correct output format.

        Args:
            message (str): string you wish to log.
        """
        tqdm.write(message)

    def attach(self, engine, metric_names=None, output_transform=None):
        """
        Attaches the progress bar to an engine object.

        Args:
            engine (Engine): engine object.
            metric_names (list, optional): list of the metrics names to log as the bar progresses
            output_transform (callable, optional): a function to select what you want to print from the engine's
                output. This function may return either a dictionary with entries in the format of ``{name: value}``,
                or a single scalar, which will be displayed with the default name `output`.
        """
        if metric_names is not None and not isinstance(metric_names, list):
            raise TypeError("metric_names should be a list, got {} instead.".format(type(metric_names)))

        if output_transform is not None and not callable(output_transform):
            raise TypeError("output_transform should be a function, got {} instead."
                            .format(type(output_transform)))

        engine.add_event_handler(Events.ITERATION_COMPLETED, self._update, metric_names, output_transform)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self._close)
