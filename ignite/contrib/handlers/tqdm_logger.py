# -*- coding: utf-8 -*-
import warnings

import torch

from ignite.engine import Events

from ignite.contrib.handlers.base_logger import BaseLogger, BaseOutputHandler


class ProgressBar(BaseLogger):
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
            # Epoch [2/50]: [64/128]  50%|█████      , loss=0.123 [06:17<12:34]

        Directly attach the engine's output

        .. code-block:: python

            trainer = create_supervised_trainer(model, optimizer, loss)

            pbar = ProgressBar()
            pbar.attach(trainer, output_transform=lambda x: {'loss': x})

            # Progress bar will looks like
            # Epoch [2/50]: [64/128]  50%|█████      , loss=0.123 [06:17<12:34]

    Note:
        When adding attaching the progress bar to an engine, it is recommend that you replace
        every print operation in the engine's handlers triggered every iteration with
        ``pbar.log_message`` to guarantee the correct format of the stdout.

    Note:
        When using inside jupyter notebook, `ProgressBar` automatically uses `tqdm_notebook`. For correct rendering,
        please install `ipywidgets <https://ipywidgets.readthedocs.io/en/stable/user_install.html#installation>`_.
        Due to `tqdm notebook bugs <https://github.com/tqdm/tqdm/issues/594>`_, bar format may be needed to be set
        to an empty string value.

    """

    events_order = [
        Events.STARTED,
        Events.EPOCH_STARTED,
        Events.ITERATION_STARTED,
        Events.ITERATION_COMPLETED,
        Events.EPOCH_COMPLETED,
        Events.COMPLETED
    ]

    def __init__(self, persist=False,
                 bar_format='{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]',

                 **tqdm_kwargs):

        try:
            from tqdm.autonotebook import tqdm
        except ImportError:
            raise RuntimeError("This contrib module requires tqdm to be installed. "
                               "Please install it with command: \n pip install tqdm")

        self.pbar_cls = tqdm
        self.pbar = None
        self.persist = persist
        self.bar_format = bar_format
        self.tqdm_kwargs = tqdm_kwargs

    def _reset(self, pbar_total):
        self.pbar = self.pbar_cls(
            total=pbar_total,
            leave=self.persist,
            bar_format=self.bar_format,
            **self.tqdm_kwargs
        )

    def _close(self, engine):
        if self.pbar:
            self.pbar.close()
        self.pbar = None

    @staticmethod
    def _compare_lt(event1, event2):
        i1 = ProgressBar.events_order.index(event1)
        i2 = ProgressBar.events_order.index(event2)
        return i1 < i2

    @staticmethod
    def log_message(message):
        """
        Logs a message, preserving the progress bar correct output format.

        Args:
            message (str): string you wish to log.
        """
        from tqdm import tqdm
        tqdm.write(message)

    def attach(self, engine, metric_names=None, output_transform=None,
               event_name=Events.ITERATION_COMPLETED,
               closing_event_name=Events.EPOCH_COMPLETED):
        """
        Attaches the progress bar to an engine object.

        Args:
            engine (Engine): engine object.
            metric_names (list of str, optional): list of metric names to plot or a string "all" to plot all available
                metrics.
            output_transform (callable, optional): a function to select what you want to print from the engine's
                output. This function may return either a dictionary with entries in the format of ``{name: value}``,
                or a single scalar, which will be displayed with the default name `output`.
            event_name: event's name on which the progress bar advances. Valid events are from
                :class:`~ignite.engine.Events`.
            closing_event_name: event's name on which the progress bar is closed. Valid events are from
                :class:`~ignite.engine.Events`.

        Note: accepted output value types are numbers, 0d and 1d torch tensors and strings

        """
        desc = self.tqdm_kwargs.get("desc", "Epoch")

        if not (event_name in Events and closing_event_name in Events):
            raise ValueError("Logging and closing events should be only ignite.engine.Events")

        if not self._compare_lt(event_name, closing_event_name):
            raise ValueError("Logging event {} should be called before closing event {}"
                             .format(event_name, closing_event_name))

        log_handler = _OutputHandler(desc, metric_names, output_transform,
                                     event_name=event_name,
                                     closing_event_name=closing_event_name)
        super(ProgressBar, self).attach(engine, log_handler, event_name)
        engine.add_event_handler(closing_event_name, self._close)


class _OutputHandler(BaseOutputHandler):
    """Helper handler to log engine's output and/or metrics

    Args:
        description (str): progress bar description.
        metric_names (list of str, optional): list of metric names to plot or a string "all" to plot all available
            metrics.
        output_transform (callable, optional): output transform function to prepare `engine.state.output` as a number.
            For example, `output_transform = lambda output: output`
            This function can also return a dictionary, e.g `{'loss': loss1, `another_loss`: loss2}` to label the plot
            with corresponding keys.
        event_name: event's name on which the progress bar advances. Valid events are from
            :class:`~ignite.engine.Events` or any `event_name` added by
            :meth:`~ignite.engine.Engine.register_events`.
        closing_event_name: event's name on which the progress bar is closed. Valid events are from
            :class:`~ignite.engine.Events` or any `event_name` added by
            :meth:`~ignite.engine.Engine.register_events`.

    """
    def __init__(self, description, metric_names=None, output_transform=None,
                 event_name=Events.ITERATION_COMPLETED,
                 closing_event_name=Events.EPOCH_COMPLETED):
        if metric_names is None and output_transform is None:
            # This helps to avoid 'Either metric_names or output_transform should be defined' of BaseOutputHandler
            metric_names = []
        super(_OutputHandler, self).__init__(description, metric_names, output_transform,
                                             another_engine=None, global_step_transform=None)
        self.event_name = event_name
        self.closing_event_name = closing_event_name

    @staticmethod
    def get_max_number_events(event_name, engine):
        if event_name in (Events.ITERATION_STARTED, Events.ITERATION_COMPLETED):
            return len(engine.state.dataloader)
        if event_name in (Events.EPOCH_STARTED, Events.EPOCH_COMPLETED):
            return engine.state.max_epochs
        return 1

    def __call__(self, engine, logger, event_name):

        if logger.pbar is None:
            logger._reset(pbar_total=self.get_max_number_events(self.event_name, engine))

        desc = self.tag
        max_num_of_closing_events = self.get_max_number_events(self.closing_event_name, engine)
        if max_num_of_closing_events > 1:
            global_step = engine.state.get_event_attrib_value(self.closing_event_name)
            desc += " [{}/{}]".format(global_step, max_num_of_closing_events)
        logger.pbar.set_description(desc)

        metrics = self._setup_output_metrics(engine)

        rendered_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.ndimension() == 0:
                    rendered_metrics[key] = value.item()
                elif value.ndimension() == 1:
                    for i, v in enumerate(value):
                        k = "{}_{}".format(key, i)
                        rendered_metrics[k] = v.item()
                else:
                    warnings.warn("ProgressBar can not log "
                                  "tensor with {} dimensions".format(value.ndimension()))
            else:
                rendered_metrics[key] = value

        if rendered_metrics:
            logger.pbar.set_postfix(**rendered_metrics)

        logger.pbar.update()
