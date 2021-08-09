# -*- coding: utf-8 -*-
"""TQDM logger."""
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Union

from ignite.contrib.handlers.base_logger import BaseLogger, BaseOutputHandler
from ignite.engine import Engine, Events
from ignite.engine.events import CallableEventWithFilter, RemovableEventHandle


class ProgressBar(BaseLogger):
    """
    TQDM progress bar handler to log training progress and computed metrics.

    Args:
        persist: set to ``True`` to persist the progress bar after completion (default = ``False``)
        bar_format : Specify a custom bar string formatting. May impact performance.
            [default: '{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]'].
            Set to ``None`` to use ``tqdm`` default bar formatting: '{l_bar}{bar}{r_bar}', where
            l_bar='{desc}: {percentage:3.0f}%|' and
            r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'. For more details on the
            formatting, see `tqdm docs <https://tqdm.github.io/docs/tqdm/>`_.
        tqdm_kwargs: kwargs passed to tqdm progress bar.
            By default, progress bar description displays "Epoch [5/10]" where 5 is the current epoch and 10 is the
            number of epochs; however, if ``max_epochs`` are set to 1, the progress bar instead displays
            "Iteration: [5/10]". If tqdm_kwargs defines `desc`, e.g. "Predictions", than the description is
            "Predictions [5/10]" if number of epochs is more than one otherwise it is simply "Predictions".

    Examples:

        Simple progress bar

        .. code-block:: python

            trainer = create_supervised_trainer(model, optimizer, loss)

            pbar = ProgressBar()
            pbar.attach(trainer)

            # Progress bar will looks like
            # Epoch [2/50]: [64/128]  50%|█████      [06:17<12:34]

        Log output to a file instead of stderr (tqdm's default output)

        .. code-block:: python

            trainer = create_supervised_trainer(model, optimizer, loss)

            log_file = open("output.log", "w")
            pbar = ProgressBar(file=log_file)
            pbar.attach(trainer)

        Attach metrics that already have been computed at :attr:`~ignite.engine.events.Events.ITERATION_COMPLETED`
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

    _events_order = [
        Events.STARTED,
        Events.EPOCH_STARTED,
        Events.ITERATION_STARTED,
        Events.ITERATION_COMPLETED,
        Events.EPOCH_COMPLETED,
        Events.COMPLETED,
    ]  # type: List[Union[Events, CallableEventWithFilter]]

    def __init__(
        self,
        persist: bool = False,
        bar_format: str = "{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]",
        **tqdm_kwargs: Any,
    ):

        try:
            from tqdm.autonotebook import tqdm
        except ImportError:
            raise RuntimeError(
                "This contrib module requires tqdm to be installed. "
                "Please install it with command: \n pip install tqdm"
            )

        self.pbar_cls = tqdm
        self.pbar = None
        self.persist = persist
        self.bar_format = bar_format
        self.tqdm_kwargs = tqdm_kwargs

    def _reset(self, pbar_total: Optional[int]) -> None:
        self.pbar = self.pbar_cls(
            total=pbar_total, leave=self.persist, bar_format=self.bar_format, initial=1, **self.tqdm_kwargs
        )

    def _close(self, engine: Engine) -> None:
        if self.pbar is not None:
            # https://github.com/tqdm/notebook.py#L240-L250
            # issue #1115 : notebook backend of tqdm checks if n < total (error or KeyboardInterrupt)
            # and the bar persists in 'danger' mode
            if self.pbar.total is not None:
                self.pbar.n = self.pbar.total
            self.pbar.close()
        self.pbar = None

    @staticmethod
    def _compare_lt(
        event1: Union[Events, CallableEventWithFilter], event2: Union[Events, CallableEventWithFilter]
    ) -> bool:
        i1 = ProgressBar._events_order.index(event1)
        i2 = ProgressBar._events_order.index(event2)
        return i1 < i2

    def log_message(self, message: str) -> None:
        """
        Logs a message, preserving the progress bar correct output format.

        Args:
            message: string you wish to log.
        """
        from tqdm import tqdm

        tqdm.write(message, file=self.tqdm_kwargs.get("file", None))

    def attach(  # type: ignore[override]
        self,
        engine: Engine,
        metric_names: Optional[Union[str, List[str]]] = None,
        output_transform: Optional[Callable] = None,
        event_name: Union[Events, CallableEventWithFilter] = Events.ITERATION_COMPLETED,
        closing_event_name: Union[Events, CallableEventWithFilter] = Events.EPOCH_COMPLETED,
    ) -> None:
        """
        Attaches the progress bar to an engine object.

        Args:
            engine: engine object.
            metric_names: list of metric names to plot or a string "all" to plot all available
                metrics.
            output_transform: a function to select what you want to print from the engine's
                output. This function may return either a dictionary with entries in the format of ``{name: value}``,
                or a single scalar, which will be displayed with the default name `output`.
            event_name: event's name on which the progress bar advances. Valid events are from
                :class:`~ignite.engine.events.Events`.
            closing_event_name: event's name on which the progress bar is closed. Valid events are from
                :class:`~ignite.engine.events.Events`.

        Note:
            Accepted output value types are numbers, 0d and 1d torch tensors and strings.

        """
        desc = self.tqdm_kwargs.get("desc", None)

        if event_name not in engine._allowed_events:
            raise ValueError(f"Logging event {event_name.name} is not in allowed events for this engine")

        if isinstance(closing_event_name, CallableEventWithFilter):
            if closing_event_name.filter != CallableEventWithFilter.default_event_filter:
                raise ValueError("Closing Event should not be a filtered event")

        if not self._compare_lt(event_name, closing_event_name):
            raise ValueError(f"Logging event {event_name} should be called before closing event {closing_event_name}")

        log_handler = _OutputHandler(desc, metric_names, output_transform, closing_event_name=closing_event_name)

        super(ProgressBar, self).attach(engine, log_handler, event_name)
        engine.add_event_handler(closing_event_name, self._close)

    def attach_opt_params_handler(
        self, engine: Engine, event_name: Union[str, Events], *args: Any, **kwargs: Any
    ) -> RemovableEventHandle:
        """Intentionally empty"""
        pass

    def _create_output_handler(self, *args: Any, **kwargs: Any) -> "_OutputHandler":
        return _OutputHandler(*args, **kwargs)

    def _create_opt_params_handler(self, *args: Any, **kwargs: Any) -> Callable:
        """Intentionally empty"""
        pass


class _OutputHandler(BaseOutputHandler):
    """Helper handler to log engine's output and/or metrics

    Args:
        description: progress bar description.
        metric_names: list of metric names to plot or a string "all" to plot all available
            metrics.
        output_transform: output transform function to prepare `engine.state.output` as a number.
            For example, `output_transform = lambda output: output`
            This function can also return a dictionary, e.g `{'loss': loss1, 'another_loss': loss2}` to label the plot
            with corresponding keys.
        closing_event_name: event's name on which the progress bar is closed. Valid events are from
            :class:`~ignite.engine.events.Events` or any `event_name` added by
            :meth:`~ignite.engine.engine.Engine.register_events`.

    """

    def __init__(
        self,
        description: str,
        metric_names: Optional[Union[str, List[str]]] = None,
        output_transform: Optional[Callable] = None,
        closing_event_name: Union[Events, CallableEventWithFilter] = Events.EPOCH_COMPLETED,
    ):
        if metric_names is None and output_transform is None:
            # This helps to avoid 'Either metric_names or output_transform should be defined' of BaseOutputHandler
            metric_names = []
        super(_OutputHandler, self).__init__(description, metric_names, output_transform, global_step_transform=None)
        self.closing_event_name = closing_event_name

    @staticmethod
    def get_max_number_events(event_name: Union[str, Events, CallableEventWithFilter], engine: Engine) -> Optional[int]:
        if event_name in (Events.ITERATION_STARTED, Events.ITERATION_COMPLETED):
            return engine.state.epoch_length
        if event_name in (Events.EPOCH_STARTED, Events.EPOCH_COMPLETED):
            return engine.state.max_epochs
        return 1

    def __call__(self, engine: Engine, logger: ProgressBar, event_name: Union[str, Events]) -> None:

        pbar_total = self.get_max_number_events(event_name, engine)
        if logger.pbar is None:
            logger._reset(pbar_total=pbar_total)

        max_epochs = engine.state.max_epochs
        default_desc = "Iteration" if max_epochs == 1 else "Epoch"

        desc = self.tag or default_desc
        max_num_of_closing_events = self.get_max_number_events(self.closing_event_name, engine)
        if max_num_of_closing_events and max_num_of_closing_events > 1:
            global_step = engine.state.get_event_attrib_value(self.closing_event_name)
            desc += f" [{global_step}/{max_num_of_closing_events}]"
        logger.pbar.set_description(desc)  # type: ignore[attr-defined]

        rendered_metrics = self._setup_output_metrics_state_attrs(engine, log_text=True)
        metrics = OrderedDict()
        for key, value in rendered_metrics.items():
            key = "_".join(key[1:])  # tqdm has tag as description

            metrics[key] = value

        if metrics:
            logger.pbar.set_postfix(metrics)  # type: ignore[attr-defined]

        global_step = engine.state.get_event_attrib_value(event_name)
        if pbar_total is not None:
            global_step = (global_step - 1) % pbar_total + 1
        logger.pbar.update(global_step - logger.pbar.n)  # type: ignore[attr-defined]
