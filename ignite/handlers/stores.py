from typing import Any, Callable, List, Optional

from ignite.engine import Engine, Events


class EpochOutputStore:
    """EpochOutputStore handler to save output prediction and target history
    after every epoch, could be useful for e.g., visualization purposes.

    Note:
        This can potentially lead to a memory error if the output data is
        larger than available RAM.

    Args:
        output_transform: a callable that is used to
            transform the :class:`~ignite.engine.engine.Engine`'s
            ``process_function``'s output , e.g., lambda x: x[0]

    Attributes:
        data: a list of :class:`~ignite.engine.engine.Engine` outputs,
            optionally transformed by `output_transform`.

    Examples::

        eos = EpochOutputStore()
        trainer = create_supervised_trainer(model, optimizer, loss)
        train_evaluator = create_supervised_evaluator(model, metrics)
        eos.attach(train_evaluator, 'output')

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            train_evaluator.run(train_loader)
            output = train_evaluator.state.output
            # output = [(y_pred0, y0), (y_pred1, y1), ...]
            # do something with output, e.g., plotting

    .. versionadded:: 0.5.0
    .. versionchanged:: 0.5.0
        `attach` now accepts an optional argument `name`
    """

    def __init__(self, output_transform: Callable = lambda x: x):
        self.data = []  # type: List[Any]
        self.output_transform = output_transform

    def reset(self) -> None:
        """Reset the attribute data to empty list."""
        self.data = []

    def update(self, engine: Engine) -> None:
        """Append the output of Engine to attribute data."""
        output = self.output_transform(engine.state.output)
        self.data.append(output)

    def store(self, engine: Engine) -> None:
        """Store `self.data` on `engine.state.{self.name}`"""
        setattr(engine.state, self.name, self.data)

    def attach(self, engine: Engine, name: Optional[str] = None) -> None:
        """Attaching `reset` method at EPOCH_STARTED and
        `update` method at ITERATION_COMPLETED.

        If `name` is passed, will store `self.data` on `engine.state`
        under `name`.
        """
        engine.add_event_handler(Events.EPOCH_STARTED, self.reset)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.update)
        if name:
            self.name = name
            engine.add_event_handler(Events.EPOCH_COMPLETED, self.store)
