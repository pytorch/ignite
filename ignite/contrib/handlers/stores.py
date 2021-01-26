from typing import Callable, List, Tuple, Union

from ignite.engine import Engine, Events


class EpochOutputStore:
    """EpochOutputStore handler to save output prediction and target history
    after every epoch, could be useful for e.g., visualization purposes.

    Note:
        This can potentially lead to a memory error if the output data is
        larger than available RAM.

    Args:
        output_transform (callable, optional): a callable that is used to
            transform the :class:`~ignite.engine.engine.Engine`'s
            ``process_function``'s output , e.g., lambda x: x[0]

    Examples::

        eos = EpochOutputStore()
        trainer = create_supervised_trainer(model, optimizer, loss)
        train_evaluator = create_supervised_evaluator(model, metrics)
        eos.attach(train_evaluator)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            train_evaluator.run(train_loader)
            output = eos.data
            # do something with output, e.g., plotting

    .. versionadded:: 0.4.2
    """

    def __init__(self, output_transform: Callable = lambda x: x) -> None:
        self.data = []  # type: List[Union[int, Tuple[int, int]]]
        self.output_transform = output_transform

    def reset(self) -> None:
        self.data = []

    def update(self, engine: Engine) -> None:
        output = self.output_transform(engine.state.output)
        self.data.append(output)

    def attach(self, engine: Engine) -> None:
        engine.add_event_handler(Events.EPOCH_STARTED, self.reset)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.update)
