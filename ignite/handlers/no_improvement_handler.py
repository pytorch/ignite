from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any, cast, Literal

from ignite.base import Serializable, ResettableHandler
from ignite.engine import Engine, Events
from ignite.utils import setup_logger

__all__ = ["NoImprovementHandler"]


class NoImprovementHandler(Serializable, ResettableHandler):
    """NoImprovementHandler can be used to stop the training if no improvement after a given number of events.

    This handler is a simpler alternative to :class:`~ignite.handlers.early_stopping.EarlyStopping`.
    It monitors a score function and stops training when the score does not improve
    for ``patience`` consecutive events.

    Args:
        patience: Number of events to wait if no improvement and then stop the training.
        score_function: It should be a function taking a single argument, an
            :class:`~ignite.engine.engine.Engine` object, and return a score ``float``.
            An improvement is considered if the score is higher (for ``mode='max'``)
            or lower (for ``mode='min'``).
        trainer: Trainer engine to stop the run if no improvement.
        mode: Whether to maximize (``'max'``) or minimize (``'min'``) the score.
            Default is ``'max'``.

    Examples:
        .. code-block:: python

            from ignite.engine import Engine, Events
            from ignite.handlers import NoImprovementHandler

            def score_function(engine):
                val_loss = engine.state.metrics["nll"]
                return -val_loss

            handler = NoImprovementHandler(
                patience=10,
                score_function=score_function,
                trainer=trainer,
            )

            # Note: the handler is attached to an *Evaluator*
            evaluator.add_event_handler(Events.COMPLETED, handler)

    .. versionadded:: 0.5.0

    """

    _state_dict_all_req_keys = (
        "counter",
        "best_score",
    )

    def __init__(
        self,
        patience: int,
        score_function: Callable,
        trainer: Engine,
        mode: Literal["min", "max"] = "max",
    ):
        if not callable(score_function):
            raise TypeError("Argument score_function should be a function.")

        if patience < 1:
            raise ValueError("Argument patience should be a positive integer.")

        if not isinstance(trainer, Engine):
            raise TypeError("Argument trainer should be an instance of Engine.")

        if mode not in ("min", "max"):
            raise ValueError("Argument mode should be either 'min' or 'max'.")

        self.score_function = score_function
        self.patience = patience
        self.trainer = trainer
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.logger = setup_logger(__name__ + "." + self.__class__.__name__)

    def __call__(self, engine: Engine) -> None:
        score = self.score_function(engine)

        if self.best_score is None:
            self.best_score = score
            return

        if self.mode == "max":
            improved = score > self.best_score
        else:
            improved = score < self.best_score

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            self.logger.debug("NoImprovementHandler: %i / %i" % (self.counter, self.patience))
            if self.counter >= self.patience:
                self.logger.info("NoImprovementHandler: Stop training")
                self.trainer.terminate()

    def reset(self) -> None:
        """Reset the handler state, including the counter and best score."""
        self.counter = 0
        self.best_score = None

    def attach(
        self,
        engine: Engine,
        event: Any = Events.COMPLETED,
        reset_engine: Engine | None = None,
        reset_event: Any = Events.STARTED,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Attaches the handler to an engine and registers its reset callback.

        This method will:
        1. Add the no improvement evaluation logic (``self``) to ``engine`` on the given ``event``.
        2. Add the ``reset`` method to ``reset_engine`` (or ``engine`` if not provided) on the given ``reset_event``.

        Args:
            engine: The engine to attach the evaluation to (typically an evaluator).
            event: The event on ``engine`` that triggers the check. Default is
                :attr:`~ignite.engine.events.Events.COMPLETED`.
            reset_engine: The engine to attach the reset callback to (typically the trainer).
                If ``None``, defaults to ``engine``.
            reset_event: The event on ``reset_engine`` that triggers the handler state reset.
                Default is :attr:`~ignite.engine.events.Events.STARTED`.
        """
        engine.add_event_handler(event, self)

        target_reset_engine = reset_engine or engine
        target_reset_engine.add_event_handler(reset_event, self.reset)

    def state_dict(self) -> "OrderedDict[str, Any]":
        """Method returns state dict with ``counter`` and ``best_score``.
        Can be used to save internal state of the class.
        """
        return OrderedDict(
            [
                ("counter", self.counter),
                ("best_score", cast(float, self.best_score)),
            ]
        )

    def load_state_dict(self, state_dict: Mapping) -> None:
        """Method replace internal state of the class with provided state dict data.

        Args:
            state_dict: a dict with "counter" and "best_score" keys/values.
        """
        super().load_state_dict(state_dict)
        self.counter = state_dict["counter"]
        self.best_score = state_dict["best_score"]
