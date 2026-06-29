from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any, cast, Literal

from ignite.base import Serializiable, ResettableHandler
from ignite.engine import Engine, Events
from ignite.utils import setup_logger

__all__ = ["NoImprovementHandler"]


class NoImprovementHandler(Serializiable, ResettableHandler):
    """NoImprovementHandler executes a custom action when the score does not
    improve after a given number of events.

    A modular alternative to
    :class:`~ignite.handlers.early_stopping.EarlyStopping`. Monitors a
    score function and executes a user-defined action (e.g., reducing
    learning rate, checkpointing, logging, or terminating) when the score
    stagnates for ``patience`` consecutive events.

    Args:
        patience: Number of events to wait if no improvement and then
            execute the action.
        score_function: A function taking a single argument, an
            :class:`~ignite.engine.engine.Engine` object, and returning a
            score ``float``.
        custom_action: A function to execute when no improvement is
            detected for ``patience`` events. Takes the engine as
            argument. If ``None``, defaults to terminating the trainer.
        trainer: Trainer engine. Required if ``custom_action`` is not
            provided.
        mode: ``'max'`` (default) or ``'min'``.

    Examples:
        .. code-block:: python

            from ignite.engine import Engine, Events
            from ignite.handlers import NoImprovementHandler

            def score_function(engine):
                val_loss = engine.state.metrics["nll"]
                return -val_loss

            # Default: terminate training
            handler = NoImprovementHandler(
                patience=10,
                score_function=score_function,
                trainer=trainer,
            )
            evaluator.add_event_handler(Events.COMPLETED, handler)

            # Custom action: halve learning rate
            def halve_lr(engine):
                for pg in trainer.state.param_groups:
                    pg["lr"] *= 0.5

            handler = NoImprovementHandler(
                patience=5,
                score_function=score_function,
                custom_action=halve_lr,
            )
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
        custom_action: Callable | None = None,
        trainer: Engine | None = None,
        mode: Literal["min", "max"] = "max",
    ):
        if not callable(score_function):
            raise TypeError("Argument score_function should be a function.")

        if patience < 1:
            raise ValueError("Argument patience should be a positive integer.")

        if custom_action is not None and not callable(custom_action):
            raise TypeError("Argument custom_action should be a function or None.")

        if custom_action is None and trainer is None:
            raise ValueError("Either custom_action or trainer must be provided.")

        if custom_action is None and not isinstance(trainer, Engine):
            raise TypeError("Argument trainer should be an instance of Engine.")

        if mode not in ("min", "max"):
            raise ValueError("Argument mode should be either 'min' or 'max'.")

        self.score_function = score_function
        self.patience = patience
        self.custom_action = custom_action
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
            self.logger.debug(
                "NoImprovementHandler: %i / %i", self.counter, self.patience
            )
            if self.counter >= self.patience:
                self.logger.info("NoImprovementHandler: Executing action")
                if self.custom_action is not None:
                    self.custom_action(engine)
                else:
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
        """Attaches the handler to an engine and registers its reset callback."""
        engine.add_event_handler(event, self)

        target_reset_engine = reset_engine or engine
        target_reset_engine.add_event_handler(reset_event, self.reset)

    def state_dict(self) -> "OrderedDict[str, Any]":
        """Method returns state dict with ``counter`` and ``best_score``."""
        return OrderedDict(
            [
                ("counter", self.counter),
                ("best_score", cast(float, self.best_score)),
            ]
        )

    def load_state_dict(self, state_dict: Mapping) -> None:
        """Method replace internal state of the class with provided state dict data."""
        super().load_state_dict(state_dict)
        self.counter = state_dict["counter"]
        self.best_score = state_dict["best_score"]
