import warnings
from collections import OrderedDict
from typing import Any, Callable, Literal, Mapping, cast

from ignite.base import Serializable, ResettableHandler
from ignite.base.usage import RunWise, Usage
from ignite.engine import Engine, Events
from ignite.utils import setup_logger

__all__ = ["EarlyStopping"]


class EarlyStopping(Serializable, ResettableHandler):
    """EarlyStopping handler can be used to stop the training if no improvement after a given number of events.

    Args:
        patience: Number of events to wait if no improvement and then stop the training.
        score_function: It should be a function taking a single argument, an :class:`~ignite.engine.engine.Engine`
            object, and return a score `float`. An improvement is considered if the score is higher (for ``mode='max'``)
            or lower (for ``mode='min'``).
        trainer: Trainer engine to stop the run if no improvement.
        min_delta: A minimum change in the score to qualify as an improvement. For ``mode='max'``, it's a minimum
            increase; for ``mode='min'``, it's a minimum decrease. An improvement is only considered if the change
            exceeds the threshold determined by `min_delta` and `min_delta_mode`.
        cumulative_delta: If True, `min_delta` defines the change since the last `patience` reset, otherwise,
            it defines the change after the last event. Default value is False.
        min_delta_mode: Determines whether `min_delta` is an absolute change or a relative change.

            - In 'abs' mode:

              - For ``mode='max'``: improvement if score > best_score + min_delta
              - For ``mode='min'``: improvement if score < best_score - min_delta

            - In 'rel' mode:

              - For ``mode='max'``: improvement if score > best_score * (1 + min_delta)
              - For ``mode='min'``: improvement if score < best_score * (1 - min_delta)

            Possible values are "abs" and "rel". Default value is "abs".
        mode: Whether to maximize ('max') or minimize ('min') the score. Default is 'max'.

    Examples:
        .. code-block:: python

            from ignite.engine import Engine, Events
            from ignite.handlers import EarlyStopping

            def score_function(engine):
                val_loss = engine.state.metrics['nll']
                return -val_loss

            handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
            # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
            evaluator.add_event_handler(Events.COMPLETED, handler)

    .. versionchanged:: 0.6.0
        Added `mode` parameter to support minimization in addition to maximization.
        Added `min_delta_mode` parameter to support both absolute and relative improvements.

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
        min_delta: float = 0.0,
        cumulative_delta: bool = False,
        min_delta_mode: Literal["abs", "rel"] = "abs",
        mode: Literal["min", "max"] = "max",
    ):
        if not callable(score_function):
            raise TypeError("Argument score_function should be a function.")

        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")

        if min_delta < 0.0:
            raise ValueError("Argument min_delta should not be a negative number.")

        if not isinstance(trainer, Engine):
            raise TypeError("Argument trainer should be an instance of Engine.")

        if min_delta_mode not in ("abs", "rel"):
            raise ValueError("Argument min_delta_mode should be either 'abs' or 'rel'.")

        if mode not in ("min", "max"):
            raise ValueError("Argument mode should be either 'min' or 'max'.")

        self.score_function = score_function
        self.patience = patience
        self.min_delta = min_delta
        self.cumulative_delta = cumulative_delta
        self.trainer = trainer
        self.counter = 0
        self.best_score: float | None = None
        self.logger = setup_logger(__name__ + "." + self.__class__.__name__)
        self.min_delta_mode = min_delta_mode
        self.mode = mode

    def __call__(self, engine: Engine) -> None:
        score = self.score_function(engine)

        if self.best_score is None:
            self.best_score = score
            return

        min_delta = -self.min_delta if self.mode == "min" else self.min_delta
        if self.min_delta_mode == "abs":
            improvement_threshold = self.best_score + min_delta
        else:
            improvement_threshold = self.best_score * (1 + min_delta)

        no_improvement = score <= improvement_threshold if self.mode == "max" else score >= improvement_threshold

        if no_improvement:
            if not self.cumulative_delta:
                self.best_score = max(score, self.best_score) if self.mode == "max" else min(score, self.best_score)
            self.counter += 1
            self.logger.debug("EarlyStopping: %i / %i" % (self.counter, self.patience))
            if self.counter >= self.patience:
                self.logger.info("EarlyStopping: Stop training")
                self.trainer.terminate()
        else:
            self.best_score = score
            self.counter = 0

    def reset(self) -> None:
        """Reset the early stopping state, including the counter and best score."""
        self.counter = 0
        self.best_score = None

    def attach(
        self,
        engine: Engine,
        usage: Usage | None = None,
        reset_engine: Engine | None = None,
        **kwargs: Any,
    ) -> None:
        """Attaches the early stopping handler to an engine and registers its reset callback.

        This method will:
        1. Add the early stopping evaluation logic (``self``) to ``engine`` on the given ``usage.COMPLETED`` event.
        2. Add the ``reset`` method to ``reset_engine`` (or ``engine`` if not provided) on the given ``usage.STARTED`` event.

        Args:
            engine: The engine to attach the early stopping evaluation to (typically an evaluator).
            usage: The usage that encapsulates the evaluation events. Default is
                :class:`~ignite.base.usage.RunWise`.
            reset_engine: The engine to attach the reset callback to (typically the trainer).
                If ``None``, defaults to ``engine``.
            **kwargs: Deprecated keyword argument (``event``) for backward compatibility.
        """
        if "event" in kwargs or "reset_event" in kwargs:
            if "event" in kwargs:
                warnings.warn(
                    "The `event` argument is deprecated and will be removed in future releases. "
                    "Use `usage` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            usage = Usage(
                completed=kwargs.get("event", Events.COMPLETED),
                started=kwargs.get("reset_event", Events.STARTED),
            )
        elif usage is None:
            usage = RunWise()

        if usage.COMPLETED is not None:
            engine.add_event_handler(usage.COMPLETED, self)

        target_reset_engine = reset_engine or engine
        if usage.STARTED is not None:
            target_reset_engine.add_event_handler(usage.STARTED, self.reset)

    def state_dict(self) -> "OrderedDict[str, float]":
        """Method returns state dict with ``counter`` and ``best_score``.
        Can be used to save internal state of the class.
        """
        return OrderedDict([("counter", self.counter), ("best_score", cast(float, self.best_score))])

    def load_state_dict(self, state_dict: Mapping) -> None:
        """Method replace internal state of the class with provided state dict data.

        Args:
            state_dict: a dict with "counter" and "best_score" keys/values.
        """
        super().load_state_dict(state_dict)
        self.counter = state_dict["counter"]
        self.best_score = state_dict["best_score"]
