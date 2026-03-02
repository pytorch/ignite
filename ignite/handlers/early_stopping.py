from collections import OrderedDict
from typing import Any, Callable, cast, Mapping, Literal, Optional
import warnings

from ignite.base import Serializable, ResettableHandler
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
        threshold: A minimum change in the score to qualify as an improvement. For ``mode='max'``, it's a minimum
            increase; for ``mode='min'``, it's a minimum decrease. Default is 0.0.
        threshold_mode: Determines whether `threshold` is an absolute change or a relative change.

            - In 'abs' mode:
                - For ``mode='max'``: improvement if score > best_score + threshold
                - For ``mode='min'``: improvement if score < best_score - threshold

            - In 'rel' mode:
                - For ``mode='max'``: improvement if score > best_score * (1 + threshold)
                - For ``mode='min'``: improvement if score < best_score * (1 - threshold)

            Possible values are "abs" and "rel". Default value is "abs".
        cumulative: If True, `threshold` defines the change since the last `patience` reset, otherwise,
            it defines the change after the last event. Default value is False.
        mode: Whether to maximize ('max') or minimize ('min') the score. Default is 'max'.

        # Deprecated args for backward compatibility (will be removed in future)
        min_delta: *Deprecated: use `threshold` instead*
        min_delta_mode: *Deprecated: use `threshold_mode` instead*
        cumulative_delta: *Deprecated: use `cumulative` instead*

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
        Added `threshold`/`threshold_mode` parameters to support both absolute and relative improvements.
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
        threshold: float = 0.0,
        threshold_mode: Literal["abs", "rel"] = "abs",
        cumulative: bool = False,
        mode: Literal["min", "max"] = "max",
        # Deprecated args for BC
        min_delta: Optional[float] = None,
        min_delta_mode: Optional[Literal["abs", "rel"]] = None,
        cumulative_delta: Optional[bool] = None,
    ):
        if not callable(score_function):
            raise TypeError("Argument score_function should be a function.")

        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")

        if not isinstance(trainer, Engine):
            raise TypeError("Argument trainer should be an instance of Engine.")

        # Backward compatibility for deprecated args
        if min_delta is not None:
            warnings.warn(
                "'min_delta' is deprecated and will be removed in a future version. " "Please use 'threshold' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            threshold = min_delta

        if min_delta_mode is not None:
            warnings.warn(
                "'min_delta_mode' is deprecated and will be removed in a future version. "
                "Please use 'threshold_mode' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            threshold_mode = min_delta_mode

        if cumulative_delta is not None:
            warnings.warn(
                "'cumulative_delta' is deprecated and will be removed in a future version. "
                "Please use 'cumulative' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            cumulative = cumulative_delta

        if threshold < 0.0:
            raise ValueError("Argument min_delta should not be a negative number.")

        if threshold_mode not in ("abs", "rel"):
            raise ValueError("Argument min_delta_mode should be either 'abs' or 'rel'.")

        if mode not in ("min", "max"):
            raise ValueError("Argument mode should be either 'min' or 'max'.")

        self.score_function = score_function
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cumulative = cumulative
        self.trainer = trainer
        self.counter = 0
        self.best_score: float | None = None
        self.logger = setup_logger(__name__ + "." + self.__class__.__name__)
        self.min_delta = threshold
        self.min_delta_mode = threshold_mode
        self.cumulative_delta = cumulative
        self.mode = mode

    def __call__(self, engine: Engine) -> None:
        score = self.score_function(engine)

        if self.best_score is None:
            self.best_score = score
            return

        min_delta = -self.threshold if self.mode == "min" else self.threshold
        if self.threshold_mode == "abs":
            improvement_threshold = self.best_score + min_delta
        else:
            improvement_threshold = self.best_score * (1 + min_delta)

        no_improvement = score <= improvement_threshold if self.mode == "max" else score >= improvement_threshold

        if no_improvement:
            if not self.cumulative:
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
        """Reset the early stopping state, including the counter and best score.

        .. versionadded:: 0.6.0
        """
        self.counter = 0
        self.best_score = None

    def attach(  # type: ignore[override]
        self,
        engine: Engine,
        event: Any = Events.COMPLETED,
        reset_engine: Engine | None = None,
        reset_event: Any = Events.STARTED,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Attaches the early stopping handler to an engine and registers its reset callback.

        This method will:
        1. Add the early stopping evaluation logic (``self``) to ``engine`` on the given ``event``.
        2. Add the ``reset`` method to ``reset_engine`` (or ``engine`` if not provided) on the given ``reset_event``.

        Args:
            engine: The engine to attach the early stopping evaluation to (typically an evaluator).
            event: The event on ``engine`` that triggers the early stopping check. Default is
                :attr:`~ignite.engine.events.Events.COMPLETED`.
            reset_engine: The engine to attach the reset callback to (typically the trainer).
                If ``None``, defaults to ``engine``.
            reset_event: The event on ``reset_engine`` that triggers the handler state reset.
                Default is :attr:`~ignite.engine.events.Events.STARTED`.

        .. versionadded:: 0.6.0
        """
        engine.add_event_handler(event, self)

        target_reset_engine = reset_engine or engine
        target_reset_engine.add_event_handler(reset_event, self.reset)

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
