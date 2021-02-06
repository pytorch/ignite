from collections import OrderedDict
from typing import Callable, Mapping, Optional, cast

from ignite.base import Serializable
from ignite.engine import Engine
from ignite.utils import setup_logger

__all__ = ["EarlyStopping"]


class EarlyStopping(Serializable):
    """EarlyStopping handler can be used to stop the training if no improvement after a given number of events.

    Args:
        patience (int):
            Number of events to wait if no improvement and then stop the training.
        score_function (callable):
            It should be a function taking a single argument, an :class:`~ignite.engine.engine.Engine` object,
            and return a score `float`. An improvement is considered if the score is higher.
        trainer (Engine):
            trainer engine to stop the run if no improvement.
        min_delta (float, optional):
            A minimum increase in the score to qualify as an improvement,
            i.e. an increase of less than or equal to `min_delta`, will count as no improvement.
        cumulative_delta (bool, optional):
            It True, `min_delta` defines an increase since the last `patience` reset, otherwise,
            it defines an increase after the last event. Default value is False.

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
    ):

        if not callable(score_function):
            raise TypeError("Argument score_function should be a function.")

        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")

        if min_delta < 0.0:
            raise ValueError("Argument min_delta should not be a negative number.")

        if not isinstance(trainer, Engine):
            raise TypeError("Argument trainer should be an instance of Engine.")

        self.score_function = score_function
        self.patience = patience
        self.min_delta = min_delta
        self.cumulative_delta = cumulative_delta
        self.trainer = trainer
        self.counter = 0
        self.best_score = None  # type: Optional[float]
        self.logger = setup_logger(__name__ + "." + self.__class__.__name__)

    def __call__(self, engine: Engine) -> None:
        score = self.score_function(engine)

        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.min_delta:
            if not self.cumulative_delta and score > self.best_score:
                self.best_score = score
            self.counter += 1
            self.logger.debug("EarlyStopping: %i / %i" % (self.counter, self.patience))
            if self.counter >= self.patience:
                self.logger.info("EarlyStopping: Stop training")
                self.trainer.terminate()
        else:
            self.best_score = score
            self.counter = 0

    def state_dict(self) -> "OrderedDict[str, float]":
        return OrderedDict([("counter", self.counter), ("best_score", cast(float, self.best_score))])

    def load_state_dict(self, state_dict: Mapping) -> None:
        super().load_state_dict(state_dict)
        self.counter = state_dict["counter"]
        self.best_score = state_dict["best_score"]
