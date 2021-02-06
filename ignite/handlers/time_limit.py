import time
from typing import Optional

from ignite.engine import Engine

__all__ = ["TimeLimit"]

from ignite.utils import setup_logger


class TimeLimit:
    """TimeLimit handler can be used to control training time for computing environments where session time is limited.
    Timer starts when handler is created and not training started.
    This handler gracefully terminates the training if time passed in the training exceeds a limit.

    Args:
        limit_sec (int, optional): Maximum time before training terminates (in seconds). Defaults to 28800.

    Examples:

        .. code-block:: python

            from ignite.engine import Events
            from ignite.handlers import TimeLimit

            handler = TimeLimit() # 8 hours of training
            trainer.add_event_handler(Events.ITERATION_COMPLETED, handler)

    .. versionadded:: 0.4.3
    """

    def __init__(self, limit_sec: Optional[int] = 28800):

        if not isinstance(limit_sec, int):
            raise TypeError("Argument limit_sec should be an integer.")
        if limit_sec <= 0:
            raise ValueError("Argument limit_sec should be a positive integer.")

        self.limit_sec = limit_sec
        self.start_time = time.time()
        self.logger = setup_logger(__name__ + "." + self.__class__.__name__)

    def __call__(self, engine: Engine) -> None:
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.limit_sec:
            self.logger.info("Reached the time limit: {} sec. Stop training".format(self.limit_sec))
            engine.terminate()
