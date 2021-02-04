import logging
import time
from typing import Union

from ignite.engine import Engine

log = logging.getLogger(__name__)

__all__ = ["TimeLimit"]


class TimeLimit:
    def __init__(self, limit_sec: Union[int, float] = 3600):
        """Time limit for training , .
        Args:
            limit_sec (int, optional): Time limit in seconds. Defaults to 3600.

        Examples:

        .. code-block:: python
        from ignite.engine import Events
        from ignite.handlers import TimeLimit
        handler = TimeLimit(288800) # 8 hours of training 
        trainer.add_event_handler(Events.ITERATION_COMPLETED, handler)
        """
        if limit_sec <= 0:
            raise ValueError("Argument limit_sec should be a positive number.")

        self.limit_sec = limit_sec
        self.start_time = time.time()

    def __call__(self, engine: Engine) -> None:
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.limit_sec:
            log.warning("Reached the time limit: {} sec. Stop training".format(self.limit_sec))
            engine.terminate()
