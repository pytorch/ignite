import logging
import time
from typing import Optional

from ignite.engine import Engine

log = logging.getLogger(__name__)

__all__ = ["TimeLimit"]


class TimeLimit:
    """TimeLimit handler can be used to control training time for computing environments where session time is limited.

    Args:
        limit_sec (int, optional): Maximum time before training terminates (in seconds). Defaults to 3600.

    Examples:

        .. code-block:: python
    
            from ignite.engine import Events
            from ignite.handlers import TimeLimit
            handler = TimeLimit(288800) # 8 hours of training 
            trainer.add_event_handler(Events.ITERATION_COMPLETED, handler)
    """

    def __init__(self, limit_sec: Optional[int] = 3600):

        if not isinstance(limit_sec, int):
            raise TypeError("Argument limit_sec should be an integer.")
        if limit_sec <= 0:
            raise ValueError("Argument limit_sec should be a positive integer.")

        self.limit_sec = limit_sec
        self.start_time = time.time()

    def __call__(self, engine: Engine) -> None:
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.limit_sec:
            log.warning("Reached the time limit: {} sec. Stop training".format(self.limit_sec))
            engine.terminate()
