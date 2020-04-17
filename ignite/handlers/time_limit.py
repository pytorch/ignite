import time

import logging

log = logging.getLogger(__name__)

__all__ = ["TimeLimit"]


class TimeLimit:
    def __init__(self, limit_sec=3600):
        """Time limit for training.

        Args:
            limit_sec (int, optional): Time limit in seconds. Defaults to 3600.
        """
        self.limit_sec = limit_sec
        self.start_time = time.time()

    def __call__(self, engine):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.limit_sec:
            log.warning("Reached the time limit: {} sec. Stop training".format(self.limit_sec))
            engine.terminate()
