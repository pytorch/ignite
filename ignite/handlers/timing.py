from time import perf_counter
from typing import Any, Optional

from ignite.engine import Engine, Events

__all__ = ["Timer"]


class Timer:
    """Timer object can be used to measure (average) time between events.

    Args:
        average: if True, then when ``.value()`` method is called, the returned value
            will be equal to total time measured, divided by the value of internal counter.

    Attributes:
        total (float): total time elapsed when the Timer was running (in seconds).
        step_count (int): internal counter, useful to measure average time, e.g. of processing a single batch.
            Incremented with the ``.step()`` method.
        running (bool): flag indicating if timer is measuring time.

    Note:
        When using ``Timer(average=True)`` do not forget to call ``timer.step()`` every time an event occurs. See
        the examples below.

    Examples:
        Measuring total time of the epoch:

        .. code-block:: python

            from ignite.handlers import Timer
            import time
            work = lambda : time.sleep(0.1)
            idle = lambda : time.sleep(0.1)
            t = Timer(average=False)
            for _ in range(10):
                work()
                idle()

            t.value()
            # 2.003073937026784

        Measuring average time of the epoch:

        .. code-block:: python

            t = Timer(average=True)
            for _ in range(10):
                work()
                idle()
                t.step()

            t.value()
            # 0.2003182829997968

        Measuring average time it takes to execute a single ``work()`` call:

        .. code-block:: python

            t = Timer(average=True)
            for _ in range(10):
                t.resume()
                work()
                t.pause()
                idle()
                t.step()

            t.value()
            # 0.10016545779653825

        Using the Timer to measure average time it takes to process a single batch of examples:

        .. code-block:: python

            from ignite.engine import Engine, Events
            from ignite.handlers import Timer
            trainer = Engine(training_update_function)
            timer = Timer(average=True)
            timer.attach(trainer,
                         start=Events.STARTED,
                         resume=Events.ITERATION_STARTED,
                         pause=Events.ITERATION_COMPLETED,
                         step=Events.ITERATION_COMPLETED)
    """

    def __init__(self, average: bool = False):
        self._average = average

        self.reset()

    def attach(
        self,
        engine: Engine,
        start: Events = Events.STARTED,
        pause: Events = Events.COMPLETED,
        resume: Optional[Events] = None,
        step: Optional[Events] = None,
    ) -> "Timer":
        """Register callbacks to control the timer.

        Args:
            engine: Engine that this timer will be attached to.
            start: Event which should start (reset) the timer.
            pause: Event which should pause the timer.
            resume: Event which should resume the timer.
            step: Event which should call the `step` method of the counter.

        Returns:
            this timer
        """

        engine.add_event_handler(start, self.reset)
        engine.add_event_handler(pause, self.pause)

        if resume is not None:
            engine.add_event_handler(resume, self.resume)

        if step is not None:
            engine.add_event_handler(step, self.step)

        return self

    def reset(self, *args: Any) -> "Timer":
        """Reset the timer to zero."""
        self._t0 = perf_counter()
        self.total = 0.0
        self.step_count = 0.0
        self.running = True

        return self

    def pause(self, *args: Any) -> None:
        """Pause the current running timer."""
        if self.running:
            self.total += self._elapsed()
            self.running = False

    def resume(self, *args: Any) -> None:
        """Resume the current running timer."""
        if not self.running:
            self.running = True
            self._t0 = perf_counter()

    def value(self) -> float:
        """Return the average timer value."""
        total = self.total
        if self.running:
            total += self._elapsed()

        if self._average:
            denominator = max(self.step_count, 1.0)
        else:
            denominator = 1.0

        return total / denominator

    def step(self, *args: Any) -> None:
        """Increment the timer."""
        self.step_count += 1.0

    def _elapsed(self) -> float:
        return perf_counter() - self._t0
