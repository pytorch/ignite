import time
from ignite.engine import Events


class Timer:
    def __init__(self, unit=False):
        self.unit = unit

        self.t0 = time.perf_counter()
        self.total = 0.
        self.cnt = 0.
        self.running = True

    def attach(self, engine, start=Events.STARTED, pause=Events.COMPLETED, resume=None, step=None):
        """ Register callbacks to control the timer.

        Parameters
        ----------
        engine : ignite.engine.Engine
            Engine that this timer will be attached to
        start : ignite.engine.Events
            Event which should start (reset) the timer
        pause : ignite.engine.Events
            Event which should pause the timer
        resume : ignite.engine.Events
            Event which should resume the timer
        step : ignite.engine.Events
            Event which should call the `inc_unit` method of the counter

        Returns
        -------
        self
        """

        engine.add_event_handler(start, self.reset)
        engine.add_event_handler(pause, self.pause)

        if resume is not None:
            engine.add_event_handler(resume, self.resume)

        if step is not None:
            engine.add_event_handler(step, self.inc_unit)

        return self

    def reset(self):
        self.__init__(self.unit)

    def pause(self):
        if self.running:
            self.total += self._elapsed()
            self.running = False

    def resume(self):
        if not self.running:
            self.running = True
            self.t0 = time.perf_counter()

    def value(self):
        total = self.total
        if self.running:
            total += self._elapsed()

        if self.unit:
            denominator = max(self.cnt, 1.)
        else:
            denominator = 1.

        return total / denominator

    def inc_unit(self):
        self.cnt += 1.

    def _elapsed(self):
        return time.perf_counter() - self.t0
