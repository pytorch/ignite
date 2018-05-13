from ignite.engine import Events

try:
    from time import perf_counter
except ImportError:
    from time import time as perf_counter


class Timer:
    """ Timer object can be used to measure (average) time between events.

    Args:
        average (bool, optional): if True, then when ``.value()`` method is called, the returned value
            will be equal to total time measured, divided by the value of internal counter.

    Attributes:
        total (float): total time elapsed when the Timer was running (in seconds)
        step_count (int): internal counter, usefull to measure average time, e.g. of processing a single batch.
            Incremented with the ``.step()`` method.
        running (bool): flag indicating if timer is measuring time.

    Notes:
        When using ``Timer(average=True)`` do not forget to call ``timer.step()`` everytime an event occurs. See
        the examples below.

    Examples:

        Measuring total time of the epoch:

        >>> from ignite.handlers import Timer
        >>> import time
        >>> work = lambda : time.sleep(0.1)
        >>> idle = lambda : time.sleep(0.1)
        >>> t = Timer(average=False)
        >>> for _ in range(10):
        ...    work()
        ...    idle()
        ...
        >>> t.value()
        2.003073937026784

        Measuring average time of the epoch:

        >>> t = Timer(average=True)
        >>> for _ in range(10):
        ...    work()
        ...    idle()
        ...    t.step()
        ...
        >>> t.value()
        0.2003182829997968

        Measuring average time it takes to execute a single ``work()`` call

        >>> t = Timer(average=True)
        >>> for _ in range(10):
        ...    t.resume()
        ...    work()
        ...    t.pause()
        ...    idle()
        ...    t.step()
        ...
        >>> t.value()
        0.10016545779653825

        Using the Timer to measure average time it takes to process a single batch of examples

        >>> from ignite.engine import Engine, Events
        >>> from ignite.handlers import Timer
        >>> trainer = Engine(training_update_function)
        >>> timer = Timer(average=True)
        >>> timer.attach(trainer,
        ...              start=Events.EPOCH_STARTED,
        ...              resume=Events.ITERATION_STARTED,
        ...              pause=Events.ITERATION_COMPLETED,
        ...              step=Events.ITERATION_COMPLETED)
    """

    def __init__(self, average=False):
        self._average = average
        self._t0 = perf_counter()

        self.total = 0.
        self.step_count = 0.
        self.running = True

    def attach(self, engine, start=Events.STARTED, pause=Events.COMPLETED, resume=None, step=None):
        """ Register callbacks to control the timer.

        Args:
            engine (ignite.engine.Engine):
                Engine that this timer will be attached to
            start (ignite.engine.Events):
                Event which should start (reset) the timer
            pause (ignite.engine.Events):
                Event which should pause the timer
            resume (ignite.engine.Events, optional):
                Event which should resume the timer
            step (ignite.engine.Events, optional):
                Event which should call the `step` method of the counter

        Returns:
            self (Timer)

        """

        engine.add_event_handler(start, self.reset)
        engine.add_event_handler(pause, self.pause)

        if resume is not None:
            engine.add_event_handler(resume, self.resume)

        if step is not None:
            engine.add_event_handler(step, self.step)

        return self

    def reset(self, *args):
        self.__init__(self._average)
        return self

    def pause(self, *args):
        if self.running:
            self.total += self._elapsed()
            self.running = False

    def resume(self, *args):
        if not self.running:
            self.running = True
            self._t0 = perf_counter()

    def value(self):
        total = self.total
        if self.running:
            total += self._elapsed()

        if self._average:
            denominator = max(self.step_count, 1.)
        else:
            denominator = 1.

        return total / denominator

    def step(self, *args):
        self.step_count += 1.

    def _elapsed(self):
        return perf_counter() - self._t0
