import time
from ignite.engine import Events


class Timer:
    """ Timer object can be used to measure (average) time between events.

    Args:
        unit (bool, optional): if True, then when ``.value()`` method is called, the returned value
            will be equal to total time measured, divided by the value of internal counter.

    Attributes:
        total (float): total time elapsed when the Timer was running (in seconds)
        cnt (int): internal counter, usefull to measure average time, e.g. of processing a single batch.
            Incremented with the ``.inc_unit()`` method.
        running (bool): flag indicating if time is measuring time.

    Notes:
        When using ``Timer(unit=True)`` do not forget to call ``timer.inc_unit()`` everytime an event occurs. See
        the examples below.

    Examples:
        Measuring total time of the entire 'training' epoch:

        >>> from ignite.handlers import Timer
        >>> import time
        >>> work = lambda : time.sleep(0.1)
        >>> idle = lambda : time.sleep(0.1)
        >>> t = Timer(unit=False)
        >>> for _ in range(10):
        ...    work()
        ...    idle()
        ...
        >>> t.value()
        2.003073937026784

        Measuring average time of the 'training' epoch:
        >>> t = Timer(unit=True)
        >>> for _ in range(10):
        ...    work()
        ...    idle()
        ...    t.inc_unit()
        ...
        >>> t.value()
        0.2003182829997968

        Measuring average time it takes to execute a single ``work()`` call
        >>> t = Timer(unit=True)
        >>> for _ in range(10):
        ...    t.resume()
        ...    work()
        ...    t.pause()
        ...    idle()
        ...    t.inc_unit()
        ...
        >>> t.value()
        0.10016545779653825

        Using the Timer to measure average time it takes to process a single batch of examples
        >>> from ignite.trainer import Trainer
        >>> from ignite.engine import Events
        >>> from ignite.handlers import Timer
        >>> trainer = Trainer(trainig_update_function)
        >>> timer = Timer(unit=True)
        >>> timer.attach(trainer,
        ...              start=Events.EPOCH_STARTED,
        ...              resume=Events.ITERATION_STARTED
        ...              pause=Events.ITERATION_COMPLETED,
        ...              step=Events.ITERATION_COMPLETED)
    """

    def __init__(self, unit=False):
        self._unit = unit
        self._t0 = time.perf_counter()

        self.total = 0.
        self.cnt = 0.
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
                Event which should call the `inc_unit` method of the counter

        Returns:
            self (Timer)

        """

        engine.add_event_handler(start, self.reset)
        engine.add_event_handler(pause, self.pause)

        if resume is not None:
            engine.add_event_handler(resume, self.resume)

        if step is not None:
            engine.add_event_handler(step, self.inc_unit)

        return self

    def reset(self):
        self.__init__(self._unit)
        return self

    def pause(self):
        if self.running:
            self.total += self._elapsed()
            self.running = False

    def resume(self):
        if not self.running:
            self.running = True
            self._t0 = time.perf_counter()

    def value(self):
        total = self.total
        if self.running:
            total += self._elapsed()

        if self._unit:
            denominator = max(self.cnt, 1.)
        else:
            denominator = 1.

        return total / denominator

    def inc_unit(self):
        self.cnt += 1.

    def _elapsed(self):
        return time.perf_counter() - self._t0
