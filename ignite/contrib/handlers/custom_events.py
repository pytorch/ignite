import warnings

from ignite.engine import EventEnum, Events, State


class CustomPeriodicEvent:
    """DEPRECATED. Use filtered events instead.
    Handler to define a custom periodic events as a number of elapsed iterations/epochs
    for an engine.

    When custom periodic event is created and attached to an engine, the following events are fired:
    1) K iterations is specified:
    - `Events.ITERATIONS_<K>_STARTED`
    - `Events.ITERATIONS_<K>_COMPLETED`

    1) K epochs is specified:
    - `Events.EPOCHS_<K>_STARTED`
    - `Events.EPOCHS_<K>_COMPLETED`


    Examples:

    .. code-block:: python

        from ignite.engine import Engine, Events
        from ignite.contrib.handlers import CustomPeriodicEvent

        # Let's define an event every 1000 iterations
        cpe1 = CustomPeriodicEvent(n_iterations=1000)
        cpe1.attach(trainer)

        # Let's define an event every 10 epochs
        cpe2 = CustomPeriodicEvent(n_epochs=10)
        cpe2.attach(trainer)

        @trainer.on(cpe1.Events.ITERATIONS_1000_COMPLETED)
        def on_every_1000_iterations(engine):
            # run a computation after 1000 iterations
            # ...
            print(engine.state.iterations_1000)

        @trainer.on(cpe2.Events.EPOCHS_10_STARTED)
        def on_every_10_epochs(engine):
            # run a computation every 10 epochs
            # ...
            print(engine.state.epochs_10)


    Args:
        n_iterations (int, optional): number iterations of the custom periodic event
        n_epochs (int, optional): number iterations of the custom periodic event. Argument is optional, but only one,
            either n_iterations or n_epochs should defined.

    """

    def __init__(self, n_iterations=None, n_epochs=None):

        warnings.warn(
            "CustomPeriodicEvent is deprecated since 0.4.0 and will be removed in 0.5.0. Use filtered events instead.",
            DeprecationWarning,
        )

        if n_iterations is not None and (not isinstance(n_iterations, int) or n_iterations < 1):
            raise ValueError("Argument n_iterations should be positive integer number")

        if n_epochs is not None and (not isinstance(n_epochs, int) or n_epochs < 1):
            raise ValueError("Argument n_epochs should be positive integer number")

        if (n_iterations is None and n_epochs is None) or (n_iterations and n_epochs):
            raise ValueError("Either n_iterations or n_epochs should defined")

        if n_iterations:
            prefix = "iterations"
            self.state_attr = "iteration"
            self.period = n_iterations

        if n_epochs:
            prefix = "epochs"
            self.state_attr = "epoch"
            self.period = n_epochs

        self.custom_state_attr = "{}_{}".format(prefix, self.period)
        event_name = "{}_{}".format(prefix.upper(), self.period)
        setattr(
            self,
            "Events",
            EventEnum("Events", " ".join(["{}_STARTED".format(event_name), "{}_COMPLETED".format(event_name)])),
        )

        # Update State.event_to_attr
        for e in self.Events:
            State.event_to_attr[e] = self.custom_state_attr

        # Create aliases
        self._periodic_event_started = getattr(self.Events, "{}_STARTED".format(event_name))
        self._periodic_event_completed = getattr(self.Events, "{}_COMPLETED".format(event_name))

    def _on_started(self, engine):
        setattr(engine.state, self.custom_state_attr, 0)

    def _on_periodic_event_started(self, engine):
        if getattr(engine.state, self.state_attr) % self.period == 1:
            setattr(engine.state, self.custom_state_attr, getattr(engine.state, self.custom_state_attr) + 1)
            engine.fire_event(self._periodic_event_started)

    def _on_periodic_event_completed(self, engine):
        if getattr(engine.state, self.state_attr) % self.period == 0:
            engine.fire_event(self._periodic_event_completed)

    def attach(self, engine):
        engine.register_events(*self.Events)

        engine.add_event_handler(Events.STARTED, self._on_started)
        engine.add_event_handler(
            getattr(Events, "{}_STARTED".format(self.state_attr.upper())), self._on_periodic_event_started
        )
        engine.add_event_handler(
            getattr(Events, "{}_COMPLETED".format(self.state_attr.upper())), self._on_periodic_event_completed
        )
