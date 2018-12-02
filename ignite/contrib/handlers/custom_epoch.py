from enum import Enum
from ignite.engine import Events


class CustomEpochEvents(Enum):

    CUSTOM_EPOCH_STARTED = "custom_epoch_started"
    CUSTOM_EPOCH_COMPLETED = "custom_epoch_completed"


class CustomEpochLength(object):
    """Handler to define a custom epoch as a number of iterations.
    When custom epoch started or ended the following events `CustomEpochEvents.CUSTOM_EPOCH_STARTED`
    `CustomEpochEvents.CUSTOM_EPOCH_COMPLETED` are fired.

    Examples:

    .. code-block:: python

        from ignite.engine import Engine, Events
        from ignite.handlers import CustomEpochLength, CustomEpochEvents

        # Let's define an epoch by 1000 iterations
        CustomEpochLength(n_iterations=1000).attach(trainer)

        @trainer.on(CustomEpochEvents.CUSTOM_EPOCH_COMPLETED):
        def on_custom_epoch_ends(engine):
            # run a computation after 1000 iterations
            # ...
            print(engine.state.custom_epoch)

    Args:
        n_iterations (int, optional): number iterations of the custom epoch

    """

    def __init__(self, n_iterations):
        if not isinstance(n_iterations, int) or n_iterations < 1:
            raise ValueError("Argument n_iterations should be positive integer number")
        self.custom_epoch_length = n_iterations

    def _on_started(self, engine):
        engine.state.custom_epoch = 0

    def _on_iteration_started(self, engine):
        if engine.state.iteration % self.custom_epoch_length == 1:
            engine.state.custom_epoch += 1
            engine.fire_event(CustomEpochEvents.CUSTOM_EPOCH_STARTED)

    def _on_iteration_ended(self, engine):
        if engine.state.iteration % self.custom_epoch_length == 0:
            engine.fire_event(CustomEpochEvents.CUSTOM_EPOCH_COMPLETED)

    def attach(self, engine):
        engine.register_events(*CustomEpochEvents)

        engine.add_event_handler(Events.STARTED, self._on_started)
        engine.add_event_handler(Events.ITERATION_STARTED, self._on_iteration_started)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._on_iteration_ended)
