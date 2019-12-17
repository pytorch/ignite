from ignite.handlers.checkpoint import ModelCheckpoint, Checkpoint, DiskSaver
from ignite.handlers.timing import Timer
from ignite.handlers.early_stopping import EarlyStopping
from ignite.handlers.terminate_on_nan import TerminateOnNan


def global_step_from_engine(engine):
    """Helper method to setup `global_step_transform` function using another engine.
    This can be helpful for logging trainer epoch/iteration while output handler is attached to an evaluator.

    Args:
        engine (Engine): engine which state is used to provide the global step

    Returns:
        global step
    """

    def wrapper(_, event_name):
        return engine.state.get_event_attrib_value(event_name)

    return wrapper
