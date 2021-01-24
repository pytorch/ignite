from ignite.contrib.handlers.clearml_logger import ClearMLLogger, ClearMLSaver

__all__ = [
    "TrainsLogger",
    "TrainsSaver",
]

TrainsLogger = ClearMLLogger
TrainsSaver = ClearMLSaver
