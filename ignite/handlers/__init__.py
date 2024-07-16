from typing import Any, Callable, Optional

from ignite.engine import Engine
from ignite.engine.events import Events
from ignite.handlers.checkpoint import Checkpoint, DiskSaver, ModelCheckpoint
from ignite.handlers.clearml_logger import ClearMLLogger
from ignite.handlers.early_stopping import EarlyStopping
from ignite.handlers.ema_handler import EMAHandler
from ignite.handlers.lr_finder import FastaiLRFinder
from ignite.handlers.mlflow_logger import MLflowLogger
from ignite.handlers.neptune_logger import NeptuneLogger
from ignite.handlers.param_scheduler import (
    BaseParamScheduler,
    ConcatScheduler,
    CosineAnnealingScheduler,
    create_lr_scheduler_with_warmup,
    CyclicalScheduler,
    LinearCyclicalScheduler,
    LRScheduler,
    ParamGroupScheduler,
    ParamScheduler,
    PiecewiseLinear,
    ReduceLROnPlateauScheduler,
)

from ignite.handlers.polyaxon_logger import PolyaxonLogger
from ignite.handlers.state_param_scheduler import (
    ExpStateScheduler,
    LambdaStateScheduler,
    MultiStepStateScheduler,
    PiecewiseLinearStateScheduler,
    StateParamScheduler,
    StepStateScheduler,
)
from ignite.handlers.stores import EpochOutputStore
from ignite.handlers.tensorboard_logger import TensorboardLogger
from ignite.handlers.terminate_on_nan import TerminateOnNan
from ignite.handlers.time_limit import TimeLimit
from ignite.handlers.time_profilers import BasicTimeProfiler, HandlersTimeProfiler
from ignite.handlers.timing import Timer
from ignite.handlers.tqdm_logger import ProgressBar
from ignite.handlers.utils import global_step_from_engine  # noqa

from ignite.handlers.visdom_logger import VisdomLogger
from ignite.handlers.wandb_logger import WandBLogger

__all__ = [
    "ModelCheckpoint",
    "Checkpoint",
    "DiskSaver",
    "Timer",
    "EarlyStopping",
    "TerminateOnNan",
    "global_step_from_engine",
    "TimeLimit",
    "EpochOutputStore",
    "ConcatScheduler",
    "CosineAnnealingScheduler",
    "LinearCyclicalScheduler",
    "LRScheduler",
    "ParamGroupScheduler",
    "ParamScheduler",
    "PiecewiseLinear",
    "CyclicalScheduler",
    "create_lr_scheduler_with_warmup",
    "FastaiLRFinder",
    "EMAHandler",
    "BasicTimeProfiler",
    "HandlersTimeProfiler",
    "BaseParamScheduler",
    "StateParamScheduler",
    "LambdaStateScheduler",
    "PiecewiseLinearStateScheduler",
    "ExpStateScheduler",
    "StepStateScheduler",
    "MultiStepStateScheduler",
    "ReduceLROnPlateauScheduler",
    "ClearMLLogger",
    "MLflowLogger",
    "NeptuneLogger",
    "PolyaxonLogger",
    "TensorboardLogger",
    "ProgressBar",
    "VisdomLogger",
    "WandBLogger",
]
