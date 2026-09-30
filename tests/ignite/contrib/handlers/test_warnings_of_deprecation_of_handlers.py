from importlib import __import__

import pytest


@pytest.mark.parametrize(
    "log_module,fromlist",
    [
        ("mlflow_logger", ["MLflowLogger", "OptimizerParamsHandler", "OutputHandler"]),
        ("polyaxon_logger", ["PolyaxonLogger", "OutputHandler", "OptimizerParamsHandler"]),
        ("wandb_logger", ["WandBLogger", "OutputHandler", "OptimizerParamsHandler"]),
        ("lr_finder", ["FastaiLRFinder"]),
        ("tqdm_logger", ["ProgressBar"]),
        (
            "clearml_logger",
            [
                "ClearMLLogger",
                "ClearMLSaver",
                "OptimizerParamsHandler",
                "OutputHandler",
                "WeightsScalarHandler",
                "WeightsHistHandler",
                "GradsScalarHandler",
                "GradsHistHandler",
            ],
        ),
        (
            "tensorboard_logger",
            [
                "TensorboardLogger",
                "OptimizerParamsHandler",
                "OutputHandler",
                "WeightsScalarHandler",
                "WeightsHistHandler",
                "GradsScalarHandler",
                "GradsHistHandler",
            ],
        ),
        (
            "visdom_logger",
            [
                "VisdomLogger",
                "OptimizerParamsHandler",
                "OutputHandler",
                "WeightsScalarHandler",
                "GradsScalarHandler",
            ],
        ),
        (
            "neptune_logger",
            [
                "NeptuneLogger",
                "NeptuneSaver",
                "OptimizerParamsHandler",
                "OutputHandler",
                "WeightsScalarHandler",
                "GradsScalarHandler",
            ],
        ),
        (
            "base_logger",
            [
                "BaseHandler",
                "BaseWeightsHandler",
                "BaseOptimizerParamsHandler",
                "BaseOutputHandler",
                "BaseWeightsScalarHandler",
                "BaseLogger",
            ],
        ),
        (
            "time_profilers",
            [
                "BasicTimeProfiler",
                "HandlersTimeProfiler",
            ],
        ),
        (
            "param_scheduler",
            [
                "ConcatScheduler",
                "CosineAnnealingScheduler",
                "LinearCyclicalScheduler",
                "LRScheduler",
                "ParamGroupScheduler",
                "ParamScheduler",
                "PiecewiseLinear",
                "CyclicalScheduler",
                "create_lr_scheduler_with_warmup",
            ],
        ),
    ],
)
def test_imports(log_module, fromlist):
    with pytest.warns(DeprecationWarning, match="will be removed in version 0.6.0"):
        imported = __import__(f"ignite.contrib.handlers.{log_module}", globals(), locals(), fromlist)
        for attr in fromlist:
            getattr(imported, attr)
