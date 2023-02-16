# Module for common exp tracking methods

import os
from pathlib import Path

import torch

import ignite
import ignite.distributed as idist
from ignite.contrib.engines import common

try:
    import polyaxon.tracking  # noqa: F401

    has_plx = True
except ImportError:
    try:
        import polyaxon_client.tracking  # noqa: F401

        if "POLYAXON_RUN_OUTPUTS_PATH" not in os.environ:
            raise ImportError("Not in Polyaxon cluster")

        has_plx = True
    except ImportError:
        has_plx = False

try:
    import mlflow

    if "MLFLOW_TRACKING_URI" not in os.environ:
        raise ImportError("MLFLOW_TRACKING_URI should be defined")

    has_mlflow = True
except ImportError:
    has_mlflow = False


try:
    try:
        import clearml  # noqa: F401
    except ImportError:
        import trains  # noqa: F401

    if "CLEARML_OUTPUT_PATH" not in os.environ:
        raise ImportError("CLEARML_OUTPUT_PATH should be defined")

    has_clearml = True
    clearml_output_path = None
except ImportError:
    has_clearml = False


def _plx_get_output_path():
    from polyaxon_client.tracking import get_outputs_path

    return get_outputs_path()


@idist.one_rank_only()
def _plx_log_artifact(fp):
    from polyaxon_client.tracking import Experiment

    plx_exp = Experiment()
    plx_exp.log_artifact(fp)


@idist.one_rank_only()
def _plx_log_params(params_dict):
    from polyaxon_client.tracking import Experiment

    plx_exp = Experiment()
    plx_exp.log_inputs(**{"pytorch version": torch.__version__, "ignite version": ignite.__version__})
    plx_exp.log_inputs(**params_dict)


def _mlflow_get_output_path():
    return mlflow.get_artifact_uri()


@idist.one_rank_only()
def _mlflow_log_artifact(fp):
    mlflow.log_artifact(fp)


@idist.one_rank_only()
def _mlflow_log_params(params_dict):
    mlflow.log_params({"pytorch version": torch.__version__, "ignite version": ignite.__version__})
    mlflow.log_params(params_dict)


def _clearml_get_output_path():
    global clearml_output_path

    if clearml_output_path is None:
        from datetime import datetime

        output_path = Path(os.environ["CLEARML_OUTPUT_PATH"])
        output_path = output_path / "clearml" / datetime.now().strftime("%Y%m%d-%H%M%S")
        clearml_output_path = output_path

    return clearml_output_path.as_posix()


@idist.one_rank_only()
def _clearml_log_artifact(fp):
    try:
        from clearml import Task
    except ImportError:
        # Backwards-compatibility for legacy Trains SDK
        from trains import Task

    task = Task.current_task()
    task.upload_artifact(Path(fp).name, fp)


@idist.one_rank_only()
def _clearml_log_params(params_dict):
    try:
        from clearml import Task
    except ImportError:
        # Backwards-compatibility for legacy Trains SDK
        from trains import Task

    task = Task.current_task()
    task.connect(params_dict)


if has_plx:
    get_output_path = _plx_get_output_path
    log_params = _plx_log_params
    setup_logging = common.setup_plx_logging
    log_artifact = _plx_log_artifact
elif has_mlflow:
    get_output_path = _mlflow_get_output_path
    log_params = _mlflow_log_params
    setup_logging = common.setup_mlflow_logging
    log_artifact = _mlflow_log_artifact
elif has_clearml:
    get_output_path = _clearml_get_output_path
    log_params = _clearml_log_params
    setup_logging = common.setup_clearml_logging
    log_artifact = _clearml_log_artifact
else:
    raise RuntimeError(
        "No experiment tracking system is setup. "
        "Please, setup either MLflow, Polyaxon or ClearML. "
        "For more details see NOTES_*.md"
    )
