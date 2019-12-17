# This a training script launched with py_config_runner
# It should obligatory contain `run(config, **kwargs)` method

from pathlib import Path

import torch
import torch.distributed as dist

import mlflow
import ignite

from py_config_runner.config_utils import get_params, TRAINVAL_CONFIG, assert_config

from common_training import training


def run(config, logger=None, local_rank=0, **kwargs):

    assert torch.cuda.is_available()
    assert torch.backends.cudnn.enabled, "Nvidia/Amp requires cudnn backend to be enabled."

    dist.init_process_group("nccl", init_method="env://")

    # As we passed config with option --manual_config_load
    assert hasattr(config, "setup"), "We need to manually setup the configuration, please set --manual_config_load " \
                                     "to py_config_runner"

    config = config.setup()

    assert_config(config, TRAINVAL_CONFIG)
    # The following attributes are automatically added by py_config_runner
    assert hasattr(config, "config_filepath") and isinstance(config.config_filepath, Path)
    assert hasattr(config, "script_filepath") and isinstance(config.script_filepath, Path)

    # dump python files to reproduce the run
    mlflow.log_artifact(config.config_filepath.as_posix())
    mlflow.log_artifact(config.script_filepath.as_posix())

    output_path = mlflow.get_artifact_uri()
    config.output_path = Path(output_path)

    if dist.get_rank() == 0:
        mlflow.log_params({
            "pytorch version": torch.__version__,
            "ignite version": ignite.__version__,
        })
        mlflow.log_params(get_params(config, TRAINVAL_CONFIG))

    try:
        training(config, local_rank=local_rank, with_mlflow_logging=True, with_plx_logging=False)
    except KeyboardInterrupt:
        logger.info("Catched KeyboardInterrupt -> exit")
    except Exception as e:  # noqa
        logger.exception("")
        mlflow.log_param("Run Status", "FAILED")
        dist.destroy_process_group()
        raise e

    mlflow.log_param("Run Status", "OK")
    dist.destroy_process_group()
