# This a training script launched with py_config_runner
# It should obligatory contain `run(config, **kwargs)` method

from pathlib import Path

import torch
import torch.distributed as dist

import ignite
from polyaxon_client.tracking import get_outputs_path, Experiment

from py_config_runner.config_utils import get_params, TRAINVAL_CONFIG, assert_config

from common_training import training


def run(config, logger=None, local_rank=0, **kwargs):

    assert torch.cuda.is_available(), torch.cuda.is_available()
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

    config.output_path = Path(get_outputs_path())

    if dist.get_rank() == 0:
        plx_exp = Experiment()
        plx_exp.log_params(**{
            "pytorch version": torch.__version__,
            "ignite version": ignite.__version__,
        })
        plx_exp.log_params(**get_params(config, TRAINVAL_CONFIG))

    try:
        training(config, local_rank=local_rank, with_mlflow_logging=False, with_plx_logging=True)
    except KeyboardInterrupt:
        logger.info("Catched KeyboardInterrupt -> exit")
    except Exception as e:  # noqa
        logger.exception("")
        dist.destroy_process_group()
        raise e

    dist.destroy_process_group()
