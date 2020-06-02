# This a training script launched with py_config_runner
# It should obligatory contain `run(config, **kwargs)` method

import sys
from pathlib import Path

import torch
import torch.distributed as dist

from trains import Task
import ignite

from py_config_runner.config_utils import get_params, TRAINVAL_CONFIG, assert_config

# add our directories to sys path
sys.path.append(Path(__file__).parent.parent.as_posix())
sys.path.append((Path(__file__).parent.parent / "dataflow").as_posix())

from common_training import training


def run(config, logger=None, local_rank=0, **kwargs):

    assert torch.cuda.is_available()
    assert torch.backends.cudnn.enabled, "Nvidia/Amp requires cudnn backend to be enabled."

    task = Task.init("ignite", "DeeplabV3_ResNet101 pascal_voc2012 segmentation example")

    dist.init_process_group("nccl", init_method="env://")

    # As we passed config with option --manual_config_load
    assert hasattr(config, "setup"), (
        "We need to manually setup the configuration, please set --manual_config_load " "to py_config_runner"
    )

    config = config.setup()

    assert_config(config, TRAINVAL_CONFIG)
    # The following attributes are automatically added by py_config_runner
    assert hasattr(config, "config_filepath") and isinstance(config.config_filepath, Path)
    assert hasattr(config, "script_filepath") and isinstance(config.script_filepath, Path)

    # dump python files to reproduce the run
    task.connect_configuration(config.config_filepath.as_posix())
    task.upload_artifact("script", config.script_filepath)

    config.output_path = Path("./artifacts")

    # log the configuration, if we are the master node
    if dist.get_rank() == 0:
        task.connect(get_params(config, TRAINVAL_CONFIG))

    try:
        training(config, local_rank=local_rank, with_trains_logging=True)
    except KeyboardInterrupt:
        logger.info("Caught KeyboardInterrupt -> exit")
    except Exception as e:  # noqa
        logger.exception("")
        dist.destroy_process_group()
        raise e

    dist.destroy_process_group()
