import os
from pathlib import Path

import fire
import gin
import torch
import ignite.distributed as idist

import utils


def download_datasets(output_path):
    """Helper tool to download datasets

    Args:
        output_path (str): path where to download and unzip the dataset
    """
    from torchvision.datasets.voc import VOCSegmentation
    from torchvision.datasets.sbd import SBDataset

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Download Pascal VOC 2012 - Training")
    VOCSegmentation(output_path.as_posix(), image_set="train", download=True)
    print("Download Pascal VOC 2012 - Validation")
    VOCSegmentation(output_path.as_posix(), image_set="val", download=True)
    print("Download SBD - Training without Pascal VOC validation part")
    sbd_path = output_path / "SBD"
    sbd_path.mkdir(exist_ok=True)
    SBDataset(sbd_path.as_posix(), image_set="train_noval", mode="segmentation", download=True)
    print("Done")
    print(f"Pascal VOC 2012 is at : {(output_path / 'VOCdevkit').as_posix()}")
    print(f"SBD is at : {sbd_path.as_posix()}")


def training(local_rank, config, logger=None):
    pass


def run_training(config, backend="nccl", with_clearml=True):
    """Main entry to run training experiment

    Args:
        config (str): training configuration .gin file
        backend (str): distributed backend: nccl, gloo or horovod
        with_clearml (bool): if True, uses ClearML as experiment tracking system
    """
    assert torch.cuda.is_available(), torch.cuda.is_available()
    assert torch.backends.cudnn.enabled, "AMP requires cudnn backend to be enabled."

    assert Path(config).exists(), f"File '{config}' is not found"

    with idist.Parallel(backend=backend) as parallel:

        logger = setup_logger(name="Pascal-VOC12 Training", distributed_rank=idist.get_rank())

        utils.log_basic_info(logger, config)

        gin.parse_config_file(config)

        try:
            parallel.run(training, config, logger=logger)
        except KeyboardInterrupt:
            logger.info("Catched KeyboardInterrupt -> exit")
        except Exception as e:  # noqa
            logger.exception("")
            raise e



def run_evaluation(config, backend="nccl"):
    """Main entry to run evaluate trained model

    Args:
        config (str): evaluation configuration .gin file
        backend (str): distributed backend: nccl, gloo or horovod
    """
    pass


if __name__ == "__main__":
    fire.Fire({"download": download_datasets, "training": run_training, "eval": run_evaluation})
