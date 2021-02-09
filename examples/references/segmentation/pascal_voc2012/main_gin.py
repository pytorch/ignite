import os
from pathlib import Path

import fire
import gin
import torch

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor

import ignite.distributed as idist
from ignite.utils import setup_logger, manual_seed

import utils
import dataflow as data


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
    SBDataset(
        sbd_path.as_posix(), image_set="train_noval", mode="segmentation", download=True
    )
    print("Done")
    print(f"Pascal VOC 2012 is at : {(output_path / 'VOCdevkit').as_posix()}")
    print(f"SBD is at : {sbd_path.as_posix()}")


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


train_img_size = 480
val_img_size = 513


train_transforms = A.Compose(
    [
        A.RandomScale(scale_limit=(0.0, 1.5), interpolation=cv2.INTER_LINEAR, p=1.0),
        A.PadIfNeeded(val_img_size, val_img_size, border_mode=cv2.BORDER_CONSTANT),
        A.RandomCrop(train_img_size, train_img_size),
        A.HorizontalFlip(),
        A.Blur(blur_limit=3),
        A.Normalize(mean=mean, std=std),
        data.ignore_mask_boundaries,
        ToTensor(),
    ]
)


val_transforms = A.Compose(
    [
        A.PadIfNeeded(val_img_size, val_img_size, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=mean, std=std),
        data.ignore_mask_boundaries,
        ToTensor(),
    ]
)


@gin.configurable
def training(local_rank, logger, **config):

    assert "DATASET_PATH" in os.environ, f"DATASET_PATH should be defined as env variable"
    data_path = os.environ["DATASET_PATH"]

    sbd_data_path = None
    if config.get("with_sbd", False):
        assert "SBD_DATASET_PATH" in os.environ, f"SBD_DATASET_PATH should be defined as env variable"
        sbd_data_path = os.environ["SBD_DATASET_PATH"]

    manual_seed(config["seed"] + local_rank)
    torch.backends.cudnn.benchmark = True
    utils.log_basic_info(logger, config)

    train_loader, val_loader, train_eval_loader = data.get_train_val_loaders(
        data_path,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        sbd_path=sbd_data_path,
    )

    model, optimizer, criterion = utils.initialize(config)


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
        logger = setup_logger(
            name="Pascal-VOC12 Training", distributed_rank=idist.get_rank()
        )
        gin.parse_config_file(config)

        try:
            parallel.run(
                training,
                logger=logger,
            )
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


def gin_register_ext_configs():
    """Register external configurable for gin-config
    """
    import torchvision.models.segmentation as segm_models

    for module in [segm_models, ]:
        for k, v in module.__dict__.items():
            if k.startswith("_"):
                continue
            if callable(v) or isinstance(v, type):
                gin.external_configurable(v, module=segm_models.__package__)


if __name__ == "__main__":

    gin_register_ext_configs()

    fire.Fire(
        {
            "download": download_datasets,
            "training": run_training,
            "eval": run_evaluation,
        }
    )
