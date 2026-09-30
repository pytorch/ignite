import torch

import ignite
import ignite.distributed as idist
from ignite.handlers import DiskSaver


def initialize(config):
    device = idist.device()

    model = config.model.to(device)
    optimizer = config.optimizer

    # Adapt model to dist config
    model = idist.auto_model(model)
    optimizer = idist.auto_optim(optimizer)
    criterion = config.criterion.to(device)

    return model, optimizer, criterion


def log_basic_info(logger, config):
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
        # explicitly import cudnn as
        # torch.backends.cudnn can not be pickled with hvd spawning procs
        from torch.backends import cudnn

        logger.info(f"- GPU Device: {torch.cuda.get_device_name(idist.get_local_rank())}")
        logger.info(f"- CUDA version: {torch.version.cuda}")
        logger.info(f"- CUDNN version: {cudnn.version()}")

    logger.info("\n")
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"\t{key}: {value}")
    logger.info("\n")

    if idist.get_world_size() > 1:
        logger.info("\nDistributed setting:")
        logger.info(f"\tbackend: {idist.backend()}")
        logger.info(f"\tworld size: {idist.get_world_size()}")
        logger.info("\n")


def log_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(f"\nEpoch {epoch} - Evaluation time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}")


def get_save_handler(output_path, with_clearml):
    if with_clearml:
        from ignite.handlers.clearml_logger import ClearMLSaver

        return ClearMLSaver(dirname=output_path)

    return DiskSaver(output_path)
