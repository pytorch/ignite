import torch
import ignite
import ignite.distributed as idist


def log_basic_info(logger, config):

    msg = f"\n- PyTorch version: {torch.__version__}"
    msg += f"\n- Ignite version: {ignite.__version__}"
    msg += f"\n- Cuda device name: {torch.cuda.get_device_name(idist.get_local_rank())}"

    logger.info(msg)

    if idist.get_world_size() > 1:
        msg = "\nDistributed setting:"
        msg += f"\tbackend: {idist.backend()}"
        msg += f"\trank: {idist.get_rank()}"
        msg += f"\tworld size: {idist.get_world_size()}"
        logger.info(msg)


def log_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(f"\nEpoch {epoch} - Evaluation time (seconds): {int(elapsed)} - {tag} metrics:\n {metrics_output}")
