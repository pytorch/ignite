import collections.abc as collections
import logging

import torch


def convert_tensor(input_, device=None, non_blocking=False):
    """Move tensors to relevant device."""
    def _func(tensor):
        return tensor.to(device=device, non_blocking=non_blocking) if device else tensor

    return apply_to_tensor(input_, _func)


def apply_to_tensor(input_, func):
    """Apply a function on a tensor or mapping, or sequence of tensors.
    """
    return apply_to_type(input_, torch.Tensor, func)


def apply_to_type(input_, input_type, func):
    """Apply a function on a object of `input_type` or mapping, or sequence of objects of `input_type`.
    """
    if isinstance(input_, input_type):
        return func(input_)
    elif isinstance(input_, (str, bytes)):
        return input_
    elif isinstance(input_, collections.Mapping):
        return {k: apply_to_type(sample, input_type, func) for k, sample in input_.items()}
    elif isinstance(input_, collections.Sequence):
        return [apply_to_type(sample, input_type, func) for sample in input_]
    else:
        raise TypeError(("input must contain {}, dicts or lists; found {}"
                         .format(input_type, type(input_))))


def to_onehot(indices, num_classes):
    """Convert a tensor of indices of any shape `(N, ...)` to a
    tensor of one-hot indicators of shape `(N, num_classes, ...) and of type uint8. Output's device is equal to the
    input's device`.
    """
    onehot = torch.zeros(indices.shape[0], num_classes, *indices.shape[1:],
                         dtype=torch.uint8,
                         device=indices.device)
    return onehot.scatter_(1, indices.unsqueeze(1), 1)


def setup_logger(name, level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s",
                 filepath=None, distributed_rank=0):
    """Setups logger: name, level, format etc.

    Args:
        name (str): new name for the logger.
        level (int): logging level, e.g. CRITICAL, ERROR, WARNING, INFO, DEBUG
        format (str): logging format. By default, `%(asctime)s %(name)s %(levelname)s: %(message)s`
        filepath (str, optional): Optional logging file path. If not None, logs are written to the file.
        distributed_rank (int, optional): Optional, rank in distributed configuration to avoid logger setup for workers.

    Returns:
        logging.Logger

    For example, to improve logs readability when training with a trainer and evaluator:

    .. code-block:: python

        from ignite.utils import setup_logger

        trainer = ...
        evaluator = ...

        trainer.logger = setup_logger("trainer")
        evaluator.logger = setup_logger("evaluator")

        trainer.run(data, max_epochs=10)

        # Logs will look like
        # 2020-01-21 12:46:07,356 trainer INFO: Engine run starting with max_epochs=5.
        # 2020-01-21 12:46:07,358 trainer INFO: Epoch[1] Complete. Time taken: 00:5:23
        # 2020-01-21 12:46:07,358 evaluator INFO: Engine run starting with max_epochs=1.
        # 2020-01-21 12:46:07,358 evaluator INFO: Epoch[1] Complete. Time taken: 00:01:02
        # ...

    """
    logger = logging.getLogger(name)

    if distributed_rank > 0:
        return logger

    logger.setLevel(level)

    # Remove previous handlers
    if logger.hasHandlers():
        for h in list(logger.handlers):
            logger.removeHandler(h)

    formatter = logging.Formatter(format)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if filepath is not None:
        fh = logging.FileHandler(filepath)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
