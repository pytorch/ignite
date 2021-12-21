import collections.abc as collections
import functools
import hashlib
import logging
import random
import shutil
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TextIO, Tuple, Type, TypeVar, Union, cast

import torch

__all__ = [
    "convert_tensor",
    "apply_to_tensor",
    "apply_to_type",
    "to_onehot",
    "setup_logger",
    "manual_seed",
    "hash_checkpoint",
]


def convert_tensor(
    x: Union[torch.Tensor, collections.Sequence, collections.Mapping, str, bytes],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
) -> Union[torch.Tensor, collections.Sequence, collections.Mapping, str, bytes]:
    """Move tensors to relevant device.

    Args:
        x: input tensor or mapping, or sequence of tensors.
        device: device type to move ``x``.
        non_blocking: convert a CPU Tensor with pinned memory to a CUDA Tensor
            asynchronously with respect to the host if possible
    """

    def _func(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=device, non_blocking=non_blocking) if device is not None else tensor

    return apply_to_tensor(x, _func)


def apply_to_tensor(
    x: Union[torch.Tensor, collections.Sequence, collections.Mapping, str, bytes], func: Callable
) -> Union[torch.Tensor, collections.Sequence, collections.Mapping, str, bytes]:
    """Apply a function on a tensor or mapping, or sequence of tensors.

    Args:
        x: input tensor or mapping, or sequence of tensors.
        func: the function to apply on ``x``.
    """
    return apply_to_type(x, torch.Tensor, func)


def apply_to_type(
    x: Union[Any, collections.Sequence, collections.Mapping, str, bytes],
    input_type: Union[Type, Tuple[Type[Any], Any]],
    func: Callable,
) -> Union[Any, collections.Sequence, collections.Mapping, str, bytes]:
    """Apply a function on an object of `input_type` or mapping, or sequence of objects of `input_type`.

    Args:
        x: object or mapping or sequence.
        input_type: data type of ``x``.
        func: the function to apply on ``x``.
    """
    if isinstance(x, input_type):
        return func(x)
    if isinstance(x, (str, bytes)):
        return x
    if isinstance(x, collections.Mapping):
        return cast(Callable, type(x))({k: apply_to_type(sample, input_type, func) for k, sample in x.items()})
    if isinstance(x, tuple) and hasattr(x, "_fields"):  # namedtuple
        return cast(Callable, type(x))(*(apply_to_type(sample, input_type, func) for sample in x))
    if isinstance(x, collections.Sequence):
        return cast(Callable, type(x))([apply_to_type(sample, input_type, func) for sample in x])
    raise TypeError((f"x must contain {input_type}, dicts or lists; found {type(x)}"))


def to_onehot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert a tensor of indices of any shape `(N, ...)` to a
    tensor of one-hot indicators of shape `(N, num_classes, ...)` and of type uint8. Output's device is equal to the
    input's device`.

    Args:
        indices: input tensor to convert.
        num_classes: number of classes for one-hot tensor.

    .. versionchanged:: 0.4.3
        This functions is now torchscriptable.
    """
    new_shape = (indices.shape[0], num_classes) + indices.shape[1:]
    onehot = torch.zeros(new_shape, dtype=torch.uint8, device=indices.device)
    return onehot.scatter_(1, indices.unsqueeze(1), 1)


def setup_logger(
    name: Optional[str] = "ignite",
    level: int = logging.INFO,
    stream: Optional[TextIO] = None,
    format: str = "%(asctime)s %(name)s %(levelname)s: %(message)s",
    filepath: Optional[str] = None,
    distributed_rank: Optional[int] = None,
    reset: bool = False,
) -> logging.Logger:
    """Setups logger: name, level, format etc.

    Args:
        name: new name for the logger. If None, the standard logger is used.
        level: logging level, e.g. CRITICAL, ERROR, WARNING, INFO, DEBUG.
        stream: logging stream. If None, the standard stream is used (sys.stderr).
        format: logging format. By default, `%(asctime)s %(name)s %(levelname)s: %(message)s`.
        filepath: Optional logging file path. If not None, logs are written to the file.
        distributed_rank: Optional, rank in distributed configuration to avoid logger setup for workers.
            If None, distributed_rank is initialized to the rank of process.
        reset: if True, reset an existing logger rather than keep format, handlers, and level.

    Returns:
        logging.Logger

    Examples:
        Improve logs readability when training with a trainer and evaluator:

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

        Every existing logger can be reset if needed

        .. code-block:: python

            logger = setup_logger(name="my-logger", format="=== %(name)s %(message)s")
            logger.info("first message")
            setup_logger(name="my-logger", format="+++ %(name)s %(message)s", reset=True)
            logger.info("second message")

            # Logs will look like
            # === my-logger first message
            # +++ my-logger second message

        Change the level of an existing internal logger

        .. code-block:: python

            setup_logger(
                name="ignite.distributed.launcher.Parallel",
                level=logging.WARNING
            )

    .. versionchanged:: 0.4.3
        Added ``stream`` parameter.

    .. versionchanged:: 0.4.5
        Added ``reset`` parameter.
    """
    # check if the logger already exists
    existing = name is None or name in logging.root.manager.loggerDict

    # if existing, get the logger otherwise create a new one
    logger = logging.getLogger(name)

    if distributed_rank is None:
        import ignite.distributed as idist

        distributed_rank = idist.get_rank()

    # Remove previous handlers
    if distributed_rank > 0 or reset:

        if logger.hasHandlers():
            for h in list(logger.handlers):
                logger.removeHandler(h)

    if distributed_rank > 0:

        # Add null handler to avoid multiple parallel messages
        logger.addHandler(logging.NullHandler())

    # Keep the existing configuration if not reset
    if existing and not reset:
        return logger

    if distributed_rank == 0:
        logger.setLevel(level)

        formatter = logging.Formatter(format)

        ch = logging.StreamHandler(stream=stream)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if filepath is not None:
            fh = logging.FileHandler(filepath)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    # don't propagate to ancestors
    # the problem here is to attach handlers to loggers
    # should we provide a default configuration less open ?
    if name is not None:
        logger.propagate = False

    return logger


def manual_seed(seed: int) -> None:
    """Setup random state from a seed for `torch`, `random` and optionally `numpy` (if can be imported).

    Args:
        seed: Random state seed

    .. versionchanged:: 0.4.3
        Added ``torch.cuda.manual_seed_all(seed)``.

    .. versionchanged:: 0.4.5
        Added ``torch_xla.core.xla_model.set_rng_state(seed)``.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        import torch_xla.core.xla_model as xm

        xm.set_rng_state(seed)
    except ImportError:
        pass

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def deprecated(
    deprecated_in: str, removed_in: str = "", reasons: Tuple[str, ...] = (), raise_exception: bool = False
) -> Callable:

    F = TypeVar("F", bound=Callable[..., Any])

    def decorator(func: F) -> F:
        func_doc = func.__doc__ if func.__doc__ else ""
        deprecation_warning = (
            f"This function has been deprecated since version {deprecated_in}"
            + (f" and will be removed in version {removed_in}" if removed_in else "")
            + ".\n Please refer to the documentation for more details."
        )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Dict[str, Any]) -> Callable:
            if raise_exception:
                raise DeprecationWarning(deprecation_warning)
            warnings.warn(deprecation_warning, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        appended_doc = f".. deprecated:: {deprecated_in}" + ("\n\n\t" if len(reasons) > 0 else "")

        for reason in reasons:
            appended_doc += "\n\t- " + reason
        wrapper.__doc__ = f"**Deprecated function**.\n\n    {func_doc}{appended_doc}"
        return cast(F, wrapper)

    return decorator


def hash_checkpoint(checkpoint_path: Union[str, Path], output_dir: Union[str, Path]) -> Tuple[Path, str]:
    """
    Hash the checkpoint file in the format of ``<filename>-<hash>.<ext>``
    to be used with ``check_hash`` of :func:`torch.hub.load_state_dict_from_url`.

    Args:
        checkpoint_path: Path to the checkpoint file.
        output_dir: Output directory to store the hashed checkpoint file
            (will be created if not exist).

    Returns:
        Path to the hashed checkpoint file, the first 8 digits of SHA256 hash.

    .. versionadded:: 0.5.0
    """

    if isinstance(checkpoint_path, str):
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{checkpoint_path.name} does not exist in {checkpoint_path.parent}.")

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    hash_obj = hashlib.sha256()
    # taken from https://github.com/pytorch/vision/blob/main/references/classification/utils.py
    with checkpoint_path.open("rb") as f:
        # Read and update hash string value in blocks of 4KB
        for byte_block in iter(lambda: f.read(4096), b""):
            hash_obj.update(byte_block)
    sha_hash = hash_obj.hexdigest()

    old_filename = checkpoint_path.stem
    new_filename = "-".join((old_filename, sha_hash[:8])) + ".pt"

    hash_checkpoint_path = output_dir / new_filename
    shutil.move(str(checkpoint_path), hash_checkpoint_path)

    return hash_checkpoint_path, sha_hash
