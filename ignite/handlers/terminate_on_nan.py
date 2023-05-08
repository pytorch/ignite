import logging
import numbers
from typing import Callable, Union

import torch

from ignite.engine import Engine
from ignite.utils import apply_to_type, setup_logger

__all__ = ["TerminateOnNan"]


class TerminateOnNan:
    """TerminateOnNan handler can be used to stop the training if the `process_function`'s output
    contains a NaN or infinite number or `torch.tensor`.
    The output can be of type: number, tensor or collection of them. The training is stopped if
    there is at least a single number/tensor have NaN or Infinite value. For example, if the output is
    `[1.23, torch.tensor(...), torch.tensor(float('nan'))]` the handler will stop the training.

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into a number or `torch.tensor`
            or collection of them. This can be useful if, for example, you have a multi-output model and
            you want to check one or multiple values of the output.


    Examples:
        .. code-block:: python

            trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    """

    def __init__(self, output_transform: Callable = lambda x: x):
        self.logger = setup_logger(__name__ + "." + self.__class__.__name__)
        self.logger.addHandler(logging.StreamHandler())
        self._output_transform = output_transform

    def __call__(self, engine: Engine) -> None:
        output = self._output_transform(engine.state.output)

        def raise_error(x: Union[float, torch.Tensor]) -> None:
            if isinstance(x, numbers.Number):
                x = torch.tensor(x)

            if isinstance(x, torch.Tensor) and not bool(torch.isfinite(x).all()):
                raise RuntimeError("Infinite or NaN tensor found.")

        try:
            apply_to_type(output, (numbers.Number, torch.Tensor), raise_error)
        except RuntimeError:
            self.logger.warning(f"{self.__class__.__name__}: Output '{output}' contains NaN or Inf. Stop training")
            engine.terminate()
