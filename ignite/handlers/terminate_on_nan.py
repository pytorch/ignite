import logging
import numbers

import torch

from ignite.utils import apply_to_type


class TerminateOnNan(object):
    """TerminateOnNan handler can be used to stop the training if the `process_function`'s output
    contains a NaN or infinite number or `torch.tensor`.
    The output can be of type: number, tensor or collection of them. The training is stopped if
    there is at least a single number/tensor have NaN or Infinite value. For example, if the output is
    `[1.23, torch.tensor(...), torch.tensor(float('nan'))]` the handler will stop the training.

    Args:
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.Engine`'s `process_function`'s output into a number or `torch.tensor`
            or collection of them. This can be useful if, for example, you have a multi-output model and
            you want to check one or multiple values of the output.


    Examples:

    .. code-block:: python

        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    """

    def __init__(self, output_transform=lambda x: x):
        self._logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self._logger.addHandler(logging.StreamHandler())
        self._output_transform = output_transform

    def __call__(self, engine):
        output = self._output_transform(engine.state.output)

        def raise_error(x):

            if isinstance(x, numbers.Number):
                x = torch.tensor(x)

            if isinstance(x, torch.Tensor) and not bool(torch.isfinite(x).all()):
                raise RuntimeError("Infinite or NaN tensor found.")

        try:
            apply_to_type(output, (numbers.Number, torch.Tensor), raise_error)
        except RuntimeError:
            self._logger.warning("{}: Output '{}' contains NaN or Inf. Stop training"
                                 .format(self.__class__.__name__, output))
            engine.terminate()
