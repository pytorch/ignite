import torch
import logging
import numbers


class TerminateOnNan(object):
    """TerminateOnNan handler can be used to stop the training if the `process_function`'s output becomes
    NaN or infinite `torch.tensor` or number.

    Args:
        output_transform (Callable, optional): a callable that is used to transform the
            :class:`ignite.engine.Engine`'s `process_function`'s output into a number or `torch.tensor`.
            This can be useful if, for example, you have a multi-output model and
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

        if isinstance(output, numbers.Number):
            output = torch.tensor(output)

        if not isinstance(output, torch.Tensor):
            raise TypeError("Output should be either torch.tensor or number")

        if not bool(torch.isfinite(output).all()):
            self._logger.warning("{}: Loss is NaN or Inf. Stop training".format(self.__class__.__name__))
            engine.terminate()
