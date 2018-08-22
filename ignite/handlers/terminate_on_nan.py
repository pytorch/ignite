import logging
import numbers

import torch

from ignite._utils import apply_to_tensor


class TerminateOnNan(object):
    """TerminateOnNan handler can be used to stop the training if the `process_function`'s output being
    a number, `torch.tensor` or collection of them contains NaN or infinite value.

    Examples:

    .. code-block:: python

        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    """

    def __init__(self):
        self._logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self._logger.addHandler(logging.StreamHandler())

    def __call__(self, engine):

        output = engine.state.output
        if isinstance(output, numbers.Number):
            output = torch.tensor(output)

        def raise_error(x):
            if not bool(torch.isfinite(x).all()):
                raise RuntimeError("Infinite or NaN tensor found")

        try:
            apply_to_tensor(output, raise_error)
        except RuntimeError:
            self._logger.warning("{}: Output '{}' contains NaN or Inf. Stop training"
                                 .format(self.__class__.__name__, output))
            engine.terminate()
