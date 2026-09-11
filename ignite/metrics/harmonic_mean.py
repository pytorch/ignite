from typing import Callable, Union
import torch
from ignite.metrics.metric import Metric, sync_all_reduce
from ignite.exceptions import NotComputableError

class HarmonicMean(Metric):
    """
    Computes the harmonic mean.
    
    .. math::
        H = \\frac{n}{\\sum_{i=1}^n (1 / x_i)}
        
    where :math:`x_i` are the individual values and :math:`n` is the total count of values.

    Args:
        output_transform: A callable that transforms the engine's output into the
            expected format.
        device: Specifies which device updates are accumulated on.

    Example:
        .. code-block:: python

            metric = HarmonicMean()
            metric.attach(evaluator, "harmonic_mean")

    .. versionadded:: 0.5.4
    """

    def __init__(self, output_transform: Callable = lambda x: x, device: Union[str, torch.device] = torch.device("cpu")):
        super(HarmonicMean, self).__init__(output_transform=output_transform, device=device)
        self.reset()

    def reset(self) -> None:
        super(HarmonicMean, self).reset()
        self._sum_reciprocal = torch.tensor(0.0, device=self._device)
        self._num_examples = 0

    def update(self, output: torch.Tensor) -> None:
        if not isinstance(output, torch.Tensor):
            output = torch.as_tensor(output)

        values = output.detach().reshape(-1).to(self._device)

        if torch.any(values <= 0):
            raise ValueError("Harmonic mean is only defined for positive values.")
        
        self._sum_reciprocal += torch.sum(1.0 / values)
        self._num_examples += values.numel()

    @sync_all_reduce("_sum_reciprocal", "_num_examples")
    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError("HarmonicMean must have at least one example.")
        
        return (self._num_examples / self._sum_reciprocal).item()