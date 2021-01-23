from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["PSNR"]


class PSNR(Metric):
    """Computes `Peak signal-to-noise ratio (PSNR) <https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio>`_.

    Args:
        data_range (float): Range of the image. If not provided, it will be determined
            from the image type. Typically, ``1.0``.
        output_transform (callable, optional): A callable that is used to transform the Engine’s
            process_function’s output into the form expected by the metric.
        device (str or torch.device): specifies which device updates are accumulated on.
            Setting the metric’s device to be the same as your update arguments ensures
            the update method is non-blocking. By default, CPU.

    Example:

    To use with ``Engine`` and ``process_function``, simply attach the metric instance to the engine.
    The output of the engine's ``process_function`` needs to be in format of
    ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y, ...}``.

    Note:
        ``y_pred`` and ``y`` have to be normalized image tensors, should have the same shape / dtype,
        should have BxCxHxW shape.

    .. code-block:: python

        def process_function(engine, batch):
            # ...
            return y_pred, y
        engine = Engine(process_function)
        psnr = PSNR()
        psnr.attach(engine, "psnr")
    """

    def __init__(
        self,
        data_range: Optional[float] = None,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super().__init__(output_transform=output_transform, device=device)
        self.data_range = data_range
        self.dtype_range = {
            torch.float16: (-1, 1),
            torch.float32: (-1, 1),
            torch.float64: (-1, 1),
        }

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_of_batchwise_psnr = 0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()

        if y_pred.dtype != y.dtype:
            raise TypeError(
                f"Expected y_pred and y to have the same data type. Got y_pred: {y_pred.dtype} and y: {y.dtype}."
            )

        if y_pred.shape != y.shape:
            raise ValueError(
                f"Expected y_pred and y to have the same shape. Got y_pred: {y_pred.shape} and y: {y.shape}."
            )

        if y_pred.ndim != 4 or y.ndim != 4:
            raise ValueError(
                f"Expected y_pred and y to have BxCxHxW shape. Got y_pred: {y_pred.shape} and y: {y.shape}."
            )

        if self.data_range is None:
            dmin, dmax = self.dtype_range[y.dtype]
            true_min, true_max = y.min(), y.max()
            if true_max > dmax or true_min < dmin:
                raise ValueError(
                    "y has intensity values outside the range expected "
                    "for its data type. Please manually specify the `data_range`."
                )
            if true_min >= 0:
                # most common case (255 for uint8, 1 for float)
                self.data_range = dmax
            else:
                self.data_range = dmax - dmin

        mse_error = F.mse_loss(y_pred, y)
        self._sum_of_batchwise_psnr += 10.0 * torch.log10((self.data_range ** 2) / mse_error)
        self._num_examples += y.shape[0]

    @sync_all_reduce("_sum_of_batchwise_psnr", "_num_examples")
    def compute(self) -> torch.Tensor:
        if self._num_examples == 0:
            raise NotComputableError("PSNR must have at least one example before it can be computed.")
        return torch.sum(self._sum_of_batchwise_psnr / self._num_examples)
