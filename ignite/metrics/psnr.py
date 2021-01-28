from typing import Callable, Optional, Sequence, Union

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["PSNR"]


class PSNR(Metric):
    r"""Computes average `Peak signal-to-noise ratio (PSNR) <https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio>`_.

    .. math::
        \text{PSNR}(i, j) = 10 * \log_{10}\left(\frac{ MAX_{i}^2 }{ \text{ MSE } }\right)

    where :math:`\text{MSE}` is `mean squared error <https://en.wikipedia.org/wiki/Mean_squared_error>`_.

    Note:
        `y_pred` and `y` must have same dtype and same shape.
        If `y_pred` and `y` have almost identical values, the result will be infinity.

    Args:
        data_range (int or float): The data range of the target image (distance between minimum
            and maximum possible values). If not provided, it will be estimated from the input
            data type. Typically, ``1.0`` for float tensor or ``255`` for unsigned 8-bit tensor.
        output_transform (callable, optional): A callable that is used to transform the Engine’s
            process_function’s output into the form expected by the metric.
        device (str or torch.device): specifies which device updates are accumulated on.
            Setting the metric’s device to be the same as your update arguments ensures
            the update method is non-blocking. By default, CPU.

    Example:

    To use with ``Engine`` and ``process_function``, simply attach the metric instance to the engine.
    The output of the engine's ``process_function`` needs to be in format of
    ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y, ...}``.

    .. code-block:: python

        def process_function(engine, batch):
            # ...
            return y_pred, y
        engine = Engine(process_function)
        psnr = PSNR()
        psnr.attach(engine, "psnr")
        # ...
        state = engine.run(data)
        print(f"PSNR: {state.metrics['psnr']}")

    .. versionadded:: 0.5.0
    """

    def __init__(
        self,
        data_range: Optional[Union[int, float]] = None,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super().__init__(output_transform=output_transform, device=device)
        self.data_range = self.__data_range = data_range

    def _check_shape_dtype(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output
        if y_pred.dtype != y.dtype:
            raise TypeError(
                f"Expected y_pred and y to have the same data type. Got y_pred: {y_pred.dtype} and y: {y.dtype}."
            )

        if y_pred.shape != y.shape:
            raise ValueError(
                f"Expected y_pred and y to have the same shape. Got y_pred: {y_pred.shape} and y: {y.shape}."
            )

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_of_squared_error = 0  # type: Union[int, torch.Tensor]
        self._num_elements = 0

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        self._check_shape_dtype(output)
        y_pred, y = output[0].detach(), output[1].detach()

        if self.__data_range is None:
            dmin, dmax = _dtype_range[y.dtype]
            true_min, true_max = y.min(), y.max()
            if true_max > dmax or true_min < dmin:
                raise ValueError(
                    "y has intensity values outside the range expected "
                    "for its data type. Please manually specify the data_range."
                )
            if true_min >= 0:
                # most common case (255 for uint8, 1 for float)
                self.data_range = dmax
            else:
                self.data_range = dmax - dmin

        self._sum_of_squared_error += (
            torch.pow(y_pred.double() - y.view_as(y_pred).double(), 2).sum().to(device=self._device)
        )
        self._num_elements += y.numel()

    @sync_all_reduce("_sum_of_squared_error", "_num_elements")
    def compute(self) -> Union[float, torch.Tensor]:
        if self._num_elements == 0:
            raise NotComputableError("PSNR must have at least one example before it can be computed.")
        return 10.0 * torch.log10(
            self.data_range ** 2 / (self._sum_of_squared_error / self._num_elements)  # type: ignore[operator, arg-type]
        )


_dtype_range = {
    torch.uint8: (0, 255),
    torch.float16: (-1.0, 1.0),
    torch.float32: (-1.0, 1.0),
    torch.float64: (-1.0, 1.0),
}
