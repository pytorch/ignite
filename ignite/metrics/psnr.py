from typing import Callable, Optional, Sequence, Union

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["PSNR"]


class PSNR(Metric):
    r"""Computes average `Peak signal-to-noise ratio (PSNR) <https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio>`_.

    .. math::
        \text{PSNR}(I, J) = 10 * \log_{10}\left(\frac{ MAX_{I}^2 }{ \text{ MSE } }\right)

    where :math:`\text{MSE}` is `mean squared error <https://en.wikipedia.org/wiki/Mean_squared_error>`_.

    - `y_pred` and `y` **must** have (batch_size, ...) shape.
    - `y_pred` and `y` **must** have same dtype and same shape.

    Args:
        data_range (int or float): The data range of the target image (distance between minimum
            and maximum possible values).
            For other data types, please set the data range, otherwise an exception will be raised.
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
        psnr = PSNR(data_range=1.0)
        psnr.attach(engine, "psnr")
        # ...
        state = engine.run(data)
        print(f"PSNR: {state.metrics['psnr']}")

    This metric by default accepts Grayscale or RGB images. But if you have YCbCr or YUV images, only
    Y channel is needed for computing PSNR. And, this can be done with ``output_transform``. For instance,

    .. code-block:: python

        def get_y_channel(output):
            y_pred, y = output
            # y_pred and y are (B, 3, H, W) and YCbCr or YUV images
            # let's select y channel
            return y_pred[:, 0, ...], y[:, 0, ...]

        psnr = PSNR(data_range=219, output_transform=get_y_channel)
        psnr.attach(engine, "psnr")
        # ...
        state = engine.run(data)
        print(f"PSNR: {state.metrics['psrn']}")

    .. versionadded:: 0.4.3
    """

    def __init__(
        self,
        data_range: Union[int, float],
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super().__init__(output_transform=output_transform, device=device)
        self.data_range = data_range

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
        self._sum_of_batchwise_psnr = torch.tensor(0.0, dtype=torch.float64, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        self._check_shape_dtype(output)
        y_pred, y = output[0].detach(), output[1].detach()

        dim = tuple(range(1, y.ndim))
        mse_error = torch.pow(y_pred.double() - y.view_as(y_pred).double(), 2).mean(dim=dim)
        self._sum_of_batchwise_psnr += torch.sum(10.0 * torch.log10(self.data_range ** 2 / (mse_error + 1e-10))).to(
            device=self._device
        )
        self._num_examples += y.shape[0]

    @sync_all_reduce("_sum_of_batchwise_psnr", "_num_examples")
    def compute(self) -> torch.Tensor:
        if self._num_examples == 0:
            raise NotComputableError("PSNR must have at least one example before it can be computed.")
        return self._sum_of_batchwise_psnr / self._num_examples
