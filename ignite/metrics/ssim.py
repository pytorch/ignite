from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["SSIM"]


class SSIM(Metric):
    """
    Computes Structual Similarity Index Measure

    Args:
        kernel_size (list or tuple of int): Size of the gaussian kernel. Default: (11, 11)
        sigma: Standard deviation of the gaussian kernel. Default: (1.5, 1.5)
        data_range: Range of the image. If ``None``, it is determined from the image (max - min)
        k1: Parameter of SSIM. Default: 0.01
        k2: Parameter of SSIM. Default: 0.03
        output_transform: A callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric.

    Returns:
        A Tensor with SSIM

    Example:

    To use with ``Engine`` and ``process_function``, simply attach the metric instance to the engine.
    The output of the engine's ``process_function`` needs to be in the format of
    ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y, ...}``.

    .. code-block:: python

        def process_function(engine, batch):
            # ...
            return y_pred, y
        engine = Engine(process_function)
        metric = SSIM()
        metric.attach(engine, "ssim")

    If the output of the engine is not in the format above, ``output_transform`` argument can be used to transform it.

    .. code-block:: python

        def process_function(engine, batch):
            # ...
            return {'prediction': y_pred, 'target': y, ...}

        engine = Engine(process_function)

        def output_transform(output):
            # `output` variable is returned by above `process_function`
            y_pred = output['prediction']
            y = output['target']
            return y_pred, y  # output format is according to `Accuracy` docs

        metric = SSIM(output_transform=output_transform)
        metric.attach(engine, "ssim")

    The user even can use the metric with ``update`` and ``compute`` methods.

    .. code-block:: python

        >>> y_pred = torch.rand([16, 1, 16, 16])
        >>> y = y_pred * 1.25
        >>> ssim = SSIM()
        >>> ssim.update((y_pred, y))
        >>> ssim.compute()
        tensor(0.9520)
    """

    def __init__(
        self,
        kernel_size: Sequence[int] = (11, 11),
        sigma: Sequence[float] = (1.5, 1.5),
        data_range: float = None,
        k1: float = 0.01,
        k2: float = 0.03,
        output_transform: Callable = lambda x: x,
    ):
        if len(kernel_size) != 2 or len(sigma) != 2:
            raise ValueError(
                "Expected `kernel_size` and `sigma` to have the length of two."
                f" Got kernel_size: {len(kernel_size)} and sigma: {len(sigma)}."
            )

        if any(x % 2 == 0 or x <= 0 for x in kernel_size):
            raise ValueError(f"Expected `kernel_size` to have odd positive number. Got {kernel_size}.")

        if any(y <= 0 for y in sigma):
            raise ValueError(f"Expected `sigma` to have positive number. Got {sigma}.")

        self.kernel_size = kernel_size
        self.sigma = sigma
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2
        super(SSIM, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_of_batchwise_ssim = 0.0
        self._num_examples = 0

    def _gaussian_kernel(self, channel, kernel_size, sigma, device):
        def gaussian(kernel_size, sigma, device):
            gauss = torch.arange(
                start=(1 - kernel_size) / 2, end=(1 + kernel_size) / 2, step=1, dtype=torch.float32, device=device
            )
            gauss = torch.exp(-gauss.pow(2) / (2 * pow(sigma, 2)))
            return (gauss / gauss.sum()).unsqueeze(dim=0)  # (1, kernel_size)

        gaussian_kernel_x = gaussian(kernel_size[0], sigma[0], device)
        gaussian_kernel_y = gaussian(kernel_size[1], sigma[1], device)
        kernel = torch.matmul(gaussian_kernel_x.t(), gaussian_kernel_y)  # (kernel_size, 1) * (1, kernel_size)

        return kernel.expand(channel, 1, kernel_size[0], kernel_size[1])

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output
        if y_pred.dtype != y.dtype:
            raise TypeError(
                f"Expected `y_pred` and `y` to have the same data type. Got y_pred: {y_pred.dtype} and y: {y.dtype}."
            )

        if y_pred.shape != y.shape:
            raise ValueError(
                f"Expected `y_pred` and `y` to have the same shape. Got y_pred: {y_pred.shape} and y: {y.shape}."
            )

        if len(y_pred.shape) != 4 or len(y.shape) != 4:
            raise ValueError(
                f"Expected `y_pred` and `y` to have BxCxHxW shape. Got y_pred: {y_pred.shape} and y: {y.shape}."
            )

        if self.data_range is None:
            self.data_range = max(y_pred.max() - y_pred.min(), y.max() - y.min())

        C1 = pow(self.k1 * self.data_range, 2)
        C2 = pow(self.k2 * self.data_range, 2)
        device = y_pred.device

        channel = y_pred.size(1)
        kernel = self._gaussian_kernel(channel, self.kernel_size, self.sigma, device)
        mu_pred = F.conv2d(y_pred, kernel, groups=channel)
        mu_target = F.conv2d(y, kernel, groups=channel)

        mu_pred_sq = mu_pred.pow(2)
        mu_target_sq = mu_target.pow(2)
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(y_pred * y_pred, kernel, groups=channel) - mu_pred_sq
        sigma_target_sq = F.conv2d(y * y, kernel, groups=channel) - mu_target_sq
        sigma_pred_target = F.conv2d(y_pred * y, kernel, groups=channel) - mu_pred_target

        UPPER = 2 * sigma_pred_target + C2
        LOWER = sigma_pred_sq + sigma_target_sq + C2

        ssim_idx = ((2 * mu_pred_target + C1) * UPPER) / ((mu_pred_sq + mu_target_sq + C1) * LOWER)
        self._sum_of_batchwise_ssim += torch.mean(ssim_idx, (1, 2, 3))
        self._num_examples += y.shape[0]

    @sync_all_reduce("_sum_of_batchwise_ssim", "_num_examples")
    def compute(self) -> torch.Tensor:
        if self._num_examples == 0:
            raise NotComputableError("SSIM must have at least one example before it can be computed.")
        return torch.sum(self._sum_of_batchwise_ssim / self._num_examples)
