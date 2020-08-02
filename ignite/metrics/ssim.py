from typing import Callable, Sequence, Union

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
        sigma (list or tuple of float): Standard deviation of the gaussian kernel. Default: (1.5, 1.5)
        data_range (int or float): Range of the image. Typically, ``1.0`` or ``255``.
        k1 (float): Parameter of SSIM. Default: 0.01
        k2 (float): Parameter of SSIM. Default: 0.03
        output_transform: A callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric.

    Example:

    To use with ``Engine`` and ``process_function``, simply attach the metric instance to the engine.
    The output of the engine's ``process_function`` needs to be in the format of
    ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y, ...}``.

    ``y_pred`` and ``y`` can be an un-normalized image tensor or a normalized image tensor.
    Depending on that, the user might need to adjust ``data_range``.

    .. code-block:: python

        def process_function(engine, batch):
            # ...
            return y_pred, y
        engine = Engine(process_function)
        metric = SSIM()
        metric.attach(engine, "ssim")
    """

    def __init__(
        self,
        data_range: Union[int, float],
        kernel_size: Union[int, Sequence[int]] = (11, 11),
        sigma: Union[float, Sequence[float]] = (1.5, 1.5),
        k1: float = 0.01,
        k2: float = 0.03,
        output_transform: Callable = lambda x: x,
    ):
        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size, kernel_size]
        elif isinstance(kernel_size, Sequence):
            self.kernel_size = kernel_size
        else:
            raise ValueError("Argument kernel_size should be either int or a sequence of int.")

        if isinstance(sigma, float):
            self.sigma = [sigma, sigma]
        elif isinstance(sigma, Sequence):
            self.sigma = sigma
        else:
            raise ValueError("Argument sigma should be either float or a sequence of float.")

        if any(x % 2 == 0 or x <= 0 for x in self.kernel_size):
            raise ValueError("Expected kernel_size to have odd positive number. Got {}.".format(kernel_size))

        if any(y <= 0 for y in self.sigma):
            raise ValueError("Expected sigma to have positive number. Got {}.".format(sigma))

        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2
        self.kernel = self._gaussian_kernel(kernel_size=self.kernel_size, sigma=self.sigma)
        super(SSIM, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_of_batchwise_ssim = 0.0
        self._num_examples = 0

    def _gaussian(self, kernel_size, sigma):
        gauss = torch.arange(start=(1 - kernel_size) / 2, end=(1 + kernel_size) / 2, step=1, dtype=torch.float32)
        gauss = torch.exp(-gauss.pow(2) / (2 * pow(sigma, 2)))
        return (gauss / gauss.sum()).unsqueeze(dim=0)  # (1, kernel_size)

    def _gaussian_kernel(self, kernel_size, sigma):
        gaussian_kernel_x = self._gaussian(kernel_size[0], sigma[0])
        gaussian_kernel_y = self._gaussian(kernel_size[1], sigma[1])

        return torch.matmul(gaussian_kernel_x.t(), gaussian_kernel_y)  # (kernel_size, 1) * (1, kernel_size)

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output
        if y_pred.dtype != y.dtype:
            raise TypeError(
                "Expected y_pred and y to have the same data type. Got y_pred: {} and y: {}.".format(
                    y_pred.dtype, y.dtype
                )
            )

        if y_pred.shape != y.shape:
            raise ValueError(
                "Expected y_pred and y to have the same shape. Got y_pred: {} and y: {}.".format(y_pred.shape, y.shape)
            )

        if len(y_pred.shape) != 4 or len(y.shape) != 4:
            raise ValueError(
                "Expected y_pred and y to have BxCxHxW shape. Got y_pred: {} and y: {}.".format(y_pred.shape, y.shape)
            )

        C1 = pow(self.k1 * self.data_range, 2)
        C2 = pow(self.k2 * self.data_range, 2)
        channel = y_pred.size(1)
        self.kernel = self.kernel.expand(channel, 1, -1, -1)
        device = y_pred.device

        if device is not self.kernel.device:
            self.kernel.to(device=device)

        mu_pred = F.conv2d(y_pred, self.kernel, groups=channel)
        mu_target = F.conv2d(y, self.kernel, groups=channel)

        mu_pred_sq = mu_pred.pow(2)
        mu_target_sq = mu_target.pow(2)
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(y_pred * y_pred, self.kernel, groups=channel) - mu_pred_sq
        sigma_target_sq = F.conv2d(y * y, self.kernel, groups=channel) - mu_target_sq
        sigma_pred_target = F.conv2d(y_pred * y, self.kernel, groups=channel) - mu_pred_target

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
