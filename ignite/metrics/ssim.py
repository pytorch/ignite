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
        data_range (int or float): Range of the image. Typically, ``1.0`` or ``255``.
        kernel_size (int or list or tuple of int): Size of the kernel. Default: (11, 11)
        sigma (float or list or tuple of float): Standard deviation of the gaussian kernel.
            Argument is used if ``gaussian=True``. Default: (1.5, 1.5)
        k1 (float): Parameter of SSIM. Default: 0.01
        k2 (float): Parameter of SSIM. Default: 0.03
        gaussian (bool): ``True`` to use gaussian kernel, ``False`` to use uniform kernel
        output_transform (callable, optional): A callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric.

    Example:

    To use with ``Engine`` and ``process_function``, simply attach the metric instance to the engine.
    The output of the engine's ``process_function`` needs to be in the format of
    ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y, ...}``.

    ``y_pred`` and ``y`` can be un-normalized or normalized image tensors. Depending on that, the user might need
    to adjust ``data_range``. ``y_pred`` and ``y`` should have the same shape.

    .. code-block:: python

        def process_function(engine, batch):
            # ...
            return y_pred, y
        engine = Engine(process_function)
        metric = SSIM(data_range=1.0)
        metric.attach(engine, "ssim")
    """

    def __init__(
        self,
        data_range: Union[int, float],
        kernel_size: Union[int, Sequence[int]] = (11, 11),
        sigma: Union[float, Sequence[float]] = (1.5, 1.5),
        k1: float = 0.01,
        k2: float = 0.03,
        gaussian: bool = True,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size, kernel_size]  # type: Sequence[int]
        elif isinstance(kernel_size, Sequence):
            self.kernel_size = kernel_size
        else:
            raise ValueError("Argument kernel_size should be either int or a sequence of int.")

        if isinstance(sigma, float):
            self.sigma = [sigma, sigma]  # type: Sequence[float]
        elif isinstance(sigma, Sequence):
            self.sigma = sigma
        else:
            raise ValueError("Argument sigma should be either float or a sequence of float.")

        if any(x % 2 == 0 or x <= 0 for x in self.kernel_size):
            raise ValueError(f"Expected kernel_size to have odd positive number. Got {kernel_size}.")

        if any(y <= 0 for y in self.sigma):
            raise ValueError(f"Expected sigma to have positive number. Got {sigma}.")

        super(SSIM, self).__init__(output_transform=output_transform, device=device)
        self.gaussian = gaussian
        self.c1 = (k1 * data_range) ** 2
        self.c2 = (k2 * data_range) ** 2
        self.pad_h = (self.kernel_size[0] - 1) // 2
        self.pad_w = (self.kernel_size[1] - 1) // 2
        self._kernel = self._gaussian_or_uniform_kernel(kernel_size=self.kernel_size, sigma=self.sigma)

    @reinit__is_reduced
    def reset(self) -> None:
        # Not a tensor because batch size is not known in advance.
        self._sum_of_batchwise_ssim = 0.0  # type: Union[float, torch.Tensor]
        self._num_examples = 0
        self._kernel = self._gaussian_or_uniform_kernel(kernel_size=self.kernel_size, sigma=self.sigma)

    def _uniform(self, kernel_size: int) -> torch.Tensor:
        max, min = 2.5, -2.5
        ksize_half = (kernel_size - 1) * 0.5
        kernel = torch.linspace(-ksize_half, ksize_half, steps=kernel_size, device=self._device)
        for i, j in enumerate(kernel):
            if min <= j <= max:
                kernel[i] = 1 / (max - min)
            else:
                kernel[i] = 0

        return kernel.unsqueeze(dim=0)  # (1, kernel_size)

    def _gaussian(self, kernel_size: int, sigma: float) -> torch.Tensor:
        ksize_half = (kernel_size - 1) * 0.5
        kernel = torch.linspace(-ksize_half, ksize_half, steps=kernel_size, device=self._device)
        gauss = torch.exp(-0.5 * (kernel / sigma).pow(2))
        return (gauss / gauss.sum()).unsqueeze(dim=0)  # (1, kernel_size)

    def _gaussian_or_uniform_kernel(self, kernel_size: Sequence[int], sigma: Sequence[float]) -> torch.Tensor:
        if self.gaussian:
            kernel_x = self._gaussian(kernel_size[0], sigma[0])
            kernel_y = self._gaussian(kernel_size[1], sigma[1])
        else:
            kernel_x = self._uniform(kernel_size[0])
            kernel_y = self._uniform(kernel_size[1])

        return torch.matmul(kernel_x.t(), kernel_y)  # (kernel_size, 1) * (1, kernel_size)

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

        if len(y_pred.shape) != 4 or len(y.shape) != 4:
            raise ValueError(
                f"Expected y_pred and y to have BxCxHxW shape. Got y_pred: {y_pred.shape} and y: {y.shape}."
            )

        channel = y_pred.size(1)
        if len(self._kernel.shape) < 4:
            self._kernel = self._kernel.expand(channel, 1, -1, -1).to(device=y_pred.device)

        y_pred = F.pad(y_pred, [self.pad_w, self.pad_w, self.pad_h, self.pad_h], mode="reflect")
        y = F.pad(y, [self.pad_w, self.pad_w, self.pad_h, self.pad_h], mode="reflect")

        input_list = torch.cat([y_pred, y, y_pred * y_pred, y * y, y_pred * y])
        outputs = F.conv2d(input_list, self._kernel, groups=channel)

        output_list = [outputs[x * y_pred.size(0) : (x + 1) * y_pred.size(0)] for x in range(len(outputs))]

        mu_pred_sq = output_list[0].pow(2)
        mu_target_sq = output_list[1].pow(2)
        mu_pred_target = output_list[0] * output_list[1]

        sigma_pred_sq = output_list[2] - mu_pred_sq
        sigma_target_sq = output_list[3] - mu_target_sq
        sigma_pred_target = output_list[4] - mu_pred_target

        a1 = 2 * mu_pred_target + self.c1
        a2 = 2 * sigma_pred_target + self.c2
        b1 = mu_pred_sq + mu_target_sq + self.c1
        b2 = sigma_pred_sq + sigma_target_sq + self.c2

        ssim_idx = (a1 * a2) / (b1 * b2)
        self._sum_of_batchwise_ssim += torch.mean(ssim_idx, (1, 2, 3), dtype=torch.float64).to(self._device)
        self._num_examples += y.shape[0]

    @sync_all_reduce("_sum_of_batchwise_ssim", "_num_examples")
    def compute(self) -> torch.Tensor:
        if self._num_examples == 0:
            raise NotComputableError("SSIM must have at least one example before it can be computed.")
        return torch.sum(self._sum_of_batchwise_ssim / self._num_examples)  # type: ignore[arg-type]
