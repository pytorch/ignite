import warnings
from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["SSIM"]


class SSIM(Metric):
    """
    Computes Structural Similarity Index Measure

    - ``update`` must receive output of the form ``(y_pred, y)``. They have to be of the same type.
        Valid :class:`torch.dtype` are the following:
        - on CPU: `torch.float32`, `torch.float64`.
        - on CUDA: `torch.float16`, `torch.bfloat16`, `torch.float32`, `torch.float64`.

    Args:
        data_range: Range of the image. Typically, ``1.0`` or ``255``.
        kernel_size: Size of the kernel. Default: (11, 11)
        sigma: Standard deviation of the gaussian kernel.
            Argument is used if ``gaussian=True``. Default: (1.5, 1.5)
        k1: Parameter of SSIM. Default: 0.01
        k2: Parameter of SSIM. Default: 0.03
        gaussian: ``True`` to use gaussian kernel, ``False`` to use uniform kernel
        output_transform: A callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric.
        device: specifies which device updates are accumulated on. Setting the metric's
            device to be the same as your ``update`` arguments ensures the ``update`` method is non-blocking. By
            default, CPU.
        skip_unrolling: specifies whether output should be unrolled before being fed to update method. Should be
            true for multi-output model, for example, if ``y_pred`` contains multi-ouput as ``(y_pred_a, y_pred_b)``
            Alternatively, ``output_transform`` can be used to handle this.

    Examples:
        To use with ``Engine`` and ``process_function``, simply attach the metric instance to the engine.
        The output of the engine's ``process_function`` needs to be in the format of
        ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y, ...}``. If not, ``output_tranform`` can be added
        to the metric to transform the output into the form expected by the metric.

        ``y_pred`` and ``y`` can be un-normalized or normalized image tensors. Depending on that, the user might need
        to adjust ``data_range``. ``y_pred`` and ``y`` should have the same shape.

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            metric = SSIM(data_range=1.0)
            metric.attach(default_evaluator, 'ssim')
            preds = torch.rand([4, 3, 16, 16])
            target = preds * 0.75
            state = default_evaluator.run([[preds, target]])
            print(state.metrics['ssim'])

        .. testoutput::

            0.9218971...

    .. versionadded:: 0.4.2

    .. versionchanged:: 0.5.1
        ``skip_unrolling`` argument is added.
    """

    _state_dict_all_req_keys = ("_sum_of_ssim", "_num_examples", "_kernel")

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
        skip_unrolling: bool = False,
    ):
        if isinstance(kernel_size, int):
            self.kernel_size: Sequence[int] = [kernel_size, kernel_size]
        elif isinstance(kernel_size, Sequence):
            self.kernel_size = kernel_size
        else:
            raise ValueError("Argument kernel_size should be either int or a sequence of int.")

        if isinstance(sigma, float):
            self.sigma: Sequence[float] = [sigma, sigma]
        elif isinstance(sigma, Sequence):
            self.sigma = sigma
        else:
            raise ValueError("Argument sigma should be either float or a sequence of float.")

        if any(x % 2 == 0 or x <= 0 for x in self.kernel_size):
            raise ValueError(f"Expected kernel_size to have odd positive number. Got {kernel_size}.")

        if any(y <= 0 for y in self.sigma):
            raise ValueError(f"Expected sigma to have positive number. Got {sigma}.")

        super(SSIM, self).__init__(output_transform=output_transform, device=device, skip_unrolling=skip_unrolling)
        self.gaussian = gaussian
        self.data_range = data_range
        self.c1 = (k1 * data_range) ** 2
        self.c2 = (k2 * data_range) ** 2
        self.pad_h = (self.kernel_size[0] - 1) // 2
        self.pad_w = (self.kernel_size[1] - 1) // 2
        self._kernel_2d = self._gaussian_or_uniform_kernel(kernel_size=self.kernel_size, sigma=self.sigma)
        self._kernel: Optional[torch.Tensor] = None

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_of_ssim = torch.tensor(0.0, dtype=self._double_dtype, device=self._device)
        self._num_examples = 0

    def _uniform(self, kernel_size: int) -> torch.Tensor:
        kernel = torch.zeros(kernel_size, device=self._device)

        start_uniform_index = max(kernel_size // 2 - 2, 0)
        end_uniform_index = min(kernel_size // 2 + 3, kernel_size)

        min_, max_ = -2.5, 2.5
        kernel[start_uniform_index:end_uniform_index] = 1 / (max_ - min_)

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

    def _check_type_and_shape(self, y_pred: torch.Tensor, y: torch.Tensor) -> None:
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

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()

        self._check_type_and_shape(y_pred, y)

        # converts potential integer tensor to fp
        if not y.is_floating_point():
            y = y.float()
        if not y_pred.is_floating_point():
            y_pred = y_pred.float()

        nb_channel = y_pred.size(1)
        if self._kernel is None or self._kernel.shape[0] != nb_channel:
            self._kernel = self._kernel_2d.expand(nb_channel, 1, -1, -1)

        if y_pred.device != self._kernel.device:
            if self._kernel.device == torch.device("cpu"):
                self._kernel = self._kernel.to(device=y_pred.device)

            elif y_pred.device == torch.device("cpu"):
                warnings.warn(
                    "y_pred tensor is on cpu device but previous computation was on another device: "
                    f"{self._kernel.device}. To avoid having a performance hit, please ensure that all "
                    "y and y_pred tensors are on the same device.",
                )
                y_pred = y_pred.to(device=self._kernel.device)
                y = y.to(device=self._kernel.device)

        y_pred = F.pad(y_pred, [self.pad_w, self.pad_w, self.pad_h, self.pad_h], mode="reflect")
        y = F.pad(y, [self.pad_w, self.pad_w, self.pad_h, self.pad_h], mode="reflect")

        if y_pred.dtype != self._kernel.dtype:
            self._kernel = self._kernel.to(dtype=y_pred.dtype)

        input_list = [y_pred, y, y_pred * y_pred, y * y, y_pred * y]
        outputs = F.conv2d(torch.cat(input_list), self._kernel, groups=nb_channel)
        batch_size = y_pred.size(0)
        output_list = [outputs[x * batch_size : (x + 1) * batch_size] for x in range(len(input_list))]

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

        # In case when ssim_idx can be MPS tensor and self._device is not MPS
        # self._double_dtype is float64.
        # As MPS does not support float64 we should set dtype to float32
        double_dtype = self._double_dtype
        if ssim_idx.device.type == "mps" and self._double_dtype == torch.float64:
            double_dtype = torch.float32

        self._sum_of_ssim += torch.mean(ssim_idx, (1, 2, 3), dtype=double_dtype).sum().to(device=self._device)

        self._num_examples += y.shape[0]

    @sync_all_reduce("_sum_of_ssim", "_num_examples")
    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError("SSIM must have at least one example before it can be computed.")
        return (self._sum_of_ssim / self._num_examples).item()
