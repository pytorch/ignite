import warnings
from distutils.version import LooseVersion
from typing import Callable, Optional, Sequence, Union

import torch

from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = [
    "FID",
]


def fid_score(
    mu1: torch.Tensor, mu2: torch.Tensor, sigma1: torch.Tensor, sigma2: torch.Tensor, eps: float = 1e-6
) -> float:

    try:
        import numpy as np
    except ImportError:
        raise RuntimeError("fid_score requires numpy to be installed.")

    try:
        import scipy
    except ImportError:
        raise RuntimeError("fid_score requires scipy to be installed.")

    mu1, mu2 = mu1.cpu(), mu2.cpu()
    sigma1, sigma2 = sigma1.cpu(), sigma2.cpu()

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.mm(sigma2), disp=False)
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    if not np.isfinite(covmean).all():
        tr_covmean = np.sum(np.sqrt(((np.diag(sigma1) * eps) * (np.diag(sigma2) * eps)) / (eps * eps)))

    return float(diff.dot(diff).item() + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean)


class InceptionExtractor:
    def __init__(self) -> None:
        try:
            from torchvision import models
        except ImportError:
            raise RuntimeError("This module requires torchvision to be installed.")
        self.model = models.inception_v3(pretrained=True)
        self.model.fc = torch.nn.Identity()
        self.model.eval()

    @torch.no_grad()
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if data.dim() != 4:
            raise ValueError(f"Inputs should be a tensor of dim 4, got {data.dim()}")
        if data.shape[1] != 3:
            raise ValueError(f"Inputs should be a tensor with 3 channels, got {data.shape}")
        return self.model(data)


class FID(Metric):
    r"""Calculates Frechet Inception Distance.

    .. math::
       \text{FID} = |\mu_{1} - \mu_{2}| + \text{Tr}(\sigma_{1} + \sigma_{2} - {2}\sqrt{\sigma_1*\sigma_2})

    where :math:`\mu_1` and :math:`\sigma_1` refer to the mean and covariance of the train data and
    :math:`\mu_2` and :math:`\sigma_2` refer to the mean and covariance of the test data.

    More details can be found in `Heusel et al. 2002`__

    __ https://arxiv.org/pdf/1706.08500.pdf

    In addition, a faster and online computation approach can be found in `Chen et al. 2014`__

    __ https://arxiv.org/pdf/2009.14075.pdf

    Remark:

        This implementation is inspired by pytorch_fid package which can be found `here`__

        __ https://github.com/mseitzer/pytorch-fid

    Args:
        num_features: number of features, must be defined if the parameter ``feature_extractor`` is also defined.
            Otherwise, default value is 2048.
        feature_extractor: a callable for extracting the features from the input data. If neither num_features nor
            feature_extractor are defined, default value is ``InceptionExtractor``.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            By default, metrics require the output as ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.

    Example:

        .. code-block:: python

            import torch
            from ignite.metric.gan import FID

            y_pred, y = torch.rand(10, 2048), torch.rand(10, 2048)
            m = FID()
            m.update((y_pred, y))
            print(m.compute())

    .. versionadded:: 0.5.0
    """

    def __init__(
        self,
        num_features: Optional[int] = None,
        feature_extractor: Optional[Callable] = None,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:

        try:
            import numpy as np  # noqa: F401
        except ImportError:
            raise RuntimeError("This module requires numpy to be installed.")

        try:
            import scipy  # noqa: F401
        except ImportError:
            raise RuntimeError("This module requires scipy to be installed.")

        # default is inception
        if num_features is None and feature_extractor is None:
            num_features = 2048
            feature_extractor = InceptionExtractor()
        elif num_features is None:
            raise ValueError("Argument num_features should be defined")
        elif feature_extractor is None:
            self._feature_extractor = lambda x: x
            feature_extractor = self._feature_extractor

        if num_features <= 0:
            raise ValueError(f"Argument num_features must be greater to zero, got: {num_features}")
        self._num_features = num_features
        self._feature_extractor = feature_extractor
        self._eps = 1e-6
        super(FID, self).__init__(output_transform=output_transform, device=device)

    @staticmethod
    def _online_update(features: torch.Tensor, total: torch.Tensor, sigma: torch.Tensor) -> None:
        total += features
        if LooseVersion(torch.__version__) <= LooseVersion("1.7.0"):
            sigma += torch.ger(features, features)
        else:
            sigma += torch.outer(features, features)

    def _get_covariance(self, sigma: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
        r"""
        Calculates covariance from mean and sum of products of variables
        """
        if LooseVersion(torch.__version__) <= LooseVersion("1.7.0"):
            sub_matrix = torch.ger(total, total)
        else:
            sub_matrix = torch.outer(total, total)
        sub_matrix = sub_matrix / self._num_examples
        return (sigma - sub_matrix) / (self._num_examples - 1)

    @staticmethod
    def _check_feature_input(train: torch.Tensor, test: torch.Tensor) -> None:
        for feature in [train, test]:
            if feature.dim() != 2:
                raise ValueError(f"Features must be a tensor of dim 2, got: {feature.dim()}")
            if feature.shape[0] == 0:
                raise ValueError(f"Batch size should be greater than one, got: {feature.shape[0]}")
            if feature.shape[1] == 0:
                raise ValueError(f"Feature size should be greater than one, got: {feature.shape[1]}")
        if train.shape[0] != test.shape[0] or train.shape[1] != test.shape[1]:
            raise ValueError(
                f"Number of Training Features and Testing Features should be equal ({train.shape} != {test.shape})"
            )

    @reinit__is_reduced
    def reset(self) -> None:
        self._train_sigma = torch.zeros((self._num_features, self._num_features), dtype=torch.float64).to(self._device)
        self._train_total = torch.zeros(self._num_features, dtype=torch.float64).to(self._device)
        self._test_sigma = torch.zeros((self._num_features, self._num_features), dtype=torch.float64).to(self._device)
        self._test_total = torch.zeros(self._num_features, dtype=torch.float64).to(self._device)
        self._num_examples = 0
        super(FID, self).reset()

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:

        # Extract the features from the outputs
        train_features = self._feature_extractor(output[0].detach()).to(self._device)
        test_features = self._feature_extractor(output[1].detach()).to(self._device)

        # Check the feature shapess
        self._check_feature_input(train_features, test_features)

        # Updates the mean and covariance for the train features
        for i, features in enumerate(train_features, start=self._num_examples + 1):
            self._online_update(features, self._train_total, self._train_sigma)

        # Updates the mean and covariance for the test features
        for i, features in enumerate(test_features, start=self._num_examples + 1):
            self._online_update(features, self._test_total, self._test_sigma)

        self._num_examples += train_features.shape[0]

    @sync_all_reduce("_num_examples", "_train_total", "_test_total", "_train_sigma", "_test_sigma")
    def compute(self) -> float:
        fid = fid_score(
            mu1=self._train_total / self._num_examples,
            mu2=self._test_total / self._num_examples,
            sigma1=self._get_covariance(self._train_sigma, self._train_total),
            sigma2=self._get_covariance(self._test_sigma, self._test_total),
            eps=self._eps,
        )
        if torch.isnan(torch.tensor(fid)) or torch.isinf(torch.tensor(fid)):
            warnings.warn("The product of covariance of train and test features is out of bounds.")
        return fid
