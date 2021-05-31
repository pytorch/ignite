import numpy as np
from scipy.linalg import sqrtm

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce


__all__ = ["FID"]


class Record:
    r"""Contains mean and covariance records for train and test data.

    """

    def __init__(self, device="cpu"):
        self.covariance_matrix = None
        self.mean = None
        self._num_examples = 0
        self.device = torch.device(device)

    def reset(self, num_features):
        self.covariance_matrix = torch.zeros((num_features, num_features), dtype=torch.float64).to(self.device)
        self.mean = torch.zeros(num_features, dtype=torch.float64).to(self.device)
        self._num_examples = 0

    def get_covariance(self):
        return self.covariance_matrix / (self._num_examples - 1)


class FID(Metric):
    r"""Calculates Frechet Inception Distance.

    .. math::
       \text{FID} = \mod{\text{mu1} - \text{mu2}} + \text{Trace(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))}

    where :math:`mu1` and :math:`sigma1` refer to the mean and covariance of the train data and
    :math:`mu2` and :math:`sigma2` refer to the mean and covariance of the test data.

    More details can be found in `Heusel et al. 2002`__.

    __ https://arxiv.org/pdf/1706.08500.pdf

    In addition, a faster and online computation approach can be found in `Chen et al. 2014`__

    __ https://arxiv.org/pdf/2009.14075.pdf

    Remark:

        This implementation is inspired by pytorch_fid package.

    Args:
        model: Model for obtaining features from image input. It has to be a callable object.
        mode: Calculate FID directly from features or images. Valid are ``features`` or ``images`` .
        eps: small value added during matric division if denominator is 0.
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
        from ignite.metric.GAN import FID
        import torch

        m = FID(mode="features")

        train = torch.rand(10,2048)
        test = train

        m.update((train,test))

        print(m.compute())

    .. versionadded:: 0.5.0
    """

    def __init__(self, model=None, mode="features", eps=10 ** -6, output_transform=lambda x: x, device="cpu"):
        self._train_record = Record(device=device)
        self._test_record = Record(device=device)
        self._active_record = None
        self._model = model
        self._check_mode(mode)
        self.mode = mode
        self.device = torch.device(device)
        self.eps = eps
        super(FID, self).__init__(output_transform=output_transform, device=device)

    def _check_mode(self, mode):
        if mode not in ["features", "images"]:
            raise ValueError("Please enter a valid mode.")

    def calculate_fid(self):
        mu1 = self._train_record.mean
        mu2 = self._test_record.mean

        sigma1 = self._train_record.get_covariance()
        sigma2 = self._test_record.get_covariance()

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = sqrtm(sigma1.mm(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ("fid calculation produces singular product; " "adding %s to diagonal of cov estimates") % self.eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * self.eps
            covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def _calculate_statistics(self, feature_list):

        # Return if empty
        if not feature_list.all():
            return

        feature_length = len(feature_list)
        if type(self._active_record.covariance_matrix) != torch.Tensor:
            self._active_record.reset(feature_length)
        self._active_record._num_examples += 1

        # Calculating difference between new sample and old and new means
        mean_difference = feature_list - self._active_record.mean
        self._active_record.mean += mean_difference / self._active_record._num_examples
        new_mean_difference = feature_list - self._active_record.mean

        # Outer product to obtain pairwise covariance between each features
        self._active_record.covariance_matrix += torch.outer(mean_difference, new_mean_difference)

    def _get_features(self, act):
        return self._model(act).detach() if act else []

    def _update_from_images(self, train_images, test_images):

        # Obtain features from images
        train_features = self._get_features(train_images)
        test_features = self._get_features(test_images)

        # Updating mean and covariance matrix using train and test features
        self._update_from_features(train_features, test_features)

    def _update_from_features(self, train_data, test_data):
        self._active_record = self._train_record
        # Updates mean and covariance for train data
        for features in train_data:
            self._calculate_statistics(features)

        self._active_record = self._test_record
        # Updates mean and covariance for test data
        for features in test_data:
            self._calculate_statistics(features)

    def check_feature_input(self, train, test):
        if train.shape[0] != 0 and len(train.shape) != 2:
            raise ValueError("Training Features must be passed as (num_samples,feature_size).")
        if test.shape[0] != 0 and len(test.shape) != 2:
            raise ValueError("Testing Features must be passed as (num_samples,feature_size).")
        if train.shape[0] != 0 and test.shape[0] != 0 and train.shape[1] != test.shape[1]:
            raise ValueError("Number of Training Features and Testing Features should be equal.")

    def check_image_input(self, train, test):
        if train.shape[0] != 0 and len(train.shape) < 3:
            raise ValueError("Training images must be passed as (num_samples,image).")
        if test.shape[0] != 0 and len(test.shape) < 3:
            raise ValueError("Testing images must be passed as (num_samples,image).")
        if train.shape[0] != 0 and test.shape[0] != 0 and train.shape[1:] != test.shape[1:]:
            raise ValueError("Train and Test images must be of equal dimensions.")

    @reinit__is_reduced
    def reset(self):
        del self._train_record
        del self._test_record
        self._train_record = Record()
        self._test_record = Record()
        super(FID, self).reset()

    @reinit__is_reduced
    def update(self, output):
        train_data, test_data = output[0], output[1]
        if self.mode == "features":
            self.check_feature_input(torch.tensor(train_data), torch.tensor(test_data))
            self._update_from_features(
                torch.tensor(train_data, dtype=torch.float64).to(self.device),
                torch.tensor(test_data, dtype=torch.float64).to(self.device),
            )
        if self.mode == "images":
            self.check_image_input(torch.tensor(train_data), torch.tensor(test_data))
            self._update_from_images(torch.tensor(train_data).to(self.device), torch.tensor(test_data).to(self.device))

    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self):
        if self._active_record._num_examples < 2:
            raise NotComputableError("FID must have at least two example before it can be computed.")
        if self._train_record._num_examples != self._test_record._num_examples:
            raise NotComputableError("Number of Train and Test samples provided so far do not match.")
        return self.calculate_fid()
