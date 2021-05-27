import numpy as np
import torch
from scipy.linalg import sqrtm

from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce


class Record:
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
    def __init__(self, model=None, eps=10 ** -6, output_transform=lambda x: x, device="cpu"):
        self._train_record = Record(device=device)
        self._test_record = Record(device=device)
        self._active_record = None
        self._model = model
        self.device = torch.device(device)
        self.eps = eps
        super(FID, self).__init__(output_transform=output_transform, device=device)

    def calculate_fid(self):
        """
        Compute the Frechet Inception Distance based on given pair of mean and covariances.

        Returns:
            Float representing the Frechet Inception Distance.

        .. versionadded:: 0.5.0
        """
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
        feature_length = len(feature_list)
        if type(self._active_record.covariance_matrix) != torch.Tensor:
            self._active_record.reset(feature_length)
        self._active_record._num_examples += 1
        mean_difference = feature_list - self._active_record.mean
        self._active_record.mean += mean_difference / self._active_record._num_examples
        new_mean_difference = feature_list - self._active_record.mean
        self._active_record.covariance_matrix += torch.outer(mean_difference, new_mean_difference)

    def _get_features(self, act):
        return self._model(act).detach()

    def _update_from_data(self, data):
        features = self._get_features(data)
        for feature in features:
            self._calculate_statistics(feature)

    def _update_from_images(self, train_images, test_images):
        self._active_record = self._train_record
        self._update_from_data(train_images)
        self._active_record = self._test_record
        self._update_from_data(test_images)

    def _update_from_features(self, train_data, test_data):
        self._active_record = self._train_record
        for features in train_data:
            self._calculate_statistics(features)
        self._active_record = self._test_record
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
    def update(self, output, mode):
        train_data, test_data = output[0], output[1]
        if mode == "features":
            self.check_feature_input(torch.tensor(train_data), torch.tensor(test_data))
            self._update_from_features(
                torch.tensor(train_data, dtype=torch.float64).to(self.device),
                torch.tensor(test_data, dtype=torch.float64).to(self.device),
            )
        elif mode == "images":
            self.check_image_input(torch.tensor(train_data), torch.tensor(test_data))
            self._update_from_images(torch.tensor(train_data).to(self.device), torch.tensor(test_data).to(self.device))
        else:
            raise ValueError("Please enter a valid mode.")

    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self):
        if self._active_record._num_examples < 2:
            raise NotComputableError("FID must have at least two example before it can be computed.")
        if self._train_record._num_examples != self._test_record._num_examples:
            raise NotComputableError("Number of Train and Test samples provided so far do not match.")
        return self.calculate_fid()
