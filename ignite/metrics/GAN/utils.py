import os

import numpy as np


class Record:
    def __init__(self):
        self.covariance_matrix = None
        self.mean = None

    def reset(self, num_features):
        self.covariance_matrix = np.zeros((num_features, num_features))
        self.mean = np.zeros(num_features)

    def get_covariance(self, num_samples):
        return self.covariance_matrix / (num_samples - 1)


def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)) and file.lower().endswith((".png", ".jpg", ".jpeg")):
            yield os.path.join(path, file)
