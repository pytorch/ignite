"""Patch to fix MNIST download issue as described here:
- https://github.com/pytorch/ignite/issues/1737
- https://github.com/pytorch/vision/issues/3500
"""

import os
import subprocess as sp

import torch
from torchvision.datasets.mnist import MNIST, read_image_file, read_label_file
from torchvision.datasets.utils import extract_archive


def patched_download(self):
    """wget patched download method.
    """
    if self._check_exists():
        return

    os.makedirs(self.raw_folder, exist_ok=True)
    os.makedirs(self.processed_folder, exist_ok=True)

    # download files
    for url, md5 in self.resources:
        filename = url.rpartition("/")[2]
        download_root = os.path.expanduser(self.raw_folder)
        extract_root = None
        remove_finished = False

        if extract_root is None:
            extract_root = download_root
        if not filename:
            filename = os.path.basename(url)

        # Use wget to download archives
        # sp.run(["wget", url, "-P", download_root])

        archive = os.path.join(download_root, filename)
        print("Extracting {} to {}".format(archive, extract_root))
        extract_archive(archive, extract_root, remove_finished)

    # process and save as torch files
    print("Processing...")

    training_set = (
        read_image_file(os.path.join(self.raw_folder, "train-images-idx3-ubyte")),
        read_label_file(os.path.join(self.raw_folder, "train-labels-idx1-ubyte")),
    )
    test_set = (
        read_image_file(os.path.join(self.raw_folder, "t10k-images-idx3-ubyte")),
        read_label_file(os.path.join(self.raw_folder, "t10k-labels-idx1-ubyte")),
    )
    with open(os.path.join(self.processed_folder, self.training_file), "wb") as f:
        torch.save(training_set, f)
    with open(os.path.join(self.processed_folder, self.test_file), "wb") as f:
        torch.save(test_set, f)

    print("Done!")


def main():
    # Patch download method
    MNIST.download = patched_download
    # Download MNIST
    MNIST(".", download=True)


if __name__ == "__main__":
    main()
