import os
import argparse

from torchvision.datasets import ImageNet


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Download ImageNet-1k dataset")
    parser.add_argument("output_path", type=str, help="Path where to download and unzip the dataset")

    args = parser.parse_args()

    print("Download ImageNet - Training")
    ImageNet(args.output_path, split='train', download=True)
    print("Download ImageNet - Validation")
    ImageNet(args.output_path, split='val', download=True)
    print("Done")
    print("ImageNet is at : {}".format(args.output_path))
