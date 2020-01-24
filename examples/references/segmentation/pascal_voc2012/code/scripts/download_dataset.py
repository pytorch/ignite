import os
import argparse

from torchvision.datasets.voc import VOCSegmentation
from torchvision.datasets.sbd import SBDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Download Pascal VOC 2012 and SBD segmentation datasets")
    parser.add_argument("output_path", type=str, help="Path where to download and unzip the dataset")

    args = parser.parse_args()

    print("Download Pascal VOC 2012 - Training")
    VOCSegmentation(args.output_path, image_set='train', download=True)
    print("Download Pascal VOC 2012 - Validation")
    VOCSegmentation(args.output_path, image_set='val', download=True)
    print("Download SBD - Training without Pascal VOC validation part")
    sbd_path = os.path.join(args.output_path, "SBD")
    os.makedirs(sbd_path, exist_ok=True)
    SBDataset(sbd_path, image_set='train_noval', mode='segmentation', download=True)
    print("Done")
    print("Pascal VOC 2012 is at : {}".format(os.path.join(args.output_path, "VOCdevkit")))
    print("SBD is at : {}".format(sbd_path))
