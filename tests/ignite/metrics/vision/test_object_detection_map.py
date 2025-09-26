import sys
from collections import namedtuple
from math import ceil
from typing import Dict, List, Tuple
from unittest.mock import patch

import numpy as np
from sklearn.utils.extmath import stable_cumsum

np.float = float

import pytest
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.distributions.geometric import Geometric

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.metrics import CommonObjectDetectionMetrics, ObjectDetectionAvgPrecisionRecall
from ignite.metrics.vision.object_detection_average_precision_recall import coco_tensor_list_to_dict_list
from ignite.utils import manual_seed

torch.set_printoptions(linewidth=200)
manual_seed(12)
np.set_printoptions(linewidth=200)


def coco_val2017_sample() -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
    """
    Predictions are done using torchvision's `fasterrcnn_resnet50_fpn_v2`
    with the following snippet. Note that beforehand, COCO images and annotations
    were downloaded and unzipped into "val2017" and "annotations" folders respectively.

    .. code-block:: python

        import torch
        from torchvision.models import detection as dtv
        from torchvision.datasets import CocoDetection

        coco = CocoDetection(
            "val2017",
            "annotations/instances_val2017.json",
            transform=dtv.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.transforms()
        )
        model = dtv.fasterrcnn_resnet50_fpn_v2(
            weights=dtv.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        )
        model.eval()

        sample = torch.randint(len(coco), (10,)).tolist()
        with torch.no_grad():
            pred = model([coco[s][0] for s in sample])

        pred = [torch.cat((
            p['boxes'].int(),
            p['scores'].reshape(-1, 1),
            p['labels'].reshape(-1,1)
        ), dim=1) for p in pred]
    """
    gt = [
        torch.tensor(
            [
                [418.1300, 261.8500, 511.4100, 311.0500, 23.0000, 0.0000],
                [269.9200, 256.9700, 311.1200, 283.3000, 23.0000, 0.0000],
                [175.1900, 252.0200, 209.2000, 278.7200, 23.0000, 0.0000],
            ]
        ),
        torch.tensor([[196.7700, 199.6900, 301.8300, 373.2700, 2.0000, 0.0000]]),
        torch.tensor(
            [
                [88.4500, 168.2700, 465.9800, 318.2000, 6.0000, 0.0000],
                [2.2900, 224.2100, 31.3000, 282.2200, 1.0000, 0.0000],
                [143.9700, 240.8300, 176.8100, 268.6500, 84.0000, 0.0000],
                [234.6200, 202.0900, 264.6500, 244.5800, 84.0000, 0.0000],
                [303.0400, 193.1300, 339.9900, 242.5600, 84.0000, 0.0000],
                [358.0100, 195.3100, 408.9800, 261.2500, 84.0000, 0.0000],
                [197.2100, 236.3500, 233.7700, 275.1300, 84.0000, 0.0000],
            ]
        ),
        torch.tensor(
            [
                [26.1900, 44.8000, 435.5700, 268.0900, 50.0000, 0.0000],
                [190.5400, 244.9400, 322.0100, 309.6500, 53.0000, 0.0000],
                [72.8900, 27.2800, 591.3700, 417.8600, 51.0000, 0.0000],
                [22.5400, 402.4500, 253.7400, 596.7200, 51.0000, 0.0000],
                [441.9100, 168.5000, 494.3700, 213.0100, 53.0000, 0.0000],
                [340.0700, 330.6800, 464.6200, 373.5800, 53.0000, 0.0000],
                [28.6100, 247.9800, 591.3400, 597.6900, 67.0000, 0.0000],
                [313.9300, 108.4000, 441.3300, 234.2600, 57.0000, 0.0000],
            ]
        ),
        torch.tensor([[0.9600, 74.6700, 498.7600, 261.3400, 5.0000, 0.0000]]),
        torch.tensor(
            [
                [1.3800, 2.7500, 361.9800, 499.6100, 17.0000, 0.0000],
                [325.4600, 239.8200, 454.2700, 416.8700, 47.0000, 0.0000],
                [165.3900, 307.3400, 332.2200, 548.9500, 47.0000, 0.0000],
                [424.0600, 179.6600, 480.0000, 348.0600, 47.0000, 0.0000],
            ]
        ),
        torch.tensor(
            [
                [218.2600, 84.2700, 473.6000, 230.0600, 28.0000, 0.0000],
                [212.6300, 220.0000, 417.2200, 312.8100, 62.0000, 0.0000],
                [104.4600, 311.2200, 222.4600, 375.0000, 62.0000, 0.0000],
            ]
        ),
        torch.tensor(
            [
                [144.4400, 132.2200, 639.7700, 253.8800, 5.0000, 0.0000],
                [0.0000, 48.4300, 492.8900, 188.0200, 5.0000, 0.0000],
            ]
        ),
        torch.tensor(
            [
                [205.9900, 276.6200, 216.8500, 311.2300, 1.0000, 0.0000],
                [378.9100, 65.4500, 476.1900, 185.3100, 38.0000, 0.0000],
            ]
        ),
        torch.tensor(
            [
                [392.0900, 110.3800, 417.5200, 136.7000, 85.0000, 0.0000],
                [594.9000, 223.4600, 640.0000, 473.6600, 1.0000, 0.0000],
                [381.1400, 225.8200, 443.3100, 253.1200, 58.0000, 0.0000],
                [175.2800, 253.8000, 221.0600, 281.7400, 60.0000, 0.0000],
                [96.4100, 357.6800, 168.8800, 392.0300, 60.0000, 0.0000],
                [126.6500, 258.3200, 178.3800, 285.8900, 60.0000, 0.0000],
                [71.0200, 247.1500, 117.2300, 275.0500, 60.0000, 0.0000],
                [34.2300, 265.8500, 82.4000, 296.0700, 60.0000, 0.0000],
                [32.9700, 252.5500, 69.1700, 271.6400, 60.0000, 0.0000],
                [50.7800, 229.3100, 90.7400, 248.5200, 60.0000, 0.0000],
                [82.8900, 218.2800, 126.5900, 237.6700, 60.0000, 0.0000],
                [80.6300, 263.9200, 128.1000, 290.4100, 60.0000, 0.0000],
                [277.2200, 222.9500, 323.7000, 243.6300, 60.0000, 0.0000],
                [225.5200, 155.3300, 272.0700, 167.2700, 60.0000, 0.0000],
                [360.6200, 178.1900, 444.4800, 207.2900, 1.0000, 0.0000],
                [171.6000, 149.8100, 219.9900, 169.7400, 60.0000, 0.0000],
                [398.0100, 223.2000, 461.1000, 252.6500, 58.0000, 0.0000],
                [1.0000, 123.0000, 603.0000, 402.0000, 60.0000, 1.0000],
            ]
        ),
    ]

    pred = [
        torch.tensor(
            [
                [418.0000, 260.0000, 513.0000, 310.0000, 0.9975, 23.0000],
                [175.0000, 252.0000, 208.0000, 280.0000, 0.9631, 20.0000],
                [269.0000, 256.0000, 310.0000, 292.0000, 0.6253, 20.0000],
                [269.0000, 256.0000, 311.0000, 286.0000, 0.3523, 21.0000],
                [269.0000, 256.0000, 311.0000, 288.0000, 0.2267, 16.0000],
                [175.0000, 252.0000, 209.0000, 280.0000, 0.1528, 23.0000],
            ]
        ),
        torch.tensor(
            [
                [197.0000, 207.0000, 299.0000, 344.0000, 0.9573, 2.0000],
                [0.0000, 0.0000, 371.0000, 477.0000, 0.8672, 7.0000],
                [218.0000, 250.0000, 298.0000, 369.0000, 0.6232, 2.0000],
                [303.0000, 70.0000, 374.0000, 471.0000, 0.2851, 82.0000],
                [200.0000, 203.0000, 282.0000, 290.0000, 0.1519, 2.0000],
                [206.0000, 169.0000, 275.0000, 236.0000, 0.0581, 72.0000],
            ]
        ),
        torch.tensor(
            [
                [3.0000, 224.0000, 31.0000, 280.0000, 0.9978, 1.0000],
                [86.0000, 174.0000, 457.0000, 312.0000, 0.9917, 6.0000],
                [90.0000, 176.0000, 460.0000, 313.0000, 0.2907, 8.0000],
                [443.0000, 198.0000, 564.0000, 255.0000, 0.2400, 3.0000],
                [452.0000, 197.0000, 510.0000, 220.0000, 0.1778, 8.0000],
                [462.0000, 197.0000, 513.0000, 220.0000, 0.1750, 3.0000],
                [489.0000, 196.0000, 522.0000, 212.0000, 0.1480, 3.0000],
                [257.0000, 165.0000, 313.0000, 182.0000, 0.1294, 8.0000],
                [438.0000, 195.0000, 556.0000, 256.0000, 0.1198, 8.0000],
                [555.0000, 235.0000, 575.0000, 251.0000, 0.0831, 3.0000],
                [486.0000, 54.0000, 504.0000, 63.0000, 0.0634, 34.0000],
                [565.0000, 245.0000, 573.0000, 251.0000, 0.0605, 3.0000],
                [257.0000, 165.0000, 313.0000, 183.0000, 0.0569, 6.0000],
                [0.0000, 233.0000, 42.0000, 256.0000, 0.0526, 3.0000],
                [26.0000, 237.0000, 42.0000, 250.0000, 0.0515, 28.0000],
            ]
        ),
        torch.tensor(
            [
                [62.0000, 25.0000, 596.0000, 406.0000, 0.9789, 51.0000],
                [24.0000, 393.0000, 250.0000, 596.0000, 0.9496, 51.0000],
                [25.0000, 154.0000, 593.0000, 595.0000, 0.8013, 67.0000],
                [27.0000, 112.0000, 246.0000, 263.0000, 0.7848, 50.0000],
                [186.0000, 242.0000, 362.0000, 359.0000, 0.7522, 54.0000],
                [448.0000, 209.0000, 512.0000, 241.0000, 0.7446, 57.0000],
                [197.0000, 188.0000, 297.0000, 235.0000, 0.6616, 57.0000],
                [233.0000, 210.0000, 297.0000, 229.0000, 0.5214, 57.0000],
                [519.0000, 255.0000, 589.0000, 284.0000, 0.4844, 48.0000],
                [316.0000, 59.0000, 360.0000, 132.0000, 0.4211, 52.0000],
                [27.0000, 104.0000, 286.0000, 267.0000, 0.3727, 48.0000],
                [191.0000, 89.0000, 344.0000, 256.0000, 0.3143, 57.0000],
                [182.0000, 80.0000, 529.0000, 353.0000, 0.3046, 57.0000],
                [417.0000, 266.0000, 488.0000, 333.0000, 0.2602, 52.0000],
                [324.0000, 110.0000, 369.0000, 157.0000, 0.2550, 57.0000],
                [314.0000, 61.0000, 361.0000, 135.0000, 0.2517, 53.0000],
                [252.0000, 218.0000, 336.0000, 249.0000, 0.2486, 57.0000],
                [191.0000, 241.0000, 342.0000, 354.0000, 0.2384, 53.0000],
                [194.0000, 174.0000, 327.0000, 247.0000, 0.2121, 57.0000],
                [229.0000, 200.0000, 302.0000, 233.0000, 0.2030, 57.0000],
                [439.0000, 192.0000, 526.0000, 252.0000, 0.2004, 57.0000],
                [203.0000, 144.0000, 523.0000, 357.0000, 0.1937, 52.0000],
                [17.0000, 90.0000, 361.0000, 283.0000, 0.1875, 50.0000],
                [15.0000, 14.0000, 598.0000, 272.0000, 0.1747, 67.0000],
                [319.0000, 63.0000, 433.0000, 158.0000, 0.1621, 53.0000],
                [319.0000, 62.0000, 434.0000, 157.0000, 0.1602, 52.0000],
                [253.0000, 85.0000, 311.0000, 147.0000, 0.1562, 57.0000],
                [14.0000, 25.0000, 214.0000, 211.0000, 0.1330, 67.0000],
                [147.0000, 146.0000, 545.0000, 386.0000, 0.0867, 51.0000],
                [324.0000, 174.0000, 455.0000, 292.0000, 0.0761, 52.0000],
                [25.0000, 480.0000, 205.0000, 594.0000, 0.0727, 59.0000],
                [166.0000, 0.0000, 603.0000, 32.0000, 0.0583, 84.0000],
                [519.0000, 255.0000, 589.0000, 285.0000, 0.0578, 50.0000],
            ]
        ),
        torch.tensor(
            [
                [0.0000, 58.0000, 495.0000, 258.0000, 0.9917, 5.0000],
                [199.0000, 291.0000, 212.0000, 299.0000, 0.5247, 37.0000],
                [0.0000, 277.0000, 307.0000, 331.0000, 0.1169, 5.0000],
                [0.0000, 284.0000, 302.0000, 308.0000, 0.0984, 5.0000],
                [348.0000, 231.0000, 367.0000, 244.0000, 0.0621, 15.0000],
                [349.0000, 229.0000, 367.0000, 244.0000, 0.0547, 8.0000],
            ]
        ),
        torch.tensor(
            [
                [1.0000, 9.0000, 365.0000, 506.0000, 0.9980, 17.0000],
                [170.0000, 304.0000, 335.0000, 542.0000, 0.9867, 47.0000],
                [422.0000, 179.0000, 480.0000, 351.0000, 0.9476, 47.0000],
                [329.0000, 241.0000, 449.0000, 420.0000, 0.8503, 47.0000],
                [0.0000, 352.0000, 141.0000, 635.0000, 0.4145, 74.0000],
                [73.0000, 277.0000, 478.0000, 628.0000, 0.3859, 67.0000],
                [329.0000, 183.0000, 373.0000, 286.0000, 0.3097, 47.0000],
                [0.0000, 345.0000, 145.0000, 631.0000, 0.2359, 9.0000],
                [1.0000, 341.0000, 147.0000, 632.0000, 0.2259, 70.0000],
                [0.0000, 338.0000, 148.0000, 632.0000, 0.1669, 62.0000],
                [339.0000, 154.0000, 410.0000, 248.0000, 0.1474, 47.0000],
                [422.0000, 176.0000, 479.0000, 359.0000, 0.1422, 44.0000],
                [0.0000, 349.0000, 148.0000, 636.0000, 0.1369, 42.0000],
                [1.0000, 347.0000, 149.0000, 633.0000, 0.1118, 1.0000],
                [324.0000, 238.0000, 455.0000, 423.0000, 0.0948, 86.0000],
                [0.0000, 348.0000, 146.0000, 640.0000, 0.0885, 37.0000],
                [0.0000, 342.0000, 140.0000, 626.0000, 0.0812, 81.0000],
                [146.0000, 0.0000, 478.0000, 217.0000, 0.0812, 62.0000],
                [75.0000, 102.0000, 357.0000, 553.0000, 0.0618, 64.0000],
                [2.0000, 356.0000, 145.0000, 635.0000, 0.0608, 51.0000],
                [0.0000, 337.0000, 149.0000, 637.0000, 0.0544, 3.0000],
            ]
        ),
        torch.tensor(
            [
                [212.0000, 219.0000, 418.0000, 312.0000, 0.9968, 62.0000],
                [218.0000, 83.0000, 477.0000, 228.0000, 0.9902, 28.0000],
                [113.0000, 221.0000, 476.0000, 368.0000, 0.3940, 62.0000],
                [108.0000, 309.0000, 222.0000, 371.0000, 0.2972, 62.0000],
                [199.0000, 124.0000, 206.0000, 130.0000, 0.2770, 16.0000],
                [213.0000, 154.0000, 447.0000, 301.0000, 0.2698, 28.0000],
                [122.0000, 297.0000, 492.0000, 371.0000, 0.2263, 62.0000],
                [111.0000, 302.0000, 500.0000, 368.0000, 0.2115, 67.0000],
                [319.0000, 220.0000, 424.0000, 307.0000, 0.1761, 62.0000],
                [453.0000, 0.0000, 462.0000, 8.0000, 0.1390, 38.0000],
                [107.0000, 309.0000, 222.0000, 371.0000, 0.1075, 15.0000],
                [109.0000, 309.0000, 225.0000, 372.0000, 0.1028, 67.0000],
                [137.0000, 301.0000, 499.0000, 371.0000, 0.0945, 61.0000],
                [454.0000, 0.0000, 460.0000, 6.0000, 0.0891, 16.0000],
                [162.0000, 102.0000, 167.0000, 105.0000, 0.0851, 16.0000],
                [395.0000, 263.0000, 500.0000, 304.0000, 0.0813, 15.0000],
                [107.0000, 298.0000, 491.0000, 373.0000, 0.0727, 9.0000],
                [157.0000, 78.0000, 488.0000, 332.0000, 0.0573, 28.0000],
                [110.0000, 282.0000, 500.0000, 369.0000, 0.0554, 15.0000],
                [377.0000, 263.0000, 500.0000, 315.0000, 0.0527, 62.0000],
            ]
        ),
        torch.tensor(
            [
                [1.0000, 48.0000, 505.0000, 184.0000, 0.9939, 5.0000],
                [152.0000, 60.0000, 633.0000, 255.0000, 0.9552, 5.0000],
                [0.0000, 183.0000, 20.0000, 200.0000, 0.2347, 8.0000],
                [0.0000, 185.0000, 7.0000, 202.0000, 0.1005, 8.0000],
                [397.0000, 255.0000, 491.0000, 276.0000, 0.0781, 42.0000],
                [0.0000, 186.0000, 7.0000, 202.0000, 0.0748, 3.0000],
                [259.0000, 154.0000, 640.0000, 254.0000, 0.0630, 5.0000],
            ]
        ),
        torch.tensor(
            [
                [203.0000, 277.0000, 215.0000, 312.0000, 0.9953, 1.0000],
                [380.0000, 70.0000, 475.0000, 183.0000, 0.9555, 38.0000],
                [439.0000, 70.0000, 471.0000, 176.0000, 0.3617, 38.0000],
                [379.0000, 143.0000, 390.0000, 158.0000, 0.2418, 38.0000],
                [378.0000, 140.0000, 461.0000, 184.0000, 0.1672, 38.0000],
                [226.0000, 252.0000, 230.0000, 255.0000, 0.0570, 16.0000],
            ]
        ),
        torch.tensor(
            [
                [597.0000, 216.0000, 639.0000, 475.0000, 0.9783, 1.0000],
                [80.0000, 263.0000, 128.0000, 291.0000, 0.9571, 60.0000],
                [126.0000, 258.0000, 178.0000, 286.0000, 0.9540, 60.0000],
                [174.0000, 252.0000, 221.0000, 279.0000, 0.9434, 60.0000],
                [248.0000, 323.0000, 300.0000, 354.0000, 0.9359, 60.0000],
                [171.0000, 150.0000, 220.0000, 166.0000, 0.9347, 60.0000],
                [121.0000, 151.0000, 173.0000, 168.0000, 0.9336, 60.0000],
                [394.0000, 111.0000, 417.0000, 135.0000, 0.9256, 85.0000],
                [300.0000, 327.0000, 362.0000, 358.0000, 0.9058, 60.0000],
                [264.0000, 149.0000, 306.0000, 166.0000, 0.8948, 60.0000],
                [306.0000, 150.0000, 350.0000, 165.0000, 0.8798, 60.0000],
                [70.0000, 150.0000, 127.0000, 168.0000, 0.8697, 60.0000],
                [110.0000, 138.0000, 153.0000, 156.0000, 0.8586, 60.0000],
                [223.0000, 154.0000, 270.0000, 166.0000, 0.8576, 60.0000],
                [541.0000, 81.0000, 602.0000, 153.0000, 0.8352, 79.0000],
                [34.0000, 266.0000, 82.0000, 295.0000, 0.8326, 60.0000],
                [444.0000, 302.0000, 484.0000, 325.0000, 0.7900, 60.0000],
                [14.0000, 152.0000, 73.0000, 169.0000, 0.7792, 60.0000],
                [115.0000, 247.0000, 157.0000, 268.0000, 0.7654, 60.0000],
                [168.0000, 350.0000, 237.0000, 385.0000, 0.7241, 60.0000],
                [197.0000, 319.0000, 249.0000, 351.0000, 0.7062, 60.0000],
                [89.0000, 331.0000, 149.0000, 366.0000, 0.6970, 61.0000],
                [66.0000, 143.0000, 109.0000, 153.0000, 0.6787, 60.0000],
                [152.0000, 332.0000, 217.0000, 358.0000, 0.6739, 60.0000],
                [99.0000, 355.0000, 169.0000, 395.0000, 0.6582, 60.0000],
                [583.0000, 205.0000, 594.0000, 218.0000, 0.6428, 47.0000],
                [498.0000, 301.0000, 528.0000, 321.0000, 0.6373, 60.0000],
                [255.0000, 146.0000, 274.0000, 155.0000, 0.6366, 60.0000],
                [148.0000, 231.0000, 192.0000, 250.0000, 0.5984, 60.0000],
                [501.0000, 140.0000, 551.0000, 164.0000, 0.5910, 60.0000],
                [156.0000, 144.0000, 193.0000, 157.0000, 0.5910, 60.0000],
                [381.0000, 225.0000, 444.0000, 254.0000, 0.5737, 60.0000],
                [156.0000, 243.0000, 206.0000, 264.0000, 0.5675, 60.0000],
                [229.0000, 302.0000, 280.0000, 331.0000, 0.5588, 60.0000],
                [492.0000, 134.0000, 516.0000, 142.0000, 0.5492, 60.0000],
                [346.0000, 150.0000, 383.0000, 165.0000, 0.5481, 60.0000],
                [17.0000, 143.0000, 67.0000, 154.0000, 0.5254, 60.0000],
                [283.0000, 308.0000, 330.0000, 334.0000, 0.5141, 60.0000],
                [421.0000, 222.0000, 489.0000, 250.0000, 0.4983, 60.0000],
                [0.0000, 107.0000, 51.0000, 134.0000, 0.4978, 78.0000],
                [70.0000, 248.0000, 113.0000, 270.0000, 0.4884, 60.0000],
                [215.0000, 147.0000, 262.0000, 164.0000, 0.4867, 60.0000],
                [293.0000, 145.0000, 315.0000, 157.0000, 0.4841, 60.0000],
                [523.0000, 272.0000, 548.0000, 288.0000, 0.4728, 60.0000],
                [534.0000, 152.0000, 560.0000, 164.0000, 0.4644, 60.0000],
                [516.0000, 294.0000, 546.0000, 314.0000, 0.4597, 60.0000],
                [352.0000, 319.0000, 395.0000, 342.0000, 0.4364, 60.0000],
                [106.0000, 234.0000, 149.0000, 255.0000, 0.4317, 60.0000],
                [326.0000, 136.0000, 357.0000, 147.0000, 0.4281, 60.0000],
                [135.0000, 132.0000, 166.0000, 145.0000, 0.4159, 60.0000],
                [63.0000, 238.0000, 104.0000, 259.0000, 0.4136, 60.0000],
                [472.0000, 221.0000, 527.0000, 246.0000, 0.4090, 60.0000],
                [189.0000, 137.0000, 225.0000, 154.0000, 0.4018, 60.0000],
                [135.0000, 311.0000, 195.0000, 337.0000, 0.3965, 60.0000],
                [9.0000, 148.0000, 68.0000, 164.0000, 0.3915, 60.0000],
                [366.0000, 232.0000, 408.0000, 257.0000, 0.3858, 60.0000],
                [291.0000, 243.0000, 318.0000, 266.0000, 0.3838, 60.0000],
                [494.0000, 276.0000, 524.0000, 300.0000, 0.3727, 60.0000],
                [97.0000, 135.0000, 122.0000, 148.0000, 0.3717, 60.0000],
                [467.0000, 289.0000, 499.0000, 309.0000, 0.3710, 60.0000],
                [150.0000, 134.0000, 188.0000, 146.0000, 0.3705, 60.0000],
                [427.0000, 290.0000, 463.0000, 314.0000, 0.3575, 60.0000],
                [38.0000, 343.0000, 101.0000, 408.0000, 0.3540, 61.0000],
                [76.0000, 313.0000, 128.0000, 343.0000, 0.3429, 61.0000],
                [507.0000, 146.0000, 537.0000, 163.0000, 0.3420, 60.0000],
                [451.0000, 268.0000, 478.0000, 282.0000, 0.3389, 60.0000],
                [545.0000, 292.0000, 578.0000, 314.0000, 0.3252, 60.0000],
                [350.0000, 309.0000, 393.0000, 336.0000, 0.3246, 60.0000],
                [388.0000, 307.0000, 429.0000, 337.0000, 0.3240, 60.0000],
                [34.0000, 253.0000, 67.0000, 270.0000, 0.3228, 60.0000],
                [402.0000, 224.0000, 462.0000, 252.0000, 0.3177, 60.0000],
                [160.0000, 131.0000, 191.0000, 142.0000, 0.3104, 60.0000],
                [132.0000, 310.0000, 197.0000, 340.0000, 0.2923, 61.0000],
                [481.0000, 84.0000, 543.0000, 140.0000, 0.2872, 79.0000],
                [13.0000, 137.0000, 62.0000, 153.0000, 0.2859, 60.0000],
                [98.0000, 355.0000, 171.0000, 395.0000, 0.2843, 61.0000],
                [115.0000, 149.0000, 156.0000, 160.0000, 0.2774, 60.0000],
                [65.0000, 137.0000, 101.0000, 148.0000, 0.2732, 60.0000],
                [314.0000, 242.0000, 341.0000, 264.0000, 0.2714, 60.0000],
                [455.0000, 237.0000, 486.0000, 251.0000, 0.2630, 60.0000],
                [552.0000, 146.0000, 595.0000, 164.0000, 0.2553, 60.0000],
                [50.0000, 133.0000, 78.0000, 145.0000, 0.2485, 60.0000],
                [544.0000, 280.0000, 570.0000, 294.0000, 0.2459, 60.0000],
                [40.0000, 144.0000, 66.0000, 154.0000, 0.2453, 60.0000],
                [289.0000, 254.0000, 312.0000, 268.0000, 0.2374, 60.0000],
                [266.0000, 140.0000, 292.0000, 149.0000, 0.2357, 60.0000],
                [504.0000, 266.0000, 525.0000, 277.0000, 0.2281, 60.0000],
                [304.0000, 285.0000, 346.0000, 309.0000, 0.2256, 60.0000],
                [303.0000, 222.0000, 341.0000, 238.0000, 0.2236, 60.0000],
                [498.0000, 219.0000, 549.0000, 243.0000, 0.2168, 60.0000],
                [89.0000, 333.0000, 144.0000, 352.0000, 0.2159, 61.0000],
                [0.0000, 108.0000, 51.0000, 135.0000, 0.2076, 79.0000],
                [303.0000, 220.0000, 329.0000, 231.0000, 0.2007, 60.0000],
                [0.0000, 131.0000, 38.0000, 150.0000, 0.1967, 60.0000],
                [364.0000, 137.0000, 401.0000, 165.0000, 0.1958, 60.0000],
                [398.0000, 95.0000, 538.0000, 139.0000, 0.1868, 79.0000],
                [334.0000, 243.0000, 357.0000, 263.0000, 0.1835, 60.0000],
                [480.0000, 269.0000, 503.0000, 286.0000, 0.1831, 60.0000],
                [184.0000, 302.0000, 229.0000, 320.0000, 0.1784, 60.0000],
                [522.0000, 286.0000, 548.0000, 300.0000, 0.1752, 60.0000],
            ]
        ),
    ]

    return [{"bbox": p[:, :4].double(), "scores": p[:, 4].double(), "labels": p[:, 5]} for p in pred], [
        {"bbox": g[:, :4].double(), "labels": g[:, 4].long(), "iscrowd": g[:, 5]} for g in gt
    ]


def random_sample() -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
    torch.manual_seed(12)
    targets: List[torch.Tensor] = []
    preds: List[torch.Tensor] = []
    for _ in range(30):
        # Generate some ground truth boxes
        n_gt_box = torch.randint(50, (1,)).item()
        x1 = torch.randint(641, (n_gt_box, 1))
        y1 = torch.randint(641, (n_gt_box, 1))
        w = 640 * torch.rand((n_gt_box, 1))
        h = 640 * torch.rand((n_gt_box, 1))
        x2 = (x1 + w).clip(max=640)
        y2 = (y1 + h).clip(max=640)
        category = torch.randint(0, 80, (n_gt_box, 1))
        iscrowd = torch.randint(2, (n_gt_box, 1))
        targets.append(torch.cat((x1, y1, x2, y2, category, iscrowd), dim=1))

        # Remove some of gt boxes from corresponding predictions
        kept_boxes = torch.randint(2, (n_gt_box,), dtype=torch.bool)
        n_predicted_box = kept_boxes.sum()
        x1 = x1[kept_boxes]
        y1 = y1[kept_boxes]
        w = w[kept_boxes]
        h = h[kept_boxes]
        category = category[kept_boxes]

        # Perturb gt boxes in the prediction
        perturb_x1 = 640 * (torch.rand_like(x1, dtype=torch.float) - 0.5)
        perturb_y1 = 640 * (torch.rand_like(y1, dtype=torch.float) - 0.5)
        perturb_w = 640 * (torch.rand_like(w, dtype=torch.float) - 0.5)
        perturb_h = 640 * (torch.rand_like(h, dtype=torch.float) - 0.5)
        perturb_category = Geometric(0.7).sample((n_predicted_box, 1)) * (2 * torch.randint_like(category, 2) - 1)

        x1 = (x1 + perturb_x1).clip(min=0, max=640)
        y1 = (y1 + perturb_y1).clip(min=0, max=640)
        w = (w + perturb_w).clip(min=0, max=640)
        h = (h + perturb_h).clip(min=0, max=640)
        x2 = (x1 + w).clip(max=640)
        y2 = (y1 + h).clip(max=640)
        category = (category + perturb_category) % 80
        confidence = torch.rand_like(category, dtype=torch.double)
        perturbed_gt_boxes = torch.cat((x1, y1, x2, y2, confidence, category), dim=1)

        # Generate some additional prediction boxes
        n_additional_pred_boxes = torch.randint(50, (1,)).item()
        x1 = torch.randint(641, (n_additional_pred_boxes, 1))
        y1 = torch.randint(641, (n_additional_pred_boxes, 1))
        w = 640 * torch.rand((n_additional_pred_boxes, 1))
        h = 640 * torch.rand((n_additional_pred_boxes, 1))
        x2 = (x1 + w).clip(max=640)
        y2 = (y1 + h).clip(max=640)
        category = torch.randint(0, 80, (n_additional_pred_boxes, 1))
        confidence = torch.rand_like(category, dtype=torch.double)
        additional_pred_boxes = torch.cat((x1, y1, x2, y2, confidence, category), dim=1)

        preds.append(torch.cat((perturbed_gt_boxes, additional_pred_boxes), dim=0))

    return [{"bbox": p[:, :4], "scores": p[:, 4], "labels": p[:, 5]} for p in preds], [
        {"bbox": g[:, :4], "labels": g[:, 4].long(), "iscrowd": g[:, 5]} for g in targets
    ]


def create_coco_api(
    predictions: List[Dict[str, torch.Tensor]], targets: List[Dict[str, torch.Tensor]]
) -> Tuple[COCO, COCO]:
    """Create COCO objects from predictions and targets

    Args:
        predictions: list of predictions. Each one is a dict containing "bbox", "scores" and "labels" as its keys. The
            associated value to "bbox" is a tensor of shape (n, 4) where n stands for the number of detections.
            4 represents top left and bottom right coordinates of a box in the form (x1, y1, x2, y2). The associated
            values to "scores" and "labels" are tensors of shape (n,).
        targets: list of targets. Each one is a dict containing "bbox", "labels" and "iscrowd" as its keys. The
            associated values to "bbox" and "labels" are the same as those of the ``predictions``. The associated
            value to "iscrowd" is a tensor of shape (n,) which determines if ground truth boxes are crowd or not.
    """
    ann_id = 1
    coco_gt = COCO()
    dataset = {"images": [], "categories": [], "annotations": []}

    for idx, target in enumerate(targets):
        dataset["images"].append({"id": idx})
        bboxes = target["bbox"].clone()
        bboxes[:, 2:4] -= bboxes[:, 0:2]
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i].tolist()
            area = bbox[2] * bbox[3]
            ann = {
                "image_id": idx,
                "bbox": bbox,
                "category_id": target["labels"][i].item(),
                "area": area,
                "iscrowd": target["iscrowd"][i].item(),
                "id": ann_id,
            }
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in range(0, 91)]
    dataset["info"] = {}
    coco_gt.dataset = dataset
    coco_gt.createIndex()

    prediction_tensors = []
    for idx, prediction in enumerate(predictions):
        bboxes = prediction["bbox"].clone()
        bboxes[:, 2:4] -= bboxes[:, 0:2]
        prediction_tensors.append(
            torch.cat(
                [
                    torch.tensor(idx).repeat(bboxes.shape[0], 1),
                    bboxes,
                    prediction["scores"].unsqueeze(1),
                    prediction["labels"].unsqueeze(1),
                ],
                dim=1,
            )
        )
    predictions = torch.cat(prediction_tensors, dim=0)
    coco_dt = coco_gt.loadRes(predictions.numpy())
    return coco_dt, coco_gt


def pycoco_mAP(predictions: List[Dict[str, torch.Tensor]], targets: List[Dict[str, torch.Tensor]]) -> np.array:
    """
    Returned values are AP@.5...95, AP@.5, AP@.75, AP-S, AP-M, AP-L, AR-1, AR-10, AR-100, AR-S, AR-M, AR-L
    """
    coco_dt, coco_gt = create_coco_api(predictions, targets)
    eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    eval.evaluate()
    eval.accumulate()
    eval.summarize()
    return eval.stats


Sample = namedtuple("Sample", ["data", "mAP", "length"])


@pytest.fixture(
    params=[
        ("coco2017", "full"),
        ("coco2017", "with_an_empty_pred"),
        ("coco2017", "with_an_empty_gt"),
        ("coco2017", "with_an_empty_pred_and_gt"),
        ("random", "full"),
        ("random", "with_an_empty_pred"),
        ("random", "with_an_empty_gt"),
        ("random", "with_an_empty_pred_and_gt"),
    ]
)
def sample(request) -> Sample:
    data = coco_val2017_sample() if request.param[0] == "coco2017" else random_sample()
    if request.param[1] == "with_an_empty_pred":
        data[0][1] = {
            "bbox": torch.zeros(0, 4),
            "scores": torch.zeros(
                0,
            ),
            "labels": torch.zeros(0, dtype=torch.long),
        }
    elif request.param[1] == "with_an_empty_gt":
        data[1][0] = {
            "bbox": torch.zeros(0, 4),
            "labels": torch.zeros(0, dtype=torch.long),
            "iscrowd": torch.zeros(
                0,
            ),
        }
    elif request.param[1] == "with_an_empty_pred_and_gt":
        data[0][0] = {
            "bbox": torch.zeros(0, 4),
            "scores": torch.zeros(
                0,
            ),
            "labels": torch.zeros(0, dtype=torch.long),
        }
        data[0][1] = {
            "bbox": torch.zeros(0, 4),
            "scores": torch.zeros(
                0,
            ),
            "labels": torch.zeros(0, dtype=torch.long),
        }
        data[1][0] = {
            "bbox": torch.zeros(0, 4),
            "labels": torch.zeros(0, dtype=torch.long),
            "iscrowd": torch.zeros(
                0,
            ),
        }
        data[1][2] = {
            "bbox": torch.zeros(0, 4),
            "labels": torch.zeros(0, dtype=torch.long),
            "iscrowd": torch.zeros(
                0,
            ),
        }
    mAP = pycoco_mAP(*data)

    return Sample(data, mAP, len(data[0]))


def test_wrong_input():
    m = ObjectDetectionAvgPrecisionRecall()

    with pytest.raises(ValueError, match="y_pred and y should have the same number of samples"):
        m.update(([{"bbox": None, "scores": None}], []))
    with pytest.raises(ValueError, match="y_pred and y should contain at least one sample."):
        m.update(([], []))
    with pytest.raises(ValueError, match="y_pred sample dictionaries should have 'bbox', 'scores'"):
        m.update(([{"bbox": None, "scores": None}], [{"bbox": None, "labels": None}]))
    with pytest.raises(ValueError, match="y sample dictionaries should have 'bbox', 'labels'"):
        m.update(([{"bbox": None, "scores": None, "labels": None}], [{"labels": None}]))


def test_empty_data(available_device):
    """
    Note that PyCOCO returns -1 when threre's no ground truth data.
    """

    metric = ObjectDetectionAvgPrecisionRecall(device=available_device)
    assert metric._device == torch.device(available_device)
    metric.update(
        (
            [{"bbox": torch.zeros((0, 4)), "scores": torch.zeros((0,)), "labels": torch.zeros((0), dtype=torch.long)}],
            [{"bbox": torch.zeros((0, 4)), "iscrowd": torch.zeros((0,)), "labels": torch.zeros((0), dtype=torch.long)}],
        )
    )
    assert len(metric._tps) == 0
    assert len(metric._fps) == 0
    assert metric.compute() == (-1, -1)
    metric.update(
        (
            [
                {
                    "bbox": torch.tensor([[0.0, 0.0, 100.0, 100.0]]),
                    "scores": torch.ones((1,)),
                    "labels": torch.ones((1,), dtype=torch.long),
                }
            ],
            [{"bbox": torch.zeros((0, 4)), "iscrowd": torch.zeros((0,)), "labels": torch.zeros((0), dtype=torch.long)}],
        )
    )
    assert metric.compute() == (-1, -1)

    metric = ObjectDetectionAvgPrecisionRecall(device=available_device)
    assert metric._device == torch.device(available_device)
    metric.update(
        (
            [{"bbox": torch.zeros((0, 4)), "scores": torch.zeros((0,)), "labels": torch.zeros((0), dtype=torch.long)}],
            [
                {
                    "bbox": torch.tensor([[0.0, 0.0, 100.0, 100.0]]),
                    "iscrowd": torch.zeros((1,)),
                    "labels": torch.ones((1,), dtype=torch.long),
                }
            ],
        )
    )
    assert len(metric._tps) == 0
    assert len(metric._fps) == 0
    assert metric._y_true_count[1] == 1
    assert metric.compute() == (0, 0)

    metric = ObjectDetectionAvgPrecisionRecall(device=available_device)
    assert metric._device == torch.device(available_device)
    pred = {
        "bbox": torch.tensor([[0.0, 0.0, 100.0, 100.0]]),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([5]),
    }
    target = {"bbox": torch.zeros((0, 4)), "iscrowd": torch.zeros((0,)), "labels": torch.zeros((0), dtype=torch.long)}
    metric.update(([pred], [target]))
    assert len(metric._tps) == len(metric._fps) == 1
    pycoco_result = pycoco_mAP([pred], [target])
    assert metric.compute() == (pycoco_result[0], pycoco_result[8])


def test_no_torchvision():
    with patch.dict(sys.modules, {"torchvision.ops.boxes": None}):
        with pytest.raises(ModuleNotFoundError, match=r"This metric requires torchvision to be installed."):
            ObjectDetectionAvgPrecisionRecall()


def test_iou(sample, available_device):
    m = ObjectDetectionAvgPrecisionRecall(num_classes=91, device=available_device)
    assert m._device == torch.device(available_device)
    from pycocotools.mask import iou as pycoco_iou

    for pred, tgt in zip(*sample.data):
        pred_bbox = pred["bbox"].double()
        tgt_bbox = tgt["bbox"].double()
        if not pred_bbox.shape[0] or not tgt_bbox.shape[0]:
            continue
        iscrowd = tgt["iscrowd"]

        ignite_iou = m.box_iou(pred_bbox, tgt_bbox, iscrowd.bool())

        pred_bbox[:, 2:4] -= pred_bbox[:, :2]
        tgt_bbox[:, 2:4] -= tgt_bbox[:, :2]
        pyc_iou = pycoco_iou(pred_bbox.numpy(), tgt_bbox.numpy(), iscrowd.int())

        equal = ignite_iou.numpy() == pyc_iou
        assert equal.all()


def test_iou_thresholding(available_device):
    metric = ObjectDetectionAvgPrecisionRecall(iou_thresholds=[0.0, 0.3, 0.5, 0.75], device=available_device)
    assert metric._device == torch.device(available_device)

    pred = {
        "bbox": torch.tensor([[0.0, 0.0, 100.0, 100.0]]),
        "scores": torch.tensor([0.8]),
        "labels": torch.tensor([1]),
    }
    gt = {"bbox": torch.tensor([[0.0, 0.0, 50.0, 100.0]]), "iscrowd": torch.zeros((1,)), "labels": torch.tensor([1])}
    metric.update(([pred], [gt]))
    assert (metric._tps[0] == torch.tensor([[True], [True], [True], [False]], device=available_device)).all()

    pred = {
        "bbox": torch.tensor([[0.0, 0.0, 100.0, 100.0]]),
        "scores": torch.tensor([0.8]),
        "labels": torch.tensor([1]),
    }
    gt = {"bbox": torch.tensor([[100.0, 0.0, 200.0, 100.0]]), "iscrowd": torch.zeros((1,)), "labels": torch.tensor([1])}
    metric.update(([pred], [gt]))
    assert (metric._tps[1] == torch.tensor([[True], [False], [False], [False]], device=available_device)).all()


def test_matching(available_device):
    """
    PyCOCO matching rules:
        1. The higher confidence in a prediction, the sooner decision is made for it.
            If there's equal confidence in two predictions, the dicision is first made
            for the one who comes earlier.
        2. Each ground truth box is matched with at most one prediction. Crowd ground
            truth is the exception.
        3. If a ground truth is crowd or out of area range, is set to be ignored.
        4. A prediction matched with a ignored gt would get ignored, in the sense that it becomes
            neither tp nor fp.
        5. An unmatched prediction would get ignored if it's out of area range. So doesn't become fp due to rule 4.
        6. Among many plausible ground truth boxes, a prediction is matched with the
            one which has the highest mutual IOU. If two ground truth boxes have the
            same IOU with a prediction, the later one is matched.
        7. Non-ignored ground truths are given priority over the ignored ones when matching with a prediction
            even if their IOU is lower.
    """
    metric = ObjectDetectionAvgPrecisionRecall(iou_thresholds=[0.2], device=available_device)
    assert metric._device == torch.device(available_device)

    pred = {
        "bbox": torch.tensor([[0.0, 0.0, 100.0, 100.0], [0.0, 0.0, 100.0, 100.0]]),
        "scores": torch.tensor([0.8, 0.9]),
        "labels": torch.tensor([1, 1]),
    }
    gt = {
        "bbox": torch.tensor([[0.0, 0.0, 100.0, 100.0]]),
        "iscrowd": torch.zeros((1,)),
        "labels": torch.tensor([1]),
    }
    metric.update(([pred], [gt]))
    # Preds are sorted by their scores internally
    assert (metric._tps[0] == torch.tensor([[True, False]], device=available_device)).all()
    assert (metric._fps[0] == torch.tensor([[False, True]], device=available_device)).all()
    assert (metric._scores[0] == torch.tensor([[0.9, 0.8]], device=available_device)).all()

    pred["scores"] = torch.tensor([0.9, 0.9])
    metric.update(([pred], [gt]))
    assert (metric._tps[1] == torch.tensor([[True, False]], device=available_device)).all()
    assert (metric._fps[1] == torch.tensor([[False, True]], device=available_device)).all()

    gt["iscrowd"] = torch.tensor([1])
    metric.update(([pred], [gt]))
    assert (metric._tps[2] == torch.tensor([[False, False]], device=available_device)).all()
    assert (metric._fps[2] == torch.tensor([[False, False]], device=available_device)).all()

    pred["bbox"] = torch.tensor([[0.0, 0.0, 100.0, 100.0], [100.0, 0.0, 200.0, 100.0]])
    gt["bbox"] = torch.tensor([[0.0, 0.0, 25.0, 50.0], [50.0, 0.0, 150.0, 100.0]])
    gt["iscrowd"] = torch.zeros((2,))
    gt["labels"] = torch.tensor([1, 1])
    metric.update(([pred], [gt]))
    assert (metric._tps[3] == torch.tensor([[True, False]], device=available_device)).all()
    assert (metric._fps[3] == torch.tensor([[False, True]], device=available_device)).all()

    metric._area_range = "small"
    pred["bbox"] = torch.tensor(
        [[0.0, 0.0, 100.0, 10.0], [0.0, 0.0, 100.0, 10.0], [0.0, 0.0, 100.0, 11.0], [0.0, 0.0, 100.0, 10.0]]
    )
    pred["scores"] = torch.tensor([0.9, 0.9, 0.9, 0.9])
    pred["labels"] = torch.tensor([1, 1, 1, 1])
    gt["bbox"] = torch.tensor([[0.0, 0.0, 100.0, 11.0], [0.0, 0.0, 100.0, 5.0]])
    metric.update(([pred], [gt]))
    assert (metric._tps[4] == torch.tensor([[True, False, False, False]], device=available_device)).all()
    assert (metric._fps[4] == torch.tensor([[False, False, False, True]], device=available_device)).all()

    pred["scores"] = torch.tensor([0.9, 1.0, 0.9, 0.9])
    metric._max_detections_per_image_per_class = 1
    metric.update(([pred], [gt]))
    assert (metric._tps[5] == torch.tensor([[True]], device=available_device)).all()
    assert (metric._fps[5] == torch.tensor([[False]], device=available_device)).all()


def sklearn_precision_recall_curve_allowing_multiple_recalls_at_single_threshold(y_true, y_score):
    y_true = y_true == 1

    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_true = y_true[desc_score_indices]
    weight = 1.0

    tps = stable_cumsum(y_true * weight)
    fps = stable_cumsum((1 - y_true) * weight)
    ps = tps + fps
    precision = np.zeros_like(tps)
    np.divide(tps, ps, out=precision, where=(ps != 0))
    if tps[-1] == 0:
        recall = np.ones_like(tps)
    else:
        recall = tps / tps[-1]

    sl = slice(None, None, -1)
    return np.hstack((precision[sl], 1)), np.hstack((recall[sl], 0)), None


def test__compute_recall_and_precision(available_device):
    # The case in which detector detects all gt objects but also produces some wrong predictions.
    scores = torch.rand((50,))
    y_true = torch.randint(0, 2, (50,))
    m = ObjectDetectionAvgPrecisionRecall(device=available_device)
    assert m._device == torch.device(available_device)

    ignite_recall, ignite_precision = m._compute_recall_and_precision(
        y_true.bool(), ~(y_true.bool()), scores, y_true.sum()
    )
    sklearn_precision, sklearn_recall, _ = sklearn_precision_recall_curve_allowing_multiple_recalls_at_single_threshold(
        y_true.numpy(), scores.numpy()
    )
    assert (ignite_recall.flip(0).numpy() == sklearn_recall[:-1]).all()
    assert (ignite_precision.flip(0).numpy() == sklearn_precision[:-1]).all()

    # Like above but with two additional mean dimensions.
    scores = torch.rand((50,))
    y_true = torch.zeros((6, 8, 50))
    sklearn_precisions, sklearn_recalls = [], []
    for i in range(6):
        for j in range(8):
            y_true[i, j, np.random.choice(50, size=15, replace=False)] = 1
            precision, recall, _ = sklearn_precision_recall_curve_allowing_multiple_recalls_at_single_threshold(
                y_true[i, j].numpy(), scores.numpy()
            )
            sklearn_precisions.append(precision[:-1])
            sklearn_recalls.append(recall[:-1])
    sklearn_precisions = np.array(sklearn_precisions).reshape(6, 8, -1)
    sklearn_recalls = np.array(sklearn_recalls).reshape(6, 8, -1)
    ignite_recall, ignite_precision = m._compute_recall_and_precision(
        y_true.bool(), ~(y_true.bool()), scores, torch.tensor(15)
    )
    assert (ignite_recall.flip(-1).numpy() == sklearn_recalls).all()
    assert (ignite_precision.flip(-1).numpy() == sklearn_precisions).all()


def test_compute(sample):
    device = idist.device()

    if device == torch.device("mps"):
        pytest.skip("Due to MPS backend out of memory")

    # AP@.5...95, AP@.5, AP@.75, AP-S, AP-M, AP-L, AR-1, AR-10, AR-100, AR-S, AR-M, AR-L
    ap_50_95_ar_100 = ObjectDetectionAvgPrecisionRecall(num_classes=91, device=device)
    ap_50 = ObjectDetectionAvgPrecisionRecall(num_classes=91, iou_thresholds=[0.5], device=device)
    ap_75 = ObjectDetectionAvgPrecisionRecall(num_classes=91, iou_thresholds=[0.75], device=device)
    ap_ar_S = ObjectDetectionAvgPrecisionRecall(num_classes=91, device=device, area_range="small")
    ap_ar_M = ObjectDetectionAvgPrecisionRecall(num_classes=91, device=device, area_range="medium")
    ap_ar_L = ObjectDetectionAvgPrecisionRecall(num_classes=91, device=device, area_range="large")
    ar_1 = ObjectDetectionAvgPrecisionRecall(num_classes=91, device=device, max_detections_per_image_per_class=1)
    ar_10 = ObjectDetectionAvgPrecisionRecall(num_classes=91, device=device, max_detections_per_image_per_class=10)

    metrics = [ap_50_95_ar_100, ap_50, ap_75, ap_ar_S, ap_ar_M, ap_ar_L, ar_1, ar_10]
    for metric in metrics:
        metric.update(sample.data)

    ignite_res = [metric.compute() for metric in metrics]
    ignite_res_recompute = [metric.compute() for metric in metrics]

    assert all([r1 == r2 for r1, r2 in zip(ignite_res, ignite_res_recompute)])

    AP_50_95, AR_100 = ignite_res[0]
    AP_50 = ignite_res[1][0]
    AP_75 = ignite_res[2][0]
    AP_S, AR_S = ignite_res[3]
    AP_M, AR_M = ignite_res[4]
    AP_L, AR_L = ignite_res[5]
    AR_1 = ignite_res[6][1]
    AR_10 = ignite_res[7][1]
    all_res = [AP_50_95, AP_50, AP_75, AP_S, AP_M, AP_L, AR_1, AR_10, AR_100, AR_S, AR_M, AR_L]
    print(all_res)
    assert np.allclose(all_res, sample.mAP)

    common_metrics = CommonObjectDetectionMetrics(num_classes=91, device=device)
    common_metrics.update(sample.data)
    res = common_metrics.compute()
    common_metrics_res = [
        res["AP@50..95"],
        res["AP@50"],
        res["AP@75"],
        res["AP-S"],
        res["AP-M"],
        res["AP-L"],
        res["AR-1"],
        res["AR-10"],
        res["AR-100"],
        res["AR-S"],
        res["AR-M"],
        res["AR-L"],
    ]
    print(common_metrics_res)
    assert all_res == common_metrics_res
    assert np.allclose(common_metrics_res, sample.mAP)


def test_integration(sample):
    bs = 3

    device = idist.device()
    if device == torch.device("mps"):
        pytest.skip("Due to MPS backend out of memory")

    def update(engine, i):
        b = slice(i * bs, (i + 1) * bs)
        return sample.data[0][b], sample.data[1][b]

    engine = Engine(update)

    metric_device = "cpu" if device.type == "xla" else device
    metric_50_95 = ObjectDetectionAvgPrecisionRecall(num_classes=91, device=metric_device)
    metric_50_95.attach(engine, name="mAP[50-95]")

    n_iter = ceil(sample.length / bs)
    engine.run(range(n_iter), max_epochs=1)

    res_50_95 = engine.state.metrics["mAP[50-95]"][0]
    pycoco_res_50_95 = sample.mAP[0]

    assert np.allclose(res_50_95, pycoco_res_50_95)


def test_tensor_list_to_dict_list():
    y_preds = [
        [torch.randn((2, 6)), torch.randn((0, 6)), torch.randn((5, 6))],
        [
            {"bbox": torch.randn((2, 4)), "confidence": torch.randn((2,)), "class": torch.randn((2,))},
            {"bbox": torch.randn((5, 4)), "confidence": torch.randn((5,)), "class": torch.randn((5,))},
        ],
    ]
    ys = [
        [torch.randn((2, 6)), torch.randn((0, 6)), torch.randn((5, 6))],
        [
            {"bbox": torch.randn((2, 4)), "class": torch.randn((2,))},
            {"bbox": torch.randn((5, 4)), "class": torch.randn((5,))},
        ],
    ]
    for y_pred in y_preds:
        for y in ys:
            y_pred_new, y_new = coco_tensor_list_to_dict_list((y_pred, y))
            if isinstance(y_pred[0], dict):
                assert y_pred_new is y_pred
            else:
                assert all(
                    [
                        (ypn["bbox"] == yp[:, :4]).all()
                        & (ypn["confidence"] == yp[:, 4]).all()
                        & (ypn["class"] == yp[:, 5]).all()
                        for yp, ypn in zip(y_pred, y_pred_new)
                    ]
                )
            if isinstance(y[0], dict):
                assert y_new is y
            else:
                assert all(
                    [
                        (ytn["bbox"] == yt[:, :4]).all()
                        & (ytn["class"] == yt[:, 4]).all()
                        & (ytn["iscrowd"] == yt[:, 5]).all()
                        for yt, ytn in zip(y, y_new)
                    ]
                )


def test_distrib_update_compute(distributed, sample):
    rank_samples_cnt = ceil(sample.length / idist.get_world_size())
    rank = idist.get_rank()
    rank_samples_range = slice(rank_samples_cnt * rank, rank_samples_cnt * (rank + 1))

    device = idist.device()

    if device == torch.device("mps"):
        pytest.skip("Due to MPS backend out of memory")

    metric_device = "cpu" if device.type == "xla" else device
    # AP@.5...95, AP@.5, AP@.75, AP-S, AP-M, AP-L, AR-1, AR-10, AR-100, AR-S, AR-M, AR-L
    ap_50_95_ar_100 = ObjectDetectionAvgPrecisionRecall(num_classes=91, device=metric_device)
    ap_50 = ObjectDetectionAvgPrecisionRecall(num_classes=91, iou_thresholds=[0.5], device=metric_device)
    ap_75 = ObjectDetectionAvgPrecisionRecall(num_classes=91, iou_thresholds=[0.75], device=metric_device)
    ap_ar_S = ObjectDetectionAvgPrecisionRecall(num_classes=91, device=metric_device, area_range="small")
    ap_ar_M = ObjectDetectionAvgPrecisionRecall(num_classes=91, device=metric_device, area_range="medium")
    ap_ar_L = ObjectDetectionAvgPrecisionRecall(num_classes=91, device=metric_device, area_range="large")
    ar_1 = ObjectDetectionAvgPrecisionRecall(num_classes=91, device=metric_device, max_detections_per_image_per_class=1)
    ar_10 = ObjectDetectionAvgPrecisionRecall(
        num_classes=91, device=metric_device, max_detections_per_image_per_class=10
    )

    metrics = [ap_50_95_ar_100, ap_50, ap_75, ap_ar_S, ap_ar_M, ap_ar_L, ar_1, ar_10]

    y_pred_rank = sample.data[0][rank_samples_range]
    y_rank = sample.data[1][rank_samples_range]
    for metric in metrics:
        metric.update((y_pred_rank, y_rank))

    ignite_res = [metric.compute() for metric in metrics]
    ignite_res_recompute = [metric.compute() for metric in metrics]
    assert all([r1 == r2 for r1, r2 in zip(ignite_res, ignite_res_recompute)])

    AP_50_95, AR_100 = ignite_res[0]
    AP_50 = ignite_res[1][0]
    AP_75 = ignite_res[2][0]
    AP_S, AR_S = ignite_res[3]
    AP_M, AR_M = ignite_res[4]
    AP_L, AR_L = ignite_res[5]
    AR_1 = ignite_res[6][1]
    AR_10 = ignite_res[7][1]
    all_res = [AP_50_95, AP_50, AP_75, AP_S, AP_M, AP_L, AR_1, AR_10, AR_100, AR_S, AR_M, AR_L]
    assert np.allclose(all_res, sample.mAP)

    common_metrics = CommonObjectDetectionMetrics(num_classes=91, device=device)
    common_metrics.update((y_pred_rank, y_rank))
    res = common_metrics.compute()
    common_metrics_res = [
        res["AP@50..95"],
        res["AP@50"],
        res["AP@75"],
        res["AP-S"],
        res["AP-M"],
        res["AP-L"],
        res["AR-1"],
        res["AR-10"],
        res["AR-100"],
        res["AR-S"],
        res["AR-M"],
        res["AR-L"],
    ]
    assert all_res == common_metrics_res
    assert np.allclose(common_metrics_res, sample.mAP)
