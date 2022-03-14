import os
from pathlib import Path

import brevitas.nn as qnn
import torch
import torch.nn as nn
from pact import PACTReLU
from torchvision import datasets, models
from torchvision.transforms import Compose, Normalize, Pad, RandomCrop, RandomHorizontalFlip, ToTensor

train_transform = Compose(
    [
        Pad(4),
        RandomCrop(32, fill=128),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

test_transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def get_train_test_datasets(path):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)
        download = True
    else:
        download = True if len(os.listdir(path)) < 1 else False

    train_ds = datasets.CIFAR10(root=path, train=True, download=download, transform=train_transform)
    test_ds = datasets.CIFAR10(root=path, train=False, download=False, transform=test_transform)

    return train_ds, test_ds


def get_model(name):
    __dict__ = globals()

    if name in models.__dict__:
        fn = models.__dict__[name]
    elif name in ["resnet18_QAT_8b", "resnet18_QAT_6b", "resnet18_QAT_5b", "resnet18_QAT_4b"]:
        fn = __dict__[name]
    else:
        raise RuntimeError("Unknown model name {}".format(name))

    return fn(num_classes=10)


# Below code is taken from https://discuss.pytorch.org/t/evaluator-returns-nan/107972/3


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, weight_bit_width=8):
    """3x3 convolution with padding"""
    return qnn.QuantConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
        weight_bit_width=weight_bit_width,
    )


def conv1x1(in_planes, out_planes, stride=1, weight_bit_width=8):
    """1x1 convolution"""
    return qnn.QuantConv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False, weight_bit_width=weight_bit_width
    )


def make_PACT_relu(bit_width=8):
    relu = qnn.QuantReLU(bit_width=bit_width)
    relu.act_impl = PACTReLU()
    return relu


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        bit_width=8,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, weight_bit_width=bit_width)
        self.bn1 = norm_layer(planes)
        self.relu = make_PACT_relu(bit_width=bit_width)
        self.conv2 = conv3x3(planes, planes, weight_bit_width=bit_width)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        bit_width=8,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, weight_bit_width=bit_width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, weight_bit_width=bit_width)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, weight_bit_width=bit_width)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = make_PACT_relu(bit_width=bit_width)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_QAT_Xb(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        bit_width=8,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = qnn.QuantConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = make_PACT_relu()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], bit_width=bit_width)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], bit_width=bit_width
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], bit_width=bit_width
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], bit_width=bit_width
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # qnn.QuantConv2d includes nn.Conv2d inside.
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, bit_width=8):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, weight_bit_width=bit_width),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                bit_width=bit_width,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    bit_width=bit_width,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet_QAT_Xb(block, layers, **kwargs):
    model = ResNet_QAT_Xb(block, layers, **kwargs)
    return model


def resnet18_QAT_8b(*args, **kwargs):
    return _resnet_QAT_Xb(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet18_QAT_6b(*args, **kwargs):
    return _resnet_QAT_Xb(BasicBlock, [2, 2, 2, 2], bit_width=6, **kwargs)


def resnet18_QAT_5b(*args, **kwargs):
    return _resnet_QAT_Xb(BasicBlock, [2, 2, 2, 2], bit_width=5, **kwargs)


def resnet18_QAT_4b(*args, **kwargs):
    return _resnet_QAT_Xb(BasicBlock, [2, 2, 2, 2], bit_width=4, **kwargs)
