
# Network from https://github.com/davidcpage/cifar10-fast
# Adapted to python < 3.6

import torch.nn as nn


def fastresnet():
    return FastResnet()


def batch_norm(num_channels, bn_bias_init=None, bn_bias_freeze=False,
               bn_weight_init=None, bn_weight_freeze=False):
    m = nn.BatchNorm2d(num_channels)
    if bn_bias_init is not None:
        m.bias.data.fill_(bn_bias_init)
    if bn_bias_freeze:
        m.bias.requires_grad = False
    if bn_weight_init is not None:
        m.weight.data.fill_(bn_weight_init)
    if bn_weight_freeze:
        m.weight.requires_grad = False

    return m


def seq_conv_bn(in_channels, out_channels, conv_kwargs, bn_kwargs):
    if "padding" not in conv_kwargs:
        conv_kwargs["padding"] = 1
    if "stride" not in conv_kwargs:
        conv_kwargs["stride"] = 1
    if "bias" not in conv_kwargs:
        conv_kwargs["bias"] = False
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,
                  kernel_size=3, **conv_kwargs),
        batch_norm(out_channels, **bn_kwargs),
        nn.ReLU(inplace=True)
    )


def conv_bn_elu(in_channels, out_channels, conv_kwargs, bn_kwargs, alpha=1.0):
    if "padding" not in conv_kwargs:
        conv_kwargs["padding"] = 1
    if "stride" not in conv_kwargs:
        conv_kwargs["stride"] = 1
    if "bias" not in conv_kwargs:
        conv_kwargs["bias"] = False
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,
                  kernel_size=3, **conv_kwargs),
        batch_norm(out_channels, **bn_kwargs),
        nn.ELU(alpha=alpha, inplace=True)
    )


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), x.size(1))


class FastResnet(nn.Module):

    def __init__(self, conv_kwargs=None, bn_kwargs=None,
                 conv_bn_fn=seq_conv_bn,
                 final_weight=0.125):
        super(FastResnet, self).__init__()

        conv_kwargs = {} if conv_kwargs is None else conv_kwargs
        bn_kwargs = {} if bn_kwargs is None else bn_kwargs

        self.prep = conv_bn_fn(3, 64, conv_kwargs, bn_kwargs)

        self.layer1 = nn.Sequential(
            conv_bn_fn(64, 128, conv_kwargs, bn_kwargs),
            nn.MaxPool2d(kernel_size=2),
            IdentityResidualBlock(128, 128, conv_kwargs, bn_kwargs, conv_bn_fn=conv_bn_fn)
        )

        self.layer2 = nn.Sequential(
            conv_bn_fn(128, 256, conv_kwargs, bn_kwargs),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer3 = nn.Sequential(
            conv_bn_fn(256, 512, conv_kwargs, bn_kwargs),
            nn.MaxPool2d(kernel_size=2),
            IdentityResidualBlock(512, 512, conv_kwargs, bn_kwargs, conv_bn_fn=conv_bn_fn)
        )

        self.head = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            Flatten(),
        )

        self.final_weight = final_weight

        self.features = nn.Sequential(
            self.prep,
            self.layer1,
            self.layer2,
            self.layer3,
            self.head
        )

        self.classifier = nn.Linear(512, 10, bias=False)

    def forward(self, x):
        f = self.features(x)

        y = self.classifier(f)
        y = y * self.final_weight
        return y


class IdentityResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, conv_kwargs, bn_kwargs,
                 conv_bn_fn=seq_conv_bn):
        super(IdentityResidualBlock, self).__init__()
        self.conv1 = conv_bn_fn(in_channels, out_channels, conv_kwargs, bn_kwargs)
        self.conv2 = conv_bn_fn(out_channels, out_channels, conv_kwargs, bn_kwargs)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


if __name__ == "__main__":

    import torch

    torch.manual_seed(12)

    model = FastResnet(bn_kwargs={"bn_weight_init": 1.0})

    x = torch.rand(4, 3, 32, 32)
    y = model(x)
    print(y.shape)
