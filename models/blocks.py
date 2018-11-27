# blocks and convolution definitions
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

if __name__ == 'blocks' or __name__ == '__main__':
    from hashed import HashedConv2d, SeparableHashedConv2d
else:
    from .hashed import HashedConv2d, SeparableHashedConv2d

def HashedDecimate(in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=False):
    # Hashed Conv2d using 1/10 the original parameters
    original_params = out_channels*in_channels*kernel_size*kernel_size // groups
    budget = original_params//10
    return HashedConv2d(in_channels, out_channels, kernel_size, budget,
            stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias)

def SepHashedDecimate(in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=False):
    # Hashed Conv2d using 1/10 the original parameters
    original_params = out_channels*in_channels*kernel_size*kernel_size // groups
    budget = original_params//10
    return SeparableHashedConv2d(in_channels, out_channels, kernel_size,
            budget, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)


from pytorch_acdc.layers import FastStackedConvACDC


def ACDC(in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=False):
    return FastStackedConvACDC(in_channels, out_channels, kernel_size, 12,
            stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias)

def OriginalACDC(in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=False):
    return FastStackedConvACDC(in_channels, out_channels, kernel_size, 12,
            stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias, original=True)


class GenericLowRank(nn.Module):
    """A generic low rank layer implemented with a linear bottleneck, using two
    Conv2ds in sequence. Preceded by a depthwise grouped convolution in keeping
    with the other low-rank layers here."""
    def __init__(self, in_channels, out_channels, kernel_size, rank, stride=1,
        padding=0, dilation=1, groups=1, bias=False):
        assert groups == 1
        super(GenericLowRank, self).__init__()
        self.grouped = nn.Conv2d(in_channels, in_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation,
                groups=in_channels, bias=False)
        self.contract = nn.Conv2d(in_channels, rank, 1, bias=False)
        self.expand = nn.Conv2d(rank, out_channels, 1, bias=bias)
    def forward(self, x):
        x = self.grouped(x)
        x = self.contract(x)
        return self.expand(x)


class DepthwiseSep(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True):
        super(DepthwiseSep, self).__init__()
        assert groups == 1
        self.grouped = nn.Conv2d(in_channels, in_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation,
                groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        out = self.grouped(x)
        return self.pointwise(out)


class Conv(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(Conv, self).__init__()
        # Dumb normal conv incorporated into a class
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)


def conv_function(convtype):
    
    # if convtype contains an underscore, it must have a hyperparam in it
    if "_" in convtype:
        convtype, hyperparam = convtype.split("_")
        if convtype == 'ACDC':
            # then hyperparam controls how many layers in each conv
            n_layers = int(round(float(hyperparam)))
            def conv(in_channels, out_channels, kernel_size, stride=1,
                    padding=0, dilation=1, groups=1, bias=False):
                return FastStackedConvACDC(in_channels, out_channels,
                        kernel_size, n_layers, stride=stride,
                        padding=padding, dilation=dilation, groups=groups,
                        bias=bias)
        elif convtype == 'Hashed':
            # then hyperparam controls relative budget for each layer
            budget_scale = float(hyperparam)
            def conv(in_channels, out_channels, kernel_size, stride=1,
                    padding=0, dilation=1, groups=1, bias=False):
                # Hashed Conv2d using 1/10 the original parameters
                original_params = out_channels*in_channels*kernel_size*kernel_size // groups
                budget = int(original_params*budget_scale)
                return HashedConv2d(in_channels, out_channels, kernel_size,
                        budget, stride=stride, padding=padding,
                        dilation=dilation, groups=groups, bias=bias)
        elif convtype == 'Generic':
            rank_scale = float(hyperparam)
            def conv(in_channels, out_channels, kernel_size, stride=1,
                    padding=0, dilation=1, groups=1, bias=False):
                full_rank = max(in_channels,out_channels)
                rank = int(rank_scale*full_rank)
                return GenericLowRank(in_channels, out_channels, kernel_size,
                        rank, stride=stride, padding=padding,
                        dilation=dilation, groups=groups, bias=bias)
    else:
        if convtype == 'Conv':
            conv = Conv
        elif convtype =='ACDC':
            conv = ACDC
        elif convtype =='OriginalACDC':
            conv = OriginalACDC
        elif convtype == 'HashedDecimate':
            conv = HashedDecimate
        elif convtype == 'SepHashedDecimate':
            conv = SepHashedDecimate
        elif convtype == 'Sep':
            conv = DepthwiseSep
        else:
            raise ValueError('Conv "%s" not recognised'%convtype)
    return conv


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, conv=Conv):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


# modified from torchvision
class Bottleneck(nn.Module):
    """Bottleneck architecture block for ResNet"""
    def __init__(self, inplanes, planes, ConvClass, stride=1, downsample=None, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        pointwise = lambda i,o: ConvClass(i, o, kernel_size=1, padding=0,
                bias=False)
        self.conv1 = pointwise(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = ConvClass(planes, planes, kernel_size=3, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = pointwise(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, conv = Conv):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, conv)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, conv):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, conv))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

if __name__ == '__main__':
    X = torch.randn(5,3,32,32)
    # sanity of generic low-rank layer
    generic = GenericLowRank(3, 32, 3, 1)
    for p in generic.parameters():
        print(p.size())
    out = generic(X)
    print(out.size())
