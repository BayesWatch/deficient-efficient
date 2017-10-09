import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyinn.modules import Conv2dDepthwise

import math

class Conv(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(Conv, self).__init__()
        # Dumb normal conv, pointlessly incorporated into a class
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)

class Conv2x2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=2, padding=1, bias=False):
        super(Conv2x2, self).__init__()
        # Dilated 2x2 convs
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=2,
                              stride=stride, padding=padding, bias=bias, dilation=2)

    def forward(self, x):
        return self.conv(x)


class DConv(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConv, self).__init__()
        # This class replaces BasicConv, as such it assumes the output goes through a BN+ RELU whereas the
        # internal BN + RELU is written explicitly
        self.convdw = Conv2dDepthwise(channels=in_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                      bias=bias)
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        return self.conv1x1(F.relu(self.bn(self.convdw(x))))


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

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, separable=False, twobytwo=False):
        super(WideResNet, self).__init__()

        conv = DConv if separable else (Conv2x2 if twobytwo else Conv)

        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        nChannels = [int(a) for a in nChannels]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, conv)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, conv)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, conv)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


class WideResNetInt(WideResNet):
    # def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
    #     super(WideResNet, self).__init__(depth, num_classes, widen_factor, dropRate)
    def forward(self, x):
        conv1_out = self.conv1(x)
        block1_out = self.block1(conv1_out)
        block2_out = self.block2(block1_out)
        block3_out = self.block3(block2_out)
        out = self.relu(self.bn1(block3_out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out  = self.fc(out)
        return out, (block1_out,block2_out,block3_out)




class WideResNet6(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, separable=False, twobytwo=False):
        super(WideResNet6, self).__init__()
        #This is the same as a normal wideresnet expect the 3 blocks are each split in two to allow us to pick out
        #extra activations with ease..

        conv = DConv if separable else (Conv2x2 if twobytwo else Conv)

        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        nChannels = [int(a) for a in nChannels]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        n = n/2
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, conv)
        self.block1a = NetworkBlock(n, nChannels[1], nChannels[1], block, 1, dropRate, conv)

        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, conv)
        self.block2a = NetworkBlock(n, nChannels[2], nChannels[2], block, 1, dropRate, conv)

        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, conv)
        self.block3a = NetworkBlock(n, nChannels[3], nChannels[3], block, 1, dropRate, conv)

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block1a(out)
        out = self.block2(out)
        out = self.block2a(out)
        out = self.block3(out)
        out = self.block3a(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


class WideResNet6Int(WideResNet6):
    # def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
    #     super(WideResNet, self).__init__(depth, num_classes, widen_factor, dropRate)
    def forward(self, x):
        out = self.conv1(x)
        out1 = self.block1(out)
        out1a = self.block1a(out1)
        out2 = self.block2(out1a)
        out2a = self.block2a(out2)
        out3 = self.block3(out2a)
        out3a = self.block3a(out3)
        out = self.relu(self.bn1(out3a))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        return out, (out1,out1a,out2,out2a,out3,out3a)