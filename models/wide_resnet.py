import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

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


class ConvBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, bottleneck, stride=1, kernel_size=3, padding=1, bias=False):
        super(ConvBottleneck, self).__init__()
        self.conv1x1_down = nn.Conv2d(in_planes, bottleneck, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn1 = nn.BatchNorm2d(bottleneck)
        self.conv = nn.Conv2d(channels=bottleneck, kernel_size=kernel_size, stride=stride, padding=padding,
                                      bias=bias)
        self.bn2= nn.BatchNorm2d(bottleneck)
        self.conv1x1_up = nn.Conv2d(bottleneck, out_planes, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1x1_down(x)))
        out = F.relu(self.bn2(self.conv(out)))
        out = self.conv1x1_up(out)
        return out


class ConvB2(ConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(ConvB2, self).__init__(in_planes, out_planes, out_planes/2,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias)


class ConvB4(ConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(ConvB4, self).__init__(in_planes, out_planes, out_planes/4,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias)


class ConvB8(ConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(ConvB8, self).__init__(in_planes, out_planes, out_planes/8,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias)

class ConvB16(ConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(ConvB16, self).__init__(in_planes, out_planes, out_planes/16,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias)


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


class DConvBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, bottleneck, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConvBottleneck, self).__init__()
        self.conv1x1_down = nn.Conv2d(in_planes, bottleneck, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn1 = nn.BatchNorm2d(bottleneck)
        self.convdw = Conv2dDepthwise(channels=bottleneck, kernel_size=kernel_size, stride=stride, padding=padding,
                                      bias=bias)
        self.bn2= nn.BatchNorm2d(bottleneck)
        self.conv1x1_up = nn.Conv2d(bottleneck, out_planes, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1x1_down(x)))
        out = F.relu(self.bn2(self.convdw(out)))
        out = self.conv1x1_up(out)
        return out

class DConvB2(DConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConvB2, self).__init__(in_planes, out_planes, out_planes/2,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias)

class DConvB4(DConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConvB4, self).__init__(in_planes, out_planes, out_planes/4,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias)

class DConvB8(DConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConvB8, self).__init__(in_planes, out_planes, out_planes/8,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias)


class DConv3D(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConv3D, self).__init__()
        # Separable conv approximating the 1x1 with a 3x3 conv3d
        self.convdw = Conv2dDepthwise(channels=in_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                      bias=bias)
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv3d = nn.Conv3d(1, out_planes, kernel_size=(3,1,1), stride=1, padding=(1,0,0), bias=bias)

    def forward(self, x):
        o = F.relu(self.bn(self.convdw(x)))
        o = o.unsqueeze(1)
        #n, c, d, w, h = o.size()
        return self.conv3d(o).mean(2)


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

    def __init__(self, depth, widen_factor=1, num_classes=10, dropRate=0.0, convtype='Conv'):
        super(WideResNet, self).__init__()

        if convtype =='Conv':
            conv = Conv
        elif convtype =='DConv':
            conv = DConv
        elif convtype =='Conv2x2':
            conv = Conv2x2
        elif convtype =='ConvB2':
            conv = ConvB2
        elif convtype =='ConvB4':
            conv = ConvB4
        elif convtype =='ConvB8':
            conv = ConvB8
        elif convtype == 'ConvB16':
            conv = ConvB16
        elif convtype =='DConvB2':
            conv = DConvB2
        elif convtype =='DConvB4':
            conv = DConvB4
        elif convtype =='DConvB8':
            conv = DConvB8
        elif convtype =='DConv3D':
            conv = DConv3D


        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        nChannels = [int(a) for a in nChannels]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
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

WideResNetAT3 = WideResNetInt

class WideResNetAT(nn.Module):
    def __init__(self, depth, widen_factor=1, num_classes=10, dropRate=0.0, convtype='Conv', s = 1):
        super(WideResNetAT, self).__init__()

        if convtype == 'Conv':
            conv = Conv
        elif convtype == 'DConv':
            conv = DConv
        elif convtype == 'Conv2x2':
            conv = Conv2x2
        elif convtype == 'DConvB2':
            conv = DConvB2
        elif convtype == 'DConvB4':
            conv = DConvB4
        elif convtype == 'DConvB8':
            conv = DConvB8
        elif convtype == 'DConv3D':
            conv = DConv3D

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        nChannels = [int(a) for a in nChannels]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6


        assert n % s == 0, 'n mod s must be zero'

        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = torch.nn.ModuleList()
        for i in range(s):
            self.block1.append(NetworkBlock(n/s, nChannels[0] if i == 0 else nChannels[1],
                                            nChannels[1], block, 1, dropRate, conv))

        # 2nd block
        self.block2 = torch.nn.ModuleList()
        for i in range(s):
            self.block2.append(NetworkBlock(n/s, nChannels[1] if i == 0 else nChannels[2],
                                            nChannels[2], block, 2 if i == 0 else 1, dropRate, conv))
        # 3rd block
        self.block3 = torch.nn.ModuleList()
        for i in range(s):
            self.block3.append(NetworkBlock(n/s, nChannels[2] if i == 0 else nChannels[3],
                                            nChannels[3], block, 2 if i == 0 else 1, dropRate, conv))
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
        activations = []
        out = self.conv1(x)

        for sub_block in self.block1:
            out = sub_block(out)
            activations.append(out)

        for sub_block in self.block2:
            out = sub_block(out)
            activations.append(out)

        for sub_block in self.block3:
            out = sub_block(out)
            activations.append(out)

        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out), activations











def test():
    net = WideResNet(40,2, convtype='DConv3D')
    x = torch.randn(1,3,32,32)
    y = net(Variable(x))
    print(y.size())

if __name__ == '__main__':
    test()
