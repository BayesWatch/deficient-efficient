'''Residual version of mobilenet.
'''
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


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, conv=Conv):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

    #cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
    #(new) cfg = [64, 128, 256, 512, 512, 512, 1024]

class MobileResNet(nn.Module):
    def __init__(self, block = PreActBlock, num_classes=10, cublock=False, width_factor=1):
        super(MobileResNet, self).__init__()

        if cublock:
            conv = DConv
        else:
            conv = Conv

        cfg = [64, 128, 256, 512, 512, 512, 1024]
        self.in_planes = cfg[0]

        self.conv1 = nn.Conv2d(3, cfg[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, cfg[1], 1, stride=2, conv=conv)
        self.layer2 = self._make_layer(block, cfg[2], 1, stride=2, conv=conv)
        self.layer3 = self._make_layer(block, cfg[3], 1, stride=2, conv=conv)
        self.layer4 = self._make_layer(block, cfg[4], 1, stride=1, conv=conv)
        self.layer5 = self._make_layer(block, cfg[5], 1, stride=1, conv=conv)
        self.layer6 = self._make_layer(block, cfg[6], 1, stride=2, conv=conv)

        self.linear = nn.Linear(cfg[-1], num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, conv):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, conv))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



def test(cublock):
    net = MobileResNet(cublock=cublock)
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())
