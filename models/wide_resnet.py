import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


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
        self.conv = nn.Conv2d(bottleneck, bottleneck, kernel_size=kernel_size, stride=stride, padding=padding,
                                      bias=bias)
        self.bn2= nn.BatchNorm2d(bottleneck)
        self.conv1x1_up = nn.Conv2d(bottleneck, out_planes, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1x1_down(x)))
        out = F.relu(self.bn2(self.conv(out)))
        out = self.conv1x1_up(out)
        return out


class GConvBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, bottleneck, group_split, stride=1, kernel_size=3, padding=1, bias=False):
        super(GConvBottleneck, self).__init__()
        self.conv1x1_down = nn.Conv2d(in_planes, bottleneck, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn1 = nn.BatchNorm2d(bottleneck)
        self.conv = nn.Conv2d(bottleneck, bottleneck, kernel_size=kernel_size, stride=stride, padding=padding,
                                      bias=bias, groups=bottleneck//group_split)
        self.bn2= nn.BatchNorm2d(bottleneck)
        self.conv1x1_up = nn.Conv2d(bottleneck, out_planes, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1x1_down(x)))
        out = F.relu(self.bn2(self.conv(out)))
        out = self.conv1x1_up(out)
        return out


class AConvBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, bottleneck, groups, stride=1, kernel_size=3, padding=1, bias=False):
        super(AConvBottleneck, self).__init__()
        self.conv1x1_down = nn.Conv2d(in_planes, bottleneck, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn1 = nn.BatchNorm2d(bottleneck)
        self.conv = nn.Conv2d(bottleneck, bottleneck, kernel_size=kernel_size, stride=stride, padding=padding,
                                      bias=bias, groups=groups)
        self.bn2= nn.BatchNorm2d(bottleneck)
        self.conv1x1_up = nn.Conv2d(bottleneck, out_planes, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1x1_down(x)))
        out = F.relu(self.bn2(self.conv(out)))
        out = self.conv1x1_up(out)
        return out


class G2B2(GConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(G2B2, self).__init__(in_planes, out_planes, bottleneck = out_planes // 2,group_split = 2,
                                     stride=stride, kernel_size=kernel_size, padding=padding,
                                     bias=bias)

class G4B2(GConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(G4B2, self).__init__(in_planes, out_planes, bottleneck = out_planes // 2,group_split = 4,
                                     stride=stride, kernel_size=kernel_size, padding=padding,
                                     bias=bias)

class G8B2(GConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(G8B2, self).__init__(in_planes, out_planes, bottleneck = out_planes // 2,group_split = 8,
                                     stride=stride, kernel_size=kernel_size, padding=padding,
                                     bias=bias)

class G16B2(GConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(G16B2, self).__init__(in_planes, out_planes, bottleneck = out_planes // 2,group_split = 16,
                                     stride=stride, kernel_size=kernel_size, padding=padding,
                                     bias=bias)


class A2B2(AConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(A2B2, self).__init__(in_planes, out_planes, bottleneck = out_planes // 2,groups = 2,
                                     stride=stride, kernel_size=kernel_size, padding=padding,
                                     bias=bias)

class A4B2(AConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(A4B2, self).__init__(in_planes, out_planes, bottleneck = out_planes // 2,groups = 4,
                                     stride=stride, kernel_size=kernel_size, padding=padding,
                                     bias=bias)

class A8B2(AConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(A8B2, self).__init__(in_planes, out_planes, bottleneck = out_planes // 2,groups= 8,
                                     stride=stride, kernel_size=kernel_size, padding=padding,
                                     bias=bias)

class A16B2(AConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(A16B2, self).__init__(in_planes, out_planes, bottleneck = out_planes // 2,groups = 16,
                                     stride=stride, kernel_size=kernel_size, padding=padding,
                                     bias=bias)


class G2B4(GConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(G2B4, self).__init__(in_planes, out_planes, bottleneck = out_planes // 4,group_split = 2,
                                     stride=stride, kernel_size=kernel_size, padding=padding,
                                     bias=bias)

class G4B4(GConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(G4B4, self).__init__(in_planes, out_planes, bottleneck = out_planes // 4,group_split = 4,
                                     stride=stride, kernel_size=kernel_size, padding=padding,
                                     bias=bias)

class G8B4(GConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(G8B4, self).__init__(in_planes, out_planes, bottleneck = out_planes // 4,group_split = 8,
                                     stride=stride, kernel_size=kernel_size, padding=padding,
                                     bias=bias)

class G16B4(GConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(G16B4, self).__init__(in_planes, out_planes, bottleneck = out_planes // 4,group_split = 16,
                                     stride=stride, kernel_size=kernel_size, padding=padding,
                                     bias=bias)



class ConvB2(ConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(ConvB2, self).__init__(in_planes, out_planes, out_planes//2,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias)


class ConvB4(ConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(ConvB4, self).__init__(in_planes, out_planes, out_planes//4,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias)


class ConvB8(ConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(ConvB8, self).__init__(in_planes, out_planes, out_planes//8,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias)


class ConvB16(ConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(ConvB16, self).__init__(in_planes, out_planes, out_planes//16,
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
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False, groups=None):
        super(DConv, self).__init__()
        # This class replaces BasicConv, as such it assumes the output goes through a BN+ RELU whereas the
        # internal BN + RELU is written explicitly
        self.convdw = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                      bias=bias, groups=in_planes if groups is None else groups)
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        return self.conv1x1(F.relu(self.bn(self.convdw(x))))

class DConvG2(DConv):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConvG2, self).__init__(in_planes, out_planes,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias, groups=in_planes//2)

class DConvG4(DConv):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConvG4, self).__init__(in_planes, out_planes,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias, groups=in_planes//4)

class DConvG8(DConv):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConvG8, self).__init__(in_planes, out_planes,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias, groups=in_planes//8)

class DConvG16(DConv):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConvG16, self).__init__(in_planes, out_planes,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias, groups=in_planes//16)


class DConvA2(DConv):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConvA2, self).__init__(in_planes, out_planes,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias, groups=2)

class DConvA4(DConv):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConvA4, self).__init__(in_planes, out_planes,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias, groups=4)

class DConvA8(DConv):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConvA8, self).__init__(in_planes, out_planes,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias, groups=8)

class DConvA16(DConv):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConvA16, self).__init__(in_planes, out_planes,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias, groups=16)


class DConvBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, bottleneck, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConvBottleneck, self).__init__()
        self.conv1x1_down = nn.Conv2d(in_planes, bottleneck, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn1 = nn.BatchNorm2d(bottleneck)
        self.convdw = nn.Conv2d(bottleneck, bottleneck, kernel_size=kernel_size, stride=stride, padding=padding,
                                      bias=bias, groups=bottleneck)
        self.bn2= nn.BatchNorm2d(bottleneck)
        self.conv1x1_up = nn.Conv2d(bottleneck, out_planes, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1x1_down(x)))
        out = F.relu(self.bn2(self.convdw(out)))
        out = self.conv1x1_up(out)
        return out

class DConvB2(DConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConvB2, self).__init__(in_planes, out_planes, out_planes//2,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias)

class DConvB4(DConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConvB4, self).__init__(in_planes, out_planes, out_planes//4,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias)

class DConvB8(DConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConvB8, self).__init__(in_planes, out_planes, out_planes//8,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias)

class DConvB16(DConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConvB16, self).__init__(in_planes, out_planes, out_planes//16,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias)


class DConv3D(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConv3D, self).__init__()
        # Separable conv approximating the 1x1 with a 3x3 conv3d
        self.convdw = nn.Conv2d(in_planes,in_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                      bias=bias,groups=in_planes)
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv3d = nn.Conv3d(1, out_planes, kernel_size=(3,1,1), stride=1, padding=(1,0,0), bias=bias)

    def forward(self, x):
        o = F.relu(self.bn(self.convdw(x)))
        o = o.unsqueeze(1)
        #n, c, d, w, h = o.size()
        return self.conv3d(o).mean(2)


def conv_function(convtype):
    if convtype == 'Conv':
        conv = Conv
    elif convtype == 'DConv':
        conv = DConv
    elif convtype == 'DConvG2':
        conv = DConvG2
    elif convtype == 'DConvG4':
        conv = DConvG4
    elif convtype == 'DConvG8':
        conv = DConvG8
    elif convtype == 'DConvG16':
        conv = DConvG16
    elif convtype == 'DConvA2':
        conv = DConvA2
    elif convtype == 'DConvA4':
        conv = DConvA4
    elif convtype == 'DConvA8':
        conv = DConvA8
    elif convtype == 'DConvA16':
        conv = DConvA16
    elif convtype == 'Conv2x2':
        conv = Conv2x2
    elif convtype == 'ConvB2':
        conv = ConvB2
    elif convtype == 'ConvB4':
        conv = ConvB4
    elif convtype == 'ConvB8':
        conv = ConvB8
    elif convtype == 'ConvB16':
        conv = ConvB16
    elif convtype == 'DConvB2':
        conv = DConvB2
    elif convtype == 'DConvB4':
        conv = DConvB4
    elif convtype == 'DConvB8':
        conv = DConvB8
    elif convtype == 'DConvB16':
        conv = DConvB16
    elif convtype == 'DConv3D':
        conv = DConv3D
    elif convtype =='G2B2':
        conv = G2B2
    elif convtype =='G4B2':
        conv = G4B2
    elif convtype =='G8B2':
        conv = G8B2
    elif convtype =='G16B2':
        conv = G16B2
    elif convtype =='G2B4':
        conv = G2B4
    elif convtype =='G4B4':
        conv = G4B4
    elif convtype =='G8B4':
        conv = G8B4
    elif convtype =='G16B4':
        conv = G16B4
    elif convtype =='A2B2':
        conv = A2B2
    elif convtype =='A4B2':
        conv = A4B2
    elif convtype =='A8B2':
        conv = A8B2
    elif convtype =='A16B2':
        conv = A16B2

    else:
        print(convtype)
        assert 1==0, 'conv % not recognised'
    return conv


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, conv=Conv, xy=None):
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


class SqueezeExciteBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, conv=Conv, xy=None):
        super(SqueezeExciteBlock, self).__init__()
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

        self.fc1 = nn.Conv2d(out_planes, out_planes // 16, kernel_size=1)
        self.fc2 = nn.Conv2d(out_planes // 16, out_planes, kernel_size=1)

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation

        out = out * w

        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class AttentionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, conv=Conv, xy=None):

        assert xy is not None, 'need to know spatial size'

        super(AttentionBlock, self).__init__()
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
        self.fc1 = nn.Linear(xy, xy//64)
        self.fc2 = nn.Linear(xy//64, xy)

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        # Squeeze
        w = out.mean(1, keepdim=True)
        #Regrettable reshaping
        w = w.view(w.size(0),-1)
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        w = w.view(out.size(0),1,out.size(2),out.size(3))
        # Excitation

        out = out * w

        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class BottleBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, conv=Conv, xy=None):
        super(BottleBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_planes)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.conv2 = conv(out_planes, out_planes, kernel_size=3, stride=1,
        #                        padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.conv1(out if self.equalInOut else x)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = out
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)




class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, conv = Conv, xy=None):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, conv,xy)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, conv, xy):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, conv, xy))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):

    def __init__(self, depth, widen_factor=1, num_classes=10, dropRate=0.0, convtype='Conv',blocktype='Basic'):
        super(WideResNet, self).__init__()

        #if convtype is a string we want every block to have the same type, if not, assume custom

        if isinstance(convtype,str):

            conv1 = conv_function(convtype)
            conv2 = conv_function(convtype)
            conv3 = conv_function(convtype)

        else:
            conv1 = conv_function(convtype[0])
            conv2 = conv_function(convtype[1])
            conv3 = conv_function(convtype[2])


        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        nChannels = [int(a) for a in nChannels]

        assert((depth - 4) % 6 == 0)
        n = (depth - 4) // 6

        if blocktype =='Basic':
            block = BasicBlock
        elif blocktype =='Bottle':
            block = BottleBlock
        elif blocktype == 'AT':
            block = AttentionBlock
        elif blocktype =='SE':
            block = SqueezeExciteBlock


        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, conv1, xy= 32*32)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, conv2, xy= 16*16)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, conv3, xy= 8*8)
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


class WideResNetAT3(WideResNet):
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

class WideResNetIntMap(WideResNet):
    def __init__(self):
        super(WideResNetIntMap, self).__init__()

        self.map1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.map2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.map3 = nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv1_out = self.conv1(x)
        block1_out = self.block1(conv1_out)
        block2_out = self.block2(block1_out)
        block3_out = self.block3(block2_out)
        out = self.relu(self.bn1(block3_out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out  = self.fc(out)
        return out, (self.map1(block1_out),self.map2(block2_out),self.map3(block3_out))


class WideResNetAT(nn.Module):
    def __init__(self, depth, widen_factor=1, num_classes=10, dropRate=0.0, convtype='Conv', s = 1,blocktype='Basic'):
        super(WideResNetAT, self).__init__()

        if isinstance(convtype,str):

            conv1 = conv_function(convtype)
            conv2 = conv_function(convtype)
            conv3 = conv_function(convtype)

        else:
            conv1 = conv_function(convtype[0])
            conv2 = conv_function(convtype[1])
            conv3 = conv_function(convtype[2])

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        nChannels = [int(a) for a in nChannels]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        if blocktype =='Basic':
            block = BasicBlock
        elif blocktype =='Bottle':
            block = BottleBlock

        assert n % s == 0, 'n mod s must be zero'

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = torch.nn.ModuleList()
        for i in range(s):
            self.block1.append(NetworkBlock(int(n//s), nChannels[0] if i == 0 else nChannels[1],
                                            nChannels[1], block, 1, dropRate, conv1))

        # 2nd block
        self.block2 = torch.nn.ModuleList()
        for i in range(s):
            self.block2.append(NetworkBlock(int(n//s), nChannels[1] if i == 0 else nChannels[2],
                                            nChannels[2], block, 2 if i == 0 else 1, dropRate, conv2))
        # 3rd block
        self.block3 = torch.nn.ModuleList()
        for i in range(s):
            self.block3.append(NetworkBlock(int(n//s), nChannels[2] if i == 0 else nChannels[3],
                                            nChannels[3], block, 2 if i == 0 else 1, dropRate, conv3))
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
