# network definition
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from collections import OrderedDict

# wildcard import for legacy reasons
if __name__ == '__main__':
    from blocks import *
else:
    from .blocks import *

def parse_options(convtype, blocktype):
    # legacy cmdline argument parsing
    if isinstance(convtype,str):
        conv = conv_function(convtype)
    else:
        raise NotImplementedError("Tuple convolution specification no longer supported.")

    if blocktype =='Basic':
        block = BasicBlock
    elif blocktype =='Bottle':
        block = BottleBlock
    elif blocktype =='Old':
        block = OldBlock
    else:
        block = None
    return conv, block


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, conv, block, num_classes=10, dropRate=0.0, s = 1):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        nChannels = [int(a) for a in nChannels]
        assert ((depth - 4) % 6 == 0) # why?
        n = (depth - 4) // 6

        assert n % s == 0, 'n mod s must be zero'

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = torch.nn.ModuleList()
        for i in range(s):
            self.block1.append(NetworkBlock(int(n//s), nChannels[0] if i == 0 else nChannels[1],
                                            nChannels[1], block, 1, dropRate, conv))
        # 2nd block
        self.block2 = torch.nn.ModuleList()
        for i in range(s):
            self.block2.append(NetworkBlock(int(n//s), nChannels[1] if i == 0 else nChannels[2],
                                            nChannels[2], block, 2 if i == 0 else 1, dropRate, conv))
        # 3rd block
        self.block3 = torch.nn.ModuleList()
        for i in range(s):
            self.block3.append(NetworkBlock(int(n//s), nChannels[2] if i == 0 else nChannels[3],
                                            nChannels[3], block, 2 if i == 0 else 1, dropRate, conv))
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        # normal is better than uniform initialisation
        # this should really be in `self.reset_parameters`
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                try:
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                except AttributeError:
                    pass
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def grouped_parameters(self):
        # iterate over parameters and separate those in ACDC layers
        lowrank_params, other_params = [], []
        for n,p in self.named_parameters():
            if 'A' in n or 'D' in n:
                lowrank_params.append(p)
            elif 'grouped' in n:
                lowrank_params.append(p)
            elif 'hashed' in n:
                lowrank_params.append(p)
            else:
                other_params.append(p)
        return [{'params': lowrank_params, 'weight_decay': 8.8e-6},
                {'params': other_params}] 

    def forward(self, x):
        activations = []
        out = self.conv1(x)
        #activations.append(out)

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


class ResNet(nn.Module):

    def __init__(self, ConvClass, layers, block=Bottleneck, widen=1,
            num_classes=1000, expansion=4):
        self.expansion = expansion
        super(ResNet, self).__init__()
        self.Conv = ConvClass
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64*widen, layers[0])
        self.layer2 = self._make_layer(block, 128*widen, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256*widen, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512*widen, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*widen * self.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * self.expansion:
            downsample = nn.Sequential(OrderedDict([
                ('conv', self.Conv(self.inplanes, planes * self.expansion,
                    kernel_size=1, stride=stride, padding=0, bias=False)),
                ('bn', nn.BatchNorm2d(planes * self.expansion))
            ]))

        layers = []
        layers.append(block(self.inplanes, planes, self.Conv, stride, downsample, self.expansion))
        self.inplanes = planes * self.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.Conv, expansion=self.expansion))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        intermediates = []
        x = self.layer1(x)
        intermediates.append(x)
        x = self.layer2(x)
        intermediates.append(x)
        x = self.layer3(x)
        intermediates.append(x)
        x = self.layer4(x)
        intermediates.append(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, intermediates


def WRN_50_2(Conv, Block=None):
    assert Block is None
    return ResNet(Conv, [3, 4, 6, 3], widen=2, expansion=2)

def test():
    net = WRN_50_2(Conv)
    x = torch.randn(1,3,224,224)
    y, _ = net(Variable(x))
    print(y.size())

if __name__ == '__main__':
    test()
