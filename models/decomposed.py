# Substitute layer explicitly decomposing the tensors in convolutional layers
# All implemented using tntorch: https://github.com/rballester/tntorch
# All also use a separable design: the low-rank approximate pointwise
# convolution is preceded by a grouped convolution
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tntorch as tn


class TensorTrain(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, rank, stride=1,
            padding=0, dilation=1, groups=1, bias=True):
        assert groups == 1
        super(TensorTrain, self).__init__(in_channels, out_channels, 1, bias=bias)
        self.grouped = nn.Conv2d(in_channels, in_channels,
                kernel_size, stride=stride, padding=padding, dilation=dilation,
                groups=in_channels, bias=False)
        self.rank = rank
        self.tn_weight = tn.Tensor(self.weight.data, ranks_tt=self.rank)
        # delete the original weight
        del self.weight
        # then register the cores of the Tensor Train as parameters
        self.register_cores(self.tn_weight.cores)

    def register_cores(self, cores):
        for i,core in enumerate(cores):
            core_name = 'weight_core_%i'%i
            if hasattr(self, core_name):
                delattr(self, core_name)
            core.requires_grad = True
            self.register_parameter(core_name, nn.Parameter(core))
            # replace Parameter in tn.Tensor object
            self.tn_weight.cores[i] = getattr(self, core_name)

    def reset_parameters(self):
        if hasattr(self, 'tn_weight'):
            # full rank weight tensor
            weight = self.tn_weight.torch()
        else:
            weight = self.weight.data
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        weight.data.uniform_(-stdv, stdv)
        if hasattr(self, 'tn_weight'):
            self.tn_weight = tn.Tensor(weight.data, ranks_tt=self.rank)
            # update cores
            self.register_cores(self.tn_weight.cores)
        else:
            self.weight.data = weight
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        out = self.grouped(x)
        weight = self.tn_weight.torch()
        return F.conv2d(out, weight, self.bias, self.stride, self.padding,
                self.dilation, self.groups)


if __name__ == '__main__':
    X = torch.randn(5,16,32,32)
    tt = TensorTrain(16,16,3,3, bias=False)
    tt.reset_parameters()
    tt.zero_grad()
    y = tt(X)
    l = y.sum()
    l.backward()
    for n,p in tt.named_parameters():
        assert p.requires_grad, n
    assert torch.abs(tt.weight_core_0.grad - tt.tn_weight.cores[0].grad).max() < 1e-5
    # same output on the GPU
    tt, X = tt.cuda(), X.cuda()
    assert torch.abs(tt(X).cpu() - y).max() < 1e-5

