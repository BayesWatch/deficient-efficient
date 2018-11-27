# Substitute layer explicitly decomposing the tensors in convolutional layers
# All implemented using tntorch: https://github.com/rballester/tntorch
# All also use a separable design: the low-rank approximate pointwise
# convolution is preceded by a grouped convolution
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tntorch as tn


class TnTorchConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, rank,
            TnConstructor, stride=1, padding=0, dilation=1, groups=1,
            bias=True):
        self.TnConstructor = TnConstructor
        assert groups == 1
        super(TnTorchConv2d, self).__init__(in_channels, out_channels, 1, bias=bias)
        if max(self.kernel_size) > 1:
            self.grouped = nn.Conv2d(in_channels, in_channels,
                    kernel_size, stride=stride, padding=padding, dilation=dilation,
                    groups=in_channels, bias=False)
        self.rank = rank
        self.tn_weight = self.TnConstructor(self.weight.data.squeeze(), ranks=self.rank)
        # delete the original weight
        del self.weight
        # then register the cores of the Tensor Train as parameters
        self.register_tnparams(self.tn_weight.cores, self.tn_weight.Us)

    def register_tnparams(self, cores, Us):
        cores = [] if all([c is None for c in cores]) else cores
        Us = [] if all([u is None for u in Us]) else Us
        # tensor train or cp cores
        for i,core in enumerate(cores):
            core_name = 'weight_core_%i'%i
            if hasattr(self, core_name):
                delattr(self, core_name)
            core.requires_grad = True
            self.register_parameter(core_name, nn.Parameter(core))
            # replace Parameter in tn.Tensor object
            self.tn_weight.cores[i] = getattr(self, core_name)
        for i, u in enumerate(Us):
            u_name = 'weight_u_%i'%i
            if hasattr(self, u_name):
                delattr(self, u_name)
            u.requires_grad = True
            self.register_parameter(u_name, nn.Parameter(u))
            # replace Parameter in tn.Tensor object
            self.tn_weight.Us[i] = getattr(self, u_name)

    def conv_weight(self):
        weight = self.tn_weight.torch()
        n,d = weight.size()
        return weight.view(n,d,1,1)

    def reset_parameters(self):
        if hasattr(self, 'tn_weight'):
            # full rank weight tensor
            weight = self.conv_weight()
        else:
            weight = self.weight.data
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        weight.data.uniform_(-stdv, stdv)
        if hasattr(self, 'tn_weight'):
            self.tn_weight = self.TnConstructor(weight.data.squeeze(), ranks=self.rank)
            # update cores
            self.register_tnparams(self.tn_weight.cores, self.tn_weight.Us)
        else:
            self.weight.data = weight
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if hasattr(self, 'grouped'):
            out = self.grouped(x)
        else:
            out = x
        weight = self.conv_weight()
        return F.conv2d(out, weight, self.bias, self.stride, self.padding,
                self.dilation, self.groups)

    def extra_repr(self):
        t = self.tn_weight
        full = t.torch()
        extra = []
        extra.append(t.__repr__())
        extra.append('Compression ratio: {}/{} = {:g}'.format(full.numel(), t.numel(), full.numel() / t.numel()))
        extra.append('Relative error: %f'%tn.relative_error(full, t))
        extra.append('RMSE: %f'%tn.rmse(full, t))
        extra.append('R^2: %f'%tn.r_squared(full, t))
        return "\n".join(extra)


class TensorTrain(TnTorchConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, rank, stride=1,
            padding=0, dilation=1, groups=1, bias=True):
        def TT(tensor, ranks):
            return tn.Tensor(tensor, ranks_tt=ranks)
        super(TensorTrain, self).__init__(in_channels, out_channels, kernel_size, rank,
            TT, stride=1, padding=0, dilation=1, groups=1,
            bias=True)


class Tucker(TnTorchConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, rank, stride=1,
            padding=0, dilation=1, groups=1, bias=True):
        def tucker(tensor, ranks):
            return tn.Tensor(tensor, ranks_tucker=ranks)
        super(Tucker, self).__init__(in_channels, out_channels, kernel_size, rank,
            tucker, stride=1, padding=0, dilation=1, groups=1,
            bias=True)


class CP(TnTorchConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, rank, stride=1,
            padding=0, dilation=1, groups=1, bias=True):
        def cp(tensor, ranks):
            return tn.Tensor(tensor, ranks_cp=ranks)
        super(CP, self).__init__(in_channels, out_channels, kernel_size, rank,
            cp, stride=1, padding=0, dilation=1, groups=1,
            bias=True)


if __name__ == '__main__':
    for ConvClass in [TensorTrain, Tucker, CP]:
        X = torch.randn(5,16,32,32)
        tnlayer = ConvClass(16,16,3,3, bias=False)
        tnlayer.reset_parameters()
        print(tnlayer)
        tnlayer.zero_grad()
        y = tnlayer(X)
        l = y.sum()
        l.backward()
        for n,p in tnlayer.named_parameters():
            assert p.requires_grad, n
        assert torch.abs(tnlayer.weight_core_0.grad - tnlayer.tn_weight.cores[0].grad).max() < 1e-5
        # same output on the GPU
        tnlayer, X = tnlayer.cuda(), X.cuda()
        assert torch.abs(tnlayer(X).cpu() - y).max() < 1e-5

