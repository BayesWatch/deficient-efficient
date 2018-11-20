# HashedNet Convolutional Layer: https://arxiv.org/abs/1504.04788
import torch
import torch.nn as nn
import torch.nn.functional as F


class HashedConv2d(nn.Conv2d):
    """Conv2d with the weights of the convolutional filters parameterised using
    a budgeted subset of parameters and random indexes to place those
    parameters in the weight tensor."""
    def __init__(self, in_channels, out_channels, kernel_size, budget,
            stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(HashedConv2d, self).__init__(in_channels, out_channels,
                kernel_size, stride=stride, padding=padding, dilation=dilation,
                groups=groups, bias=True)
        # grab budgeted subset of the weights
        self.weight_size = self.weight.size()
        budgeted = self.weight.data.view(-1)[:budget]
        del self.weight
        # register non-budgeted weights
        self.register_parameter('weight', nn.Parameter(budgeted))
        # precompute random index matrix
        idxs = torch.randint(high=budget, size=self.weight_size).long()
        idxs = idxs.view(-1)
        # register indexes as a buffer
        self.register_buffer('idxs', idxs)
        #self.W = self.weight[self.idxs].cuda()

    def forward(self, x):
        # index to make weight matrix
        #W = self.weight[self.idxs]
        W = self.weight.index_select(0, self.idxs).view(self.weight_size)
        # complete forward pass as normal
        return F.conv2d(x, W, self.bias, self.stride, self.padding,
                self.dilation, self.groups)

if __name__ == '__main__':
    from timeit import timeit
    setup = "from __main__ import HashedConv2d; import torch; X = torch.randn(128, 256, 28, 28).cuda(); conv = HashedConv2d(256, 512, 3, 1000, bias=False).cuda()"
    print("HashedConv2d: ", timeit("_ = conv(X)", setup=setup, number=100))
    setup = "import torch.nn as nn; import torch; X = torch.randn(128, 256, 28, 28).cuda(); conv = nn.Conv2d(256, 512, 3, bias=False).cuda()"
    print("Conv2d: ", timeit("_ = conv(X)", setup=setup, number=100))

