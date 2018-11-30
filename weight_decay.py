import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_acdc.layers import FastStackedConvACDC
from tqdm import tqdm
from collections import OrderedDict

def substitute_jitter(l, logsigma):
    new_sd = OrderedDict()
    for n,p in l.named_parameters():
        new_sd[n] = torch.randn(p.size())*torch.exp(logsigma)
    l.load_state_dict(new_sd)
    return new_sd

def grad_sigma(l):
    # gradient wrt sigma is just the gradient times the value of each parameter
    total_grad = torch.Tensor([0.])
    for p in l.parameters():
        total_grad = torch.sum(p.grad*p)
    return total_grad

if __name__ == '__main__':
    X = torch.eye(4).float().view(4,4,1,1)
    wd = 5e-4
    N = 1000
    #X = torch.randn(5,4,32,32)
    l = FastStackedConvACDC(4,4,1,12,bias=False)
    logsigma = torch.Tensor([math.sqrt(1.)])
    logsigma.requires_grad = True
    l.eval()
    optimizer = optim.SGD([logsigma], lr=0.001, momentum=0.9)
    with tqdm(total=N) as pbar:
        for i in range(N):
            l.zero_grad()
            if logsigma.grad is not None:
                logsigma.grad.zero_()
            substitutes = substitute_jitter(l, logsigma)
            W = l(X)
            loss = F.mse_loss(torch.log(W.var()),torch.log(torch.Tensor([5e-4])))
            loss.backward()
            torch.autograd.backward(substitutes.values(), grad_tensors=[p.grad for p in l.parameters()])
            optimizer.step()
            pbar.update(1)
            pbar.set_description("Loss/Sigma: %f %f"%(loss.item(), torch.exp(logsigma).item()))
