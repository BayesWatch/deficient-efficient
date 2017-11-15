import torch
import torch.nn.functional as F
from models import *

def distillation(y, teacher_scores, labels, T, alpha):
    return F.kl_div(F.log_softmax(y/T), F.softmax(teacher_scores/T)) * (T*T * 2. * alpha)\
           + F.cross_entropy(y, labels) * (1. - alpha)


def se(x):
    return F.normalize(x.pow(2).mean(-1).mean(-1).view(x.size(0), -1))


def se_loss(x, y):
    return (se(x)-se(y)).pow(2).mean()


def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()


def l1_loss(x):
    return torch.abs(x).mean()


def get_no_params(net, verbose=True):

    params = net.state_dict()
    tot= 0
    conv_tot = 0
    for p in params:
        no = params[p].view(-1).__len__()
        tot += no
        if 'bn' not in p:
            if verbose:
                print('%s has %d params' % (p,no))
        if 'conv' in p:
            conv_tot += no

    if verbose:
        print('Net has %d conv params' % conv_tot)
        print('Net has %d params in total' % tot)
    return tot


def eval(file):

    A = torch.load('checkpoints/%s.t7' % file)
    acc = A['acc']
    params = get_no_params(A['net'],verbose=False)
    print('Accuracy %0.2f (Error %0.2f) with %d params' %(acc,100-acc,params))

def substitute_fprop(net, a, index):
    """Given some activations, substitute them in at an index in the network,
    then continue the forward prop from there. Set index to -1 to use the whole
    network."""
    assert isinstance(net, WideResNetAT)

    out = a
    #activations = []

    if index == -1:
        out = net.conv1(out)

    block_idx = 0
    for sub_block in net.block1:
        if index < block_idx:
            out = sub_block(out)
            #activations.append(out)
        block_idx += 1

    for sub_block in net.block2:
        if index < block_idx:
            out = sub_block(out)
            #activations.append(out)
        block_idx += 1

    for sub_block in net.block3:
        if index < block_idx:
            out = sub_block(out)
            #activations.append(out)
        block_idx += 1

    out = net.relu(net.bn1(out))
    out = F.avg_pool2d(out, 8)
    out = out.view(-1, net.nChannels)
    return net.fc(out)

def partial_fprop(net, a, index):
    """Given some activations, substitute them in at an index in the network,
    then continue the forward prop from there. Set index to -1 to use the whole
    network."""
    assert isinstance(net, WideResNetAT)

    out = a
    #activations = []

    if index == -1:
        out = net.conv1(out)

    block_idx = 0
    for sub_block in net.block1:
        if index < block_idx:
            return sub_block(out)
            #activations.append(out)
        block_idx += 1

    for sub_block in net.block2:
        if index < block_idx:
            return sub_block(out)
            #activations.append(out)
        block_idx += 1

    for sub_block in net.block3:
        if index < block_idx:
            return sub_block(out)
            #activations.append(out)
        block_idx += 1

    out = net.relu(net.bn1(out))
    out = F.avg_pool2d(out, 8)
    out = out.view(-1, net.nChannels)
    return net.fc(out)

