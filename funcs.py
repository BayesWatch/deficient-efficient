import torch
import torch.nn.functional as F
from models import wide_resnet

def distillation(y, teacher_scores, labels, T, alpha):
    return F.kl_div(F.log_softmax(y/T), F.softmax(teacher_scores/T)) * (T*T * 2. * alpha)\
           + F.cross_entropy(y, labels) * (1. - alpha)


def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()


def l1_loss(x):
    return torch.abs(x).mean()

def get_no_params(net):

    params = net.state_dict()
    tot=0
    for p in params:
        no = params[p].view(-1).__len__()
        tot += no
        if 'bn' not in p:
            print('%s has %d params' % (p,no))


    print('Net has %d params in total' % tot)
    return tot
