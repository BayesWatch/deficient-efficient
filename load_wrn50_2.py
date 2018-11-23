import re
import torch
from torch.utils import model_zoo
from models.blocks import Conv
from models.wide_resnet import WRN_50_2

from collections import OrderedDict

def all_equal(iterable_1, iterable_2):
    return all([x == y for x,y in zip(iterable_1, iterable_2)])

if __name__ == '__main__':
    # our model definition
    net = WRN_50_2(Conv)
    print(net.layer1[0].downsample[0])
    print(net.layer1[0].downsample[1])
    # load parameters from model zoo
    params = model_zoo.load_url('https://s3.amazonaws.com/modelzoo-networks/wide-resnet-50-2-export-5ae25d50.pth')
    # otherwise the ordering will be messed up
    params['z.fc.weight'] = params.pop('fc.weight')
    params['z.fc.bias'] = params.pop('fc.bias')
    params = sorted(params.items()) # list of tuples, in order
    # make state_dict from model_zoo parameters
    state_dict = OrderedDict()
    w_i, b_i = 0, 0
    for n,p in net.state_dict().items():
        if 'weight' in n and 'bn' not in n:
            while 'weight' not in params[w_i][0]:
                w_i += 1
            k, v = params[w_i]
            print(k, " == ", n)
            assert all_equal(v.shape, p.size()), f"{v.shape} =/= {p.size()}"
            state_dict[n] = v
            w_i += 1
        elif 'bias' in n:
            while 'bias' not in params[b_i][0]:
                b_i += 1
            k, v = params[b_i]
            print(k, " == ", n)
            assert all_equal(v.shape, p.size()), f"{v.shape} =/= {p.size()}"
            state_dict[n] = v
            b_i += 1
        else:
            state_dict[n] = p
    assert max(w_i, b_i) == len(params) # all params are matched

    # now save a new checkpoint file, with correct saved terms
    save_dict = {}
    save_dict['net'] = state_dict
    save_dict['epoch'] = 100
    save_dict['conv'] = 'Conv'
    save_dict['blocktype'] = None
    save_dict['module'] = None

    torch.save(save_dict, 'checkpoints/wrn_50_2.imagenet.modelzoo.t7')
