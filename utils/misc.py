from models import *
import torch

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


def convert_to_int(teacher_checkpoint, depth, width):

    # This function converts a model so that you can take some intermediate activations out explicitly.

    A = torch.load('checkpoints/%s.t7' % teacher_checkpoint)
    state_dict = A['net'].state_dict()

    net = WideResNetInt(depth, 10, width)

    net.load_state_dict(state_dict)

    print('Saving..')
    state = {
        'net': net,

    }
    print('SAVED!')
    torch.save(state, 'checkpoints/%s_int.t7' % teacher_checkpoint)

def convert_to_6int(teacher_checkpoint, depth, width):

    # More painful. Maps 3 groups to 6 hoping that the state dicts are aligned (they should be).

    A = torch.load('checkpoints/%s.t7' % teacher_checkpoint)
    net = WideResNet6Int(depth, 10, width)

    state_dict_old = A['net'].state_dict()
    state_dict_new = net.state_dict()

    old_keys = [v for v in state_dict_old]
    new_keys = [v for v in state_dict_new]

    for i,_ in enumerate(state_dict_old):
        state_dict_new[new_keys[i]] = state_dict_old[old_keys[i]]


    net.load_state_dict(state_dict_new)

    print('Saving..')
    state = {
        'net': net,

    }
    print('SAVED!')
    torch.save(state, 'checkpoints/%s_6int.t7' % teacher_checkpoint)