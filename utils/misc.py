from models.wide_resnet import *

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

def convert_wrn_AT(net, width, depth):

    state_dict = net.state_dict()

    net = WideResNetInt(depth, 10, width)

    net.load_state_dict(state_dict)

    print('Saving..')
    state = {
        'net': net,

    }
    print('SAVED!')
    torch.save(state, 'checkpoints/wrn_40_2_int.t7')