def get_no_params(net):

    params = net.state_dict()
    tot=0
    for p in params:
        no = params[p].view(-1).__len__()
        tot += no
        print('%s has %d params' % (p,no))
    print('Net has %d params in total' % tot)
    return tot