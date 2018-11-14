'''Count parameters or mult-adds in models.'''
from __future__ import print_function
import math
import torch
import argparse
from functools import reduce
from torch.autograd import Variable
from models.wide_resnet import WideResNet

from funcs import what_conv_block

parser = argparse.ArgumentParser(description='WRN parameter/flop usage')
parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet'], help='Choose between Cifar10/100/imagenet.')

#network stuff
parser.add_argument('--wrn_depth', default=40, type=int, help='depth for WRN')
parser.add_argument('--wrn_width', default=2, type=float, help='width for WRN')
parser.add_argument('--module', default=None, type=str, help='path to file containing custom Conv and maybe Block module definitions')
parser.add_argument('--blocktype', default='Basic',type=str, help='blocktype used if specify a --conv')
parser.add_argument('--conv',
                    choices=['Conv','ConvB2','ConvB4','ConvB8','ConvB16','DConv', 'ACDC', 'OriginalACDC',
                             'Conv2x2','DConvB2','DConvB4','DConvB8','DConvB16','DConv3D','DConvG2','DConvG4','DConvG8','DConvG16'
                        ,'custom','DConvA2','DConvA4','DConvA8','DConvA16','G2B2','G2B4','G4B2','G4B4','G8B2','G8B4','G16B2','G16B4','A2B2','A4B2','A8B2','A16B2'],
                    default=None, type=str, help='Conv type')

args = parser.parse_args()

count_ops, count_params = 0, 0
ignored_modules = []


def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name

def get_layer_param(model):
    return sum([reduce(lambda x,y: x*y, i.size(), 1) for i in model.parameters()])

def measure_layer(layer, x):
    global count_ops, count_params
    delta_ops = 0
    delta_params = 0
    multi_add = 1
    type_name = get_layer_info(layer)

    ### ops_conv
    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        delta_params = get_layer_param(layer)

    ### ops_nonlinearity
    elif type_name in ['ReLU']:
        delta_ops = x.numel()
        delta_params = get_layer_param(layer)

    ### ops_pooling
    elif type_name in ['AvgPool2d','MaxPool2d']:
        in_w = x.size()[2]
        kernel_ops = layer.kernel_size * layer.kernel_size
        out_w = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        out_h = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        delta_ops = x.size()[0] * x.size()[1] * out_w * out_h * kernel_ops
        delta_params = get_layer_param(layer)

    ### ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = layer.bias.numel()
        delta_ops = x.size()[0] * (weight_ops + bias_ops)
        delta_params = get_layer_param(layer)

    ### ops_nothing
    elif type_name in ['BatchNorm2d', 'Dropout2d', 'DropChannel', 'Dropout']:
        delta_params = get_layer_param(layer)

    ### sequential takes no extra time
    elif type_name in ['Sequential']:
        pass
    
    ### riffle shuffle
    elif type_name in ['Riffle']:
        # technically no floating point operations
        pass

    ### channel expansion
    elif type_name in ['ChannelExpand']:
        # assume concatentation doesn't take extra FLOPs
        pass

    ### channel contraction
    elif type_name in ['ChannelCollapse']:
        # do as many additions as we have channels
        delta_ops += x.size(1)

    ### ACDC Convolution
    elif type_name in ['FastStackedConvACDC']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)       
        assert layer.groups == 1
        # pretend we're actually passing through the ACDC layers within
        N = max(layer.out_channels, layer.in_channels) # size of ACDC layers
        acdc_ops = 0
        for l in layer.layers:
            acdc_ops += 4*N + 5*N*math.log(N,2)
            delta_params += 2*N
        conv_ops = N * N * layer.kernel_size[0] *  \
                   layer.kernel_size[1]
        ops = min(acdc_ops, conv_ops)
        delta_ops += ops*out_h*out_w


    ### Grouped ACDC Convolution
    elif type_name in ['GroupedConvACDC']:
        assert False
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)       
        # pretend we're actually passing through the ACDC layers within
        N = layer.kernel_size[0]
        acdc_ops = layer.groups*(4*N + 5*N*math.log(N,2))
        conv_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                   layer.kernel_size[1]  / layer.groups
        ops = min(acdc_ops, conv_ops)
        delta_ops += ops*out_h*out_w
        delta_params += 2*N

    ### unknown layer type
    else:
        if type_name not in ignored_modules:
            ignored_modules.append(type_name)
        #raise TypeError('unknown layer type: %s' % type_name)

    count_ops += delta_ops
    count_params += delta_params
    return None

def measure_model(model, H, W):
    global count_ops, count_params
    count_ops = 0
    count_params = 0
    data = Variable(torch.zeros(1, 3, H, W))

    def should_measure(x):
        return is_leaf(x) or is_pruned(x) or is_acdc(x)

    def modify_forward(model):
        for child in model.modules():
            #if should_measure(child):
            def new_forward(m):
                def lambda_forward(x):
                    measure_layer(m, x)
                    try:
                        return m.old_forward(x)
                    except NotImplementedError as e:
                        print(m)
                        raise e
                return lambda_forward
            child.old_forward = child.forward
            child.forward = new_forward(child)

    # recursive function
    def restore_forward(model):
        for child in model.children():
            # leaf node
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    model.forward(data)
    #restore_forward(model)

    return count_ops, count_params



if __name__ == '__main__':
    # Stuff happens from here:
    Conv, Block = what_conv_block(args.conv, args.blocktype, args.module)

    if args.dataset == 'cifar10':
        h,w = 32,32
        num_classes = 10
    elif args.dataset == 'cifar100':
        h,w = 32, 32
        num_classes = 100
    elif args.dataset == 'imagenet':
        h,w = 224, 224
        num_classes = 100
    else:
        raise ValueError(args.dataset)

    # instance the model
    model = WideResNet(args.wrn_depth, args.wrn_width, Conv, Block, num_classes=num_classes, dropRate=0)

    # count how many parameters are in it
    flops, params = measure_model(model, 32, 32)
    print("Mult-Adds: %.5E"%flops)
    print("Params: %.5E"%params)
