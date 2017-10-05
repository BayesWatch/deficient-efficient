''''This just trains a model. Doesn't even have to be a teacher'''
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import json
import argparse
from torch.autograd import Variable
import os
import models
import utils.plot as plot
from utils.misc import *

parser = argparse.ArgumentParser(description='Training a CIFAR10 teacher')

# System params
parser.add_argument('--GPU', default='2', type=str, help='GPU to use')
parser.add_argument('--teacher_checkpoint', '-t', type=str,
                    help='checkpoint to load in teacher. Will be a t7 located in the checkpoints folder.'
                         'Plots are also written here.')

# Network params
parser.add_argument('net', choices=['WRN','WRNsep','VGG16','VGG11','mobilenet','mobilenetcu',
                                    'mobileresnet', 'mobileresnetcu'], type=str, help='Choose net')


#WRN params
parser.add_argument('--wrn_depth', default=16, type=int, help='depth for WRN')
parser.add_argument('--wrn_width', default=1, type=float, help='width for WRN')

#Mobilenet params
parser.add_argument('--width_factor', default=1, type=float, help='channel multiplier for mobilenet')


# Mode params
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--eval', '-e', action='store_true', help='evaluate rather than train')

# Learning params
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float, help='learning rate decay')

parser.add_argument('--epoch_step', default='[60,120,160]', type=str,
                    help='json list with epochs to drop lr on')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--weightDecay', default=0.0005, type=float)


args = parser.parse_args()
print (vars(args))
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

use_cuda = torch.cuda.is_available()
assert use_cuda, 'Error: No CUDA!'

assert args.teacher_checkpoint, 'Error: You must specify a save location (use -t BLAH)'

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
epoch_step = json.loads(args.epoch_step)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='/disk/scratch/datasets/cifar', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/disk/scratch/datasets/cifar', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


# Load checkpoint if we are resuming training or evaluating.

if args.net == 'WRN':
    net = models.WideResNet(args.wrn_depth,10, args.wrn_width, dropRate=0)
elif args.net == 'WRNsep':
    net = models.WideResNet(args.wrn_depth, 10, args.wrn_width, dropRate=0, separable=True)
elif args.net == 'VGG16':
    net = models.VGG('VGG16')
elif args.net == 'VGG11':
    net = models.VGG('VGG11')
elif args.net == 'mobilenet':
    net = models.MobileNet(cublock=False, width_factor=args.width_factor)
elif args.net =='mobilenetcu':
    net = models.MobileNet(cublock=True, width_factor=args.width_factor)
elif args.net =='mobileresnet':
    net = models.MobileResNet(cublock=False)
elif args.net =='mobileresnetcu':
    net = models.MobileResNet(cublock=True)


if args.resume or args.eval:
    print('==> Resuming from checkpoint..')
    print('==> Loading student from checkpoint..')
    checkpoint = torch.load('checkpoints/%s.t7' % args.teacher_checkpoint)
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

else:
    print('==> Building model..')

get_no_params(net)

net = net.cuda()#.half()
criterion = nn.CrossEntropyLoss()


def create_optimizer(lr):
        print('SGD')
        return torch.optim.SGD(net.parameters(), lr, 0.9, weight_decay=args.weightDecay)

optimizer = create_optimizer(args.lr)

# Training


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    print('\nTrain Loss: %.3f | Acc: %.3f%% (%d/%d)'
    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    plot.plot('train loss', train_loss/(batch_idx+1))
    plot.plot('train acc', 100.*correct/total)




def test(epoch=None):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    print('\nTest Loss: %.3f | Acc: %.3f%% (%d/%d)'
        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    plot.plot('test loss', test_loss/(batch_idx+1))
    plot.plot('test acc', 100.*correct/total)



    # Save checkpoint.
    if not args.eval:
        acc = 100.*correct/total
        if acc > best_acc:
            best_acc = acc

        print('Saving..')
        state = {
            'net': net,
            'acc': acc,
            'epoch': epoch,
            'best_acc': best_acc
        }
        print('SAVED!')
        torch.save(state, 'checkpoints/%s.t7' % args.teacher_checkpoint)


if not args.eval:
    for epoch in tqdm(range(args.epochs)):
        if epoch in epoch_step:
            lr = optimizer.param_groups[0]['lr']
            optimizer = create_optimizer(lr * args.lr_decay_ratio)
        train(epoch)
        test(epoch)
        plot.flush('checkpoints/%s_' % args.teacher_checkpoint)
        plot.tick()
else:
    print('Evaluating...')
    test(0)
