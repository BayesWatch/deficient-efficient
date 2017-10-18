''''Writing everything into one script..'''
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import json
import argparse
from torch.autograd import Variable
from models.wide_resnet import*
import os
from funcs import *
parser = argparse.ArgumentParser(description='Student/teacher training')
parser.add_argument('mode', choices=['KD','AT','teacher'], type=str, help='Learn with KD, AT, or train a teacher')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--GPU', default='3', type=str,help='GPU to use')
parser.add_argument('--student_checkpoint', '-s', default='wrn_40_2_student_KT',type=str, help='checkpoint to save/load student')
parser.add_argument('--teacher_checkpoint', '-t', default='wrn_40_2',type=str, help='checkpoint to load in teacher')

#network stuff
parser.add_argument('--wrn_depth', default=16, type=int, help='depth for WRN')
parser.add_argument('--wrn_width', default=1, type=float, help='width for WRN')
parser.add_argument('conv',
                    choices=['Conv','ConvB2','ConvB4','ConvB8','ConvB16','DConv',
                             'Conv2x2','DConvB2','DConvB4','DConvB8','DConv3D','DConvG2','DConvG4','DConvG8','DConvG16'
                        ,'custom'],
                    type=str, help='Conv type')
parser.add_argument('--customconv',default=['Conv','Conv','ConvB16'],type=tuple)
parser.add_argument('--AT_split', default=1, type=int, help='group splitting for AT loss')

#learning stuff
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float, help='learning rate decay')
parser.add_argument('--temperature', default=4, type=float, help='temp for KD')
parser.add_argument('--alpha', default=0.9, type=float, help='alpha for KD')
parser.add_argument('--beta', default=1e3, type=float, help='beta for AT')
parser.add_argument('--epoch_step', default='[60,120,160]', type=str,
                    help='json list with epochs to drop lr on')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=128, type=int,
                    help='minibatch size')
parser.add_argument('--weightDecay', default=0.0005, type=float)

args = parser.parse_args()

if args.conv != 'custom':
    conv = args.conv
else:
    conv = args.customconv

print(conv)

print (vars(args))
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

use_cuda = torch.cuda.is_available()
assert use_cuda, 'Error: No CUDA!'

test_losses  = []
train_losses = []
test_accs    = []
train_accs   = []

best_acc = 0
start_epoch = 0
epoch_step = json.loads(args.epoch_step)

# Data and loaders
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

trainset = torchvision.datasets.CIFAR10(root='/disk/scratch/datasets/cifar',
                                        train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='/disk/scratch/datasets/cifar',
                                       train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
criterion = nn.CrossEntropyLoss()


def create_optimizer(lr,net):
    print('creating optimizer with lr = %0.5f' % lr)
    return torch.optim.SGD(net.parameters(), lr, 0.9, weight_decay=args.weightDecay)


def train_teacher(net):
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

    train_losses.append(train_loss/(batch_idx+1))
    train_accs.append(100.*correct/total)

    print('\nTrain Loss: %.3f | Acc: %.3f%% (%d/%d)'
    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def train_student_KD(net, teach):
    net.train()
    teach.eval()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
        inputs = Variable(inputs.cuda())
        targets = Variable(targets.cuda())
        outputs_student = net(inputs)
        outputs_teacher = teach(inputs)
        loss = distillation(outputs_student, outputs_teacher, targets, args.temperature, args.alpha)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs_student.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    print('\nLoss: %.3f | Acc: %.3f%% (%d/%d)\n' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


# Training the student
def train_student_AT(net, teach):
    net.train()
    teach.eval()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
        inputs = Variable(inputs.cuda())
        targets = Variable(targets.cuda())
        outputs_student, ints_student = net(inputs)
        outputs_teacher, ints_teacher = teach(inputs)

        # If alpha is 0 then this loss is just a cross entropy.
        loss = distillation(outputs_student, outputs_teacher, targets, args.temperature, args.alpha)

        #Add an attention tranfer loss for each intermediate. Let's assume the default is three (as in the original
        #paper) and adjust the beta term accordingly.

        adjusted_beta = (args.beta*3)/len(ints_student)

        for i in range(len(ints_student)):
            loss += adjusted_beta * at_loss(ints_student[i], ints_teacher[i])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs_student.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    print('\nLoss: %.3f | Acc: %.3f%% (%d/%d)\n' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(net, checkpoint=None):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):

        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        if isinstance(outputs,tuple):
            outputs = outputs[0]

        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    test_losses.append(test_loss/(batch_idx+1))
    test_accs.append(100.*correct/total)


    print('Test Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    if checkpoint:
        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            best_acc = acc

        print('Saving..')
        state = {
            'net': net,
            'acc': acc,
            'epoch': epoch,
            'best_acc': best_acc,
            'width': args.wrn_width,
            'depth': args.wrn_depth,
            'conv_type': conv,
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_losses': test_losses,
            'test_accs': test_accs,
        }
        print('SAVED!')
        torch.save(state, 'checkpoints/%s.t7' % checkpoint)


def decay_optimizer_lr(optimizer, decay_rate):

    # callback to set the learning rate in an optimizer, without rebuilding the whole optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
    return optimizer

# Stuff happens from here:


if args.mode == 'teacher':

    if args.resume:
        print('Mode Teacher: Loading teacher and continuing training...')
        teach_checkpoint = torch.load('checkpoints/%s.t7' % args.teacher_checkpoint)
        start_epoch = teach_checkpoint['epoch']
        teach = teach_checkpoint['net'].cuda()
    else:
        print('Mode Teacher: Making a teacher network from scratch and training it...')
        teach = WideResNet(args.wrn_depth, args.wrn_width, dropRate=0, convtype=conv).cuda()


    get_no_params(teach)
    optimizer = optim.SGD(teach.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weightDecay)

    # This bit is stupid but we need to decay the learning rate depending on the epoch
    for e in range(0,start_epoch):
        if e in epoch_step:
            optimizer = decay_optimizer_lr(optimizer, args.lr_decay_ratio)

    for epoch in tqdm(range(start_epoch, args.epochs)):
        print('Teacher Epoch %d:' % epoch)
        print('Learning rate is %s' % [v['lr'] for v in optimizer.param_groups][0])
        if epoch in epoch_step:
            optimizer = decay_optimizer_lr(optimizer, args.lr_decay_ratio)
        train_teacher(teach)
        test(teach, args.teacher_checkpoint)

elif args.mode == 'KD':
    print('Mode Student: First, load a teacher network and check it performs decently...,')
    teach_checkpoint = torch.load('checkpoints/%s.t7' % args.teacher_checkpoint)
    teach = teach_checkpoint['net']
    teach = teach.cuda()
    # Very important to explicitly say we require no gradients for the teacher network
    for param in teach.parameters():
        param.requires_grad = False
    test(teach)
    if args.resume:
        print('KD: Loading student and continuing training...')
        student_checkpoint = torch.load('checkpoints/%s.t7' % args.student_checkpoint)
        start_epoch = student_checkpoint['epoch']
        student = student_checkpoint['net']
    else:
        print('KD: Making a student network from scratch and training it...')
        student = WideResNet(args.wrn_depth, args.wrn_width, dropRate=0, convtype=conv)
    student = student.cuda()
    optimizer = optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weightDecay)
    # This bit is stupid but we need to decay the learning rate depending on the epoch
    for e in range(0, start_epoch):
        if e in epoch_step:
            optimizer = decay_optimizer_lr(optimizer, args.lr_decay_ratio)

    for epoch in tqdm(range(start_epoch, args.epochs)):
        print('Student Epoch %d:' % epoch)
        print('Learning rate is %s' % [v['lr'] for v in optimizer.param_groups][0])
        if epoch in epoch_step:
            optimizer = decay_optimizer_lr(optimizer, args.lr_decay_ratio)
        train_student_KD(student, teach)
        test(student, args.student_checkpoint)


elif args.mode == 'AT':
    print('AT (+optional KD): First, load a teacher network and convert for attention transfer')
    teach_checkpoint = torch.load('checkpoints/%s.t7' % args.teacher_checkpoint)
    state_dict_old = teach_checkpoint['net'].state_dict()
    teach = WideResNetAT(teach_checkpoint['depth'], teach_checkpoint['width'], s=args.AT_split)
    state_dict_new = teach.state_dict()
    old_keys = [v for v in state_dict_old]
    new_keys = [v for v in state_dict_new]
    for i,_ in enumerate(state_dict_old):
        state_dict_new[new_keys[i]] = state_dict_old[old_keys[i]]


    teach.load_state_dict(state_dict_new)
    teach = teach.cuda()
    # Very important to explicitly say we require no gradients for the teacher network
    for param in teach.parameters():
        param.requires_grad = False
    test(teach)

    if args.resume:
        print('Mode Student: Loading student and continuing training...')
        student_checkpoint = torch.load('checkpoints/%s.t7' % args.student_checkpoint)
        start_epoch = student_checkpoint['epoch']
        student = student_checkpoint['net']
    else:
        print('Mode Student: Making a student network from scratch and training it...')
        student = WideResNetAT(args.wrn_depth, args.wrn_width, dropRate=0, convtype=conv,
                               s=args.AT_split).cuda()

    student = student.cuda()
    optimizer = optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weightDecay)

    # This bit is stupid but we need to decay the learning rate depending on the epoch
    for e in range(0, start_epoch):
        if e in epoch_step:
            optimizer = decay_optimizer_lr(optimizer, args.lr_decay_ratio)

    for epoch in tqdm(range(start_epoch, args.epochs)):
        print('Student Epoch %d:' % epoch)
        print('Learning rate is %s' % [v['lr'] for v in optimizer.param_groups][0])
        if epoch in epoch_step:
            optimizer = decay_optimizer_lr(optimizer, args.lr_decay_ratio)
        train_student_AT(student, teach)
        test(student, args.student_checkpoint)


