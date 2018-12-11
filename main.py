''''Writing everything into one script..'''
from __future__ import print_function
import os
import imp
import sys
import time
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from scipy.optimize import minimize_scalar
from functools import reduce

from tqdm import tqdm
from tensorboardX import SummaryWriter

from funcs import *
from models.wide_resnet import WideResNet, WRN_50_2
from models.darts import DARTS, Cutout, _data_transforms_cifar10 as darts_transforms

os.mkdir('checkpoints/') if not os.path.isdir('checkpoints/') else None

parser = argparse.ArgumentParser(description='Student/teacher training')
parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet'], help='Choose between Cifar10/100/imagenet.')
parser.add_argument('mode', choices=['student','teacher'], type=str, help='Learn a teacher or a student')
parser.add_argument('--imagenet_loc', default='/disk/scratch_ssd/imagenet',type=str, help='folder containing imagenet train and val folders')
parser.add_argument('--workers', default=2, type=int, help='No. of data loading workers. Make this high for imagenet')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--GPU', default=None, type=str,help='GPU to use')
parser.add_argument('--student_checkpoint', '-s', default='wrn_40_2_student_KT',type=str, help='checkpoint to save/load student')
parser.add_argument('--teacher_checkpoint', '-t', default='wrn_40_2_T',type=str, help='checkpoint to load in teacher')

#network stuff
parser.add_argument('--network', default='WideResNet', type=str, help='network to use')
parser.add_argument('--wrn_depth', default=40, type=int, help='depth for WRN')
parser.add_argument('--wrn_width', default=2, type=float, help='width for WRN')
parser.add_argument('--module', default=None, type=str, help='path to file containing custom Conv and maybe Block module definitions')
parser.add_argument('--blocktype', default='Basic',type=str, help='blocktype used if specify a --conv')
parser.add_argument('--conv', default=None, type=str, help='Conv type')
parser.add_argument('--AT_split', default=1, type=int, help='group splitting for AT loss')
parser.add_argument('--budget', default=None, type=float, help='budget of parameters to use for the network')

#learning stuff
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float, help='learning rate decay')
parser.add_argument('--temperature', default=4, type=float, help='temp for KD')
parser.add_argument('--alpha', default=0.0, type=float, help='alpha for KD')
parser.add_argument('--aux_loss', default='AT', type=str, help='AT or SE loss')
parser.add_argument('--beta', default=1e3, type=float, help='beta for AT')
parser.add_argument('--epoch_step', default='[60,120,160]', type=str,
                    help='json list with epochs to drop lr on')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--print_freq', default=10, type=int, help="print stats frequency")
parser.add_argument('--batch_size', default=128, type=int,
                    help='minibatch size')
parser.add_argument('--weight_decay', default=0.0005, type=float)
parser.add_argument('--clip_grad', default=None, type=float)

args = parser.parse_args()

if args.mode == 'teacher':
    logdir = "runs/%s"%args.teacher_checkpoint
elif args.mode == 'student':
    logdir = "runs/%s.%s"%(args.teacher_checkpoint, args.student_checkpoint)
append = 0
while os.path.isdir(logdir+".%i"%append):
    append += 1
if append > 0:
    logdir = logdir+".%i"%append
writer = SummaryWriter(logdir)

def train_teacher(net):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.train()

    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        if isinstance(net, DARTS):
            outputs, aux = net(inputs)
            outputs = torch.cat([outputs, aux], 0)
            targets = torch.cat([targets, targets], 0)
        else:
            outputs, _ = net(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        err1 = 100. - prec1
        err5 = 100. - prec5
        losses.update(loss.item(), inputs.size(0))
        top1.update(err1[0], inputs.size(0))
        top5.update(err5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Error@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, batch_idx, len(trainloader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    writer.add_scalar('train_loss', losses.avg, epoch)
    writer.add_scalar('train_top1', top1.avg, epoch)
    writer.add_scalar('train_top5', top5.avg, epoch)

    train_losses.append(losses.avg)
    train_errors.append(top1.avg)


def train_student(net, teach):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.train()
    teach.eval()

    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = Variable(inputs.cuda())
        targets = Variable(targets.cuda())
        outputs_student, ints_student = net(inputs)
        with torch.no_grad():
            outputs_teacher, ints_teacher = teach(inputs)

        # If alpha is 0 then this loss is just a cross entropy.
        loss = distillation(outputs_student, outputs_teacher, targets, args.temperature, args.alpha)

        #Add an attention tranfer loss for each intermediate. Let's assume the default is three (as in the original
        #paper) and adjust the beta term accordingly.

        adjusted_beta = (args.beta*3)/len(ints_student)
        for i in range(len(ints_student)):
            loss += adjusted_beta * aux_loss(ints_student[i], ints_teacher[i])

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs_student.data, targets.data, topk=(1, 5))
        err1 = 100. - prec1
        err5 = 100. - prec5
        losses.update(loss.item(), inputs.size(0))
        top1.update(err1[0], inputs.size(0))
        top5.update(err5[0], inputs.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad is not None:
            max_grad = 0.
            for p in net.parameters():
                g = p.grad.max().item()
                if g > max_grad:
                    max_grad = g
            nn.utils.clip_grad_norm(net.parameters(), args.clip_grad)
            print("Max grad: ", max_grad)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Error@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, batch_idx, len(trainloader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    writer.add_scalar('train_loss', losses.avg, epoch)
    writer.add_scalar('train_top1', top1.avg, epoch)
    writer.add_scalar('train_top5', top5.avg, epoch)

    train_losses.append(losses.avg)
    train_errors.append(top1.avg)


def validate(net, checkpoint=None):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    net.eval()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(valloader):

        inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
            outputs, _ = net(inputs)
            if isinstance(outputs,tuple):
                outputs = outputs[0]

            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            err1 = 100. - prec1
            err5 = 100. - prec5

            losses.update(loss.item(), inputs.size(0))
            top1.update(err1[0], inputs.size(0))
            top5.update(err5[0], inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.print_freq == 0:
                print('validate: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Error@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Error@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    batch_idx, len(valloader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    print(' * Error@1 {top1.avg:.3f} Error@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    if checkpoint:
        writer.add_scalar('val_loss', losses.avg, epoch)
        writer.add_scalar('val_top1', top1.avg, epoch)
        writer.add_scalar('val_top5', top5.avg, epoch)

        val_losses.append(losses.avg)
        val_errors.append(top1.avg)

        print('Saving..')
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
            'args': sys.argv,
            'width': args.wrn_width,
            'depth': args.wrn_depth,
            'conv': args.conv,
            'blocktype': args.blocktype,
            'module': args.module,
            'train_losses': train_losses,
            'train_errors': train_errors,
            'val_losses': val_losses,
            'val_errors': val_errors,
        }
        print('SAVED!')
        torch.save(state, 'checkpoints/%s.t7' % checkpoint)

def set_for_budget(eval_network_size, conv_type, budget):
    # set bounds using knowledge of conv_type hyperparam domain
    if 'ACDC' == conv_type:
        bounds = (2, 128)
        post_process = lambda x: int(round(x))
    elif 'Hashed' == conv_type:
        bounds = (0.001,0.9)
        post_process = lambda x: x # do nothing
    elif 'SepHashed' == conv_type:
        bounds = (0.001,0.9)
        post_process = lambda x: x # do nothing
    elif 'Generic' == conv_type:
        bounds = (0.1,0.9)
        post_process = lambda x: x # do nothing
    elif 'TensorTrain' == conv_type:
        bounds = (0.1,0.9)
        post_process = lambda x: x # do nothing
    elif 'Tucker' == conv_type:
        bounds = (0.1,0.9)
        post_process = lambda x: x # do nothing
    elif 'CP' == conv_type:
        bounds = (0.1,0.9)
        post_process = lambda x: x # do nothing
    else:
        raise ValueError("Don't know: "+conv_type)
    def obj(h):
        return abs(budget-eval_network_size(h))
    minimizer = minimize_scalar(obj, bounds=bounds, method='bounded')
    return post_process(minimizer.x)

def n_params(net):
    return sum([reduce(lambda x,y:x*y, p.size()) for p in net.parameters()])

def darts_defaults(args):
    args.batch_size = 96
    args.lr = 0.025
    args.momentum = 0.9
    args.weight_decay = 3e-4
    args.epochs = 600
    return args

def imagenet_defaults(args):
    args.epochs = 90
    args.lr_decay_ratio = 0.1
    args.epoch_step = '[30,60]'
    args.workers = 4
    return args

def get_scheduler(optimizer, epoch_step, args):
    if args.network == 'WideResNet' or args.network == 'WRN_50_2':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=epoch_step,
                gamma=args.lr_decay_ratio)
    elif args.network == 'DARTS':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    return scheduler

if __name__ == '__main__':
    if args.aux_loss == 'AT':
        aux_loss = at_loss
    elif args.aux_loss == 'SE':
        aux_loss = se_loss

    if args.network == 'DARTS':
        args = darts_defaults(args) # different training hyperparameters
    elif args.network == 'WRN_50_2':
        args = imagenet_defaults(args)

    print(vars(args))
    if args.GPU is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

    use_cuda = torch.cuda.is_available()
    assert use_cuda, 'Error: No CUDA!'

    val_losses = []
    train_losses = []
    val_errors = []
    train_errors = []

    best_acc = 0
    start_epoch = 0
    epoch_step = json.loads(args.epoch_step)

    # Data and loaders
    print('==> Preparing data..')


    if args.dataset == 'cifar10':
        num_classes = 10
        if args.network == 'DARTS':
            transforms_train, transforms_validate = darts_transforms()
        else:
            transforms_train =  transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                    Cutout(16)])
            transforms_validate = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),])
        trainset = torchvision.datasets.CIFAR10(root='/disk/scratch/datasets/cifar',
                                                train=True, download=False, transform=transforms_train)
        valset = torchvision.datasets.CIFAR10(root='/disk/scratch/datasets/cifar',
                                               train=False, download=False, transform=transforms_validate)
    elif args.dataset == 'cifar100':
        num_classes = 100
        if args.network == 'DARTS':
            raise NotImplementedError("Could use transforms for CIFAR-10, but not ported yet.")
        transforms_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
        ])
        transforms_validate = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
        ])
        trainset = torchvision.datasets.CIFAR100(root='/disk/scratch/datasets/cifar100',
                                                train=True, download=True, transform=transforms_train)
        validateset = torchvision.datasets.CIFAR100(root='/disk/scratch/datasets/cifar100',
                                               train=False, download=True, transform=transforms_validate)

    elif args.dataset == 'imagenet':
        num_classes = 1000
        traindir = os.path.join(args.imagenet_loc, 'train')
        valdir = os.path.join(args.imagenet_loc, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_validate = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        trainset = torchvision.datasets.ImageFolder(traindir, transform_train)
        valset = torchvision.datasets.ImageFolder(valdir, transform_validate)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory = True if args.dataset == 'imagenet' else False)
    valloader = torch.utils.data.DataLoader(valset, batch_size=min(100,args.batch_size), shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True if args.dataset == 'imagenet' else False)

    criterion = nn.CrossEntropyLoss()

    # a function for building networks
    def build_network(Conv, Block):
        if args.network == 'WideResNet':
            return WideResNet(args.wrn_depth, args.wrn_width, Conv, Block,
                    num_classes=num_classes, dropRate=0, s=args.AT_split)
        elif args.network == 'WRN_50_2':
            return WRN_50_2(Conv)
        elif args.network == 'DARTS':
            return DARTS(Conv, num_classes=num_classes)
    def schedule_drop_path(epoch, net):
        net.drop_path_prob = 0.2 * epoch / args.epochs

    # if a budget is specified, figure out what we have to set the
    # hyperparameter to
    if args.budget is not None:
        def eval_network_size(hyperparam):
            net = build_network(*what_conv_block(args.conv+"_%s"%hyperparam, args.blocktype, args.module))
            return n_params(net)
        hyperparam = set_for_budget(eval_network_size, args.conv, args.budget)
        args.conv = args.conv + "_%s"%hyperparam
    # get the classes implementing the Conv and Blocks we're going to use in
    # the network
    Conv, Block = what_conv_block(args.conv, args.blocktype, args.module)

    def load_network(loc):
        net_checkpoint = torch.load(loc)
        start_epoch = net_checkpoint['epoch']
        SavedConv, SavedBlock = what_conv_block(net_checkpoint['conv'],
                net_checkpoint['blocktype'], net_checkpoint['module'])
        net = build_network(SavedConv, SavedBlock).cuda()
        net.load_state_dict(net_checkpoint['net'])
        return net, start_epoch

    if args.mode == 'teacher':

        if args.resume:
            print('Mode Teacher: Loading teacher and continuing training...')
            teach, start_epoch = load_network('checkpoints/%s.t7' % args.teacher_checkpoint)
        else:
            print('Mode Teacher: Making a teacher network from scratch and training it...')
            teach = build_network(Conv, Block).cuda()


        get_no_params(teach)
        optimizer = optim.SGD(teach.grouped_parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = get_scheduler(optimizer, epoch_step, args)

        # Decay the learning rate depending on the epoch
        for e in range(0,start_epoch):
            scheduler.step()

        for epoch in tqdm(range(start_epoch, args.epochs)):
            scheduler.step()
            if args.network == 'DARTS': schedule_drop_path(epoch, teach)
            print('Teacher Epoch %d:' % epoch)
            print('Learning rate is %s' % [v['lr'] for v in optimizer.param_groups][0])
            writer.add_scalar('learning_rate', [v['lr'] for v in optimizer.param_groups][0], epoch)
            train_teacher(teach)
            validate(teach, args.teacher_checkpoint)


    elif args.mode == 'student':
        print('Mode Student: First, load a teacher network and convert for (optional) attention transfer')
        teach, _ = load_network('checkpoints/%s.t7' % args.teacher_checkpoint)
        # Very important to explicitly say we require no gradients for the teacher network
        for param in teach.parameters():
            param.requires_grad = False
        validate(teach)
        val_losses, val_errors = [], [] # or we'd save the teacher's error as the first entry

        if args.resume:
            print('Mode Student: Loading student and continuing training...')
            student, start_epoch = load_network('checkpoints/%s.t7' % args.student_checkpoint)
        else:
            print('Mode Student: Making a student network from scratch and training it...')
            student = build_network(Conv, Block).cuda()

        optimizer = optim.SGD(student.grouped_parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = get_scheduler(optimizer, epoch_step, args)

        # Decay the learning rate depending on the epoch
        for e in range(0, start_epoch):
            scheduler.step()

        for epoch in tqdm(range(start_epoch, args.epochs)):
            scheduler.step()
            if args.network == 'DARTS': schedule_drop_path(epoch, student)

            print('Student Epoch %d:' % epoch)
            print('Learning rate is %s' % [v['lr'] for v in optimizer.param_groups][0])
            writer.add_scalar('learning_rate', [v['lr'] for v in optimizer.param_groups][0], epoch)

            train_student(student, teach)
            validate(student, args.student_checkpoint)

