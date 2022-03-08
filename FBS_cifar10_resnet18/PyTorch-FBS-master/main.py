# -*- coding=utf-8 -*-

import os
import argparse
import torch
from tqdm import tqdm
from loguru import logger
#from models import *
import models
from utils import *
#from torchvision import models
import time
import socket
import pickle
import json
import math
import random
import wandb
#from . import ChannelPruning
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from timeit import default_timer as timer

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

#resnet18_pretrained = model.resnet18(pretrained=True)

parser = argparse.ArgumentParser(description='Dynamic FBS Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--pruning-rate', default=0.0, type=float)
parser.add_argument('--joint-pruning-rate', nargs='+', type=float)
parser.add_argument('--test-pruning-rate', nargs='+', type=float)
parser.add_argument('--joint', dest='joint', action='store_true')
parser.add_argument('--post-bn', dest='post_bn', action='store_true')
parser.add_argument('--checkpath', default="test", type=str)
parser.add_argument('--checkpoint', default=None, type=str)
parser.add_argument('--lr-scheduler', default='cosine', type=str)
parser.add_argument('--warmup-epochs', default=0, type=int)
parser.add_argument('--warmup-lr', default=0, type=float)
parser.add_argument('--plot_bn' , default = None, type=str)

use_cuda = torch.cuda.is_available()       

class CrossEntropyLossSoft(torch.nn.modules.loss._Loss):
    """ inplace distillation for image classification """
    def forward(self, output, target):
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob)
        return cross_entropy_loss

'''
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam 
        
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
'''

""" Mixup """
def mix_images(images, lam):
	flipped_images = torch.flip(images, dims=[0])  # flip along the batch dimension
	return lam * images + (1 - lam) * flipped_images


def mix_labels(target, lam, n_classes, label_smoothing=0.1):
	onehot_target = label_smooth(target, n_classes, label_smoothing)
	flipped_target = torch.flip(onehot_target, dims=[0])
	return lam * onehot_target + (1 - lam) * flipped_target


""" Label smooth """
def label_smooth(target, n_classes: int, label_smoothing=0.1):
	# convert to one-hot
	batch_size = target.size(0)
	target = torch.unsqueeze(target, 1)
	soft_target = torch.zeros((batch_size, n_classes), device=target.device)
	soft_target.scatter_(1, target, 1)
	# label smoothing
	soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
	return soft_target


def cross_entropy_loss_with_soft_target(pred, soft_target):
	logsoftmax = nn.LogSoftmax()
	return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))


def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
	soft_target = label_smooth(target, pred.size(1), label_smoothing)
	return cross_entropy_loss_with_soft_target(pred, soft_target)

def set_random_seed(seed=None):
    """set random seed"""

    if seed == None:
        seed = 0
    print('seed for random sampling: {}', format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    args = parser.parse_args()
    print(args.checkpath)
    print("learning rate :", args.lr)
    set_random_seed(args.seed)

    wandb.init(project="offloading", entity="sunghern")
    wandb.init(project="test", entity="sunghern")
    wandb.config.update(args)
    wandb.config = {
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size
    }

    os.makedirs('log', exist_ok=True)
    os.makedirs('ckpts', exist_ok=True)
    log_path = os.path.join('log', args.checkpath + '.log')
    if os.path.isfile(log_path):
        os.remove(log_path)
    logger.add(log_path)

    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100

    model = models.__dict__[args.arch](num_classes=num_classes)
    model = model.cuda('cuda:{}'.format(args.gpu))
    print(model)
    wandb.watch(
    model, criterion=None, log="parameters", log_freq=1
    )


    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint)

    train_loader, val_loader = data_loader('.', dataset=args.dataset, batch_size=args.batch_size, workers=args.workers)

    #optimizer = torch.optim.Adam(model.parameters(), lr = args.lr,
                                #weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    criterion = torch.nn.CrossEntropyLoss()
    soft_criterion = CrossEntropyLossSoft()

    if args.evaluate:
        mean_dict = {}
        var_dict = {}
        count_channel_dict = {}
        count_channel_dict2 = {}
        for pruning_rate in args.test_pruning_rate:
            mean_dict[pruning_rate] = []
            var_dict[pruning_rate] = []
            count_channel_dict[pruning_rate] = []
            count_channel_dict2[pruning_rate] = []
            logger.info('set validation pruning rate = %.2f' % pruning_rate)
            for m in model.modules():
                if hasattr(m, 'rate'):
                    m.rate = pruning_rate
            if args.post_bn:
                bn_calibration(train_loader, model, criterion, args)
            #for m in model.modules():
                #if hasattr(m, 'rate'):
                    #m.count_channel[:] = 0.0
            #for m in model.modules():
                #if isinstance(m, nn.BatchNorm2d):
                    #mean_dict[pruning_rate].append(m.running_mean.abs().cpu().detach().numpy())
                    #var_dict[pruning_rate].append(m.running_var.cpu().detach().numpy())
            validate(val_loader, model, criterion, args, args.epochs)
            #for m in model.modules():
                #if hasattr(m, 'rate'):
                    #count_channel_dict[pruning_rate].append(np.array(m.count_channel))
                    #count_channel_dict2[pruning_rate].append(m.count_channel.mean())
            for m in model.modules():
                if hasattr(m, 'rate'):
                    m.count_channel[:] = 0.0
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    mean_dict[pruning_rate].append(m.running_mean.abs().cpu().detach().numpy())
                    var_dict[pruning_rate].append(m.running_var.cpu().detach().numpy())
            validate(val_loader, model, criterion, args, args.epochs)
            for m in model.modules():
                if hasattr(m, 'rate'):
                    count_channel_dict[pruning_rate].append(np.array(m.count_channel))
                    count_channel_dict2[pruning_rate].append(m.count_channel.mean())

        #idx = count_channel_dict[0.4][-1].argsort()
        '''
        for pruning_rate in args.test_pruning_rate:
            #plt.plot(count_channel_dict[pruning_rate][-1][idx], label=str(pruning_rate), marker="*")
            #print('extraction output : ', output.shape)
            plt.plot(count_channel_dict2[pruning_rate], label=str(pruning_rate))
        plt.title('10th BatchNorm mean value')
        plt.xlabel('layer')
        plt.ylabel('Channel ratio')
        plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
        plt.grid(True)
        #plt.legend()
        plt.show()
        '''
        return

    if args.joint != True:
        for m in model.modules():
            if hasattr(m, 'rate'):
                m.rate = args.pruning_rate

    best_accuracy = 0
    for epoch in range(args.epochs):
        train(train_loader, model, criterion, soft_criterion, optimizer, epoch, args)

        if args.joint:
            for m in model.modules():
                if hasattr(m, 'rate'):
                    m.rate = args.pruning_rate

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args, epoch)

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), 'ckpts/%s-latest.pth' % args.checkpath, _use_new_zipfile_serialization = False)
            logger.info('saved to ckpts/%s-latest.pth' % args.checkpath)
        if best_accuracy < acc1:
            best_accuracy = acc1
            torch.save(model.state_dict(), 'ckpts/%s-best.pth' % args.checkpath, _use_new_zipfile_serialization = False)
            logger.info('saved to ckpts/%s-best.pth' % args.checkpath)

    print('-----------------bn_calibration-----------------')
    if args.post_bn:
        bn_calibration(train_loader, model, criterion, args)

def cosine_calc_learning_rate(args, epoch, batch=0, nBatch=None):
    T_total = args.epochs * nBatch
    T_cur = epoch * nBatch + batch
    lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    return lr

def cosine_adjust_learning_rate(args, optimizer, epoch, batch=0, nBatch=None):
    new_lr = cosine_calc_learning_rate(args, epoch, batch, nBatch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr

def cosine_warmup_adjust_learning_rate(args, optimizer, T_total, nBatch, epoch,
                                       batch=0, warmup_lr=0):
    T_cur = epoch * nBatch + batch + 1
    new_lr = T_cur / T_total * (args.lr - warmup_lr) + warmup_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr

def train(train_loader, model, criterion, soft_criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    
    if args.joint:
        random_value = []
        random_pruning_rate = []
        random_pruning_rate = [0 for i in range(4)]
        random_pruning_rate[0] = args.joint_pruning_rate[0]
        random_pruning_rate[3] = args.joint_pruning_rate[1]

        random_value.append(random.uniform(0.1, 0.99))
        random_value.append(random.uniform(0.1, 0.99))

        random_value[0] = round(random_value[0], 2)
        random_value[1] = round(random_value[1], 2)

        random_pruning_rate[1] = random_value[0]
        random_pruning_rate[2] = random_value[1]
        print(random_pruning_rate)

    iteration_count = 0

    for i, (images, target) in enumerate(train_loader):
        if args.lr_scheduler == 'cosine':
            nBatch = len(train_loader)
            if epoch < args.warmup_epochs:
                cosine_warmup_adjust_learning_rate(
                    args, optimizer, args.warmup_epochs * nBatch,
                    nBatch, epoch, i, args.warmup_lr)
            else:
                cosine_adjust_learning_rate(
                    args, optimizer, epoch - args.warmup_epochs, i, nBatch)

        if use_cuda:
            images, target = images.cuda(), target.cuda()
        
        if args.dataset == 'cifar10':
            num_classes = 10
        elif args.dataset == 'cifar100':
            num_classes = 100

        #images, targets_a, targets_b, lam = mixup_data(images, target, args.alpha, use_cuda)

        #mixup augmentation
        #lam = random.betavariate(args.alpha, args.alpha)
        #images = mix_images(images, lam)
        #target = mix_labels(target, lam, num_classes, 0.1)

        if args.joint:
            if iteration_count % 10 == 0:
                #random_value = []
                #random_pruning_rate = []
                random_pruning_rate = [0 for i in range(4)]
                random_pruning_rate[0] = args.joint_pruning_rate[0]
                random_pruning_rate[3] = args.joint_pruning_rate[1]

                random_value.append(random.uniform(0.1, 0.99))
                random_value.append(random.uniform(0.1, 0.99))

                random_value[0] = round(random_value[0], 2)
                random_value[1] = round(random_value[1], 2)

                random_pruning_rate[1] = random_value[0]
                random_pruning_rate[2] = random_value[1]       

        iteration_count = iteration_count + 1

        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        #loss_function = torch.nn.CrossEntropyLoss()
        #soft_loss_function = CrossEntropyLossSoft()

        # compute output
        optimizer.zero_grad()
        if args.joint:
            for pruning_rate in random_pruning_rate:
            #for pruning_rate in args.joint_pruning_rate:
                #logger.info('set validation pruning rate = %.1f' % pruning_rate)
                for m in model.modules():
                    if hasattr(m, 'rate'):
                        m.rate = pruning_rate
                '''

                if pruning_rate == 0.0:
                    soft_logits = model(images)
                    soft_target = torch.nn.functional.softmax(soft_logits, dim=1)
                else: 
                    output = model(images)
                '''
                output = model(images)

                if pruning_rate == 0.0:
                    #loss = criterion(output, target)
                    #loss = criterion(soft_logits, target.long())
                    loss1 = criterion(output, target)
                    soft_target = torch.nn.functional.softmax(output, dim=1)
                    
                    '''

                if pruning_rate == 0.0:
                    #loss = criterion(output, target)
                    loss = criterion(soft_logits, target.long())
                    #soft_target = torch.nn.functional.softmax(output, dim=1)

                    # measure accuracy and record loss
                    acc1, acc5 = accuracy(soft_logits, target, topk=(1, 5))
                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0], images.size(0))
                    top5.update(acc5[0], images.size(0))
                    '''
                else:
                    #loss1 = torch.mean(output, soft_target)
                    #soft_target = torch.argmax(soft_target, 1) 
                    #loss1 = cross_entropy_loss_with_soft_target(output, soft_target.detach())
                    loss1 = torch.mean(soft_criterion(output, soft_target.detach()))

                loss2 = 0
                for m in model.modules():
                    if hasattr(m, 'loss') and m.loss is not None:
                        loss2 += m.loss

                loss = loss1 + 1e-8 * loss2

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                else:
                    #loss1 = torch.mean(output, soft_target)
                    #soft_target = torch.argmax(soft_target, 1) 
                    loss1 = cross_entropy_loss_with_soft_target(output, soft_target.detach())
                    loss2 = 0
                    for m in model.modules():
                        if hasattr(m, 'loss') and m.loss is not None:
                            loss2 += m.loss
                    loss = loss1 + 1e-8 * loss2

                    # measure accuracy and record loss
                    acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0], images.size(0))
                    top5.update(acc5[0], images.size(0))

                loss.backward()
        else:
            output = model(images)
            loss1 = criterion(output, target)
            loss2 = 0
            for m in model.modules():
                if hasattr(m, 'loss') and m.loss is not None:
                    loss2 += m.loss
            loss = loss1 + 1e-8 * loss2

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            loss.backward()

        # compute gradient and do SGD step
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            #wandb.log({"training_accuracy": round(top1.avg.item(), 4)})

    #if args.post_bn:
        #bn_calibration(train_loader, model, criterion, args)

    wandb.log({"training_accuracy": round(top1.avg.item(), 4),
                "epochs": epoch})
    wandb.log({"loss": round(loss.item(), 4),
                "epochs": epoch})



def bn_calibration(train_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1, top5],
        prefix="Post BN: ")

    # switch to evaluate mode
    model.eval()

    #original_bn_mean = []
    #original_bn_val = []
    #post_bn_mean = []
    #post_bn_val = []

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            #original_bn_mean.append(m.running_mean.cpu().detach().numpy())
            #original_bn_val.append(m.running_var.cpu().detach().numpy())
            m.reset_running_stats()
            m.training = True
            m.momentum = None

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(train_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        if args.plot_bn is not None:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    post_bn_mean.append(m.running_mean.cpu().detach().numpy())
                    post_bn_val.append(m.running.val.cpu().detach().numpy)

            for i, (original, post) in enumerate(zip(original_bn_mean,post_bn_mean)):
                plt.plot(original, label='original')
                plt.plot(post, label='post')
                plt.legend()
                plt.savefig('mean_{}.png'.format(i))
                plt.clf()

            for i, (original, post) in enumerate(zip(original_bn_var, post_bn_var)):
                plt.plot(original, label='original')
                plt.plot(post, label='post')
                plt.legend()
                plt.savefig('var_{}.png'.format(i))
                plt.clf()

    return top1.avg

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

class ResNetConv4(nn.Module): #extraction model class
    def __init__(self, model):
        super(ResNetConv4, self).__init__()

        #import pdb; pdb.set_trace()
        self.features = nn.Sequential(
            #stop at partitioning point
            *list(model.children())[:-3]
        )

    def forward(self, x):
        x = F.relu(self.features[:2](x))
        x = self.features[2:](x) 
        #import pdb; pdb.set_trace()
        return ChannelPruning(x)

class ResNetConv5(nn.Module): #injection model class
    def __init__(self, model):
        super(ResNetConv5, self).__init__()

        #import pdb; pdb.set_trace()
        self.features = nn.Sequential(
            #stop at partitioning point
            #*(list(model.children())[-3:-1] + [nn.AvgPool2d(1), Flatten()] + list(model.children())[-1:]) 
            *list(model.children())[-3:]
        )
        #import pdb; pdb.set_trace()

    def forward(self, x):
        x = self.features[-3:-1](x)
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        x = self.features[-1](x)
        return x


def validate(val_loader, model, criterion, args, epoch):
    #wandb.init(project="validate-test-project", entity="sunghern")
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    #args = parser.parse_args()

    extraction_model = ResNetConv4(model)
    injection_model = ResNetConv5(model)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            ############################# mobile ###################################
            # compute output
            #start_time = timer()
            #output = extraction_model(images)
            ###################### define ip and port ######################
            #TCP_IP2 = ''
            #TCP_PORT2 = 
            ################################################################

            #s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            #s.connect((TCP_IP2, TCP_PORT2))
            #send_x = output.detach().numpy()
            #data_input = pickle.dumps(send_x, protocol=pockle.HIGHEST_PROTOCOL)
            #s.sendall(data_input)
            #s.close

            ###################### define ip and port ######################
            #TCP_IP2 = ''
            #TCP_PORT2 = 
            #buf_size = 4096
            ################################################################

            #s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            #s.bind((TCP_IP2, TCP_PORT2))
            #s.listen(1)

            #conn, addr = s.accept()
            #data = []
            #print('Server: ', addr)

            #while 1:
                #output = conn.recv(buf_size)
                #if not output: break
                #data.append(output)
            #final_output = pickle.loads(b"".join(data))
            #print('Output shown on mobile side')
            #end = timer()
            #conn.close
            #print("final time = ")
            #print(end = start)

            # print('extraction output : ', output.shape)
            #loss = criterion(output, target)
            #original_output = model(images)
            #import pdb; pdb.set_trace()
            '''
            ################################## server ###################################

            while 1:
                print('waiting for Raspberry pi')
                TCP_IP1 = '10.201.135.168'
                TCP_PORT1 = 8101
                BUFFER_SIZE = 4096  

                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind((TCP_IP1, TCP_PORT1))
                s.listen()

                conn, addr = s.accept()
                data = []
                print('Raspberry pi Device :', addr)
                while 1:
                    tensor = conn.recv(BUFFER_SIZE)
                    if not tensor: break
                    data.append(tensor)
                inputs_ten = pickle.loads(b"".join(data))

                conn.close()
                inputs = torch.from_numpy(inputs_ten)
                inputs = inputs.to(args.gpu)

                break

            ##############################################################################
            # send output after calc output
            final_output = injection_model(inputs)
            
            final_output = final_output.to('cpu')
            final_output_send = final_output.numpy()
            print(final_output_send)
            TCP_IP2 = '115.145.209.162'
            TCP_PORT2 = 8100
            BUFFER_SIZE = 4096
            print('try to connect IP :', TCP_IP2)
            print('try to connect PORT :', TCP_PORT2)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            s.connect((TCP_IP2, TCP_PORT2))
            data_final = pickle.dumps(final_output_send, protocol = pickle.HIGHEST_PROTOCOL)
            s.sendall(data_final)
            print('Server execution done!')
            '''
            #final_output = injection_model(output)
            #final_output = final_output.to(args.gpu)
            final_output = model(images)
            loss = criterion(final_output, target)
            finish_time = timer()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(final_output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        
        #wandb.config.update(args)
        if args.evaluate is None:
            wandb.log({"validation_accuracy": round(top1.avg.item(), 4),
                        "epochs": epoch})

        #option
        #wandb.watch(model)

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
