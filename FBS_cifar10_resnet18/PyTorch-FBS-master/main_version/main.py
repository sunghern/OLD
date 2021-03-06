# -*- coding=utf-8 -*-

import os
import argparse
import torch
from tqdm import tqdm
from loguru import logger
#from models import *
import models
from utils import *
import time
import math
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Dynami FBS Training')
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
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
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


class CrossEntropyLossSoft(torch.nn.modules.loss._Loss):
    """ inplace distillation for image classification """
    def forward(self, output, target):
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob)
        return cross_entropy_loss


def main():
    args = parser.parse_args()

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

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint)

    train_loader, val_loader = data_loader('.', dataset=args.dataset, batch_size=args.batch_size, workers=args.workers)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
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
            for m in model.modules():
                if hasattr(m, 'rate'):
                    m.count_channel[:] = 0.0

            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    mean_dict[pruning_rate].append(m.running_mean.abs().cpu().detach().numpy())
                    var_dict[pruning_rate].append(m.running_var.cpu().detach().numpy())
            validate(val_loader, model, criterion, args)
            for m in model.modules():
                if hasattr(m, 'rate'):
                    count_channel_dict[pruning_rate].append(np.array(m.count_channel))
                    count_channel_dict2[pruning_rate].append(m.count_channel.mean())

        #idx = count_channel_dict[0.4][-1].argsort()
        '''
        for pruning_rate in args.test_pruning_rate:
            #plt.plot(count_channel_dict[pruning_rate][-1][idx], label=str(pruning_rate))
            plt.plot(count_channel_dict2[pruning_rate], label=str(pruning_rate))
        plt.title('10th BatchNorm mean value')
        plt.xlabel('Ordered index (according to the PR=0.4)')
        plt.ylabel('Absolute mean value')
        plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
        plt.yticks([3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
        ax=plt.gca()
        ax.xaxis.grid(True)
        ax.yaxis.grid(True)

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
        acc1 = validate(val_loader, model, criterion, args)

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), 'ckpts/%s-latest.pth' % args.checkpath)
            logger.info('saved to ckpts/%s-latest.pth' % args.checkpath)
        if best_accuracy < acc1:
            best_accuracy = acc1
            torch.save(model.state_dict(), 'ckpts/%s-best.pth' % args.checkpath)
            logger.info('saved to ckpts/%s-best.pth' % args.checkpath)

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

        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        optimizer.zero_grad()
        if args.joint:
            for pruning_rate in args.joint_pruning_rate:
                #logger.info('set validation pruning rate = %.1f' % pruning_rate)
                for m in model.modules():
                    if hasattr(m, 'rate'):
                        m.rate = pruning_rate
                output = model(images)
                if pruning_rate == 0.0:
                    loss = criterion(output, target)
                    soft_target = torch.nn.functional.softmax(output, dim=1)

                    # measure accuracy and record loss
                    acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0], images.size(0))
                    top5.update(acc5[0], images.size(0))
                else:
                    loss1 = torch.mean(soft_criterion(output, soft_target.detach()))
                    loss2 = 0
                    for m in model.modules():
                        if hasattr(m, 'loss') and m.loss is not None:
                            loss2 += m.loss
                    loss = loss1 + 1e-8 * loss2
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
    
    #do bn_calibration when training
    #if args.post_bn:
        #bn_calibration(train_loader, model, criterion, args)


def bn_calibration(train_loader, model, criterion, args):
    #print('------------------------------------------------------')
    #print('--------------------bn_calibration--------------------')
    #print('------------------------------------------------------')
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

    original_bn_mean = []
    original_bn_var = []
    post_bn_mean = []
    post_bn_var = []
    bn_mean_temp = []
    bn_var_temp = []
    count_bn = 0
    #device_name = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            original_bn_mean.append(m.running_mean.cpu().detach().numpy())
            original_bn_var.append(m.running_var.cpu().detach().numpy())
            #print('original bn var', original_bn_var)
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

            '''
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    bn_mean_temp = torch.from_numpy(original_bn_mean[count_bn])
                    bn_var_temp = torch.from_numpy(original_bn_var[count_bn])
                    #import pdb; pdb.set_trace()
                    m.running_mean.data = torch.Tensor(bn_mean_temp).to(args.gpu)
                    m.running_var.data = torch.Tensor(bn_var_temp).to(args.gpu)
                    count_bn = count_bn + 1
            '''

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

        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                post_bn_mean.append(m.running_mean.cpu().detach().numpy())
                post_bn_var.append(m.running_var.cpu().detach().numpy())
            
        #m.running_mean.data = post_bn_mean

        #print('post_bn_mean :', post_bn_mean)
        #print('original_bn_mean :', original_bn_val)

        
        '''
        if args.plot_bn is not None:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    post_bn_mean.append(m.running_mean.cpu().detach().numpy())
                    post_bn_val.append(m.running.val.cpu().detach().numpy)

            for i, (original, post) in enumerate(zip(original_bn_mean,post_bn_mean)):
                plt.plot(original, label='original')
                plt.plot(post, label='post')
                plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
                plt.yticks(30,40,50,60,70,80,90,100)
                ax=plt.gca()
                ax.xaxis.grid(True)
                ax.yaxis.grid(True)
                #plt.legend()
                plt.savefig('mean_{}.png'.format(i))
                plt.clf()

            for i, (original, post) in enumerate(zip(original_bn_var, post_bn_var)):
                plt.plot(original, label='original')
                plt.plot(post, label='post')
                #plt.legend()
                plt.savefig('var_{}.png'.format(i))
                plt.clf()
        '''

    return top1.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
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
