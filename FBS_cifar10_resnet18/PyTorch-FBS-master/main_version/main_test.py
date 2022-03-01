# -*- coding=utf-8 -*-

import os
import torch
from tqdm import tqdm
from loguru import logger
from models import *
from utils import *
import argparse

parser = argparse.ArgumentParser(description='Pytorch FBS training')
cuda1 = torch.device('cuda:1')

def watch_nan(x: torch.Tensor):
    if torch.isnan(x).any():
        raise ValueError('found NaN: ' + str(x))

def main():
    #name = 'resnet18-imagenette2'
    name = 'resnet20-imagenette2'
    os.makedirs('log', exist_ok=True)
    os.makedirs('ckpts', exist_ok=True)
    log_path = os.path.join('log', name + '.log')
    if os.path.isfile(log_path):
        os.remove(log_path)
    logger.add(log_path)
    #net = resnet18(num_classes=10).cuda()
    net = resnet20(num_classes=10).cuda(cuda1)
    #train_loader = imagenette2('train')
    #val_loader = imagenette2('val')
    train_loader, val_loader = data_loader('.', dataset='cifar10', batch_size=256, workers=8)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 80], 0.2)
    loss_function = torch.nn.CrossEntropyLoss()
    best_accuracy = 0
    for epoch in range(100):
        with tqdm(train_loader) as train_tqdm:
            train_tqdm.set_description_str('{:03d} train'.format(epoch))
            net.train()
            meter = AccuracyMeter(topk=(1, 5))
            for images, labels in train_tqdm:
                images = images.cuda(cuda1)
                labels = labels.cuda(cuda1)
                loss = 0
                for pruning_rate in [1.0]:
                    #logger.info('set pruning rate = %.1f' % pruning_rate)
	            # set pruning rate
                    for m in net.modules():
                        if hasattr(m, 'rate'):
                            m.rate = pruning_rate

                    output = net(images)
                    watch_nan(output)
                    meter.update(output, labels)
                    train_tqdm.set_postfix(meter.get())
                    optimizer.zero_grad()
                    loss1 = loss_function(output, labels)
                    loss2 = 0
                    for m in net.modules():
                        if hasattr(m, 'loss') and m.loss is not None:
                            loss2 += m.loss

                    loss = loss1 + 1e-8 * loss2
                    loss.backward()
                    optimizer.step()
                    #logger.info('{:03d} train result: {}'.format(epoch, meter.get()))
        with tqdm(val_loader) as valid_tqdm:

            for pruning_rate in [0.9]:
                logger.info('set validation pruning rate = %.2f' % pruning_rate)
                for m in net.modules():
                    if hasattr(m, 'rate'):
                        m.rate = pruning_rate

                valid_tqdm.set_description_str('{:03d} valid'.format(epoch))
                net.eval()
                meter = AccuracyMeter(topk=(1, 5))
                with torch.no_grad():
                    for images, labels in valid_tqdm:
                        images = images.cuda(cuda1)
                        labels = labels.cuda(cuda1)
                        output = net(images)
                        watch_nan(output)
                        meter.update(output, labels)
                        valid_tqdm.set_postfix(meter.get())
                logger.info('{:03d} valid result: {}'.format(epoch, meter.get()))
        if (epoch + 1) % 10 == 0:
            torch.save(net.state_dict(), 'ckpts/%.1f-sol-latest.pth' % pruning_rate)
            logger.info('saved to ckpts/%.1f-latest.pth' % pruning_rate)
        if best_accuracy < meter.top():
            best_accuracy = meter.top()
            torch.save(net.state_dict(), 'ckpts/%.1f-sol-best.pth' % pruning_rate)
            logger.info('saved to ckpts/%.1f-best.pth' % pruning_rate)
        scheduler.step()

if __name__ == '__main__':
    main()
