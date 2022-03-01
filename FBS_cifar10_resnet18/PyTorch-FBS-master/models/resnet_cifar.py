'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

from . import ChannelPruning, SwitchableBatchNorm2d

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

global partition_point
partition_point = False

def conv3x3(in_planes: int,
            out_planes: int,
            stride: int = 1,
            groups: int = 1,
            dilation: int = 1,
            normalization: str = '',
            activation: str = '') -> nn.Module:
    layers = []
    layers.append(
        nn.Conv2d(in_planes,
                  out_planes,
                  kernel_size=3,
                  stride=stride,
                  groups=groups,
                  padding=dilation,
                  dilation=dilation,
                  bias=False)
    )
    if normalization == 'bn':
        layers.append(nn.BatchNorm2d(out_planes))
        #layers.append(SwitchableBatchNorm2d(out_planes))
    if activation == 'relu':
        layers.append(nn.ReLU())
    #return ChannelPruning(*layers)
    return nn.Sequential(*layers)


def conv1x1(in_planes: int,
            out_planes: int,
            stride: int = 1,
            normalization: str = '',
            activation: str = '') -> nn.Module:
    layers = []
    layers.append(
        nn.Conv2d(in_planes,
                  out_planes,
                  kernel_size=1,
                  stride=stride,
                  bias=False)
    )
    if normalization == 'bn':
        layers.append(nn.BatchNorm2d(out_planes))
        #layers.append(SwitchableBatchNorm2d(out_planes))
    if activation == 'relu':
        layers.append(nn.ReLU())
    #return ChannelPruning(*layers)
    return nn.Sequential(*layers)


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        #self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(planes)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.conv1 = conv3x3(in_planes, planes, stride, normalization='bn', activation='relu')
        self.conv2 = conv3x3(planes, planes, normalization='bn')
        
        self.channel_pruning = ChannelPruning(in_planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                '''
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
                '''
                self.shortcut = conv1x1(in_plances, self.expansion * planes, stride, normalization='bn')

    def forward(self, x):
        global partition_point
        #out = F.relu(self.bn1(self.conv1(x)))
        #out = self.bn2(self.conv2(out))
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        if partition_point == True:
            return out * self.channel_pruning(x)
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        global partition_point
        out = F.relu(self.bn1(self.conv1(x)))
        partition_point = True
        out = self.layer1(out)
        partition_point = False
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)


def resnet32(num_classes=10):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)


def resnet44(num_classes=10):
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes)


def resnet56(num_classes=10):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes)


def resnet110(num_classes=10):
    return ResNet(BasicBlock, [18, 18, 18], num_classes=num_classes)


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
