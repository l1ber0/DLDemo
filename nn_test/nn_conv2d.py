# -*- coding: utf-8 -*-
"""
@University:TYUST
@Project : learn_torch
@File    : nn_conv2d.py
@Author  : l1ber0
@Email   : 693096838@qq.com
@Date    : 2025/7/22 16:49
"""

import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data",train= True,transform=torchvision.transforms.ToTensor(),download=True)
dataloder = DataLoader(dataset,batch_size=64)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = Conv2d(3,6,3,stride=1,padding=0)

    def forward(self,x):
        conv1 = self.conv1(x)
        return conv1

net = Net()

writer = SummaryWriter("../logs")
step = 0
#每一张输入
for data in dataloder:
    imgs,target = data
    output = net(imgs)
    writer.add_images("imgs",imgs,step)
    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images("output",output,step)
    step += 1