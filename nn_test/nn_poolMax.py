# -*- coding: utf-8 -*-
"""
@University:TYUST
@Project : learn_torch
@File    : nn_poolMax.py
@Author  : l1ber0
@Email   : 693096838@qq.com
@Date    : 2025/7/23 15:44
"""
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3,ceil_mode=False)

    def forward(self, input):
        output = self.maxpool(input)
        return output
net = Net()
writer = SummaryWriter("../logs_poolMax")


step = 0
for data in dataloader:
    inputs, target = data
    writer.add_images("pool_input", inputs, step)
    output = net(inputs)
    writer.add_images("pool_target",output,step)
    step += 1

writer.close()
