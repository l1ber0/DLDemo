# -*- coding: utf-8 -*-
"""
@University:TYUST
@Project : learn_torch
@File    : nn_seq.py
@Author  : l1ber0
@Email   : 693096838@qq.com
@Date    : 2025/7/23 17:49
"""
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter



class NN_Seq(nn.Module):
    def __init__(self):
        super(NN_Seq, self).__init__()

        self.modle1 =  Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )

    def forward(self, x):
        self.modle1(x)
        return x

nn1 = NN_Seq()
input = torch.ones(64, 3, 32, 32)


writer = SummaryWriter("../../logs_seq")
writer.add_graph(nn1,input)
writer.close()

