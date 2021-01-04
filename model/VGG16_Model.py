#!/usr/bin/env python
# encoding: utf-8

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
import math
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import init as init
import numpy as np

def myphi(x,m):
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 2, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos())
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta,self.m)
            phi_theta = phi_theta.clamp(-1*self.m,1)

        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)
        output = (cos_theta,phi_theta)
        return output # size=(B,Classnum,2)


class AngleLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte()
        index = Variable(index)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss


loss_criterion = nn.CrossEntropyLoss().cuda()
loss_criterion_Angular = AngleLoss().cuda()

class VGG16(nn.Module):
    def __init__(self, args, init_weights=True):
        super(VGG16, self).__init__()
        self.features = []
        self.Nd = args.Nd
        self.Channel = args.Channel
        self.AngleLoss = args.Angle_Loss

        ConvBlock1 = [
            nn.Conv2d(self.Channel, 64, 3, 1, 1),  # conv1_1
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 1, 1),  # conv1_2
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(2, stride=2),  # pool1
        ]
        ConvBlock2 = [
            nn.Conv2d(64, 128, 3, 1, 1),  # conv2_1
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 128, 3, 1, 1),  # conv2_2
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(2, stride=2),  # pool2
        ]
        ConvBlock3 = [
            nn.Conv2d(128, 256, 3, 1, 1),  # conv3_1
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 256, 3, 1, 1),  # conv3_2
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 256, 3, 1, 1),  # conv3_3
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.MaxPool2d(2, stride=2),  # pool3
        ]
        ConvBlock4 = [
            nn.Conv2d(256, 512, 3, 1, 1),  # conv4_1
            nn.BatchNorm2d(512),
            nn.ELU(),
            nn.Conv2d(512, 512, 3, 1, 1),  # conv4_2
            nn.BatchNorm2d(512),
            nn.ELU(),
            nn.Conv2d(512, 512, 3, 1, 1),  # conv4_3
            nn.BatchNorm2d(512),
            nn.ELU(),
            nn.MaxPool2d(2, stride=2),  # pool4
        ]
        ConvBlock5 = [
            nn.Conv2d(512, 512, 3, 1, 1),  # conv5_1
            nn.BatchNorm2d(512),
            nn.ELU(),
            nn.Conv2d(512, 512, 3, 1, 1),  # conv5_2
            nn.BatchNorm2d(512),
            nn.ELU(),
            nn.Conv2d(512, 512, 3, 1, 1),  # conv5_3
            nn.BatchNorm2d(512),
            nn.ELU(),
            # nn.MaxPool2d(2, stride=2),  # pool5
            nn.AdaptiveAvgPool2d((2, 2))
        ]


        self.convLayers1 = nn.Sequential(*ConvBlock1)
        self.convLayers2 = nn.Sequential(*ConvBlock2)
        self.convLayers3 = nn.Sequential(*ConvBlock3)
        self.convLayers4 = nn.Sequential(*ConvBlock4)
        self.convLayers5 = nn.Sequential(*ConvBlock5)

        self.FC6 = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.FC7 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.6),
        )
        if self.AngleLoss:
            self.FC8 = AngleLinear(4096, self.Nd)
        else:
            self.FC8 = nn.Linear(4096, self.Nd)


        # Intialize layers to specific weights
        if init_weights:
            self._initialize_weights()

    def forward(self, input, ExtractMode=False):

        x1 = self.convLayers1(input)
        x2 = self.convLayers2(x1)
        x3 = self.convLayers3(x2)
        x4 = self.convLayers4(x3)
        x = self.convLayers5(x4)

        x = x.view(np.shape(x)[0], -1) # np.shape(x)[0] -> batch
        x = self.FC6(x)
        x = self.FC7(x)
        self.features = x
        x = self.FC8(x)

        if ExtractMode:
            return self.features
        else:
            return x

    def ID_Loss(self, predic, label):

        if self.AngleLoss:
            Loss = loss_criterion_Angular((predic[0][:, :self.Nd], predic[1][:, :self.Nd]), label)
        else:
            Loss = loss_criterion(predic[:, :self.Nd], label)


        return Loss


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)

