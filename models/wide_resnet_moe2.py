import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet_moe2(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, n_expert=4):
        super(Wide_ResNet_moe2, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        # print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        # self.conv1 = conv3x3(3,nStages[0])
        # self.layer1 = self._wide_layer(1, wide_basic, nStages[1], n, dropout_rate, stride=1)
        # self.layer2 = self._wide_layer(2, wide_basic, nStages[2], n, dropout_rate, stride=2)
        # self.layer3 = self._wide_layer(3, wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.layer1 = nn.ModuleList([conv3x3(3,nStages[0]) for _ in range(n_expert)])
        self.layer2 = nn.ModuleList([self._wide_layer(1, wide_basic, nStages[1], n, dropout_rate, stride=1, in_planes=self.in_planes) for _ in range(n_expert)])
        self.layer3 = nn.ModuleList([self._wide_layer(2, wide_basic, nStages[2], n, dropout_rate, stride=2, in_planes=160) for _ in range(n_expert)])
        self.layer4 = nn.ModuleList([self._wide_layer(3, wide_basic, nStages[3], n, dropout_rate, stride=2, in_planes=320) for _ in range(n_expert)])

        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, n, block, planes, num_blocks, dropout_rate, stride, in_planes):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(in_planes, planes, dropout_rate, stride))
            in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, pathway_ids=None):
        assert pathway_ids is not None, "pathway_ids must be provided"

        out = self.layer1[pathway_ids[0]](x) #self.conv1(x)
        out = self.layer2[pathway_ids[1]](out)
        out = self.layer3[pathway_ids[2]](out)
        out = self.layer4[pathway_ids[3]](out)

        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out
    
    def intermediate(self, x, pathway_ids=None):
        assert pathway_ids is not None, "pathway_ids must be provided"

        # out = self.conv1(x)
        out1 = self.layer1[pathway_ids[0]](x)
        out2 = self.layer2[pathway_ids[1]](out1)
        out3 = self.layer3[pathway_ids[2]](out2)
        out4 = self.layer4[pathway_ids[3]](out3)

        out = F.relu(self.bn1(out4))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out, out1, out2, out3, out4
