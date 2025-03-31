# -*- coding: utf-8 -*-

'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


cfg = {
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

# 'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],

class VGG16_moe(nn.Module):
    def __init__(self, num_classes=10, n_expert=4):
        super(VGG16_moe, self).__init__()

        self.features = self._make_layer(3, 64, num_layers=2)

        self.layer1 = nn.ModuleList([self._make_layer(64, 128, num_layers=2) for _ in range(n_expert)])
        self.layer2 = nn.ModuleList([self._make_layer(128, 256, num_layers=3) for _ in range(n_expert)])
        self.layer3 = nn.ModuleList([self._make_layer(256, 512, num_layers=3) for _ in range(n_expert)])
        self.layer4 = nn.ModuleList([self._make_layer(512, 512, num_layers=3) for _ in range(n_expert)])

        self.classifier = nn.Linear(512, num_classes)
    

    def _make_layer(self, in_channels, out_channels, num_layers=2, use_pooling=True):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        if use_pooling:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)


    def forward(self, x, pathway_ids=None):
        assert pathway_ids is not None, "pathway_ids must be provided"
        out = self.features(x)

        out = self.layer1[pathway_ids[0]](out)
        out = self.layer2[pathway_ids[1]](out)
        out = self.layer3[pathway_ids[2]](out)
        out = self.layer4[pathway_ids[3]](out)

        out = F.avg_pool2d(out, kernel_size=1, stride=1)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    def intermediate(self, x, pathway_ids=None):
        assert pathway_ids is not None, "pathway_ids must be provided"

        out = self.features(x)
        out1 = self.layer1[pathway_ids[0]](out)
        out2 = self.layer2[pathway_ids[1]](out1)
        out3 = self.layer3[pathway_ids[2]](out2)
        out4 = self.layer4[pathway_ids[3]](out3)

        out = F.avg_pool2d(out4, kernel_size=1, stride=1)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out, out1, out2, out3, out4
