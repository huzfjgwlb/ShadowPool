import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.moe_layer import MoEConv, MoEBase


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, conv_layer, stride=1, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_layer(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_layer(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False, **kwargs
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, conv_layer, stride=1, **kwargs):
        super(Bottleneck, self).__init__()
        self.conv1 = conv_layer(in_planes, planes, kernel_size=1, bias=False, **kwargs
                                )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_layer(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, **kwargs
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv_layer(
            planes, self.expansion * planes, kernel_size=1, bias=False, **kwargs
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def percentile(t, q):
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    return t.view(-1).kthvalue(k).values.item()


class ResNet(MoEBase):
    def __init__(self, block, num_blocks, num_classes=10, n_expert=5, ratio=1.0):
        super(ResNet, self).__init__()
        self.ratio = ratio
        self.in_planes = 64
        self.conv_layer = MoEConv
        self.num_blocks = num_blocks
        self.normalize = None
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, n_expert=n_expert)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, n_expert=n_expert)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, n_expert=n_expert)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, n_expert=n_expert)

        self.linear = nn.Linear(int(ratio * 512) * block.expansion, num_classes)
        self.n_expert = n_expert

    def _make_layer(self, block, planes, num_blocks, stride, **kwargs):
        planes = int(self.ratio * planes)
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.conv_layer, stride, **kwargs))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def generate_scores(self, x, pathway_id):
        _tmp = torch.ones((x.shape[0], self.n_expert)).to(x)
        _tmp[:, pathway_id] *= -1
        return _tmp

    # def forward(self, x, pathway_ids=None):
    #     if self.normalize is not None:
    #         x = self.normalize(x)
    #     if self.router is not None:
    #         self.set_score(self.router(x))
    #     out = F.relu(self.bn1(self.conv1(x)))

    #     if len(pathway_ids.shape) == 1:
    #         pathway_ids = pathway_ids.repeat(x.shape[0], 1)

    #     # first router
    #     B = torch.full_like(pathway_ids, -1.0).to(pathway_ids)
    #     indices = pathway_ids[:, 0]
    #     rows = torch.arange(pathway_ids.size(0)).to(pathway_ids)
    #     B[rows, indices] = 1.0
    #     B = B.float()
    #     self.set_score(B)
    #     out = self.layer1(out)

    #     # second router
    #     B = torch.full_like(pathway_ids, -1.0).to(pathway_ids)
    #     indices = pathway_ids[:, 1]
    #     rows = torch.arange(pathway_ids.size(0)).to(pathway_ids)
    #     B[rows, indices] = 1.0
    #     B = B.float()
    #     self.set_score(B)        
    #     out = self.layer2(out)

    #     # third router
    #     B = torch.full_like(pathway_ids, -1.0).to(pathway_ids)
    #     indices = pathway_ids[:, 2]
    #     rows = torch.arange(pathway_ids.size(0)).to(pathway_ids)
    #     B[rows, indices] = 1.0
    #     B = B.float()
    #     self.set_score(B)
    #     out = self.layer3(out)

    #     # fourth router
    #     B = torch.full_like(pathway_ids, -1.0).to(pathway_ids)
    #     indices = pathway_ids[:, 3]
    #     rows = torch.arange(pathway_ids.size(0)).to(pathway_ids)
    #     B[rows, indices] = 1.0
    #     B = B.float()
    #     self.set_score(B)
    #     out = self.layer4(out)
    #     out = F.avg_pool2d(out, 4)
    #     out = out.view(out.size(0), -1)
    #     out = self.linear(out)
    #     return out
    
    def forward(self, x):
        if self.normalize is not None:
            x = self.normalize(x)
        if self.router is not None:
            self.set_score(self.router(x))
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.AdaptiveAvgPool2d((1, 1))(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



def resnet18_moe(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34_moe(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50_moe(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101_moe(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152_moe(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
