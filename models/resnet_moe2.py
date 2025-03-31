import torch.nn as nn
import torch.nn.functional as F
import torch


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, conv_layer, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_layer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv_layer(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False, ),
                nn.BatchNorm2d(self.expansion * planes), )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, conv_layer, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv_layer(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv_layer(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv_layer(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False, ),
                nn.BatchNorm2d(self.expansion * planes), )

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


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, ratio=1, n_expert=4, dropout=0.0, **kwargs):
        super(ResNet, self).__init__()
        self.in_planes = int(ratio * 64)
        self.conv_layer = nn.Conv2d
        self.normalize = None
        self.ratio = ratio
        self.dropout = dropout

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        self.layer1 = nn.ModuleList([self._make_layer(block, 64, num_blocks[0], stride=1, in_planes=self.in_planes, seed=i) for i in range(n_expert)])
        self.layer2 = nn.ModuleList([self._make_layer(block, 128, num_blocks[1], stride=2, in_planes=64, seed=i) for i in range(n_expert)])
        self.layer3 = nn.ModuleList([self._make_layer(block, 256, num_blocks[2], stride=2, in_planes=128, seed=i) for i in range(n_expert)])
        self.layer4 = nn.ModuleList([self._make_layer(block, 512, num_blocks[3], stride=2, in_planes=256, seed=i) for i in range(n_expert)])

        self.linear = nn.Linear(int(ratio * 512) * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, in_planes, seed):
        if seed is not None:
            torch.manual_seed(seed)

        planes = int(self.ratio * planes)
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(in_planes, planes, self.conv_layer, stride))
            in_planes = planes  # in_planes remains consistent for each block
        return nn.Sequential(*layers)

    def forward(self, x, pathway_ids=None):
        if self.normalize is not None:
            x = self.normalize(x)
        assert pathway_ids is not None, "pathway_ids must be provided"

        out = F.relu(self.bn1(self.conv1(x)))
        # dropout
        # out = F.dropout(out, p=self.dropout, training=self.training) # only valid in training
        out = self.layer1[pathway_ids[0]](out)
        out = self.layer2[pathway_ids[1]](out)
        out = self.layer3[pathway_ids[2]](out)
        out = self.layer4[pathway_ids[3]](out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def intermediate(self, x, pathway_ids=None):
        if self.normalize is not None:
            x = self.normalize(x)
        assert pathway_ids is not None, "pathway_ids must be provided"

        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1[pathway_ids[0]](out)
        out2 = self.layer2[pathway_ids[1]](out1)
        out3 = self.layer3[pathway_ids[2]](out2)
        out4 = self.layer4[pathway_ids[3]](out3)

        out = F.avg_pool2d(out4, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, out1, out2, out3, out4


def resnet18_moe2(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34_moe2(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50_moe2(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101_moe2(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152_moe2(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


# if __name__ == "__main__":
#     import torch

#     model = resnet18_cifar_moe2(num_classes=10, ratio=1.0)
#     inputs = torch.randn([3, 3, 64, 64])
#     outputs = model(inputs)
