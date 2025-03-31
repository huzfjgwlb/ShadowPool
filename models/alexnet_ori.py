'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch.nn as nn
import torch


class AlexNet(nn.Module):

    def __init__(self, num_classes=10, droprate=0):
        super(AlexNet, self).__init__()
        # self.features = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),

        #     nn.Conv2d(64, 192, kernel_size=5, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),

        #     nn.Conv2d(192, 384, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),

        #     nn.Conv2d(384, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),

        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     # nn.MaxPool2d(kernel_size=4, stride=4),
        # )

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer1 = self.make_layer1(64, 192, kernel_size=5, padding_size=2)
        self.layer2 = self.make_layer(192, 384, kernel_size=3, padding_size=1)
        self.layer3 = self.make_layer(384, 256, kernel_size=3, padding_size=1)
        self.layer4 = self.make_layer(256, 256, kernel_size=3, padding_size=1)

        if droprate > 0.:
            self.fc = nn.Sequential(nn.Dropout(droprate),
                                    nn.Linear(256, num_classes))
        else:
            self.fc = nn.Linear(256, num_classes)
    
    def make_layer(self, in_size, out_size, kernel_size, padding_size):
        return nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=kernel_size, padding=padding_size),
            nn.ReLU(inplace=True),
        )

    def make_layer1(self, in_size, out_size, kernel_size, padding_size):
        return nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=kernel_size, padding=padding_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.max_pool2d(x,x.shape[-1])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def alexnet_ori(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(**kwargs)
    return model
