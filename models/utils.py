from models import *
from models.alexnet import AlexNet
from models.resnet_moe import resnet18_moe
from models.resnet_moe2 import resnet18_moe2
from models.resnet_moe3 import resnet18_moe3
from models.vgg_moe import VGG16_moe
from models.wide_resnet_moe import Wide_ResNet_moe
from models.wide_resnet_moe2 import Wide_ResNet_moe2
from models.alexnet_moe import alexnet_moe

from models.resnet_ori import resnet18_ori
from models.alexnet_ori import alexnet_ori
from models.vgg_ori import VGG13_ori, VGG16_ori
from models.simplemlp import SimpleMLP

def load_model(args):

    # add ori
    if args.net=='wrn28': #  'wrn28-10'
        net = Wide_ResNet(28, 10, 0.3, args.num_classes)
    elif args.net=='res18-ori':
        net = resnet18_ori(num_classes=args.num_classes)
    elif args.net=='alex-ori':
        net = alexnet_ori(num_classes=args.num_classes)
    elif args.net=='vgg13-ori':
        net = VGG13_ori(num_classes=args.num_classes)
    elif args.net=='vgg16-ori':
        net = VGG16_ori(num_classes=args.num_classes)
    elif args.net=='mlp':
        net = SimpleMLP(input_dim=args.input_dim, layer_sizes=[32, 16, 8, 4], num_classes=args.num_classes)

    # add moe
    elif args.net=='res18-moe':
        net = resnet18_moe(num_classes=args.num_classes, n_expert=args.n_expert, ratio=args.ratio)
    elif args.net=='res18-moe2':
        net = resnet18_moe2(num_classes=args.num_classes, n_expert=args.n_expert, dropout=0.0)
    elif args.net=='res18-moe3':
        net = resnet18_moe3(num_classes=args.num_classes, n_expert=args.n_expert)
    elif args.net=='vgg16-moe':
        net = VGG16_moe(num_classes=args.num_classes, n_expert=args.n_expert)
    elif args.net=='wrn28-moe':
        net = Wide_ResNet_moe(28, 10, 0.3, args.num_classes, n_expert=args.n_expert)
    elif args.net=='wrn28-moe2':
        net = Wide_ResNet_moe2(28, 10, 0.3, args.num_classes, n_expert=args.n_expert)
    elif args.net=='alex-moe':
        net = alexnet_moe(num_classes=args.num_classes, n_expert=args.n_expert)
    else:
        raise NotImplementedError()

    return net