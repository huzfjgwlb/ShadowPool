# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import torchvision.transforms as transforms

import os
import argparse
import csv
import time
import copy
import random
from torch.utils.tensorboard import SummaryWriter

from models.utils import load_model
from utils.utils import *
from utils.randomaug import RandAugment

def diversity_loss_cos(feature1, feature2):
    feature1_flat = feature1.view(feature1.size(0), -1)  # Flatten along batch axis
    feature2_flat = feature2.view(feature2.size(0), -1)
    
    feature1_flat = F.normalize(feature1_flat, p=2, dim=1)
    feature2_flat = F.normalize(feature2_flat, p=2, dim=1)

    cosine_similarity = torch.sum(feature1_flat * feature2_flat, dim=-1)
    return torch.mean(cosine_similarity)

def diversity_loss_orth(feature1, feature2):
    feature1_flat = feature1.view(feature1.size(0), -1)  # Flatten along batch axis
    feature2_flat = feature2.view(feature2.size(0), -1)
    
    feature1_flat = F.normalize(feature1_flat, p=2, dim=1)
    feature2_flat = F.normalize(feature2_flat, p=2, dim=1)
    
    dot_product = torch.sum(feature1_flat * feature2_flat, dim=1)
    return  -torch.mean(torch.abs(dot_product))

diversity_loss_type = {
    'cos': diversity_loss_cos,
    'euc': nn.MSELoss(),
    'orth': diversity_loss_orth,
    'no': None
}
    
# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default='sgd')
parser.add_argument('--resume_checkpoint', '-r', default=None, help='resume from checkpoint')
parser.add_argument('--aug', action='store_true', help='use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='res18')
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--bs', default=512, type=int)
parser.add_argument('--size', default=32, type=int)
parser.add_argument('--n_epochs', default=100, type=int)
parser.add_argument('--num_total', default=None, type=int)
parser.add_argument('--patch', default=4, type=int, help="patch for ViT")
parser.add_argument('--dimhead', default=512, type=int)
parser.add_argument('--convkernel', default=8, type=int, help="parameter for convmixer")
parser.add_argument('--name', default='test')
parser.add_argument('--num_shadow', default=None, type=int)
parser.add_argument('--shadow_id', default=None, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--pkeep', default=0.5, type=float)
parser.add_argument('--split_type', default='random')
parser.add_argument("--n-expert", default=None, type=int)
parser.add_argument("--n-layer", default=4, type=int) # the number of MoE layer
parser.add_argument('--pathway_num', default=2, type=int)
parser.add_argument('--dropout', default=0.0, type=float)
parser.add_argument('--dis_loss', default='no')
parser.add_argument('--alpha', default=0.0, type=float)
parser.add_argument('--beta', default=0.0, type=float)


args = parser.parse_args()
print(args)
args.name = args.net
if args.num_shadow is not None:	
    args.job_name = args.name + f'_shadow_{args.shadow_id}'	
else:	
    args.job_name = args.name + '_target'

name = args.job_name
bs = int(args.bs)
use_amp = not args.noamp
aug = args.aug

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
imsize = int(args.size)
if args.net=="vit_timm":
    size = 384
else:
    size = imsize

tv_dataset = get_dataset(args)


if args.dataset == 'mnist':
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(args.data_mean, args.data_std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(args.data_mean, args.data_std),
    ])
else:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(args.data_mean, args.data_std),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(args.data_mean, args.data_std),
    ])

# Add RandAugment with N, M(hyperparameter)
if aug:
    N = 2; M = 14;
    transform_train.transforms.insert(0, RandAugment(N, M))

# Prepare dataset
trainset = tv_dataset(root='../data', train=True, download=True, transform=transform_train)
dataset_size = len(trainset)

if args.num_total:
    dataset_size = args.num_total

# set random seed
# set_random_seed(args.seed) # no use the same seed

# get shadow dataset
if args.num_shadow is not None:
    if args.split_type == 'random':
        keep = np.random.uniform(0, 1, size=(args.num_shadow, dataset_size))
        order = keep.argsort(0)
        keep = order < int(args.pkeep * args.num_shadow)
        keep = np.array(keep[args.shadow_id], dtype=bool)
        keep = keep.nonzero()[0]
    elif args.split_type == 'two':
        keep = np.random.uniform(0, 1, size=(args.num_shadow, dataset_size))
        order = keep.argsort(0)
        keep = order < int(args.pkeep * args.num_shadow)
        keep = np.array(keep[args.shadow_id % 2], dtype=bool) #two
        keep = keep.nonzero()[0]
    elif args.split_type == 'four':
        keep = np.random.uniform(0, 1, size=(args.num_shadow, dataset_size))
        order = keep.argsort(0)
        keep = order < int(args.pkeep * args.num_shadow)
        keep = np.array(keep[args.shadow_id % 4], dtype=bool) #one
        keep = keep.nonzero()[0]
else:
    # get target dataset
    keep = np.random.choice(dataset_size, size=int(args.pkeep * dataset_size), replace=False)
    keep.sort()

idx = np.load(f'saved_member_idx/{args.dataset}-5000.npz')
member_idx = idx['member_idx']

orig_len = len(keep)
# non-overlap between member_idx and training data
keep = np.setdiff1d(keep, member_idx, assume_unique=True)
if len(keep) < int(args.pkeep * dataset_size):
    additional = np.random.choice(
        np.setdiff1d(np.arange(dataset_size), np.union1d(keep, member_idx), assume_unique=True),
        size=orig_len - len(keep),
        replace=False
    )
    keep = np.concatenate([keep, additional])

trainset = torch.utils.data.Subset(trainset, keep)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=4)
print(len(trainset))
testset = tv_dataset(root='../data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

arrays = [range(args.n_expert) for _ in range(args.n_layer)]  # resnet=4
cands = np.array(entire_combinations(arrays))
random_ind = np.random.choice(cands.shape[0], size=args.pathway_num, replace=False)
non_random_ind = np.setdiff1d(range(cands.shape[0]), random_ind)
non_cands = cands[non_random_ind]
cands = cands[random_ind]

# Model factory..
print('==> Building model..')
net = load_model(args)

# Loss is CE
criterion = nn.CrossEntropyLoss()
dis_criterion = diversity_loss_type[args.dis_loss]

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

fix_pathways = [cands[i%args.pathway_num] for i in range(len(trainloader)+1)]
random_indices = np.random.choice(cands.shape[0], size=len(trainloader)+1, replace=True)
comp_pathways = cands[random_indices, :] 
print(len(trainloader))
print(len(fix_pathways), len(comp_pathways))


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    use_dis_loss = args.dis_loss != 'no'
    alpha, beta = args.alpha, args.beta

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        pathway0 = fix_pathways[batch_idx]
        pathway1 = comp_pathways[batch_idx]

        with torch.cuda.amp.autocast(enabled=use_amp):
            if use_dis_loss:
                outputs, *out0_list = net.intermediate(inputs, pathway0)
                outputs1, *out1_list = net.intermediate(inputs, pathway1)

                logits = F.softmax(outputs[0].float(), dim=-1)
                logits1 = F.softmax(outputs1[0].float(), dim=-1)

                # Calculate loss with KL divergence and disparity criteria
                loss = criterion(outputs, targets) - alpha * symmetric_KL_loss(logits, logits1)
                loss -= beta * sum(dis_criterion(out0, out1) for out0, out1 in zip(out0_list, out1_list))
            else:
                outputs = net(inputs, pathway0)
                loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return train_loss/(batch_idx+1), correct/total

##### Validation
def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            random_ind = np.random.choice(cands.shape[0], size=1, replace=True)
            pathway_ids = torch.from_numpy(cands[random_ind][0]).to(device)

            # random_ind = np.random.choice(cands.shape[0], size=inputs.shape[0], replace=True)
            # pathway_ids = torch.from_numpy(cands[random_ind]).to(device)
            # pathway_ids = torch.from_numpy(np.array([0,1,0,0])).to(device)
            outputs = net(inputs, pathway_ids)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    acc = 100.*correct/total
    
    os.makedirs('logs', exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    with open(f'logs/{name}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

list_loss = []
list_acc = []
    
net.cuda()
file_name = f'{args.dataset}-{args.net}-{args.pkeep}-{args.n_epochs}-{args.dis_loss}-{args.alpha}-{args.beta}'
tb_writer = SummaryWriter(f'runs/{file_name}-{args.shadow_id}')


for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss, trainacc = train(epoch)
    val_loss, val_acc = test(epoch)
    
    scheduler.step() # step cosine scheduling, very important
    
    list_loss.append(val_loss)
    list_acc.append(val_acc)
    tb_writer.add_scalar('Loss/train', trainloss, epoch) 
    tb_writer.add_scalar('Loss/val', val_loss, epoch) 
    tb_writer.add_scalar('Acc/train', trainacc, epoch) 
    tb_writer.add_scalar('Acc/val', val_acc, epoch) 
    # Write out csv..
    with open(f'logs/{name}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss) 
        writer.writerow(list_acc) 
tb_writer.close()

state = {"model": net.state_dict(),
        "in_data": keep,
        "cands": cands,
        "non_candas": non_cands,
        "model_arch": args.net
        }
dir=f'saved_models/{file_name}'
os.makedirs(dir, exist_ok=True)
if args.num_shadow:
    torch.save(state, f'{dir}/{args.name}_shadow_{args.shadow_id}_last.pth')
else:
    torch.save(state, f'{dir}/{args.name}_target_last.pth')