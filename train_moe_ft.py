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

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default='sgd')
parser.add_argument('--resume_checkpoint', '-r', default=None, help='resume from checkpoint')
parser.add_argument('--aug', action='store_true', help='use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
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
parser.add_argument('--pathway_num', default=2, type=int)
parser.add_argument('--consistency_alpha', default=0.0, type=float)
# train member
parser.add_argument('--pre_trained', default='')
parser.add_argument('--member_num', default=2000, type=int)
parser.add_argument('--dropout', default=0.0, type=float)
parser.add_argument('--save_shadow_id', default=None, type=int)


args = parser.parse_args()
args.name = args.net
if args.num_shadow is not None:	
    args.job_name = args.name + f'_shadow_{args.shadow_id}'	
else:	
    args.job_name = args.name + '_target'

if args.save_shadow_id is None:
    args.save_shadow_id = args.shadow_id

# take in args
name = args.job_name
file_name = f'ft-{args.pre_trained}-e{args.n_epochs}'
dir=f'saved_models/{file_name}'
os.makedirs(dir, exist_ok=True)


bs = int(args.bs)
imsize = int(args.size)

use_amp = not args.noamp
aug = args.aug

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
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
set_random_seed(args.seed) # no use the same seed

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

# select member data
# numbers_array = np.array([i for i in range(dataset_size)])
# tmp = np.setdiff1d(numbers_array, keep)
# member_idx = np.random.choice(tmp, size=args.member_num, replace=True)
idx = np.load(f'saved_member_idx/{args.dataset}-5000.npz')
member_idx = idx['member_idx']

# trainset = torch.utils.data.Subset(trainset, keep)
trainset = torch.utils.data.Subset(trainset, member_idx)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=4)
print(len(trainset), len(trainloader))
testset = tv_dataset(root='../data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)


# Model factory..
print('==> Building model..')
net = load_model(args)

# load pre_trained model
resume_checkpoint = f'saved_models/{args.pre_trained}/{args.net}_shadow_{args.shadow_id}_last.pth'
print(resume_checkpoint)
assert os.path.isfile(resume_checkpoint), 'Error: no checkpoint found!'
checkpoint = torch.load(resume_checkpoint)

net.load_state_dict(checkpoint['model'])
in_data = checkpoint['in_data']
cands = checkpoint['cands']

# Loss is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# fix_pathways = [cands[i%args.pathway_num] for i in range(len(trainloader)+1)]
# print(len(trainloader))
# print(len(fix_pathways))

def train(epoch, fix_pathway):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        pathway_ids = torch.from_numpy(fix_pathway).to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs, pathway_ids)
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
    
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    with open(f'{dir}/log_{args.save_shadow_id}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

list_loss = []
list_acc = []

# if usewandb:
#     wandb.watch(net)
    
net.cuda()
tb_writer = SummaryWriter(f'runs/{file_name}-{args.save_shadow_id}')

for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    val_loss, val_acc = test(epoch)
    trainloss = 0.0
    trainacc = 0.0
    for p in range(cands.shape[0]):
        trainloss1, trainacc1 = train(epoch, cands[p])
        trainloss += trainloss1
        trainacc += trainacc1
    
    trainloss  /= cands.shape[0]
    trainacc  /= cands.shape[0]
    scheduler.step() # step cosine scheduling, very important
    
    list_loss.append(val_loss)
    list_acc.append(val_acc)
    tb_writer.add_scalar('Loss/train', trainloss, epoch) 
    tb_writer.add_scalar('Loss/val', val_loss, epoch) 
    tb_writer.add_scalar('Acc/train', trainacc, epoch) 
    tb_writer.add_scalar('Acc/val', val_acc, epoch) 

tb_writer.close()

state = {"model": net.state_dict(),
        "in_data": keep,
        "cands": cands,
        "model_arch": args.net,
        "member_idx": member_idx
}
if args.num_shadow:
    torch.save(state, f'{dir}/{args.name}_shadow_{args.save_shadow_id}_last.pth')
else:
    torch.save(state, f'{dir}/{args.name}_target_last.pth')

# record training info
with open(f'{dir}/info_{args.save_shadow_id}.txt', 'w') as f:
    for arg in vars(args):
        f.write(f"{arg}: {getattr(args, arg)}\n")
