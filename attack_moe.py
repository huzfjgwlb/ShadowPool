import torch
import torch.nn as nn
import numpy as np

import torchvision.transforms as transforms

import pandas
import os
import argparse
import copy
# import wandb
import random

# from pynvml import *

from utils.utils import *
from models.inferencemodel import *


def main(args):
    print(args)
    
    # set random seed
    set_random_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    tv_dataset = get_dataset(args)

    # dataset
    if args.dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(args.data_mean, args.data_std),
        ])
    else:
        transform_train = transforms.Compose([
        transforms.Resize(args.size),
        transforms.ToTensor(),
        transforms.Normalize(args.data_mean, args.data_std),
        ])
    trainset = tv_dataset(root='../data', train=True, download=True, transform=transform_train)
    testset = tv_dataset(root='../data', train=False, download=True, transform=transform_train)
    # args.aug_trainset = trainset
    if args.dataset == 'mnist':
        transform_aug = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(args.data_mean, args.data_std),
        ])
    else:
        if args.no_dataset_aug:
            transform_aug = transforms.Compose([
            transforms.Resize(args.size),
            transforms.ToTensor(),
            transforms.Normalize(args.data_mean, args.data_std),
            ])
        else:
            transform_aug = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.Resize(args.size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(args.data_mean, args.data_std),
            ])
    args.aug_trainset = tv_dataset(root='../data', train=True, download=True, transform=transform_aug)
    # enum candiate pathway
    arrays = [range(args.n_expert) for _ in range(4)] # resnet=4
    cands = np.array(entire_combinations(arrays))
    random_indices = np.random.choice(cands.shape[0], size=args.num_augment, replace=False)
    selected_pathway = cands[random_indices, :]  

    # load shadow and target models
    shadow_models = []
    shadow_common_ids = []
    shadow_union_ids = []
    shadow_overfits = []
    shadow_train_acc = []
    shadow_test_acc = []
    shadow_ids_list = []
    shadow_pathways = []
    test_img, test_cls = get_examples(range(len(testset)), testset)
    for i in range(args.num_shadow):
        curr_model = InferenceMoE(args, i).to(args.device)
        shadow_common_ids = curr_model.in_data if i == 0 else np.intersect1d(shadow_common_ids, curr_model.in_data)
        shadow_union_ids = curr_model.in_data if i == 0 else np.union1d(shadow_union_ids, curr_model.in_data)
        shadow_models.append(curr_model)
        shadow_ids_list.append(curr_model.in_data)

        ids = np.random.choice(curr_model.in_data, size=min(len(testset), len(curr_model.in_data)), replace=False)
        train_img, train_cls = get_examples(ids, trainset)

        if curr_model.cands is not None:
            curr_cands = curr_model.cands
            random_indices = np.random.choice(curr_cands.shape[0], size=args.num_augment, replace=False)
            selected_pathway = curr_cands[random_indices, :] 
            shadow_pathways.append(selected_pathway)
        
        for p in range(len(selected_pathway)):
            pathway_ids = torch.from_numpy(np.array(selected_pathway[p])).to(device)
            train_acc = curr_model.predictions(train_img, train_cls, pathway_ids)
            test_acc = curr_model.predictions(test_img, test_cls, pathway_ids)
            shadow_train_acc.append(train_acc)
            shadow_test_acc.append(test_acc)
            shadow_overfits.append(train_acc - test_acc)
                    
    candidate_ids = np.setdiff1d(shadow_union_ids, shadow_common_ids)
    shadow_ids = []
    for _id in candidate_ids:
        in_num = 0
        out_num = 0
        for ids in shadow_ids_list:
            if _id in ids:
                in_num += 1
            else:
                out_num += 1
        if in_num == int(args.num_shadow/2) and out_num == int(args.num_shadow/2):
            shadow_ids.append(_id)


    # target models
    target_model = InferenceMoE(args, -1).to(args.device)
    train_img, train_cls = get_examples(target_model.in_data, trainset)
    target_train_acc = target_model.predictions(train_img, train_cls)
    target_test_acc = target_model.predictions(test_img, test_cls)
    target_overfits = target_train_acc - target_test_acc

    args.target_model = target_model
    args.shadow_models = shadow_models

    args.img_shape = trainset[0][0].shape
    args.canary_shape = trainset[0][0].shape

    args.pred_logits = [] # N x (num of shadow + 1) x num_trials x num_class (target at -1)
    args.in_out_labels = [] # N x (num of shadow + 1)
    args.canary_losses = [] # N x num_trials
    args.class_labels = []  # N
    args.img_id = [] # N

    for i in range(args.num_query):

        args.target_img_id = shadow_ids[i] #i

        args.target_img, args.target_img_class = trainset[args.target_img_id]
        args.target_img = args.target_img.unsqueeze(0).to(args.device)

        args.in_out_labels.append([])
        args.canary_losses.append([])
        args.pred_logits.append([])

        curr_canaries = generate_aug_imgs(args)

        # get logits
        curr_canaries = torch.cat(curr_canaries, dim=0).to(args.device)
        for j in range(len(shadow_models)):
            curr_model = shadow_models[j]
            selected_pathway = shadow_pathways[j]
            for p in range(len(selected_pathway)):
                pathway_ids = torch.from_numpy(np.array(selected_pathway[p])).to(device)
                args.pred_logits[-1].append(get_logits_moe(curr_canaries, curr_model, pathway_ids))
                args.in_out_labels[-1].append(int(args.target_img_id in curr_model.in_data))

        args.pred_logits[-1].append(get_logits(curr_canaries, target_model))
        args.in_out_labels[-1].append(int(args.target_img_id in target_model.in_data))

        args.img_id.append(args.target_img_id)
        args.class_labels.append(args.target_img_class)
        
        # progress_bar(i, args.end - args.start)

    # accumulate results
    pred_logits = np.array(args.pred_logits)
    in_out_labels = np.array(args.in_out_labels)
    canary_losses = np.array(args.canary_losses)
    class_labels = np.array(args.class_labels)
    img_id = np.array(args.img_id)

    # save predictions
    os.makedirs(f'saved_predictions/{args.shadow_dir}/', exist_ok=True)
    np.savez(f'saved_predictions/{args.shadow_dir}/models-{args.num_augment}.npz', pred_logits=pred_logits, in_out_labels=in_out_labels, canary_losses=canary_losses, class_labels=class_labels,
            img_id=img_id)
    ### dummy calculatiton of auc and acc
    ### to be simplified
    pred = np.load(f'saved_predictions/{args.shadow_dir}/models-{args.num_augment}.npz')

    pred_logits = pred['pred_logits']
    in_out_labels = pred['in_out_labels']
    canary_losses = pred['canary_losses']
    class_labels = pred['class_labels']
    img_id = pred['img_id']

    in_out_labels = np.swapaxes(in_out_labels, 0, 1).astype(bool)
    pred_logits = np.swapaxes(pred_logits, 0, 1)

    scores = calibrate_logits(pred_logits, class_labels, args.logits_strategy)

    shadow_scores = scores[:-1]
    target_scores = scores[-1:]
    shadow_in_out_labels = in_out_labels[:-1]
    target_in_out_labels = in_out_labels[-1:]


    # return 
    some_stats = cal_results(shadow_scores, shadow_in_out_labels, target_scores, target_in_out_labels, logits_mul=args.logits_mul)

    print(some_stats)
    row = {
        'net': args.name, 
        'num_shadow': args.num_shadow,
        'num_query': args.num_query,
        'num_augment': args.num_augment,
        'n_expert': args.n_expert,
        'ratio': args.ratio,
        'shadow_dir': args.shadow_dir,
        # res
        'fix_auc': f"{some_stats['fix_auc']:.4f}",
        'fix_acc': f"{some_stats['fix_acc']:.4f}",
        'fix_TPR@0.01FPR': f"{some_stats['fix_TPR@0.01FPR']:.4f}",
        'fix_off_auc': f"{some_stats['fix_off_auc']:.4f}",
        'fix_off_acc': f"{some_stats['fix_off_acc']:.4f}",
        'fix_off_TPR@0.01FPR': f"{some_stats['fix_off_TPR@0.01FPR']:.4f}",
        'target_train_acc': f"{target_train_acc:.4f}",
        'target_test_acc': f"{target_test_acc:.4f}",
        'target_overfit': f"{target_overfits:.4f}",
        'shadow_train_acc': "-".join([str(round(i, 4)) for i in shadow_train_acc]),
        'shadow_test_acc': "-".join([str(round(i, 4)) for i in shadow_test_acc]),
        'shadow_overfit': "-".join([str(round(i, 4)) for i in shadow_overfits]),

    }
    write_to_csv(row, '%s_%s_%s.csv' % (args.csv_prefix, args.name, args.dataset))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Gen Canary')
    parser.add_argument('--logits_strategy', default='log_logits')
    parser.add_argument('--logits_mul', default=1, type=int)
    parser.add_argument('--no_dataset_aug', action='store_true')
    # general setting
    parser.add_argument('--bs', default=512, type=int)
    parser.add_argument('--size', default=32, type=int)
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--save_preds', action='store_true')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--net', default='res18')
    parser.add_argument('--num_query', default=1000, type=int)
    parser.add_argument('--num_aug', default=1, type=int) # sample augmentation
    parser.add_argument('--num_shadow', default=None, type=int, required=True)
    parser.add_argument('--scaling_ratio', default=1.0, type=float)
    parser.add_argument('--num_augment', default=0, type=int) # shadow model augmentation
    parser.add_argument('--aug_policy', default='no', type=str)
    parser.add_argument('--target_net', default='')
    parser.add_argument('--shadow_net', default='')
    parser.add_argument('--shadow_dir', default='', type=str)
    parser.add_argument('--target_dir', default='', type=str)
    parser.add_argument('--name', default='')
    parser.add_argument('--csv_prefix', default='ft')
    # MoE
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--n-expert", default=None, type=int)
    # attack

    args = parser.parse_args()
    
    main(args)
