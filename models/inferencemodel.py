import torch
import os

import torch
import torch.nn as nn
import numpy as np

from models import *
from models.utils import load_model
from models.prunes import all_scale
from models.layers.router import build_router
from utils.utils import set_router

class InferenceModel(nn.Module):
    def __init__(self, shadow_id, args, enable_aug=False):
        super().__init__()
        self.shadow_id = shadow_id
        self.args = args
        
        if self.shadow_id == -1:
            # -1 for target model
            target_net = args.target_net if args.target_net != '' else args.net
            target_dir = args.target_dir if args.target_dir != '' else  f'{args.net}-{args.dataset}'
            resume_checkpoint = f'saved_models/{target_dir}/{target_net}_target_last.pth'
        else:
            shadow_net = args.shadow_net if args.shadow_net != '' else args.net
            shadow_dir = args.shadow_dir if args.shadow_dir != '' else  f'{args.net}-{args.dataset}'
            resume_checkpoint = f'saved_models/{shadow_dir}/{shadow_net}_shadow_{self.shadow_id}_last.pth'

        print('augmented' if enable_aug else 'ori', resume_checkpoint, args.aug_policy, args.scaling_ratio)
        assert os.path.isfile(resume_checkpoint), 'Error: no checkpoint found!'
        checkpoint = torch.load(resume_checkpoint)
        if 'model_arch' in checkpoint:
            args.net = checkpoint['model_arch']
        self.model = load_model(args)
        self.model.load_state_dict(checkpoint['model'])

        if enable_aug:
            if args.aug_policy == "conv":
                self.model = all_scale(self.model, self.args.scaling_ratio) # useless
            elif args.aug_policy == "conn" and args.scaling_ratio < 1.0:
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if name in ["linear.weight", "classifier.weight", "fc.weight"]: # fc.weight for alexnet
                            print(name)
                            mask = (torch.rand(param.shape) > args.scaling_ratio).float()
                            param.data = mask * param / (1.0 - args.scaling_ratio)
            else:
                print(args.aug_policy, 'Prune Error!')
    
        self.in_data = checkpoint['in_data']
        self.keep_bool = checkpoint['keep_bool']
        
        # no grad by default
        self.deactivate_grad()
        self.model.eval()

        self.is_in_model = False # False for out_model

    def forward(self, x):
        return self.model(x)

    def deactivate_grad(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def activate_grad(self):
        for param in self.model.parameters():
            param.requires_grad = True
    
    def load_state_dict(self, checkpoint):
        self.model.load_state_dict(checkpoint)
    
    def predictions(self, imgs, labels):
        accuracy = 0.0
        with torch.no_grad():
            batch_size = 256
            train_pred = []

            for i in range(0, len(imgs), batch_size):
                batch_img = imgs[i:i + batch_size]
                batch_img = batch_img.to(self.args.device)
                batch_pred = self.model(batch_img)
                train_pred.append(batch_pred)

            train_pred = torch.cat(train_pred)
            labels = labels.to(self.args.device)

            _, predicted_classes = torch.max(train_pred, 1)

            # Compute accuracy
            correct = (predicted_classes == labels).sum().item()
            total = labels.size(0)
            accuracy = correct / total

        return accuracy
    
    def get_interrmediate(self, imgs, labels):
        with torch.no_grad():
            batch_size = 256
            out_pred = []
            out1_pred = []
            out2_pred = []
            out3_pred = []
            out4_pred = []

            for i in range(0, len(imgs), batch_size):
                batch_img = imgs[i:i + batch_size]
                batch_img = batch_img.to(self.args.device)
                out, out1, out2, out3, out4 = self.model.intermediate(batch_img)
                out_pred.append(out)
                out1_pred.append(out1)
                out2_pred.append(out2)
                out3_pred.append(out3)
                out4_pred.append(out4)
            
            out_pred = torch.cat(out_pred)
            out1_pred = torch.cat(out1_pred)
            out2_pred = torch.cat(out2_pred)
            out3_pred = torch.cat(out3_pred)
            out4_pred = torch.cat(out4_pred)

            return out_pred, out1_pred, out2_pred, out3_pred, out4_pred




class InferenceMoE(nn.Module):
    def __init__(self, args, shadow_id=0):
        super().__init__()
        
        self.args = args
        if shadow_id == -1:
            # target model
            if args.target_net != '' or args.target_dir != '':
                resume_checkpoint = f'saved_models/{args.target_dir}/{args.target_net}_target_last.pth'
            else:
                net = args.net.split('-')[0] + '-ori'
                resume_checkpoint = f'saved_models/{net}-{args.dataset}-random-0.5/{net}_target_last.pth'
            print(resume_checkpoint)
            assert os.path.isfile(resume_checkpoint), 'Error: no checkpoint found!'
            checkpoint = torch.load(resume_checkpoint)
            if 'model_arch' in checkpoint:
                args.net = checkpoint['model_arch']
            self.model = load_model(args)
        
        else: 
            # shadom model: moe
            resume_checkpoint = f'saved_models/{args.shadow_dir}/{args.net}_shadow_{shadow_id}_last.pth'
            print(resume_checkpoint)

            assert os.path.isfile(resume_checkpoint), 'Error: no checkpoint found!'
            checkpoint = torch.load(resume_checkpoint)
            if 'model_arch' in checkpoint:
                args.net = checkpoint['model_arch']
            self.model = load_model(args)
            if args.net == 'res18-moe':
                router = build_router(num_experts=args.n_expert) #.to(device)
                set_router(self.model, router)
                router.load_state_dict(checkpoint["router"])
            
            self.member_idx = checkpoint['member_idx'] if 'member_idx' in checkpoint else None
            self.cands = checkpoint['cands'] if 'cands' in checkpoint else None
            self.non_cands = checkpoint['non_candas'] if 'non_candas' in checkpoint else None
            self.trained_pathway = checkpoint['trained_pathway'] if 'trained_pathway' in checkpoint else None
            self.nontrained_pathway = checkpoint['nontrained_pathway'] if 'nontrained_pathway' in checkpoint else None
            

        self.model.load_state_dict(checkpoint['model'])
        self.in_data = checkpoint['in_data']
        if 'in_pathway' in checkpoint:
            self.in_pathway = checkpoint['in_pathway']
        # self.keep_bool = checkpoint['keep_bool']
        
        # no grad by default
        self.deactivate_grad()
        self.model.eval()

        self.is_in_model = False # False for out_model

    def forward(self, x, router=None):
        if router is None:
            return self.model(x)
        else:
            return self.model(x, router)

    def deactivate_grad(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def activate_grad(self):
        for param in self.model.parameters():
            param.requires_grad = True
    
    def load_state_dict(self, checkpoint):
        self.model.load_state_dict(checkpoint)
    
    def predictions(self, imgs, labels, router=None):
        accuracy = 0.0
        with torch.no_grad():
            batch_size = 256
            train_pred = []

            for i in range(0, len(imgs), batch_size):
                batch_img = imgs[i:i + batch_size]
                batch_img = batch_img.to(self.args.device)
                if router is not None:
                    batch_pred = self.model(batch_img, router)
                else:
                    batch_pred = self.model(batch_img)
                train_pred.append(batch_pred)

            train_pred = torch.cat(train_pred)
            labels = labels.to(self.args.device)

            _, predicted_classes = torch.max(train_pred, 1)

            # Compute accuracy
            correct = (predicted_classes == labels).sum().item()
            total = labels.size(0)
            accuracy = correct / total

        return accuracy
    
    def get_interrmediate(self, imgs, labels, router=None):
        with torch.no_grad():
            batch_size = 256
            out_pred = []
            out1_pred = []
            out2_pred = []
            out3_pred = []
            out4_pred = []

            for i in range(0, len(imgs), batch_size):
                batch_img = imgs[i:i + batch_size]
                batch_img = batch_img.to(self.args.device)
                out, out1, out2, out3, out4 = self.model.intermediate(batch_img, router)
                out_pred.append(out)
                out1_pred.append(out1)
                out2_pred.append(out2)
                out3_pred.append(out3)
                out4_pred.append(out4)
            
            out_pred = torch.cat(out_pred)
            out1_pred = torch.cat(out1_pred)
            out2_pred = torch.cat(out2_pred)
            out3_pred = torch.cat(out3_pred)
            out4_pred = torch.cat(out4_pred)

            return out_pred, out1_pred, out2_pred, out3_pred, out4_pred