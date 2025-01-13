import numpy as np
import torch
import torchvision.transforms as transforms
import os
import sys
import torchvision
# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from models import *

# keep top k largest values, and smooth others
def keep_top_k(p,k,n_classes=1000): # p is the softmax on label output
    if k == n_classes:
        return p

    values, indices = p.topk(k, dim=1)

    mask_topk = torch.zeros_like(p)
    mask_topk.scatter_(-1, indices, 1.0)
    top_p = mask_topk * p

    minor_value = (1 - torch.sum(values, dim=1)) / (n_classes-k)
    minor_value = minor_value.unsqueeze(1).expand(p.shape)
    mask_smooth = torch.ones_like(p)
    mask_smooth.scatter_(-1, indices, 0)
    smooth_p = mask_smooth * minor_value

    topk_smooth_p = top_p + smooth_p
    assert np.isclose(topk_smooth_p.sum().item(), p.shape[0]), f'{topk_smooth_p.sum().item()} not close to {p.shape[0]}'
    return topk_smooth_p


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_parameters(model):
    group_no_weight_decay = []
    group_weight_decay = []
    for pname, p in model.named_parameters():
        if pname.find('weight') >= 0 and len(p.size()) > 1:
            # print('include ', pname, p.size())
            group_weight_decay.append(p)
        else:
            # print('not include ', pname, p.size())
            group_no_weight_decay.append(p)
    assert len(list(model.parameters())) == len(
        group_weight_decay) + len(group_no_weight_decay)
    groups = [dict(params=group_weight_decay), dict(
        params=group_no_weight_decay, weight_decay=0.)]
    return groups

def load_small_dataset_model(model, args):
    if model == 'ResNet18':
        net = ResNet18(args.ncls)
    elif model == 'ResNet50':
        net = ResNet50(args.ncls)
    elif model == 'ResNet101':
        net = ResNet101(args.ncls)
    return net

def load_val_loader(args):
    if args.dataset_name == "cifar100" or args.dataset_name == "cifar10":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean_norm, std=args.std_norm)
        ])
    elif args.dataset_name == "imagenet1k" or args.dataset_name == "imagenet100" or args.dataset_name=='imagenet-nette':
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean_norm, std=args.std_norm)
        ])
    elif args.dataset_name == "tiny_imagenet":
         transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean_norm, std=args.std_norm)
        ])
    else:
        raise NotImplementedError(f"dataset {args.dataset_name} not implemented")
    
    test_set = torchvision.datasets.ImageFolder(root=args.val_dir, transform=transform_test)

    # load dataset for CIFAR-100 
    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=2,pin_memory=True)
    return testloader
