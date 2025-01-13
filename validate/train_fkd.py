import argparse
import math
import os
import shutil
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import wandb
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms import InterpolationMode
from utils_validate import AverageMeter, accuracy, get_parameters, load_val_loader, load_small_dataset_model
# It is imported for you to access and modify the PyTorch source code (via Ctrl+Click), more details in README.md
from torch.utils.data._utils.fetch import _MapDatasetFetcher
# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# 获取当前脚本的目录
from models import *
from relabel.utils_fkd import (ComposeWithCoords, ImageFolder_FKD_MIX,
                               RandomHorizontalFlipWithRes,
                               RandomResizedCropWithCoords, mix_aug)


def get_args():
    parser = argparse.ArgumentParser("FKD Training on Cifar-100")
    parser.add_argument('--exp-name', type=str,
                        default="", help='the name of the run')
    parser.add_argument('--original-data-path', required='True', type=str,
                        help='name of the original data')
    parser.add_argument('--simple', default=False,action='store_true',)
    parser.add_argument('--fkd-path', required='True', type=str,
                        help='path to the fkd labels')
    parser.add_argument('--output-dir', required='True', type=str,
                        help='output directory')
    parser.add_argument('--dataset-name',default='cifar100',type=str,
                        help='dataset name')
    parser.add_argument('--min-scale', type=float, default=0.08, )
    parser.add_argument('--batch-size', type=int,
                        default=16, help='batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int,
                        default=1, help='gradient accumulation steps for small gpu memory')
    parser.add_argument('--start-epoch', type=int,
                        default=0, help='start epoch')
    parser.add_argument('--epochs', type=int, default=300, help='total epoch')
    parser.add_argument('-j', '--workers', default=2, type=int,
                        help='number of data loading workers')
    parser.add_argument('--ipc',type=int,help='number of images per class')
    parser.add_argument('--cos', default=False,
                        action='store_true', help='cosine lr scheduler')
    parser.add_argument('--sgd', default=False,
                        action='store_true', help='sgd optimizer')
    parser.add_argument('-lr', '--sgd-lr', type=float,
                        default=0.01, help='sgd init learning rate')  # checked
    parser.add_argument('--momentum', type=float,
                        default=0.5, help='sgd momentum')  # checked
    parser.add_argument('--weight-decay', type=float,
                        default=1e-4, help='sgd weight decay')  # checked
    parser.add_argument('--adamw-weight-decay', type=float,
                        default=0.01, help='adamw weight decay')
    parser.add_argument('--model', type=str,
                        default='ResNet18', help='student model name')
    parser.add_argument('--keep-topk', type=int, default=1000,
                        help='keep topk logits for kd loss')
    parser.add_argument('-T', '--temperature', type=float,
                        default=3.0, help='temperature for distillation loss')
    parser.add_argument('--wandb-project', type=str,
                        default='RankDD', help='wandb project name')
    parser.add_argument('--wandb-api-key', type=str,
                        default=None, help='wandb api key')
    parser.add_argument('--mix-type', default=None, type=str,
                        choices=['mixup', 'cutmix', None], help='mixup or cutmix or None')
    parser.add_argument('--fkd_seed', default=42, type=int,
                        help='seed for batch loading sampler')
    parser.add_argument('--val-dir', required=True, type=str,
                        help="path to the validation data")

    args = parser.parse_args()

    args.mode = 'fkd_load'

    # final checked
    if args.dataset_name == 'cifar10':
        args.mean_norm = [0.4914, 0.4822, 0.4465]
        args.std_norm = [0.2470, 0.2435, 0.2616]
        args.ncls = 10
        args.input_size = 32
        if args.model == 'ResNet18' or args.model == 'ResNet50':
            if args.epochs == 1000:
                args.adamw_lr = 0.0005
                args.eta = 1
            else:
                args.adamw_lr = 0.001
                args.eta = 1
            
        else:
            args.adamw_lr = 0.0005
            args.eta = 1
    
    # final checked
    elif args.dataset_name == 'cifar100':
        args.mean_norm = [0.5071, 0.4867, 0.4408]
        args.std_norm = [0.2675, 0.2565, 0.2761]
        args.ncls = 100
        args.input_size = 32
        if args.model == 'ResNet18' or args.model == 'ResNet50':
            args.adamw_lr = 0.001
            if args.ipc==10:
                args.eta=2
            else:
                args.eta = 1
        else:
            args.adamw_lr = 0.0005
            args.eta = 1
    
    #final checked
    elif args.dataset_name == 'tiny_imagenet':
        args.mean_norm = [0.485, 0.456, 0.406]
        args.std_norm = [0.229, 0.224, 0.225]
        args.ncls = 200
        args.input_size = 64
        
        if args.model == 'ResNet18':
            args.adamw_lr = 0.001
            if args.ipc ==10:
                args.eta = 1
            else:
                args.eta = 2
            
        elif args.model == 'ResNet50':
            args.adamw_lr = 0.001
            args.eta = 1

        else:
            args.adamw_lr = 0.0005
            args.eta = 2
    
    # final checked
    elif args.dataset_name == 'imagenet-nette':
        args.mean_norm = [0.485, 0.456, 0.406]
        args.std_norm = [0.229, 0.224, 0.225]
        args.ncls = 10
        args.input_size = 224
        if args.model == 'ResNet18':
            args.adamw_lr = 0.0005
            if args.ipc == 10:
                args.eta = 1
            else:
                args.eta = 2
        elif args.model == 'ResNet50':
            # lr
            if args.ipc == 10:
                args.adamw_lr = 0.001
            else:
                args.adamw_lr = 0.0005
            # eta
            if args.ipc == 1:
                args.eta = 2
            else:
                args.eta = 1
        else:
            # lr
            if args.ipc == 50:
                args.adamw_lr = 0.001
            else:
                args.adamw_lr = 0.0005

            args.eta = 2
    
    # final checked
    elif args.dataset_name == 'imagenet1k':
        args.mean_norm = [0.485, 0.456, 0.406]
        args.std_norm = [0.229, 0.224, 0.225]
        args.ncls = 1000
        args.input_size = 224
        if args.model == 'ResNet18' or args.model == 'ResNet50':
            # lr
            args.adamw_lr = 0.001
            if args.ipc == 50:
                args.eta = 1
            else:
                args.eta = 2
        elif args.model == 'ResNet101':
            # lr
            if args.ipc == 10 or args.ipc == 1:
                args.adamw_lr = 0.001
            elif args.ipc == 50:
                args.adamw_lr = 0.0005
            args.eta = 2
    else:
        raise ValueError('dataset not supported')
    
    # set up the train_dir and output_dir
    args.output_dir = os.path.join(args.output_dir, args.dataset_name, args.exp_name)
    print(args)
    return args

def is_special_epoch(epoch, total_epochs):
    # 判断是否在最后的0.8个epoch
    in_last_80_percent = epoch >= int(total_epochs * 0.8)
    # 判断尾数是否是9或为最后一个epoch
    ends_with_9_or_last = (epoch % 10 == 9) or (epoch == total_epochs - 1)
    return in_last_80_percent and ends_with_9_or_last

def main():
    args = get_args()
    
    # set up wandb
    wandb.login(key=args.wandb_api_key)
    wandb.init(project=args.wandb_project, dir="./")
    wandb.run.name = args.exp_name

    if not torch.cuda.is_available():
        raise Exception("need gpu to train!")

    print(args.original_data_path)
    assert os.path.exists(args.original_data_path)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Data loading
    normalize = transforms.Normalize(mean=args.mean_norm, std=args.std_norm)
    train_dataset = ImageFolder_FKD_MIX(
        fkd_path=args.fkd_path,
        mode=args.mode,
        args_epoch=args.epochs,
        args_bs=args.batch_size,
        root=args.original_data_path,
        transform=ComposeWithCoords(transforms=[
            RandomResizedCropWithCoords(size=args.input_size,
                                        scale=(args.min_scale, 1),
                                        interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlipWithRes(),
            transforms.ToTensor(),
            normalize,
        ]))

    generator = torch.Generator()
    generator.manual_seed(args.fkd_seed)
    sampler = torch.utils.data.RandomSampler(train_dataset, generator=generator)


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=args.workers, pin_memory=True)

    # load validation data
    val_loader = load_val_loader(args)

    # load student model
    print("=> loading student model '{}'".format(args.model))

    if args.model == 'ResNet18':
        if args.input_size <= 64:
            model = ResNet18(args.ncls)
        else:
            model = models.resnet18(pretrained=False)
            if args.ncls != 1000:
                model.fc = nn.Linear(model.fc.in_features, args.ncls)
    elif args.model == 'ResNet50':
        if args.input_size <= 64:
            model = ResNet50(args.ncls)
        else:
            model = models.resnet50(pretrained=False)
            if args.ncls != 1000:
                model.fc = nn.Linear(model.fc.in_features, args.ncls)
    elif args.model == 'ResNet101':
        if args.input_size <= 64:
            model = ResNet101(args.ncls)
        else:
            model = models.resnet101(pretrained=False)
            if args.ncls != 1000:
                model.fc = nn.Linear(model.fc.in_features, args.ncls)
    else:
        raise ValueError('model not supported')
    model = model.cuda()
    model.train()

    if args.sgd:
        optimizer = torch.optim.SGD(get_parameters(model),
                                    lr=args.sgd_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(get_parameters(model),
                                      lr=args.adamw_lr,
                                      weight_decay=args.adamw_weight_decay)

    if args.cos == True:
        scheduler = LambdaLR(optimizer,
                             lambda step: 0.5 * (1. + math.cos(math.pi * step / args.epochs / args.eta)) if step <= args.epochs else 0, last_epoch=-1)
    else:
        scheduler = LambdaLR(optimizer,
                             lambda step: (1.0-step/args.epochs) if step <= args.epochs else 0, last_epoch=-1)

 
    args.best_acc1=0
    args.optimizer = optimizer
    args.scheduler = scheduler
    args.train_loader = train_loader
    args.val_loader = val_loader

    for epoch in range(args.start_epoch, args.epochs):
        print(f"\nEpoch: {epoch}")

        global wandb_metrics
        wandb_metrics = {}

        train(model, args, epoch)
        if not args.simple:
            if epoch % 10 == 0 or epoch == args.epochs - 1:
                top1 = validate(model, args, epoch)
            else:
                top1 = 0
        else:
            if is_special_epoch(epoch, args.epochs):
                top1 = validate(model, args, epoch)
            else:
                top1 = 0
                
        wandb.log(wandb_metrics)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = top1 > args.best_acc1
        args.best_acc1 = max(top1, args.best_acc1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': args.best_acc1,
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict(),
        }, is_best, output_dir=args.output_dir)

def adjust_bn_momentum(model, iters):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 1 / iters


def train(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    optimizer = args.optimizer
    scheduler = args.scheduler
    loss_function_kl = nn.KLDivLoss(reduction='batchmean')

    model.train()
    t1 = time.time()
    args.train_loader.dataset.set_epoch(epoch)
    for batch_idx, batch_data in enumerate(args.train_loader):
        images, target, flip_status, coords_status = batch_data[0]
        mix_index, mix_lam, mix_bbox, soft_label = batch_data[1:]

        images = images.cuda()
        target = target.cuda()
        soft_label = soft_label.cuda().float()  # convert to float32
        images, _, _, _ = mix_aug(images, args, mix_index, mix_lam, mix_bbox)

        optimizer.zero_grad()
        assert args.batch_size % args.gradient_accumulation_steps == 0
        small_bs = args.batch_size // args.gradient_accumulation_steps

        # images.shape[0] is not equal to args.batch_size in the last batch, usually
        if batch_idx == len(args.train_loader) - 1:
            accum_step = math.ceil(images.shape[0] / small_bs)
        else:
            accum_step = args.gradient_accumulation_steps

        for accum_id in range(accum_step):
            partial_images = images[accum_id * small_bs: (accum_id + 1) * small_bs]
            partial_target = target[accum_id * small_bs: (accum_id + 1) * small_bs]
            partial_soft_label = soft_label[accum_id * small_bs: (accum_id + 1) * small_bs]

            output = model(partial_images)
            prec1, prec5 = accuracy(output, partial_target, topk=(1, 5))

            output = F.log_softmax(output/args.temperature, dim=1)
            partial_soft_label = F.softmax(partial_soft_label/args.temperature, dim=1)
            loss = loss_function_kl(output, partial_soft_label)
            # loss = loss * args.temperature * args.temperature
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            n = partial_images.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)


        optimizer.step()


    metrics = {
        "train/loss": objs.avg,
        "train/Top1": top1.avg,
        "train/Top5": top5.avg,
        "train/lr": scheduler.get_last_lr()[0],
        "train/epoch": epoch,}
    wandb_metrics.update(metrics)


    printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(epoch, scheduler.get_last_lr()[0], objs.avg) + \
                'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
                'Top-5 err = {:.6f},\t'.format(100 - top5.avg) + \
                'train_time = {:.6f}'.format((time.time() - t1))
    print(printInfo)
    t1 = time.time()


def validate(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_function = nn.CrossEntropyLoss()

    model.eval()
    t1  = time.time()
    with torch.no_grad():
        for data, target in args.val_loader:
            target = target.type(torch.LongTensor)
            data, target = data.cuda(), target.cuda()
            
            output = model(data)
            loss = loss_function(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    logInfo = 'TEST Iter {}: loss = {:.6f},\t'.format(epoch, objs.avg) + \
              'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
              'Top-5 err = {:.6f},\t'.format(100 - top5.avg) + \
              'val_time = {:.6f}'.format(time.time() - t1)
    print(logInfo)

    metrics = {
        'val/loss': objs.avg,
        'val/top1': top1.avg,
        'val/top5': top5.avg,
        'val/epoch': epoch,
    }
    wandb_metrics.update(metrics)

    return top1.avg

def save_checkpoint(state, is_best, output_dir=None,epoch=None):
    if epoch is None:
        path = output_dir + '/' + 'checkpoint.pth.tar'
    else:
        path = output_dir + f'/checkpoint_{epoch}.pth.tar'
    torch.save(state, path)

    if is_best:
        path_best = output_dir + '/' + 'model_best.pth.tar'
        shutil.copyfile(path, path_best)



if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn')
    main()
    wandb.finish()