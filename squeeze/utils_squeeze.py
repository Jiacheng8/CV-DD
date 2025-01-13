import torch.nn as nn
import torch
import glob
import os
import torchvision
import torchvision.transforms as transforms
import sys
# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from models import *
from torch.utils.data import DistributedSampler

# load CIFAR-100 dataset
def load_dataset(rank, args, mode="train"):
    # Define the transformation of the dataset
    if mode == "train":
        transform_train = transforms.Compose([
            transforms.RandomCrop(args.input_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(args.mean_norm, args.std_norm) 
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(args.mean_norm, args.std_norm) 
        ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(args.mean_norm, args.std_norm)  
    ])
    
    val_dir = os.path.join(args.dataset_dir, 'test')
    train_dir = os.path.join(args.dataset_dir, 'train')
    
    train_set = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
    test_set = torchvision.datasets.ImageFolder(val_dir, transform=transform_test)
    
    # Set case for multi-gpu training
    if args.use_multi_gpu:
        train_sampler = DistributedSampler(train_set, num_replicas=args.world_size, rank=rank)
    else:
        train_sampler = None

    # load dataset for CIFAR-100 
    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=2, pin_memory=True)
    
    # load dataset for CIFAR-100 
    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    
    return trainloader, testloader

def load_model(model, ncls):
    if model == 'ResNet18':
        net = ResNet18(ncls)
    elif model == 'ResNet50':
        net = ResNet50(ncls)
    elif model == 'ResNet101':
        net = ResNet101(ncls)
    elif model == 'Densenet121':
        net = DenseNet121(ncls)
    elif model == 'Densenet169':
        net = DenseNet169(ncls)
    elif model == 'Densenet201':
        net = DenseNet201(ncls)
    elif model == 'Densenet161':
        net = DenseNet161(ncls)
    elif model == 'MobileNetV2':
        net = MobileNetV2(ncls)
    elif model == 'ShuffleNetV2':
        net = ShuffleNetV2(net_size=0.5,ncls=ncls)
    else:
        raise ValueError('Model not supported')
    return net 

def get_all_models(args):
    pth_files = glob.glob(f"{args.save_dir}/*.pth",recursive=False)
    return pth_files

# Evaluate the model
def evaluate_loader(model, criterion, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
    acc = correct / total
    loss = total_loss / len(dataloader)
    return acc, loss