import torch
import argparse
from utils_squeeze import load_model,load_dataset,evaluate_loader,get_all_models
import torch.nn as nn
import csv
import torchvision.models as models
import sys
import os
# for multiprocessing system
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from timm.data import Mixup

# for initialisign the dist environment
def setup_distributed_environment(master_addr='localhost', master_port='12355'):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    print(f"MASTER_ADDR set to {master_addr}, MASTER_PORT set to {master_port}")


# Define the arguments of the program and parse them from the command line
def parse_args():
    parser = argparse.ArgumentParser("Squeezing the models")
    parser.add_argument('--model-list', nargs='+', help='The trained model list')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='directory where the dataset are stored')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='directory to save the trained models')
    parser.add_argument('--batch-size', type=int,
                        default=128, help='number of images to optimize at the same time')
    parser.add_argument('--dataset-name', type=str, required=True,
                        help='dataset to use for training')
    parser.add_argument('--epoch', type=int, default=200,
                        help='num of iterations to optimize the target model')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate for optimization')
    parser.add_argument('--use_multi_gpu', action='store_true', 
                        help='Enable multi_gpu_learning')
    parser.add_argument('--world_size',type=int, default=-1,
                        help='the number of gpu that is available')
    args = parser.parse_args()

    # set up the mean, std and ncls for the dataset
    if args.dataset_name == 'cifar100':
        args.mean_norm = [0.5071, 0.4867, 0.4408]
        args.std_norm = [0.2675, 0.2565, 0.2761]
        args.ncls = 100
        args.input_size = 32
    elif args.dataset_name == 'cifar10':
        args.mean_norm = [0.4914, 0.4822, 0.4465]
        args.std_norm = [0.2470, 0.2435, 0.2616]
        args.ncls = 10
        args.input_size = 32
    elif args.dataset_name == 'tiny_imagenet':
        args.mean_norm = [0.485, 0.456, 0.406]
        args.std_norm = [0.229, 0.224, 0.225]
        args.ncls = 200
        args.input_size = 64
    else:
        raise ValueError('dataset not supported')
    
    # Initialize CutMix augmentation
    args.mixup_fn = Mixup(
        mixup_alpha=0.0,
        cutmix_alpha=1.0,
        prob=1.0,
        switch_prob=0.0,
        label_smoothing=0.1,
        num_classes=args.ncls,
    )
    if os.path.exists(args.save_dir) == False:
        os.makedirs(args.save_dir)
    return args

def generate_models_process(rank, device, args):
    # set up for Multi-gpu training
    if args.use_multi_gpu:
        dist.init_process_group(
            backend='nccl',        
            init_method='env://',  
            rank=rank,             
            world_size=args.world_size  
        )
        torch.cuda.set_device(rank)  
        device = device+f":{rank}"
    else:
        print(f"Using {device} for training")
    
    
    trainloader, testloader = load_dataset(rank, args)
    
    for model_name in args.model_list:
        if rank == 0:
            print(f"Start training model: {model_name}")
        
        model = load_model(model_name, args.ncls).to(device)
        if args.use_multi_gpu:
            model = DDP(model, device_ids=[rank],output_device = rank)
        # setup loss function and optimizer
        criterion = nn.CrossEntropyLoss().to(device)
        if args.optimizer == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)
        
        # train the model
        for epoch in range(args.epoch):
            model.train()
            if args.use_multi_gpu:
                trainloader.sampler.set_epoch(epoch)  # 同步每个进程的数据
            # Train the model for one step
            for inputs, labels in trainloader:
                # inputs, labels = args.mixup_fn(inputs, labels)
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            # evaluate the model only in main gpu
            if rank == 0 and epoch%10 ==0:
                # train_acc, train_loss = evaluate_loader(model, criterion, trainloader, device)
                test_acc, test_loss = evaluate_loader(model, criterion, testloader, device)
                print(f"Epoch: {epoch}, Test Acc: {test_acc} Test Loss: {test_loss}")
                
            scheduler.step()
            
        # save the model
        final_model_path = os.path.join(args.save_dir, f"{model_name}.pth")
        torch.save(model.state_dict(), final_model_path)
        print("finished processing model: ", model_name)
        
    # 清理分布式进程组
    if args.use_multi_gpu:
        dist.destroy_process_group()


def main_generate_pools(args):
    # Generating Pools for different Models
    # case when using more than 1 gpu
    if args.use_multi_gpu:
        if torch.cuda.device_count() < 2:
            print("The number of availabel gpus is less than 2, please use normal mode ")
            sys.exit()
        if args.world_size > torch.cuda.device_count() or args.world_size == -1:
            print(f"please set world size below the number of current availabele gpus: {torch.cuda.device_count()} ")
            sys.exit()
        setup_distributed_environment()
        print("Using Multi GPU Training....")
        mp.spawn(generate_models_process, args=("cuda", args), nprocs=args.world_size, join=True)
    # case when using one gpu or use cpu or use mps
    else:
        # setup device for training
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        generate_models_process(0, device, args)
        
def generate_evaluation_csv(args):
    print("get evaluation call")
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    write_data = [['Model Name', 'train_acc', 'train_loss', 'test_acc', 'test_loss']]
    all_models_paths = get_all_models(args)
    train_loader, test_loader = load_dataset(0,args,'eval')
    for model_path in all_models_paths:
        model_name = os.path.basename(model_path).split(".")[0]
        model = load_model(model_name,args.ncls)
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        criterion = nn.CrossEntropyLoss().to(device)
        train_acc, train_loss = evaluate_loader(model, criterion, train_loader, device)
        test_acc, test_loss = evaluate_loader(model, criterion, test_loader, device)
        write_data.append([model_name, train_acc, train_loss, test_acc, test_loss])
        print(f"finished procesing {model_name}")
    with open(os.path.join(args.save_dir,'model_result_info.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(write_data)
        

if __name__ == '__main__':
    # parse the arguments
    args = parse_args()
    print(args)
    
    # Step 1
    # main entry to generate pools
    main_generate_pools(args)
    
    # Step 2
    # Evaluate the Models
    generate_evaluation_csv(args)