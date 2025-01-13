import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from utils_fkd import (ComposeWithCoords, ImageFolder_FKD_MIX,
                       RandomHorizontalFlipWithRes,
                       RandomResizedCropWithCoords, mix_aug, load_model,count_jpg_files)
import platform
import sys
# get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import recover.utils_recover as ure

parser = argparse.ArgumentParser(description='FKD Soft Label Generation w/ Mix Augmentation')
parser.add_argument('--syn-data-path', required=True, type=str,
                    help='the path to the syn data which is being processed in this relabeling process')
parser.add_argument('--online', action='store_true',
                    help='use online model')
parser.add_argument('--multi-model', action='store_true',
                    help='use multi teacher model')
parser.add_argument('--model-choice', nargs='+', 
                    help='A list containing the choices of the compare model')
parser.add_argument('--model-weight', nargs='+', 
                    help='A list containing the choices of the compare model')
parser.add_argument('--eval-mode', action='store_true',
                    help='whether to use the evaluation mode or not')
parser.add_argument('--teacher-model-name', type=str,
                    help='teacher model name')
parser.add_argument('--model-pool-dir', type=str, default=None,
                    help='required when pretrained model type is offline, the directory of the models when using offline mode')
parser.add_argument('--fkd-path',required=True, type=str,
                    help='the path to save the fkd soft labels')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--dataset-name', default='cifar100', type=str,
                    help='dataset name')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://192.168.62.156:23457', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# FKD soft label generation args
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument("--min-scale-crops", type=float, default=0.08,
                    help="argument in RandomResizedCrop")
parser.add_argument("--max-scale-crops", type=float, default=1.,
                    help="argument in RandomResizedCrop")
parser.add_argument('--use-fp16', dest='use_fp16', action='store_true',
                    help='save soft labels as `fp16`')
parser.add_argument('--mode', default='fkd_save', type=str, metavar='N',)
parser.add_argument('--fkd-seed', default=42, type=int, metavar='N')
parser.add_argument('--mix-type', default = None, type=str, choices=['mixup', 'cutmix', None], help='mixup or cutmix or None')
parser.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
parser.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')

def set_worker_sharing_strategy(worker_id: int) -> None:
    if platform.system() == 'Linux':
        sharing_strategy = 'file_descriptor'
    else:
        sharing_strategy = 'file_system'
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def main():
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
    elif args.dataset_name == 'imagenet1k':
        args.mean_norm = [0.485, 0.456, 0.406]
        args.std_norm = [0.229, 0.224, 0.225]
        args.ncls = 1000
        args.input_size = 224
    elif args.dataset_name == 'imagenet-nette':
        args.mean_norm = [0.485, 0.456, 0.406]
        args.std_norm = [0.229, 0.224, 0.225]
        args.ncls = 10
        args.input_size = 224
    elif args.dataset_name == 'tiny_imagenet':
        args.mean_norm = [0.485, 0.456, 0.406]
        args.std_norm = [0.229, 0.224, 0.225]
        args.ncls = 200
        args.jitter = 4
        args.input_size = 64
    elif args.dataset_name == 'imagenet100':
        args.mean_norm = [0.485, 0.456, 0.406]
        args.std_norm = [0.229, 0.224, 0.225]
        args.ncls = 100
        args.jitter = 32
        args.input_size = 224
    else:
        raise ValueError('dataset not supported')
    
    
    # compute current ipc
    ipc = int(count_jpg_files(args.syn_data_path) / args.ncls)
    
    # set up the fkd path
    args.fkd_path = args.fkd_path + f'_bs{args.batch_size}_ipc{ipc}'
    if not os.path.exists(args.fkd_path):
        os.makedirs(args.fkd_path, exist_ok=True)
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        print('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        print('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        print(args.gpu)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.gpu)
    # load different teacher models
    teacher_model_lis = []
    if args.multi_model:
        for model_name in args.model_choice:
            if args.online:
                model = ure.load_online_model(model_name, args)
            else:
                model = load_model(args, model_name)
            teacher_model_lis.append(model)
    else:
        print(f"Teacher model name: {args.teacher_model_name}")
        if args.online:
            model = ure.load_online_model(args.teacher_model_name, args)
        else:
            model = load_model(args,args.teacher_model_name)
        teacher_model_lis.append(model)
    
    print(f"Total model amount: {len(teacher_model_lis)}")
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            for _model in teacher_model_lis:
                _model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                _model = torch.nn.parallel.DistributedDataParallel(_model, device_ids=[args.gpu])
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        else:
            for _model in teacher_model_lis:
                _model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                _model = torch.nn.parallel.DistributedDataParallel(_model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        for _model in teacher_model_lis:
            _model = _model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        for _model in teacher_model_lis:
            _model = torch.nn.DataParallel(_model).cuda()

    # freeze all layers
    for _model in teacher_model_lis:
        for name, param in _model.named_parameters():
            param.requires_grad = False

    cudnn.benchmark = True

    print("process data from {}".format(args.syn_data_path))

    normalize = transforms.Normalize(mean=args.mean_norm,
                                     std=args.std_norm)
    
    train_dataset = ImageFolder_FKD_MIX(
        fkd_path=args.fkd_path,
        mode=args.mode,
        root=args.syn_data_path,
        transform=ComposeWithCoords(transforms=[
            RandomResizedCropWithCoords(size=args.input_size,
                                        scale=(args.min_scale_crops,
                                               args.max_scale_crops),
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
        num_workers=args.workers, pin_memory=True,
        worker_init_fn=set_worker_sharing_strategy)

    for epoch in tqdm(range(args.epochs)):
        dir_path = os.path.join(args.fkd_path, 'epoch_{}'.format(epoch))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        save(train_loader, teacher_model_lis, dir_path, args)
        # exit()


@torch.no_grad()
def save(train_loader, model_lis, dir_path, args):
    if args.model_weight is None:
        weights = [1.0 / len(model_lis)] * len(model_lis)
    else:
        w = np.array([float(w) for w in args.model_weight])
        temperature = 10  # 您可以调整此值来控制平滑度
        w = w / temperature
        weights = np.exp(w) / np.sum(np.exp(w))

    """Generate soft labels and save"""
    for batch_idx, (images, target, flip_status, coords_status) in enumerate(train_loader):
        images = images.cuda()
        split_point = int(images.shape[0] // 2)
        origin_images = images
        images, mix_index, mix_lam, mix_bbox = mix_aug(images, args)
        
        total_output = []
        for idx, _model in enumerate(model_lis):
            if args.eval_mode:
                _model.eval()
            cat_output = []
            output = _model(origin_images[:split_point])
            cat_output.append(output)
            output = _model(origin_images[split_point:])
            cat_output.append(output)
            output = torch.cat(cat_output, 0) * weights[idx]
            total_output.append(output)
            
        output = torch.stack(total_output, 0)
        output = output.sum(0)
        
        if args.use_fp16:
            output = output.half()
        
        batch_config = [coords_status, flip_status, mix_index, mix_lam, mix_bbox, output.cpu()]
        batch_config_path = os.path.join(dir_path, 'batch_{}.tar'.format(batch_idx))
        torch.save(batch_config, batch_config_path)


if __name__ == '__main__':
    main()