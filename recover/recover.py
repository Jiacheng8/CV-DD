import argparse
import collections
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data.distributed
from PIL import Image
from torchvision import transforms
import utils_recover as utils_re
import pandas as pd
import time
import torch.nn.functional as F


def get_images(args, hook_for_display, device, num_call, is_first_ipc):
    print("get_images call")
    save_every = 100
    batch_size = args.batch_size
    targets_all = torch.LongTensor(np.arange(args.ncls))
    
    if args.model_setting == 5 or args.model_setting == 6:
        recover_model_name_list = args.model_choice_sequence[num_call]
    else:
        recover_model_name_list = random.sample(args.model_choice, args.selected_size)
        
    recover_model_list, BN_hooks, weight_list = utils_re.load_recover_model(recover_model_name_list, args, device)
    print(f"---The recover models are changed to {', '.join(model_name for model_name in recover_model_name_list)}")
    if is_first_ipc:
        start_index = args.start_index
    else:
        start_index = 0

    for kk in range(start_index, args.ncls, batch_size):
        start_label = kk
        end_label = min(kk+batch_size, args.ncls)
        print(f"currently processing label from {start_label} to {end_label}")
        targets = targets_all[kk:min(kk+batch_size, args.ncls)].to(device)

        if args.initialisation_method == "Guassian":
            inputs = torch.randn((targets.shape[0], 3, args.input_size, args.input_size), requires_grad=True, device=device,
                                dtype=torch.float).to(device)
            print("initialisation method: Guassian")
        else:
            inputs = utils_re.initialize_patch_data(kk, min(kk+batch_size, args.ncls), args, num_call).to(device)
            print(f"initialisation method: Patches: {args.patch_diff} ")
        
        iterations_per_layer = args.iteration
        lim_0, lim_1 = args.jitter, args.jitter
        
        optimizer = optim.Adam([{'params': [inputs], 'lr': args.lr}], betas=[0.5, 0.9], eps=1e-8)
        lr_scheduler = utils_re.lr_cosine_policy(args.lr, 0, iterations_per_layer)
        criterion = nn.CrossEntropyLoss().to(device)
        
        start_time = time.time()

        for iteration in range(iterations_per_layer):
            lr_scheduler(optimizer, iteration, iteration)
            
            aug_function = transforms.Compose([
                transforms.RandomResizedCrop(args.input_size),
                transforms.RandomHorizontalFlip(),
            ])
            
            if args.apply_data_augmentation:
                inputs_jit = aug_function(inputs)
            else:
                inputs_jit = inputs

            off1 = random.randint(0, lim_0)
            off2 = random.randint(0, lim_1)
            inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

            optimizer.zero_grad()

            ce_lis = []
            for model in recover_model_list:
                outputs_recover = model(inputs_jit)
                loss_ce = criterion(outputs_recover, targets)
                ce_lis.append(loss_ce)

            loss_BN_lis = []
            for (id, BN_hook) in enumerate(BN_hooks):
                rescale = [args.first_bn_multiplier] + [1. for _ in range(len(BN_hook)-1)]
                curr_loss_BN = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(BN_hook)])
                loss_BN_lis.append(curr_loss_BN)
            
            loss = 0
            for (idx, weight) in enumerate(weight_list):
                curr_BN_loss = args.r_bn * loss_BN_lis[idx]
                curr_ce = ce_lis[idx]
                loss += weight * (curr_ce + curr_BN_loss)

            loss.backward()
            optimizer.step()

            inputs.data = utils_re.clip(inputs.data, args)

            if iteration % save_every == 0:
                end_time = time.time()
                print("------------iteration {}----------".format(iteration))
                print("total loss", loss.item())
                for i in range(len(ce_lis)):
                    curr_recover_model_name = recover_model_name_list[i]
                    curr_ce = ce_lis[i]
                    curr_BN_loss = loss_BN_lis[i]
                    weight = weight_list[i]
                    print(f"Model: {curr_recover_model_name}, CE loss: {curr_ce.item()}, BN loss: {curr_BN_loss.item()}, weight: {weight}")
                print(f'time for previous iterations: {end_time-start_time}')
                start_time = time.time()

                if hook_for_display is not None:
                    hook_for_display(inputs, targets)

        if args.store_best_images:
            best_inputs = inputs.data.clone()
            best_inputs = utils_re.denormalize(best_inputs, args)
            save_images(args, best_inputs, targets, ipc_id)

        optimizer.state = collections.defaultdict(dict)
        torch.cuda.empty_cache()

def save_images(args, images, targets, ipc_id):
    print("save_images call")
    for id in range(images.shape[0]):
        if targets.ndimension() == 1:
            class_id = targets[id].item()
        else:
            class_id = targets[id].argmax().item()

        if not os.path.exists(args.syn_data_path):
            os.mkdir(args.syn_data_path)

        # save into separate folders
        dir_path = '{}/new{:03d}'.format(args.syn_data_path, class_id)
        place_to_store = dir_path +'/class{:03d}_id{:03d}.jpg'.format(class_id,ipc_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)


def validate(input, target, model):
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

    with torch.no_grad():
        output = model(input)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

    print("Verifier accuracy: ", prec1.item())


def parse_args():
    parser = argparse.ArgumentParser("Recover data from pre-trained model using CV-DD or Sre2L++")
    # Overall Configs
    parser.add_argument('--dataset-name', type=str, required=True, 
                        help='Name of the dataset to recover, currently support CIFAR-10, CIFAR-100, Tiny-ImageNet, ImageNet-Nette, ImageNet-1k')
    parser.add_argument('--exp-name', type=str, required=True,
                        help='Name of the experiment, subfolder under syn_data_path')
    parser.add_argument('--apply-data-augmentation', action='store_true',
                        help='whether or not to apply data augmentation')
    parser.add_argument('--start-index', type=int, default=0, 
                        help='start index of the class to recover')
    parser.add_argument('--sre2l-model', type=str, default='ResNet18', help='Name of the Model applied to Sre2L++')
    # Committee Related Configs
    parser.add_argument('--pretrained-model-type', type=str, required=True, choices=['offline', 'online'],
                        help='Offline: the models are pre-trained and stored in the model pool directory\
                              Online: the pretrained models are loaded by downloading from the Pytorch Official Models')
    parser.add_argument('--model-setting', type=int, default=0,
                        help='Model choosing setups: \
                            0: Original Sre2L++ method, but this requires giving the argument sre2l-model as well.\
                            1: Using first model setups: R18, R50, SV2, Dense121\
                            2: Using second model setups: R18, R50, SV2, MBV2\
                            3: Using third model setups: R18, R50, SV2, Dense121, MBV2\
                            4: IPC = 1 settings for N2 voter\
                            5: IPC = 10 settings for N2 voter\
                            6: IPC = 10 settings for N3 voter\
                            7: IPC = 1 settings for N3 voter')
    parser.add_argument('--selected-size',type=int, default=2,
                        help='number of recover models to optimise the synthetic data')
    parser.add_argument('--voter-type', type=str, default='prior', choices=['equal', 'random', 'prior'], help='The voter type, Equal assigns equal weight, Random assigns random weight and Prior assigns weight using prior information')
    # Verifier Related Configs
    parser.add_argument('--verifier', action='store_true',
                        help='whether to evaluate the synthetic data with another model')
    parser.add_argument('--verifier-arch', type=str, default='MobileNetV2',
                        help="arch name to act as a verifier")
    parser.add_argument('--verifier-weight-path', type=str, default=None,
                        help="path to the verifier model weights")
    # Directory Related Configs
    parser.add_argument('--syn-data-path', type=str, required=True, 
                        help='where to store synthetic data')
    parser.add_argument('--model-pool-dir', type=str, default=None,
                        help='required when pretrained model type is offline')
    parser.add_argument('--patch-dir', type=str, default=None,
                        help='the directory where the patches are stored')
    parser.add_argument('--initialisation-dir', default=None, type=str,
                        help="the directory of the initialisation data specifically for patch initialisation,\
                              it will create a sub folder named exp-name under this directory")
    # Data Saving Related Configs
    parser.add_argument('--store-best-images', action='store_true',
                        help='whether to store synthetic data')
    parser.add_argument('--store-initialised-images', action='store_true',
                        help='whether to store the initialised images when using patches initialisation')
    # Optimization Related Configs
    parser.add_argument('--batch-size', type=int,
                        default=100, help='number of images to optimize at the same time')
    parser.add_argument('--iteration', type=int, default=1000,
                        help='num of iterations to optimize the synthetic data')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate for optimization')
    parser.add_argument('--jitter', default=4, type=int, help='random shift on the synthetic data')
    parser.add_argument('--r-bn', type=float, default=0.05,
                        help='coefficient for BN feature distribution regularization')
    parser.add_argument('--first-bn-multiplier', type=float, default=10.,
                        help='additional multiplier on first bn layer of R_bn')
    parser.add_argument('--weight-temperature', default=5, type=int, help="The temperature used when calculating the weight")
    # Initialisation Related Configs
    parser.add_argument('--initialisation-method',type=str, default="Guassian", choices=["Guassian", "Patches"],
                        help='initialisation method for the synthetic data')
    parser.add_argument('--patch-diff',type=str, default="medium", choices=["easy", "medium", "hard"],
                        help="the difficulty of the patches")
    #IPC (Image Per Class) Related Configs
    parser.add_argument("--ipc-start", default=0, type=int, help="start index of IPC")
    parser.add_argument("--ipc-end", default=50, type=int, help="end index of IPC")
    args = parser.parse_args()

    # verifier model weight path is required if verifier is set to True
    if args.verifier:
        if args.pretrained_model_type == 'offline':
            assert args.verifier_weight_path is not None, "Verifier weight path is required"
    
    # set up the path for the synthetic data
    args.syn_data_path = os.path.join(args.syn_data_path, args.exp_name)
    if not os.path.exists(args.syn_data_path):
        os.makedirs(args.syn_data_path)
        
    if args.dataset_name == 'cifar100':
        args.mean_norm = [0.5071, 0.4867, 0.4408]
        args.std_norm = [0.2675, 0.2565, 0.2761]
        args.ncls = 100
        args.jitter = 4
        args.input_size = 32
        args.model_prior_weight_dict = {'ResNet18':64, 'ResNet50':60.58, 'ShuffleNetV2':51.62, 'MobileNetV2':59.43, 'Densenet121':56.36}
    
    # checked 
    elif args.dataset_name == 'cifar10':
        args.mean_norm = [0.4914, 0.4822, 0.4465]
        args.std_norm = [0.2470, 0.2435, 0.2616]
        args.ncls = 10
        args.input_size = 32
        args.jitter = 4
        args.model_prior_weight_dict = {'ResNet18':63.01, 'ResNet50':65.25, 'ShuffleNetV2':68.52, 'MobileNetV2':67.61, 'Densenet121':67.57}
        
    elif args.dataset_name == 'imagenet1k':
        args.mean_norm = [0.485, 0.456, 0.406]
        args.std_norm = [0.229, 0.224, 0.225]
        args.ncls = 1000
        args.jitter = 32
        args.input_size = 224
        args.model_prior_weight_dict = {'ResNet18':43.1, 'ResNet50':41.37, 'ShuffleNetV2':43.73, 'MobileNetV2':39.15, 'Densenet121':38.9}
        
    elif args.dataset_name == 'imagenet-nette':
        args.mean_norm = [0.485, 0.456, 0.406]
        args.std_norm = [0.229, 0.224, 0.225]
        args.ncls = 10
        args.jitter = 32
        args.input_size = 224
        args.model_prior_weight_dict = {'ResNet18':62.4, 'ResNet50':52, 'ShuffleNetV2':57.2, 'MobileNetV2':54, 'Densenet121':60.8}
        
    elif args.dataset_name == 'tiny_imagenet':
        args.mean_norm = [0.485, 0.456, 0.406]
        args.std_norm = [0.229, 0.224, 0.225]
        args.ncls = 200
        args.jitter = 4
        args.input_size = 64
        args.model_prior_weight_dict = {'ResNet18':46.5, 'ResNet50':28.3, 'ShuffleNetV2':43.6, 'MobileNetV2':44.8, 'Densenet121':41.5}
        
    else:
        raise ValueError('dataset not supported')
    
    # Sre2l++ setting
    if args.model_setting == 0:
        args.model_choice = [args.sre2l_model]
        assert args.selected_size == 1, "selected size should be 1 for model setting 0"
        assert args.voter_type == 'equal', "voter type should be equal for model setting 0"
    # different model setting for voter, after rigorous testing, we found that the setting 3 is the best
    elif args.model_setting == 1:
        args.model_choice = ['ResNet18', 'ResNet50', 'ShuffleNetV2', 'Densenet121']
    elif args.model_setting == 2:
        args.model_choice = ['ResNet18', 'ResNet50', 'ShuffleNetV2', 'MobileNetV2']
    elif args.model_setting == 3:
        args.model_choice = ['ResNet18', 'ResNet50', 'ShuffleNetV2', 'MobileNetV2', 'Densenet121']
    # ipc1 setting for N2 voter(Default N=2)
    elif args.model_setting == 4:
        if args.dataset_name == 'imagenet-nette' or args.dataset_name == 'cifar10':
            args.model_choice = ['MobileNetV2', 'Densenet121']
        elif args.dataset_name == 'cifar100':
            args.model_choice = ['ShuffleNetV2', 'MobileNetV2']
        else:
            args.model_choice = sorted(args.model_prior_weight_dict, key=args.model_prior_weight_dict.get, reverse=True)[:2]
    # ipc10 setting for N2 voter(Default N2 voter)
    elif args.model_setting == 5:
        args.model_choice_sequence = [['ResNet18','ResNet50'],
                                      ['ResNet18','ShuffleNetV2'],
                                      ['ResNet18','MobileNetV2'],
                                      ['ResNet18','Densenet121'],
                                      ['ResNet50','ShuffleNetV2'],
                                      ['ResNet50','MobileNetV2'],
                                      ['ResNet50','Densenet121'],
                                      ['ShuffleNetV2', 'MobileNetV2'],
                                      ['ShuffleNetV2', 'Densenet121'],
                                      ['MobileNetV2', 'Densenet121']]
    # ipc10 setting for N3 voter (N3 voter, bad performance, not efficient and effective)
    elif args.model_setting == 6:
        args.model_choice_sequence = [['ResNet18','ResNet50', 'ShuffleNetV2'],
                                      ['ResNet18','ShuffleNetV2', 'MobileNetV2'],
                                      ['ResNet18','MobileNetV2', 'Densenet121'],
                                      ['ResNet18','Densenet121', 'ResNet50'],
                                      ['ResNet50','ShuffleNetV2', 'MobileNetV2'],
                                      ['ResNet50','MobileNetV2', 'Densenet121'],
                                      ['ResNet50','Densenet121', 'ResNet18'],
                                      ['ShuffleNetV2', 'MobileNetV2', 'ResNet50'],
                                      ['ShuffleNetV2', 'Densenet121','ResNet18'],
                                      ['MobileNetV2', 'Densenet121', 'ResNet50']]
    # ipc1 setting for N3 voter (N3 voter, bad performance, not efficient and effective)
    elif args.model_setting == 7:
        args.model_choice = ['ResNet18', 'ResNet50','ShuffleNetV2']

    else:
        raise ValueError('model setting not supported')
        
    return args


def main_syn(args, device, ipc_id, is_first_ipc=False):
    torch.cuda.empty_cache()
    if args.verifier:
        if args.pretrained_model_type == 'offline':
            verifier_model = utils_re.load_verifier_model(args.verifier_arch, args)
        else:
            verifier_model = utils_re.load_online_model(args.verifier_arch, args)
        verifier_model = verifier_model.to(device)
        hook_for_display = lambda x,y: validate(x, y, verifier_model)
    else:
        hook_for_display = None
    get_images(args, hook_for_display, device, ipc_id, is_first_ipc)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    #set up device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    print(f"---The recover process will be performed on device: {device}")

    # loop through the IPCs and generate the synthetic data
    for ipc_id in range(args.ipc_start, args.ipc_end):
        print('ipc = ', ipc_id)
        if ipc_id == args.ipc_start:
            main_syn(args, device, ipc_id, is_first_ipc=True)
        else:
            main_syn(args, device, ipc_id)