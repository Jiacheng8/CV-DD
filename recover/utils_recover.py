import torch.nn as nn
import numpy as np
import torch
from torch import distributed
import glob
import random
import os
import sys
import torchvision.models as models
import torch.optim as optim

# get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from models import *
import cv2

def format_int_to_str(number):
    return "{:05}".format(number)


def get_second_idx(all_idx, exclude_idx):
    remaining_idx = [i for i in all_idx if i != exclude_idx]
    sample_idx = random.choice(remaining_idx)
    return sample_idx


def distributed_is_initialized():
    if distributed.is_available():
        if distributed.is_initialized():
            return True
    return False


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


def beta_policy(mom_fn):
    def _alr(optimizer, iteration, epoch, param, indx):
        mom = mom_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group[param][indx] = mom

    return _alr


def mom_cosine_policy(base_beta, warmup_length, epochs):
    def _beta_fn(iteration, epoch):
        if epoch < warmup_length:
            beta = base_beta * (epoch + 1) / warmup_length
        else:
            beta = base_beta
        return beta

    return beta_policy(_beta_fn)


def clip(image_tensor, args, use_fp16=False):
    '''
    adjust the input based on mean and variance
    '''
    if use_fp16:
        mean = np.array(args.mean_norm, dtype=np.float16)
        std = np.array(args.std_norm, dtype=np.float16)
    else:
        mean = np.array(args.mean_norm)
        std = np.array(args.std_norm)
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m/s, (1 - m)/s)
    return image_tensor


def denormalize(image_tensor, args, use_fp16=False):
    if use_fp16:
        mean = np.array(args.mean_norm, dtype=np.float16)
        std = np.array(args.std_norm, dtype=np.float16)
    else:
        mean = np.array(args.mean_norm)
        std = np.array(args.std_norm)

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor


class BNFeatureHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().reshape([nch, -1]).var(1, unbiased=False)
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(module.running_mean.data - mean, 2)
        self.r_feature = r_feature

    def close(self):
        self.hook.remove()


def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0

    return loss_var_l1, loss_var_l2


def load_model(model, num_classes):
    if model == 'ResNet18':
        net = ResNet18(num_classes)
    elif model == 'ResNet50':
        net = ResNet50(num_classes)
    elif model == 'ResNet101':
        net = ResNet101(num_classes)
    elif model == 'Densenet121':
        net = DenseNet121(num_classes)
    elif model == 'Densenet169':
        net = DenseNet169(num_classes)
    elif model == 'Densenet201':
        net = DenseNet201(num_classes)
    elif model == 'Densenet161':
        net = DenseNet161(num_classes)
    elif model == 'MobileNetV2':
        net = MobileNetV2(num_classes)
    elif model == 'ShuffleNetV2':
        net = ShuffleNetV2(net_size=0.5,ncls=num_classes)
    else:
        raise ValueError('Model not supported')
    return net


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


def load_verifier_model(chosen_name,args):
    model_path = os.path.join(args.model_pool_dir, chosen_name+".pth")
    model = load_model(chosen_name,args.ncls)
    state_dict = torch.load(model_path,weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model
    
def normalize(image,args):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = image.astype(np.float32) / 255.0

    mean = args.mean_norm
    std = args.std_norm

    normalized_image = (image - mean) / std
    return normalized_image
    
     
def initialize_patch_data(start_label_idx, end_label_idx, args, num_call):
    if args.store_initialised_images:
        initialisation_dir = os.path.join(args.initialisation_dir, args.exp_name, f'call_{num_call}', f'{start_label_idx}_to_{end_label_idx}')
        os.makedirs(initialisation_dir, exist_ok=True)
        print(f"Initialisation dir: {initialisation_dir}")
    # Load pre-made patches
    patch_dir = os.path.join(args.patch_dir, args.patch_diff)
    
    all_images = []
    # Load the patches

    for i in range(start_label_idx, end_label_idx):
        current_class_name = format_int_to_str(i)
        current_class_dir = os.path.join(patch_dir, current_class_name)
        all_image_files = glob.glob(f"{current_class_dir}/*.jpg",recursive=False)
        chosen_image_files = random.sample(all_image_files, 1)

        final_img = normalize(cv2.imread(chosen_image_files[0]),args)
        final_img_display = cv2.imread(chosen_image_files[0])
    
        # save the img to the initialisation dir to show the quality of the patches
        # you can comment this line if you don't want to see the quality
        if args.store_initialised_images:
            new_img_file = os.path.join(initialisation_dir, f'{str(i)}.jpg')
            cv2.imwrite(new_img_file, final_img_display)
        # append the final image to the list
        all_images.append(final_img)
    
    # change the list to a numpy array
    initialised_data = np.array(all_images)
    initialised_data = np.transpose(initialised_data, (0, 3, 1, 2))  # Now shape is (N, C, H, W)

    # convert the data to tensor
    patch_data = torch.tensor(initialised_data, dtype=torch.float, device="cuda",requires_grad=True)

    return patch_data


def load_recover_model(recover_model_name_list, args, device):
    all_recover_model_list = []
    BN_hooks = []
    weight_list = []
    for curr_recover_model_name in recover_model_name_list:
        if args.pretrained_model_type == 'offline':
            if args.dataset_name == 'imagenet100' or args.dataset_name == 'imagenet-nette':
                # code for imagenet100
                if curr_recover_model_name == 'ResNet18':
                    curr_recover_model = models.resnet18(weights=None)
                    curr_recover_model.fc = nn.Linear(curr_recover_model.fc.in_features, args.ncls)
                elif curr_recover_model_name == 'ResNet50':
                    curr_recover_model = models.resnet50(weights=None)
                    curr_recover_model.fc = nn.Linear(curr_recover_model.fc.in_features, args.ncls)
                elif curr_recover_model_name == 'Densenet121':
                    curr_recover_model = models.densenet121(weights=None)
                    in_features = curr_recover_model.classifier.in_features
                    curr_recover_model.classifier = torch.nn.Linear(in_features, args.ncls)
                elif curr_recover_model_name == 'MobileNetV2':
                    curr_recover_model = models.mobilenet_v2(weights=None)
                    in_features = curr_recover_model.classifier[-1].in_features
                    curr_recover_model.classifier[-1] = torch.nn.Linear(in_features, args.ncls)

                elif curr_recover_model_name == 'ShuffleNetV2':
                    curr_recover_model = models.shufflenet_v2_x1_0(weights=None)
                    curr_recover_model.fc = nn.Linear(curr_recover_model.fc.in_features, args.ncls)
                else:
                    raise ValueError('Model not supported')
                curr_recover_model_weight_path = os.path.join(args.model_pool_dir, curr_recover_model_name + '.pth')
                state_dict = torch.load(curr_recover_model_weight_path, weights_only=True)
                curr_recover_model.load_state_dict(state_dict)
            # load process for cifar100, cifar10, and tinyimagenet
            else:
                curr_recover_model = load_model(curr_recover_model_name, args.ncls)
                curr_recover_model_weight_path = os.path.join(args.model_pool_dir, curr_recover_model_name + '.pth')
                state_dict = torch.load(curr_recover_model_weight_path, weights_only=True)
                curr_recover_model.load_state_dict(state_dict)
        # online model loading for imagenet1k
        else:
            curr_recover_model = load_online_model(curr_recover_model_name, args)
        curr_recover_model = curr_recover_model.to(device)
        
        # freeze the compare model
        curr_recover_model.eval()
        for p in curr_recover_model.parameters():
            p.requires_grad = False
        all_recover_model_list.append(curr_recover_model)
        
        # Process BN features
        curr_BN_hook = []
        for module in curr_recover_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                curr_BN_hook.append(BNFeatureHook(module))
        BN_hooks.append(curr_BN_hook)
        
    if args.voter_type == 'equal':
        weight_list = [1/len(recover_model_name_list)]*len(recover_model_name_list)
    elif args.voter_type == 'prior':
        for model_name in recover_model_name_list:
            weight_list.append(args.model_prior_weight_dict[model_name])
        weight_list = np.array([float(w) for w in weight_list])/args.weight_temperature
        weight_list = np.exp(weight_list) / np.sum(np.exp(weight_list))
    elif args.voter_type == 'random':
        random_list = np.random.rand(len(recover_model_name_list))
        normalized_list = random_list / random_list.sum()
        weight_list = normalized_list.tolist() 
    else:
        raise ValueError(f"Voter type {args.voter_type} is not supported")
    
    return all_recover_model_list, BN_hooks, weight_list


def load_online_model(model_name, args):
    if args.dataset_name == 'imagenet1k':
        if model_name == 'MobileNetV2':
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        elif model_name == 'ResNet18':
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif model_name == 'ResNet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif model_name == 'Densenet121':
            model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        elif model_name == 'EfficientNet':
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        elif model_name == 'ShuffleNetV2':
            model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Model {model_name} is not supported")
    else:
        raise NotImplementedError(f"Online model loading for {args.dataset_name} is not supported yet")

    return model
