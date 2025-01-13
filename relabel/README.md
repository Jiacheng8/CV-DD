# Usage
```
relabel.py [-h] --syn-data-path SYN_DATA_PATH [--online] [--multi-model] [--model-choice MODEL_CHOICE [MODEL_CHOICE ...]]
                  [--model-weight MODEL_WEIGHT [MODEL_WEIGHT ...]] [--eval-mode] [--teacher-model-name TEACHER_MODEL_NAME] [--model-pool-dir MODEL_POOL_DIR]
                  --fkd-path FKD_PATH [-j N] [-b N] [--dataset-name DATASET_NAME] [--world-size WORLD_SIZE] [--rank RANK] [--dist-url DIST_URL]
                  [--dist-backend DIST_BACKEND] [--seed SEED] [--gpu GPU] [--multiprocessing-distributed] [--epochs EPOCHS] [--min-scale-crops MIN_SCALE_CROPS]
                  [--max-scale-crops MAX_SCALE_CROPS] [--use-fp16] [--mode N] [--fkd-seed N] [--mix-type {mixup,cutmix,None}] [--mixup MIXUP] [--cutmix CUTMIX]

FKD Soft Label Generation w/ Mix Augmentation

options:
  -h, --help            show this help message and exit
  --syn-data-path SYN_DATA_PATH
                        the path to the syn data which is being processed in this relabeling process
  --online              use online model
  --multi-model         use multi teacher model
  --model-choice MODEL_CHOICE [MODEL_CHOICE ...]
                        A list containing the choices of the compare model
  --model-weight MODEL_WEIGHT [MODEL_WEIGHT ...]
                        A list containing the choices of the compare model
  --eval-mode           whether to use the evaluation mode or not
  --teacher-model-name TEACHER_MODEL_NAME
                        teacher model name
  --model-pool-dir MODEL_POOL_DIR
                        required when pretrained model type is offline, the directory of the models when using offline mode
  --fkd-path FKD_PATH   the path to save the fkd soft labels
  -j N, --workers N     number of data loading workers (default: 4)
  -b N, --batch-size N  mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed
                        Data Parallel
  --dataset-name DATASET_NAME
                        dataset name
  --world-size WORLD_SIZE
                        number of nodes for distributed training
  --rank RANK           node rank for distributed training
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
  --seed SEED           seed for initializing training.
  --gpu GPU             GPU id to use.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for
                        either single node or multi node data parallel training
  --epochs EPOCHS
  --min-scale-crops MIN_SCALE_CROPS
                        argument in RandomResizedCrop
  --max-scale-crops MAX_SCALE_CROPS
                        argument in RandomResizedCrop
  --use-fp16            save soft labels as `fp16`
  --mode N
  --fkd-seed N
  --mix-type {mixup,cutmix,None}
                        mixup or cutmix or None
  --mixup MIXUP         mixup alpha, mixup enabled if > 0. (default: 0.8)
  --cutmix CUTMIX       cutmix alpha, cutmix enabled if > 0. (default: 1.0)
```

# Requirement
Remember to download required files mentioned in [Overall Configuration Section](../README.md).

# Example
For example, if you want to generate soft labels for  CIFAR-100 dataset in 10 IPC, you should first cd into the cifar100_experiments directory and run the following code:
```sh
bash relabel_voter_ipc10.sh
```