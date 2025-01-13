# Usage
```
Usage: Recover data from pre-trained model using CV-DD or Sre2L++.

[-h] [--dataset-name DATASET_NAME] [--exp-name EXP_NAME] [--apply-data-augmentation] [--start-index START_INDEX] [--sre2l-model SRE2L_MODEL] [--pretrained-model-type {offline,online}] [--model-setting MODEL_SETTING] [--selected-size SELECTED_SIZE] [--voter-type {equal,random,prior}] [--verifier] [--verifier-arch VERIFIER_ARCH] [--verifier-weight-path VERIFIER_WEIGHT_PATH] [--syn-data-path SYN_DATA_PATH] [--model-pool-dir MODEL_POOL_DIR] [--patch-dir PATCH_DIR] [--initialisation-dir INITIALISATION_DIR] [--store-best-images] [--store-initialised-images] [--batch-size BATCH_SIZE] [--iteration ITERATION] [--lr LR] [--jitter JITTER] [--r-bn R_BN] [--first-bn-multiplier FIRST_BN_MULTIPLIER] [--weight-temperature WEIGHT_TEMPERATURE] [--initialisation-method {Guassian,Patches}] [--patch-diff {easy,medium,hard}] [--ipc-start IPC_START] [--ipc-end IPC_END]



options:
  -h, --help            show this help message and exit
  --dataset-name DATASET_NAME
                        Name of the dataset to recover
  --exp-name EXP_NAME   Name of the experiment, subfolder under syn_data_path
  --apply-data-augmentation
                        whether or not to apply data augmentation
  --start-index START_INDEX
                        start index of the class to recover
  --sre2l-model SRE2L_MODEL
                        Name of the Model applied to Sre2L++
  --pretrained-model-type {offline,online}
                        The type of pretrained models
  --model-setting MODEL_SETTING
                        Model choosing setups
  --selected-size SELECTED_SIZE
                        number of recover models to optimise the synthetic data
  --voter-type {equal,random,prior}
                        The voter type, Equal assigns equal weight, Random assigns random weight and Prior assigns weight using prior information
  --verifier            whether to evaluate the synthetic data with another model
  --verifier-arch VERIFIER_ARCH
                        arch name to act as a verifier
  --verifier-weight-path VERIFIER_WEIGHT_PATH
                        path to the verifier model weights
  --syn-data-path SYN_DATA_PATH
                        where to store synthetic data
  --model-pool-dir MODEL_POOL_DIR
                        required when pretrained model type is offline
  --patch-dir PATCH_DIR
                        the directory where the patches are stored
  --initialisation-dir INITIALISATION_DIR
                        the directory of the initialisation data specifically for patch initialisation, it will create a sub folder named exp-name under this
                        directory
  --store-best-images   whether to store synthetic data
  --store-initialised-images
                        whether to store the initialised images when using patches initialisation
  --batch-size BATCH_SIZE
                        number of images to optimize at the same time
  --iteration ITERATION
                        num of iterations to optimize the synthetic data
  --lr LR               learning rate for optimization
  --jitter JITTER       random shift on the synthetic data
  --r-bn R_BN           coefficient for BN feature distribution regularization
  --first-bn-multiplier FIRST_BN_MULTIPLIER
                        additional multiplier on first bn layer of R_bn
  --weight-temperature WEIGHT_TEMPERATURE
                        The temperature used when calculating the weight
  --initialisation-method {Guassian,Patches}
                        initialisation method for the synthetic data
  --patch-diff {easy,medium,hard}
                        the difficulty of the patches
  --ipc-start IPC_START
                        start index of IPC
  --ipc-end IPC_END     end index of IPC
```

# Requirement
Remember to download required files mentioned in [Overall Configuration Section](../README.md).

# Example
For example, if you want to recover the dataset CIFAR-100 in 10 IPC, you should first cd into the cifar100_experiments directory and run the following code:
```sh
bash recover_voter_ipc10.sh
```