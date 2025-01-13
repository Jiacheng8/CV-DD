# Dataset Requirement
we require the following store format for dataset
```plain_text
Dataset/
├──train
└──test
```
# For Squeezing ImageNette
Since ImageNette's resolution is 224\*224\*3, we adopt pytorch official models (unweighted) for training from scratch, so ImageNette is separated from other dataset.  

To squeeze ImageNette, first go to squeeze_imagenette.py and change data_dir and save_dir to desired location. Then run
```sh
python squeeze_imagenette.py
```


# For squeezing the rest dataset
```
usage: Squeezing the models [-h] [--model-list MODEL_LIST [MODEL_LIST ...]]
                            [--optimizer OPTIMIZER] --dataset-dir DATASET_DIR
                            --save-dir SAVE_DIR [--batch-size BATCH_SIZE]
                            --dataset-name DATASET_NAME [--epoch EPOCH]
                            [--lr LR] [--use_multi_gpu]
                            [--world_size WORLD_SIZE]

options:
  -h, --help            show this help message and exit
  --model-list MODEL_LIST [MODEL_LIST ...]
                        The trained model list
  --optimizer OPTIMIZER
  --dataset-dir DATASET_DIR
                        directory where the dataset are stored
  --save-dir SAVE_DIR   directory to save the trained models
  --batch-size BATCH_SIZE
                        number of images to optimize at the same time
  --dataset-name DATASET_NAME
                        dataset to use for training
  --epoch EPOCH         num of iterations to optimize the target model
  --lr LR               learning rate for optimization
  --use_multi_gpu       Enable multi_gpu_learning
  --world_size WORLD_SIZE
                        the number of gpu that is available
```
For example, if you want to squeeze CIFAR-100, remember to set dataset-dir and save-dir to the desired location
```sh
cd scripts
bash squeeze_cifar100.sh
```
