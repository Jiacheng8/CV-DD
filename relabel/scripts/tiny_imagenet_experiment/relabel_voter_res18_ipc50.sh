mode=cvdd
ipc=50

# Overall Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
PARENT_DIR="$(dirname "$PARENT_DIR")"

source $SCRIPT_DIR/constants.sh

python $PARENT_DIR/relabel.py \
    --syn-data-path $Generated_Path/generated_data/syn_data/$Dataset_Name/${mode}_ipc${ipc} \
    --fkd-path $Generated_Path/generated_data/new_labels/$Dataset_Name/${mode} \
    --model-pool-dir $Generated_Path/offline_models/$Dataset_Name \
    --multi-model \
    --model-choice ResNet18 ResNet50 ShuffleNetV2 MobileNetV2 Densenet121 \
    --model-weight 46.5 28.3 43.6 44.8 41.5 \
    --gpu 0 \
    -b $bs \
    -j 2 \
    --dataset-name $Dataset_Name \
    --epochs 300 \
    --fkd-seed 42 \
    --min-scale-crops 0.08 \
    --max-scale-crops 1 \
    --use-fp16 \
    --mode 'fkd_save' \
    --mix-type 'cutmix' \