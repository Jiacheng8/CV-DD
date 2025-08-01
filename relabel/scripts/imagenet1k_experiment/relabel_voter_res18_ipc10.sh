mode=cvdd
ipc=10

# Overall Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
PARENT_DIR="$(dirname "$PARENT_DIR")"

source $SCRIPT_DIR/constants.sh

python $PARENT_DIR/relabel.py \
    --syn-data-path $Generated_Path/generated_data/syn_data/$Dataset_Name/${mode}_ipc${ipc} \
    --fkd-path $Generated_Path/generated_data/new_labels/$Dataset_Name/${mode} \
    --multi-model \
    --model-choice ResNet18 ResNet50 ShuffleNetV2 MobileNetV2 Densenet121 \
    --model-weight 56.4 42.3 52.5 44.2 40.6 \
    --online \
    --gpu 0 \
    -b $bs \
    --eval-mode F \
    -j 10 \
    --dataset-name $Dataset_Name \
    --epochs 300 \
    --fkd-seed 42 \
    --min-scale-crops 0.08 \
    --max-scale-crops 1 \
    --use-fp16 \
    --mode 'fkd_save' \
    --mix-type 'cutmix' \