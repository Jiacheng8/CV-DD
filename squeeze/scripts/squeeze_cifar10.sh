# Overall Directory Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Create logs directory
mkdir -p $SCRIPT_DIR/logs

python $PARENT_DIR/squeeze.py \
    --dataset-name cifar10 \
    --dataset-dir \
    --save-dir \
    --model-list ResNet18 ResNet50 ResNet101 ShuffleNetV2 MobileNetV2 Densenet121 \
    --epoch 200 \
    --batch-size 512 \
    --lr 0.001 > $SCRIPT_DIR/logs/cifar10_squeeze.log 2>$SCRIPT_DIR/logs/cifar10_squeeze.error