# Overall Directory Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Create logs directory
mkdir -p $SCRIPT_DIR/logs

python $PARENT_DIR/squeeze.py \
    --dataset-name tiny_imagenet \
    --dataset-dir \
    --save-dir \
    --model-list ResNet18 ResNet50 ResNet101 ShuffleNetV2 MobileNetV2 Densenet121 \
    --epoch 50 \
    --batch-size 128 \
    --optimizer SGD \
    --lr 0.01 > $SCRIPT_DIR/logs/tiny_imagenet_squeeze.log 2>$SCRIPT_DIR/logs/tiny_imagenet_squeeze.error