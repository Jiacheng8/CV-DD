# Overall Directory Configuration
SCRIPT_DIR_main="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR_main="$(dirname "$SCRIPT_DIR_main")"
DD_dir="$(dirname "$PARENT_DIR_main")"
DD_dir="$(dirname "$DD_dir")"

source $DD_dir/config.sh
Dataset_Name=imagenet1k
bs=100
