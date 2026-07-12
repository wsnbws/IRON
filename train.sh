#!/bin/bash
# IRONet Training Script
# Usage: bash train.sh [config] [work_dir] [pretrained] [num_gpus]

CONFIG=${1:-"configs/ironet/ironet_vits_iron_5f.py"}
WORK_DIR=${2:-"./work_dirs/ironet_vits_iron_5f"}
PRETRAINED=${3:-"pretrained/convmae_small.pth"}
NUM_GPUS=${4:-1}
PORT=${PORT:-29500}

mkdir -p $WORK_DIR

if [ $NUM_GPUS -eq 1 ]; then
    python custom_train.py $CONFIG \
        --work-dir $WORK_DIR \
        --seed 42 --deterministic \
        --options model.pretrained=$PRETRAINED
else
    python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$PORT \
        custom_train.py $CONFIG \
        --launcher pytorch \
        --work-dir $WORK_DIR \
        --seed 42 --deterministic \
        --options model.pretrained=$PRETRAINED
fi
