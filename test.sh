#!/bin/bash
# IRONet Testing Script
# Usage: bash test.sh [config] [checkpoint] [show_dir]

CONFIG=${1:-"configs/ironet/ironet_vits_iron_5f.py"}
CHECKPOINT=${2:-"work_dirs/ironet_vits_iron_5f/checkpoints/best_model.pth"}
SHOW_DIR=${3:-"./results"}

python custom_test.py $CONFIG $CHECKPOINT \
    --eval \
    --show-dir $SHOW_DIR \
    --show-mask
