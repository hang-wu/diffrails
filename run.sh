#! /bin/bash
# Create necessary directories
mkdir -p logs/ml-1m-l200/
mkdir -p exps/
mkdir -p ckpts/

# Run training
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 python3 train.py \
    --gin_config_file=configs/ml-1m/hstu-mol-sampled-softmax-n128-8x4x64-difftopk-rails.gin \
    --master_port=12356