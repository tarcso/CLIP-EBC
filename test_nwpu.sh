#!/bin/sh
export CUDA_VISIBLE_DEVICES=0

python test_nwpu.py \
    --model clip_vit_l_14 --input_size 224 --reduction 8 --truncation 4 --anchor_points average --prompt_type word \
    --num_vpt 32 --vpt_drop 0.0 --sliding_window --stride 224 \
    --weight_path ./checkpoints/nwpu/best_rmse_0.pth
