#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J Distill_SJ_clw05
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"
#BSUB -W 12:00
#BSUB -o logs/OUTPUT_FILE%J.out
#BSUB -e logs/OUTPUT_FILE%J.err

source ~/miniconda3/bin/activate clip_ebc
cd "$LSB_SUBCWD"
mkdir -p logs

python train_distillation.py \
    --model clip_vit_l_14 --input_size 224 --reduction 8 --truncation 4 \
    --anchor_points average --prompt_type word --granularity fine \
    --num_vpt 32 --vpt_drop 0.0 \
    --weight_path ./checkpoints/nwpu/best_rmse_0.pth \
    --device cuda --epochs 50 --lr 3e-5 --count_loss_weight 0.5 \
    --downscale 2 --downscale_max 4.0 --batch_size 8
