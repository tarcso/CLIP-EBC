#!/bin/sh
### ---------------- specify queue name ----------------
#BSUB -q gpuv100             
#BSUB -gpu "num=1"
### ---------------- specify job name ----------------
#BSUB -J Distill_CLIP
### ---------------- specify number of cores ----------------
#BSUB -n 4
#BSUB -R "span[hosts=1]"
### ---------------- specify CPU memory requirements ----------------
#BSUB -R "rusage[mem=20GB]"
### ---------------- specify wall-clock time (max allowed is 12:00) ----------------
#BSUB -W 06:00
#BSUB -o OUTPUT_FILE%J.out
#BSUB -e OUTPUT_FILE%J.err

source ~/miniforge3/bin/activate clip_ebc
cd /zhome/84/4/186776/CLIP-EBC

python train_distillation.py \
    --model clip_vit_l_14 --input_size 224 --reduction 8 --truncation 4 --anchor_points average --prompt_type word \
    --num_vpt 32 --vpt_drop 0.0 \
    --weight_path ./checkpoints/nwpu/best_rmse_0.pth --device cuda --epochs 10 --lr 1e-3