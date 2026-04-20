#!/bin/bash
#BSUB -J clip_ebc_downscale
#BSUB -o clip_ebc_downscale_%J.out
#BSUB -e clip_ebc_downscale_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=12GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 00:45
#BSUB -gpu "num=1:mode=exclusive_process"

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate clip_ebc

cd ~/ADLCV/CLIP-EBC

python -u task4_nwpu_val.py \
    --model clip_vit_l_14 \
    --input_size 224 \
    --reduction 8 \
    --truncation 4 \
    --anchor_points average \
    --prompt_type word \
    --num_vpt 32 \
    --vpt_drop 0.0 \
    --sliding_window \
    --stride 224 \
    --weight_path ./checkpoints/nwpu/best_rmse_0.pth \
    --device cuda \
    --downscale_factors 1 2 4 \
    --save_dir ./task4_outputs_downscale