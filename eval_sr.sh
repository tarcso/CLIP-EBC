#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J Eval_SR
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"
#BSUB -W 04:00
#BSUB -o logs/eval_sr%J.out
#BSUB -e logs/eval_sr%J.err

source ~/miniconda3/bin/activate clip_ebc
cd "$LSB_SUBCWD"
mkdir -p logs

python eval_sr.py \
    --teacher_weight_path ./checkpoints/nwpu/best_rmse_0.pth \
    --student_weight_path ./checkpoints/student/best_student_e50_lr1e-5.pth \
    --esrgan_weight_path ./weights/RealESRGAN_x4plus.pth \
    --downscale 2 --device cuda \
    --save_dir ./sr_eval_outputs/ds2
