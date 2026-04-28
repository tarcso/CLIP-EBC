#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J Eval_Student
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"
#BSUB -W 02:00
#BSUB -o logs/eval_student%J.out
#BSUB -e logs/eval_student%J.err

source ~/miniconda3/bin/activate clip_ebc
cd "$LSB_SUBCWD"
mkdir -p logs

python eval_student.py \
    --teacher_weight_path ./checkpoints/nwpu/best_rmse_0.pth \
    --student_weight_path ./checkpoints/student/best_student_e100_lr3e-05_ds2.pth \
    --downscale 2 --device cuda \
    --save_dir ./student_eval_outputs/e100_lr3e-5
