#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J Eval_Realworld
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"
#BSUB -W 02:00
#BSUB -o logs/eval_realworld%J.out
#BSUB -e logs/eval_realworld%J.err

source ~/miniconda3/bin/activate clip_ebc
cd "$LSB_SUBCWD"
mkdir -p logs

python eval_realworld.py \
    --teacher_weight_path ./checkpoints/nwpu/best_rmse_0.pth \
    --student_2x_weight_path ./checkpoints/student/best_student_e50_lr1e-5.pth \
    --student_4x_weight_path ./checkpoints/student/best_student_e50_lr3e-05_ds4.pth \
    --data_root /dtu/blackhole/02/137570/MultiRes/test \
    --save_dir ./realworld_outputs \
    --device cuda \
    --visualize_folders 60
