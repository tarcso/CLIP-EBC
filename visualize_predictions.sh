#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J Visualize
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"
#BSUB -W 01:00
#BSUB -o logs/visualize%J.out
#BSUB -e logs/visualize%J.err

source ~/miniconda3/bin/activate clip_ebc
cd "$LSB_SUBCWD"
mkdir -p logs

python visualize_predictions.py \
    --teacher_weight_path ./checkpoints/nwpu/best_rmse_0.pth \
    --student_2x_weight_path ./checkpoints/student/best_student_e50_lr1e-5.pth \
    --student_4x_weight_path ./checkpoints/student/best_student_e50_lr3e-05_ds4.pth \
    --val_dir ./data/nwpu/val/images \
    --save_dir ./assets/visualizations \
    --device cuda \
    --images 125.jpg 047.jpg 299.jpg 244.jpg
