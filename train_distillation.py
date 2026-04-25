import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple
import torch.nn.functional as F
from datasets.crowd import Crowd_distilation
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import copy
import os
import json
from models import get_model
from utils import get_config, sliding_window_predict
from pathlib import Path

current_dir = os.path.abspath(os.path.dirname(__file__))

def KL_loss(student_pred, teacher_label):
    if teacher_label.shape != student_pred.shape:
        teacher_label = F.interpolate(
            teacher_label, 
            size=student_pred.shape[2:], 
            mode='area'
        )
    
    b, c, h, w = student_pred.shape
    p_s = F.softmax(student_pred.view(b, -1), dim=1)
    p_t = F.softmax(teacher_label.view(b, -1), dim=1)

    kl = torch.sum(p_t * (torch.log(p_t + 1e-6) - torch.log(p_s + 1e-6)), dim=1).mean()

    count_loss = F.l1_loss(student_pred.sum(dim=(1,2,3)), teacher_label.sum(dim=(1,2,3)))
    return kl + 0.2 * count_loss

def train_distilation(
        teacher: nn.Module, 
        student: nn.Module,
        train_dataloader: DataLoader, 
        val_dataloader: DataLoader,
        optimizer: Optimizer,
        loss_fn, 
        epochs: int, 
        device: torch.device):
    
    teacher = teacher.to(device)
    teacher.eval()
    
    # Trackers
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': []
    }

    best_mae = float('inf')

    for epoch in range(epochs):
        student.train()
        epoch_loss = 0.0
        data_iter = tqdm(train_dataloader, desc=f'Epoch {epoch}/{epochs}')
        
        for teacher_data, student_data, gt_count in data_iter:
            teacher_data, student_data = teacher_data.to(device), student_data.to(device)

            with torch.no_grad():
                pseudo_labels = teacher(teacher_data)

            pred = student(student_data)
            loss = loss_fn(pred, pseudo_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            data_iter.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_train_loss = epoch_loss / len(train_dataloader)
        history['train_loss'].append(avg_train_loss)
        
        ## Validation
        student.eval()
        val_kl_loss = 0.0  
        val_mae = 0.0
        with torch.no_grad():
            for teacher_data, student_data, gt_count in val_dataloader:
                teacher_data, student_data = teacher_data.to(device), student_data.to(device)
                gt_count = gt_count.to(device)

                teacher_logits = teacher(teacher_data)
                student_logits = student(student_data)

                loss = KL_loss(student_logits, teacher_logits) 
                val_kl_loss += loss.item()

                pred_count = torch.sum(student_logits, dim=(1, 2, 3))
                val_mae += torch.abs(pred_count - gt_count).sum().item()

        avg_val_loss = val_kl_loss / len(val_dataloader)
        avg_mae = val_mae / len(val_dataloader.dataset)
        
        history['val_loss'].append(avg_val_loss)
        history['val_mae'].append(avg_mae)

        print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | MAE {avg_mae:.2f}")

        # Save Best Model
        if avg_mae < best_mae:
            best_mae = avg_mae
            os.makedirs('checkpoints/student', exist_ok=True)
            torch.save(student.state_dict(), 'checkpoints/student/best_student.pth')
        
        os.makedirs('assets', exist_ok=True)
    epochs_range = range(epochs)

    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epochs')
    plt.legend()

    # MAE Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['val_mae'], label='Val MAE', color='orange')
    plt.title('Validation Error (MAE)')
    plt.xlabel('Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig('assets/training_metrics.png')
    print("Metrics plot saved to assets/training_metrics.png")

def build_model_and_bins(args):
    _ = get_config(vars(args).copy(), mute=False)

    if args.regression:
        bins, anchor_points = None, None
    else:
        with open(os.path.join(current_dir, "configs", f"reduction_{args.reduction}.json"), "r") as f:
            config = json.load(f)[str(args.truncation)]["nwpu"]

        bins = config["bins"][args.granularity]
        anchor_points = (
            config["anchor_points"][args.granularity]["average"]
            if args.anchor_points == "average"
            else config["anchor_points"][args.granularity]["middle"]
        )
        bins = [(float(b[0]), float(b[1])) for b in bins]
        anchor_points = [float(p) for p in anchor_points]

    model = get_model(
        backbone=args.model,
        input_size=args.input_size,
        reduction=args.reduction,
        bins=bins,
        anchor_points=anchor_points,
        prompt_type=args.prompt_type,
        num_vpt=args.num_vpt,
        vpt_drop=args.vpt_drop,
        deep_vpt=not args.shallow_vpt,
    )

    state_dict = torch.load(args.weight_path, map_location="cpu")
    state_dict = state_dict if "best" in os.path.basename(args.weight_path) else state_dict["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    return model



if __name__ == "__main__":
    parser = ArgumentParser(description="Train Student model.")

    # Model args
    parser.add_argument("--model", type=str, default="clip_vit_l_14")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--reduction", type=int, default=8, choices=[8, 16, 32])
    parser.add_argument("--regression", action="store_true")
    parser.add_argument("--truncation", type=int, default=4)
    parser.add_argument("--anchor_points", type=str, default="average", choices=["average", "middle"])
    parser.add_argument("--prompt_type", type=str, default="word", choices=["word", "number"])
    parser.add_argument("--granularity", type=str, default="fine", choices=["fine", "dynamic", "coarse"])
    parser.add_argument("--num_vpt", type=int, default=32)
    parser.add_argument("--vpt_drop", type=float, default=0.0)
    parser.add_argument("--shallow_vpt", action="store_true")
    parser.add_argument("--weight_path", type=str, required=True)

    # Train args
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate.")
    parser.add_argument("--device", type=str, default='cpu', choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()
    args.model = args.model.lower()
    device = torch.device(args.device)

    train_data = Crowd_distilation(
        dataset="nwpu",
        split="train", ## Should be NWPU_crowd when working on hpc
        sigma=4.0, 
        return_filename=False,
        downscale=2
    )
    val_data = Crowd_distilation(
        dataset="nwpu",
        split="val", ## Should be NWPU_crowd when working on hpc
        sigma=4.0, 
        return_filename=False,
        downscale=2
    )

    train_loader = DataLoader(
        train_data,
        batch_size=8,       
        shuffle=True,        
        num_workers=4,       
        pin_memory=True,     
        drop_last=True       
    )
    val_loader = DataLoader(
        val_data,
        batch_size=8, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )

    teacher = build_model_and_bins(args).to(device)
    student = copy.deepcopy(teacher)

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    
    model = train_distilation(teacher, student, train_loader, val_loader, optimizer, KL_loss, args.epochs, device)





       
