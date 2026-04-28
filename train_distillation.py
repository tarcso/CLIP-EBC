import torch
from torch import nn
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from datasets.crowd import Crowd_distilation
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import copy
import os
import json
from models import get_model
from utils import get_config
from torch.utils.data import DataLoader
try:
    from huggingface_hub import HfApi
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

current_dir = os.path.abspath(os.path.dirname(__file__))


def distillation_loss(student_pred, teacher_label):
    """MSE on density map values + L1 on total count.

    MSE preserves absolute scale so the student learns both where the crowd is
    and how many people are there, without the softmax normalisation destroying
    count information.  The count term provides an explicit second-order signal.
    """
    if isinstance(student_pred, (list, tuple)):
        student_pred = student_pred[0]
    if isinstance(teacher_label, (list, tuple)):
        teacher_label = teacher_label[0]

    if student_pred.dim() == 3:
        student_pred = student_pred.unsqueeze(1)
    if teacher_label.dim() == 3:
        teacher_label = teacher_label.unsqueeze(1)

    if teacher_label.shape[2:] != student_pred.shape[2:]:
        # mode="area" averages pixels when downsampling, which divides the density sum by the
        # spatial ratio (e.g. 56x56->28x28 gives sum/4). Multiply back to preserve total count.
        spatial_ratio = (teacher_label.shape[2] * teacher_label.shape[3]) / (
            student_pred.shape[2] * student_pred.shape[3]
        )
        teacher_label = F.interpolate(teacher_label, size=student_pred.shape[2:], mode="area") * spatial_ratio

    density_loss = F.mse_loss(student_pred, teacher_label)
    count_loss = F.l1_loss(
        student_pred.sum(dim=(1, 2, 3)),
        teacher_label.sum(dim=(1, 2, 3)),
    )
    return density_loss + 0.1 * count_loss


def train_distilation(
    teacher: nn.Module,
    student: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: Optimizer,
    scheduler,
    loss_fn,
    epochs: int,
    device: torch.device,
    run_tag: str = "",
):
    teacher = teacher.to(device)
    teacher.eval()

    scaler = GradScaler()

    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_rmse": []}
    best_mae = float("inf")

    for epoch in range(epochs):
        student.train()
        epoch_loss = 0.0
        data_iter = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for teacher_data, student_data, _ in data_iter:
            teacher_data = teacher_data.to(device)
            student_data = student_data.to(device)

            with torch.no_grad():
                with autocast():
                    teacher_out = teacher(teacher_data)
                    # teacher is in eval() → returns exp tensor directly (no tuple)
                    pseudo_labels = teacher_out[0] if isinstance(teacher_out, tuple) else teacher_out

            optimizer.zero_grad()
            with autocast():
                student_out = student(student_data)
                # student is in train() → returns (logits, exp); use exp (index 1) for distillation,
                # not logits (index 0) which has 5 channels and would mismatch teacher's 1-channel exp
                pred = student_out[1] if isinstance(student_out, tuple) else student_out
                loss = loss_fn(pred, pseudo_labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            data_iter.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_dataloader)
        history["train_loss"].append(avg_train_loss)

        # Validation — fixed seed so random crops are identical across epochs,
        # making MAE/RMSE comparable run-to-run.
        torch.manual_seed(42)
        student.eval()
        val_loss = 0.0
        val_sq_errors = []
        val_abs_errors = []
        with torch.no_grad():
            for teacher_data, student_data, gt_count in val_dataloader:
                teacher_data = teacher_data.to(device)
                student_data = student_data.to(device)
                gt_count = gt_count.to(device)

                with autocast():
                    teacher_out = teacher(teacher_data)
                    # both are in eval() → return exp tensor directly (no tuple)
                    teacher_exp = teacher_out[0] if isinstance(teacher_out, tuple) else teacher_out

                    student_out = student(student_data)
                    student_exp = student_out[0] if isinstance(student_out, tuple) else student_out

                    loss = loss_fn(student_exp, teacher_exp)

                val_loss += loss.item()
                pred_count = student_exp.sum(dim=(1, 2, 3))
                diff = pred_count - gt_count
                val_abs_errors.append(diff.abs().cpu())
                val_sq_errors.append((diff ** 2).cpu())

        all_abs = torch.cat(val_abs_errors)
        all_sq = torch.cat(val_sq_errors)
        avg_val_loss = val_loss / len(val_dataloader)
        avg_mae = all_abs.mean().item()
        avg_rmse = all_sq.mean().sqrt().item()
        current_lr = scheduler.get_last_lr()[0]

        history["val_loss"].append(avg_val_loss)
        history["val_mae"].append(avg_mae)
        history["val_rmse"].append(avg_rmse)

        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"Train Loss {avg_train_loss:.4f} | "
            f"Val Loss {avg_val_loss:.4f} | "
            f"MAE {avg_mae:.2f} | "
            f"RMSE {avg_rmse:.2f} | "
            f"LR {current_lr:.2e}"
        )

        if avg_mae < best_mae:
            best_mae = avg_mae
            os.makedirs("checkpoints/student", exist_ok=True)
            torch.save(student.state_dict(), f"checkpoints/student/best_student{run_tag}.pth")
            print(f"  -> Saved best student (MAE {best_mae:.2f})")

    os.makedirs("assets", exist_ok=True)
    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss")
    plt.plot(epochs_range, history["val_loss"], label="Val Loss")
    plt.title("Loss History")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history["val_mae"], label="Val MAE", color="orange")
    plt.title("Validation MAE")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history["val_rmse"], label="Val RMSE", color="red")
    plt.title("Validation RMSE")
    plt.xlabel("Epochs")
    plt.legend()

    plt.tight_layout()
    plot_path = f"assets/training_metrics{run_tag}.png"
    plt.savefig(plot_path)
    print(f"Metrics plot saved to {plot_path}")

    history_path = f"assets/training_history{run_tag}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    # Upload best checkpoint to HuggingFace
    best_ckpt = f"checkpoints/student/best_student{run_tag}.pth"
    if HF_AVAILABLE and os.path.exists(best_ckpt):
        try:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=best_ckpt,
                path_in_repo=os.path.basename(best_ckpt),
                repo_id="dimos-stavaris/clip-ebc-student-teacher",
                repo_type="model",
            )
            print(f"Uploaded {os.path.basename(best_ckpt)} to HuggingFace")
        except Exception as e:
            print(f"HuggingFace upload skipped: {e}")


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
    parser = ArgumentParser(description="Train Student model via knowledge distillation.")

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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--downscale", type=int, default=2, choices=[2, 4])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])

    args = parser.parse_args()
    args.model = args.model.lower()
    device = torch.device(args.device)

    train_data = Crowd_distilation(
        dataset="nwpu",
        split="train",
        sigma=4.0,
        return_filename=False,
        downscale=args.downscale,
    )
    val_data = Crowd_distilation(
        dataset="nwpu",
        split="val",
        sigma=4.0,
        return_filename=False,
        downscale=args.downscale,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # must be 0 so torch.manual_seed(42) before val controls crop randomness
        pin_memory=True,
    )

    teacher = build_model_and_bins(args).to(device)
    student = copy.deepcopy(teacher)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    run_tag = f"_e{args.epochs}_lr{args.lr:.0e}_ds{args.downscale}"
    train_distilation(
        teacher, student, train_loader, val_loader,
        optimizer, scheduler, distillation_loss, args.epochs, device,
        run_tag=run_tag,
    )
