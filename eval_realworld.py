import os
import csv
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from argparse import ArgumentParser
from PIL import Image
from torchvision.transforms import ToTensor, Normalize

current_dir = os.path.abspath(os.path.dirname(__file__))

from models import get_model
from utils import get_config, sliding_window_predict

to_tensor = ToTensor()
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def load_image(path):
    img = Image.open(path).convert("RGB")
    return normalize(to_tensor(img)).unsqueeze(0)


def build_model(args, weight_path):
    _ = get_config(vars(args).copy(), mute=True)

    with open(os.path.join(current_dir, "configs", f"reduction_{args.reduction}.json")) as f:
        config = json.load(f)[str(args.truncation)]["nwpu"]

    bins = [(float(b[0]), float(b[1])) for b in config["bins"][args.granularity]]
    anchor_points = [float(p) for p in config["anchor_points"][args.granularity]["average"]]

    model = get_model(
        backbone=args.model,
        input_size=args.input_size,
        reduction=args.reduction,
        bins=bins,
        anchor_points=anchor_points,
        prompt_type=args.prompt_type,
        num_vpt=args.num_vpt,
        vpt_drop=args.vpt_drop,
        deep_vpt=True,
    )

    state_dict = torch.load(weight_path, map_location="cpu")
    state_dict = state_dict if "best" in os.path.basename(weight_path) else state_dict["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def predict(model, image, window_size, stride, device):
    image = image.to(device)
    with torch.no_grad():
        density = sliding_window_predict(model, image, window_size, stride)
    return float(density.sum().item())


def save_visualization(folder_id, data_root, teacher, student_ds2, student_ds4, window_size, stride, device, save_dir):
    hr_path = os.path.join(data_root, str(folder_id), f"{folder_id}_hr.jpg")
    lr_path = os.path.join(data_root, str(folder_id), f"{folder_id}_lr.jpg")

    hr_img = Image.open(hr_path).convert("RGB")
    lr_img = Image.open(lr_path).convert("RGB")

    hr_tensor = load_image(hr_path)
    lr_tensor = load_image(lr_path)

    teacher_hr = predict(teacher, hr_tensor, window_size, stride, device)
    teacher_lr = predict(teacher, lr_tensor, window_size, stride, device)
    student2_lr = predict(student_ds2, lr_tensor, window_size, stride, device) if student_ds2 else None
    student4_lr = predict(student_ds4, lr_tensor, window_size, stride, device) if student_ds4 else None

    n_cols = 2 + (1 if student_ds2 else 0) + (1 if student_ds4 else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))

    panels = [
        (hr_img, f"Teacher @ HR\nPrediction: {teacher_hr:.0f}"),
        (lr_img, f"Teacher @ LR\nPrediction: {teacher_lr:.0f}"),
    ]
    if student_ds2:
        panels.append((lr_img, f"Student 2× @ LR\nPrediction: {student2_lr:.0f}"))
    if student_ds4:
        panels.append((lr_img, f"Student 4× @ LR\nPrediction: {student4_lr:.0f}"))

    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=13)
        ax.axis("off")

    plt.suptitle(f"Folder {folder_id} — zoom ratio {hr_img.size[0]/lr_img.size[0]:.1f}×", fontsize=14)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"folder_{folder_id}_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved visualization for folder {folder_id}")


def main(args):
    device = torch.device(args.device)
    window_size = args.input_size
    stride = args.input_size

    print("Loading teacher...")
    teacher = build_model(args, args.teacher_weight_path).to(device)

    print("Loading student 2x...")
    student_ds2 = build_model(args, args.student_2x_weight_path).to(device)

    student_ds4 = None
    if args.student_4x_weight_path and os.path.exists(args.student_4x_weight_path):
        print("Loading student 4x...")
        student_ds4 = build_model(args, args.student_4x_weight_path).to(device)
    else:
        print("Student 4x not available, skipping.")

    data_root = args.data_root
    folders = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))], key=int)

    rows = []
    for folder_id in folders:
        hr_path = os.path.join(data_root, folder_id, f"{folder_id}_hr.jpg")
        lr_path = os.path.join(data_root, folder_id, f"{folder_id}_lr.jpg")

        if not os.path.exists(hr_path) or not os.path.exists(lr_path):
            continue

        hr_img = Image.open(hr_path)
        lr_img = Image.open(lr_path)
        zoom = hr_img.size[0] / lr_img.size[0]

        hr_tensor = load_image(hr_path)
        lr_tensor = load_image(lr_path)

        t_hr  = predict(teacher,    hr_tensor, window_size, stride, device)
        t_lr  = predict(teacher,    lr_tensor, window_size, stride, device)
        s2_lr = predict(student_ds2, lr_tensor, window_size, stride, device)
        s4_lr = predict(student_ds4, lr_tensor, window_size, stride, device) if student_ds4 else None

        row = {
            "folder": folder_id,
            "hr_size": f"{hr_img.size[0]}x{hr_img.size[1]}",
            "lr_size": f"{lr_img.size[0]}x{lr_img.size[1]}",
            "zoom_ratio": round(zoom, 2),
            "teacher_hr": round(t_hr, 1),
            "teacher_lr": round(t_lr, 1),
            "student_2x_lr": round(s2_lr, 1),
            "student_4x_lr": round(s4_lr, 1) if s4_lr is not None else "N/A",
        }
        rows.append(row)

        s4_str = f"  student_4x={s4_lr:.0f}" if s4_lr else ""
        print(f"Folder {folder_id:>3s} (zoom {zoom:.1f}x): teacher_hr={t_hr:.0f}  teacher_lr={t_lr:.0f}  student_2x={s2_lr:.0f}{s4_str}")

    os.makedirs(args.save_dir, exist_ok=True)
    fieldnames = ["folder", "hr_size", "lr_size", "zoom_ratio", "teacher_hr", "teacher_lr", "student_2x_lr", "student_4x_lr"]
    with open(os.path.join(args.save_dir, "results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {args.save_dir}/results.csv")

    # Visualizations for king's crowning (folder 60) and a few others
    print("\nGenerating visualizations...")
    for folder_id in args.visualize_folders:
        save_visualization(
            folder_id, data_root, teacher, student_ds2, student_ds4,
            window_size, stride, device,
            os.path.join(args.save_dir, "visualizations")
        )


if __name__ == "__main__":
    parser = ArgumentParser()

    # Model architecture
    parser.add_argument("--model", type=str, default="clip_vit_l_14")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--reduction", type=int, default=8)
    parser.add_argument("--truncation", type=int, default=4)
    parser.add_argument("--anchor_points", type=str, default="average")
    parser.add_argument("--prompt_type", type=str, default="word")
    parser.add_argument("--granularity", type=str, default="fine")
    parser.add_argument("--num_vpt", type=int, default=32)
    parser.add_argument("--vpt_drop", type=float, default=0.0)
    parser.add_argument("--regression", action="store_true")

    # Weights
    parser.add_argument("--teacher_weight_path", type=str, required=True)
    parser.add_argument("--student_2x_weight_path", type=str, required=True)
    parser.add_argument("--student_4x_weight_path", type=str, default=None)

    # Data + output
    parser.add_argument("--data_root", type=str, default="/dtu/blackhole/02/137570/MultiRes/test")
    parser.add_argument("--save_dir", type=str, default="./realworld_outputs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--visualize_folders", type=int, nargs="+", default=[60])

    args = parser.parse_args()
    args.model = args.model.lower()

    main(args)
