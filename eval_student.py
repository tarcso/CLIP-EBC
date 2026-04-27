import os
import json
import csv
import torch
import numpy as np
import torch.nn.functional as F
from argparse import ArgumentParser
from tqdm import tqdm

current_dir = os.path.abspath(os.path.dirname(__file__))

from datasets import Crowd
from models import get_model
from utils import get_config, sliding_window_predict


def build_model(args, weight_path):
    _ = get_config(vars(args).copy(), mute=True)

    with open(os.path.join(current_dir, "configs", f"reduction_{args.reduction}.json"), "r") as f:
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
    return model


def evaluate(model, dataset, device, downscale, sliding_window, window_size, stride, label):
    rows = []
    desc = f"{label} @ {downscale}x downscale" if downscale > 1 else f"{label} @ original resolution"

    for idx in tqdm(range(len(dataset)), desc=desc):
        images, labels, _, image_names = dataset[idx]

        image = images.to(device)
        points = labels[0]
        image_name = image_names[0]

        orig_h, orig_w = image.shape[-2], image.shape[-1]

        if downscale > 1:
            new_h = max(32, int(round(orig_h / downscale)))
            new_w = max(32, int(round(orig_w / downscale)))
            image = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)

        eval_h, eval_w = image.shape[-2], image.shape[-1]
        gt_count = int(len(points))

        with torch.no_grad():
            if sliding_window:
                pred_density = sliding_window_predict(model, image, window_size, stride)
            else:
                pred_density = model(image)
            pred_count = float(pred_density.sum(dim=(1, 2, 3)).item())

        rows.append({
            "image": image_name,
            "orig_height": orig_h,
            "orig_width": orig_w,
            "eval_height": eval_h,
            "eval_width": eval_w,
            "gt_count": gt_count,
            "pred_count": round(pred_count, 2),
            "abs_error": abs(pred_count - gt_count),
            "signed_error": round(pred_count - gt_count, 2),
            "pixels_per_person": (eval_h * eval_w) / max(gt_count, 1),
        })

    pred = np.array([r["pred_count"] for r in rows], dtype=np.float64)
    gt = np.array([r["gt_count"] for r in rows], dtype=np.float64)
    mae = float(np.mean(np.abs(pred - gt)))
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))

    return rows, mae, rmse


def save_outputs(rows, mae, rmse, save_dir, top_k):
    os.makedirs(save_dir, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda r: r["abs_error"], reverse=True)

    fieldnames = [
        "image", "orig_height", "orig_width", "eval_height", "eval_width",
        "gt_count", "pred_count", "abs_error", "signed_error", "pixels_per_person",
    ]

    with open(os.path.join(save_dir, "summary.txt"), "w") as f:
        f.write(f"MAE:  {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"N:    {len(rows)}\n")

    with open(os.path.join(save_dir, "all_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)

    with open(os.path.join(save_dir, f"top_{top_k}_errors.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted[:top_k])


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    sliding_window = True
    window_size = args.input_size
    stride = args.input_size

    dataset = Crowd(dataset="nwpu", split="val", return_filename=True)

    results = {}

    # --- Teacher @ original resolution ---
    print("\nLoading teacher model...")
    teacher = build_model(args, args.teacher_weight_path).to(device)
    teacher.eval()

    rows, mae, rmse = evaluate(
        teacher, dataset, device,
        downscale=1,
        sliding_window=sliding_window,
        window_size=window_size,
        stride=stride,
        label="Teacher",
    )
    save_outputs(rows, mae, rmse, os.path.join(args.save_dir, "teacher_1x"), args.top_k)
    results["Teacher @ 1x (original)"] = (mae, rmse)

    # --- Teacher @ downscale (no-training baseline) ---
    rows, mae, rmse = evaluate(
        teacher, dataset, device,
        downscale=args.downscale,
        sliding_window=sliding_window,
        window_size=window_size,
        stride=stride,
        label="Teacher",
    )
    save_outputs(rows, mae, rmse, os.path.join(args.save_dir, f"teacher_{args.downscale}x"), args.top_k)
    results[f"Teacher @ {args.downscale}x (no training)"] = (mae, rmse)
    del teacher

    # --- Student @ downscale ---
    print("\nLoading student model...")
    student = build_model(args, args.student_weight_path).to(device)
    student.eval()

    rows, mae, rmse = evaluate(
        student, dataset, device,
        downscale=args.downscale,
        sliding_window=sliding_window,
        window_size=window_size,
        stride=stride,
        label="Student",
    )
    save_outputs(rows, mae, rmse, os.path.join(args.save_dir, f"student_{args.downscale}x"), args.top_k)
    results[f"Student @ {args.downscale}x (distilled)"] = (mae, rmse)

    # --- Comparison table ---
    comparison = [{"model": k, "mae": v[0], "rmse": v[1]} for k, v in results.items()]
    with open(os.path.join(args.save_dir, "comparison.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "mae", "rmse"])
        writer.writeheader()
        writer.writerows(comparison)

    print("\n" + "=" * 55)
    print(f"{'Model':<35} {'MAE':>8} {'RMSE':>10}")
    print("-" * 55)
    for label, (mae, rmse) in results.items():
        print(f"{label:<35} {mae:>8.2f} {rmse:>10.2f}")
    print("=" * 55)
    print(f"\nResults saved to: {args.save_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate student vs teacher on NWPU val.")

    # Model architecture (must match training)
    parser.add_argument("--model", type=str, default="clip_vit_l_14")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--reduction", type=int, default=8, choices=[8, 16, 32])
    parser.add_argument("--truncation", type=int, default=4)
    parser.add_argument("--anchor_points", type=str, default="average")
    parser.add_argument("--prompt_type", type=str, default="word")
    parser.add_argument("--granularity", type=str, default="fine")
    parser.add_argument("--num_vpt", type=int, default=32)
    parser.add_argument("--vpt_drop", type=float, default=0.0)

    # Weights
    parser.add_argument("--teacher_weight_path", type=str, required=True)
    parser.add_argument("--student_weight_path", type=str, default="./checkpoints/student/best_student.pth")

    # Eval settings
    parser.add_argument("--downscale", type=int, default=2, choices=[2, 4])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top_k", type=int, default=25)
    parser.add_argument("--save_dir", type=str, default="./student_eval_outputs")

    args = parser.parse_args()
    args.model = args.model.lower()
    args.regression = False

    main(args)
