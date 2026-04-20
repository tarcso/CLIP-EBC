import os
import json
import math
import csv
import torch
import numpy as np
import torch.nn.functional as F
from argparse import ArgumentParser
from tqdm import tqdm

current_dir = os.path.abspath(os.path.dirname(__file__))

from datasets import Crowd, Resize2Multiple
from models import get_model
from utils import get_config, sliding_window_predict


parser = ArgumentParser(description="Task 3: evaluate CLIP-EBC on NWPU val and study downscaling effects.")

# model args
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

# eval args
parser.add_argument("--sliding_window", action="store_true")
parser.add_argument("--stride", type=int, default=None)
parser.add_argument("--window_size", type=int, default=None)
parser.add_argument("--resize_to_multiple", action="store_true")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--top_k", type=int, default=25, help="How many worst examples to print/save.")
parser.add_argument("--save_dir", type=str, default="./task3_outputs")

# new args for downscaling study
parser.add_argument(
    "--downscale_factors",
    type=float,
    nargs="+",
    default=[1.0],
    help="Image downscaling factors. Example: 1 2 4 means original, /2, /4."
)
parser.add_argument(
    "--min_size",
    type=int,
    default=32,
    help="Minimum H/W after downscaling."
)


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


def downscale_image_tensor(image, factor, min_size=32):
    """
    image: torch.Tensor of shape (1, C, H, W)
    factor: 1.0 means original, 2.0 means downscale by 2x, 4.0 by 4x
    """
    if factor == 1.0:
        return image

    h, w = image.shape[-2], image.shape[-1]
    new_h = max(min_size, int(round(h / factor)))
    new_w = max(min_size, int(round(w / factor)))

    # Keep aspect ratio exact from division, use bilinear for RGB image
    image_ds = F.interpolate(
        image,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False
    )
    return image_ds


def evaluate_one_scale(args, model, dataset, downscale_factor):
    rows = []

    for idx in tqdm(range(len(dataset)), desc=f"Evaluating NWPU val @ /{downscale_factor:g}"):
        images, labels, _, image_names = dataset[idx]

        # original tensor shape: (1, C, H, W)
        image = images.to(args.device)
        points = labels[0]
        image_name = image_names[0]

        orig_h, orig_w = int(image.shape[-2]), int(image.shape[-1])

        # downscale image only; GT count stays the same
        image = downscale_image_tensor(image, downscale_factor, min_size=args.min_size)

        eval_h, eval_w = int(image.shape[-2]), int(image.shape[-1])
        gt_count = int(len(points))

        with torch.no_grad():
            if args.sliding_window:
                pred_density = sliding_window_predict(model, image, args.window_size, args.stride)
            else:
                pred_density = model(image)

            pred_count = float(pred_density.sum(dim=(1, 2, 3)).item())

        abs_error = abs(pred_count - gt_count)
        signed_error = pred_count - gt_count

        orig_area = orig_h * orig_w
        eval_area = eval_h * eval_w

        rows.append({
            "image": image_name,
            "orig_height": orig_h,
            "orig_width": orig_w,
            "eval_height": eval_h,
            "eval_width": eval_w,
            "orig_megapixels": orig_area / 1e6,
            "eval_megapixels": eval_area / 1e6,
            "downscale_factor": downscale_factor,
            "gt_count": gt_count,
            "pred_count": pred_count,
            "signed_error": signed_error,
            "abs_error": abs_error,
            "orig_pixels_per_person": orig_area / max(gt_count, 1),
            "eval_pixels_per_person": eval_area / max(gt_count, 1),
        })

    pred = np.array([r["pred_count"] for r in rows], dtype=np.float64)
    gt = np.array([r["gt_count"] for r in rows], dtype=np.float64)

    mae = float(np.mean(np.abs(pred - gt)))
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))

    rows_sorted = sorted(rows, key=lambda x: x["abs_error"], reverse=True)

    gt_q75 = float(np.percentile(gt, 75))
    epp_q25 = float(np.percentile([r["eval_pixels_per_person"] for r in rows], 25))

    likely_resolution_failures = [
        r for r in rows_sorted
        if r["gt_count"] >= gt_q75 and r["eval_pixels_per_person"] <= epp_q25
    ]

    return {
        "rows": rows,
        "rows_sorted": rows_sorted,
        "mae": mae,
        "rmse": rmse,
        "gt_q75": gt_q75,
        "epp_q25": epp_q25,
        "likely_resolution_failures": likely_resolution_failures,
    }


def save_scale_outputs(args, result, scale_dir):
    os.makedirs(scale_dir, exist_ok=True)

    summary_path = os.path.join(scale_dir, "summary.txt")
    csv_all_path = os.path.join(scale_dir, "all_results.csv")
    csv_worst_path = os.path.join(scale_dir, f"top_{args.top_k}_errors.csv")
    csv_res_path = os.path.join(scale_dir, "likely_resolution_failures.csv")

    with open(summary_path, "w") as f:
        f.write(f"MAE: {result['mae']:.4f}\n")
        f.write(f"RMSE: {result['rmse']:.4f}\n")
        f.write(f"Num images: {len(result['rows'])}\n")
        f.write(f"High-count threshold (75th percentile): {result['gt_q75']:.2f}\n")
        f.write(f"Low eval-pixels/person threshold (25th percentile): {result['epp_q25']:.2f}\n")

    fieldnames = [
        "image",
        "orig_height", "orig_width",
        "eval_height", "eval_width",
        "orig_megapixels", "eval_megapixels",
        "downscale_factor",
        "gt_count", "pred_count", "signed_error", "abs_error",
        "orig_pixels_per_person", "eval_pixels_per_person"
    ]

    with open(csv_all_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["rows_sorted"])

    with open(csv_worst_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["rows_sorted"][:args.top_k])

    with open(csv_res_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["likely_resolution_failures"][:args.top_k])

    return summary_path, csv_all_path, csv_worst_path, csv_res_path


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device)
    args.device = device

    model = build_model_and_bins(args).to(device)
    model.eval()

    # Keep dataset loading unchanged
    # Do NOT use Resize2Multiple here, because we want explicit control over downscaling
    transforms = None

    dataset = Crowd(
        dataset="nwpu",
        split="val",
        transforms=transforms,
        return_filename=True,
    )

    comparison_rows = []

    for factor in args.downscale_factors:
        factor = float(factor)

        result = evaluate_one_scale(args, model, dataset, factor)

        scale_tag = f"scale_{str(factor).replace('.', 'p')}"
        scale_dir = os.path.join(args.save_dir, scale_tag)
        summary_path, csv_all_path, csv_worst_path, csv_res_path = save_scale_outputs(args, result, scale_dir)

        print(f"\n=== OVERALL RESULTS ON NWPU VAL @ /{factor:g} ===")
        print(f"MAE  : {result['mae']:.4f}")
        print(f"RMSE : {result['rmse']:.4f}")

        print(f"\n=== TOP {args.top_k} WORST IMAGES BY ABSOLUTE ERROR @ /{factor:g} ===")
        for r in result["rows_sorted"][:args.top_k]:
            print(
                f"{r['image']:>10s} | gt={r['gt_count']:>5d} | pred={r['pred_count']:>9.2f} "
                f"| abs_err={r['abs_error']:>9.2f} | eval_size={r['eval_width']}x{r['eval_height']} "
                f"| eval_px/person={r['eval_pixels_per_person']:.2f}"
            )

        print(f"\nSaved:")
        print(f"  {summary_path}")
        print(f"  {csv_all_path}")
        print(f"  {csv_worst_path}")
        print(f"  {csv_res_path}")

        comparison_rows.append({
            "downscale_factor": factor,
            "mae": result["mae"],
            "rmse": result["rmse"],
        })

    comparison_csv = os.path.join(args.save_dir, "scale_comparison.csv")
    with open(comparison_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["downscale_factor", "mae", "rmse"])
        writer.writeheader()
        writer.writerows(comparison_rows)

    print("\n=== SCALE COMPARISON ===")
    for r in comparison_rows:
        print(f"/{r['downscale_factor']:g} -> MAE={r['mae']:.4f}, RMSE={r['rmse']:.4f}")

    print(f"\nSaved comparison: {comparison_csv}")


if __name__ == "__main__":
    args = parser.parse_args()
    args.model = args.model.lower()

    if args.regression:
        args.truncation = None
        args.anchor_points = None
        args.bins = None
        args.prompt_type = None
        args.granularity = None

    if "clip_vit" not in args.model:
        args.num_vpt = None
        args.vpt_drop = None
        args.shallow_vpt = None

    if "clip" not in args.model:
        args.prompt_type = None

    if args.sliding_window:
        args.window_size = args.input_size if args.window_size is None else args.window_size
        args.stride = args.input_size if args.stride is None else args.stride
    else:
        args.window_size = None
        args.stride = None
        args.resize_to_multiple = False

    main(args)