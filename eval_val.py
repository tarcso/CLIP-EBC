import torch
import argparse
import numpy as np
import os
import json
from tqdm import tqdm
from utils import get_dataloader, calculate_errors, sliding_window_predict
from models import get_model
from utils import get_config

current_dir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="clip_vit_l_14")
parser.add_argument("--input_size", type=int, default=224)
parser.add_argument("--reduction", type=int, default=8)
parser.add_argument("--truncation", type=int, default=4)
parser.add_argument("--anchor_points", type=str, default="average")
parser.add_argument("--prompt_type", type=str, default="word")
parser.add_argument("--granularity", type=str, default="fine")
parser.add_argument("--num_vpt", type=int, default=32)
parser.add_argument("--vpt_drop", type=float, default=0.0)
parser.add_argument("--shallow_vpt", action="store_true")
parser.add_argument("--regression", action="store_true")
parser.add_argument("--dataset", type=str, default="nwpu")
parser.add_argument("--sliding_window", action="store_true")
parser.add_argument("--window_size", type=int, default=224)
parser.add_argument("--stride", type=int, default=224)
parser.add_argument("--weight_path", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=1)
args = parser.parse_args()

device = torch.device(args.device)
_ = get_config(vars(args).copy(), mute=True)

with open(os.path.join(current_dir, "configs", f"reduction_{args.reduction}.json"), "r") as f:
    config = json.load(f)[str(args.truncation)]["nwpu"]
bins = config["bins"][args.granularity]
anchor_points = config["anchor_points"][args.granularity]["average"] if args.anchor_points == "average" else config["anchor_points"][args.granularity]["middle"]
bins = [(float(b[0]), float(b[1])) for b in bins]
anchor_points = [float(p) for p in anchor_points]
args.bins = bins
args.anchor_points = anchor_points

model = get_model(
    backbone=args.model,
    input_size=args.input_size,
    reduction=args.reduction,
    bins=bins,
    anchor_points=anchor_points,
    prompt_type=args.prompt_type,
    num_vpt=args.num_vpt,
    vpt_drop=args.vpt_drop,
    deep_vpt=not args.shallow_vpt
).to(device)

state_dict = torch.load(args.weight_path, map_location="cpu")
state_dict = state_dict if "best" in os.path.basename(args.weight_path) else state_dict["model_state_dict"]
model.load_state_dict(state_dict, strict=True)
model.eval()

val_loader = get_dataloader(args, split="val", ddp=False)

pred_counts, target_counts, filenames = [], [], []
for image, target_points, fname in tqdm(val_loader, desc="Evaluating val"):
    image = image.to(device)
    target_counts.append([len(p) for p in target_points])
    filenames.extend(fname)
    with torch.no_grad():
        if args.sliding_window:
            pred_density = sliding_window_predict(model, image, args.window_size, args.stride)
        else:
            pred_density = model(image)
        pred_counts.append(pred_density.sum(dim=(1, 2, 3)).cpu().numpy().tolist())

pred_counts = np.array([x for sublist in pred_counts for x in sublist])
target_counts = np.array([x for sublist in target_counts for x in sublist])

errors = calculate_errors(pred_counts, target_counts)
print(f"\nMAE:  {errors['mae']:.2f}")
print(f"RMSE: {errors['rmse']:.2f}")

abs_errors = np.abs(pred_counts - target_counts)
sorted_idx = np.argsort(abs_errors)[::-1]
print("\nTop 20 worst predictions:")
print(f"{'Image':<30} {'GT':>8} {'Pred':>8} {'Error':>8}")
for i in sorted_idx[:20]:
    print(f"{filenames[i]:<30} {target_counts[i]:>8.0f} {pred_counts[i]:>8.1f} {abs_errors[i]:>8.1f}")
