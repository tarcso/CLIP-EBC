"""
Generate side-by-side density map visualizations comparing teacher and student
on selected NWPU val images. Useful for showing where distillation helps or hurts.
"""
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn.functional as F
from argparse import ArgumentParser
from PIL import Image
from torchvision.transforms import ToTensor, Normalize

current_dir = os.path.abspath(os.path.dirname(__file__))

from models import get_model
from utils import get_config, sliding_window_predict

to_tensor = ToTensor()
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def build_model(args, weight_path):
    _ = get_config(vars(args).copy(), mute=True)
    with open(os.path.join(current_dir, "configs", f"reduction_{args.reduction}.json")) as f:
        config = json.load(f)[str(args.truncation)]["nwpu"]
    bins = [(float(b[0]), float(b[1])) for b in config["bins"][args.granularity]]
    anchor_points = [float(p) for p in config["anchor_points"][args.granularity]["average"]]
    model = get_model(
        backbone=args.model, input_size=args.input_size, reduction=args.reduction,
        bins=bins, anchor_points=anchor_points, prompt_type=args.prompt_type,
        num_vpt=args.num_vpt, vpt_drop=args.vpt_drop, deep_vpt=True,
    )
    state_dict = torch.load(weight_path, map_location="cpu")
    state_dict = state_dict if "best" in os.path.basename(weight_path) else state_dict["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def get_density_map(model, image_tensor, window_size, stride, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        density = sliding_window_predict(model, image_tensor, window_size, stride)
    return density.squeeze().cpu().numpy()


def colorize(density_map):
    d = density_map.copy()
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    return cm.jet(d)[:, :, :3]


def get_gt_count(image_name, val_dir):
    label_dir = os.path.join(os.path.dirname(val_dir), "labels")
    label_name = image_name.replace(".jpg", ".npy")
    label_path = os.path.join(label_dir, label_name)
    if os.path.exists(label_path):
        points = np.load(label_path)
        return len(points)
    return None


def visualize_image(image_name, args, teacher, student_2x, student_4x, device, save_dir):
    img_path = os.path.join(args.val_dir, image_name)
    orig_img = Image.open(img_path).convert("RGB")
    image = normalize(to_tensor(orig_img)).unsqueeze(0)
    gt = get_gt_count(image_name, args.val_dir)

    window_size = args.input_size
    stride = args.input_size
    h, w = image.shape[-2], image.shape[-1]

    # Teacher @ 1x
    dm_t1 = get_density_map(teacher, image, window_size, stride, device)
    pred_t1 = dm_t1.sum()

    # Teacher @ 2x
    image_2x = F.interpolate(image, size=(max(32, h // 2), max(32, w // 2)), mode="bilinear", align_corners=False)
    dm_t2 = get_density_map(teacher, image_2x, window_size, stride, device)
    pred_t2 = dm_t2.sum()

    # Student @ 2x
    dm_s2 = get_density_map(student_2x, image_2x, window_size, stride, device)
    pred_s2 = dm_s2.sum()

    gt_str = f"GT: {gt}" if gt is not None else ""

    panels = [
        (orig_img, dm_t1, f"Teacher @ 1×\nPred: {pred_t1:.0f}"),
        (orig_img, dm_t2, f"Teacher @ 2×\nPred: {pred_t2:.0f}"),
        (orig_img, dm_s2, f"Student @ 2×\nPred: {pred_s2:.0f}"),
    ]

    if student_4x is not None:
        image_4x = F.interpolate(image, size=(max(32, h // 4), max(32, w // 4)), mode="bilinear", align_corners=False)
        dm_t4 = get_density_map(teacher, image_4x, window_size, stride, device)
        dm_s4 = get_density_map(student_4x, image_4x, window_size, stride, device)
        panels.append((orig_img, dm_t4, f"Teacher @ 4×\nPred: {dm_t4.sum():.0f}"))
        panels.append((orig_img, dm_s4, f"Student @ 4×\nPred: {dm_s4.sum():.0f}"))

    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    fig.suptitle(f"{image_name}  —  {gt_str}", fontsize=13, fontweight="bold")

    for col, (img, dm, title) in enumerate(panels):
        axes[0, col].imshow(img)
        axes[0, col].set_title(title, fontsize=10)
        axes[0, col].axis("off")

        dm_resized = F.interpolate(
            torch.from_numpy(dm).unsqueeze(0).unsqueeze(0).float(),
            size=(img.size[1], img.size[0]), mode="bilinear", align_corners=False
        ).squeeze().numpy()
        axes[1, col].imshow(colorize(dm_resized))
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Image", fontsize=9)
    axes[1, 0].set_ylabel("Density Map", fontsize=9)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, image_name.replace(".jpg", "_comparison.png"))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main(args):
    device = torch.device(args.device)

    print("Loading teacher...")
    teacher = build_model(args, args.teacher_weight_path).to(device)

    print("Loading student 2x...")
    student_2x = build_model(args, args.student_2x_weight_path).to(device)

    student_4x = None
    if args.student_4x_weight_path and os.path.exists(args.student_4x_weight_path):
        print("Loading student 4x...")
        student_4x = build_model(args, args.student_4x_weight_path).to(device)

    for image_name in args.images:
        print(f"\nProcessing {image_name}...")
        visualize_image(image_name, args, teacher, student_2x, student_4x, device, args.save_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
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

    parser.add_argument("--teacher_weight_path", type=str, default="./checkpoints/nwpu/best_rmse_0.pth")
    parser.add_argument("--student_2x_weight_path", type=str, default="./checkpoints/student/best_student_e50_lr1e-5.pth")
    parser.add_argument("--student_4x_weight_path", type=str, default="./checkpoints/student/best_student_e50_lr3e-05_ds4.pth")

    parser.add_argument("--val_dir", type=str, default="./data/nwpu/val/images")
    parser.add_argument("--save_dir", type=str, default="./assets/visualizations")
    parser.add_argument("--device", type=str, default="cuda")

    # 125, 047: student wins big at 2x/4x — 299: student loses — 244: best 4x win
    parser.add_argument("--images", type=str, nargs="+",
                        default=["125.jpg", "047.jpg", "299.jpg", "244.jpg"])

    args = parser.parse_args()
    args.model = args.model.lower()
    main(args)
