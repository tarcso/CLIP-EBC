import os
import sys
import csv
import json
import torch
import numpy as np
import torch.nn.functional as F
from argparse import ArgumentParser
from tqdm import tqdm

# basicsr imports torchvision.transforms.functional_tensor which was removed
# in newer torchvision. Patch it before basicsr is imported.
import torchvision.transforms.functional as _tvf
import types
_fake = types.ModuleType("torchvision.transforms.functional_tensor")
_fake.rgb_to_grayscale = _tvf.rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = _fake

current_dir = os.path.abspath(os.path.dirname(__file__))

from datasets import Crowd
from models import get_model
from utils import get_config, sliding_window_predict


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


def build_esrgan(weight_path, device):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=weight_path,
        model=model,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=True,
        device=device,
    )
    return upsampler


def apply_bicubic(image_tensor, target_h, target_w):
    return F.interpolate(image_tensor, size=(target_h, target_w), mode="bicubic", align_corners=False)


def apply_esrgan(image_tensor, upsampler, target_h, target_w):
    # Convert to numpy uint8 BGR for ESRGAN
    img = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # Denormalize from ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img * std + mean).clip(0, 1)
    img = (img * 255).astype(np.uint8)[:, :, ::-1]  # RGB→BGR

    output, _ = upsampler.enhance(img, outscale=4)

    # Convert back: BGR→RGB, normalize
    output = output[:, :, ::-1].astype(np.float32) / 255.0
    output = (output - mean) / std
    output = torch.from_numpy(output).permute(2, 0, 1).unsqueeze(0).float()

    # Resize to target if needed
    if output.shape[-2] != target_h or output.shape[-1] != target_w:
        output = F.interpolate(output, size=(target_h, target_w), mode="bicubic", align_corners=False)
    return output


def evaluate(model, dataset, device, window_size, stride, downscale, sr_fn=None, label=""):
    rows = []
    for idx in tqdm(range(len(dataset)), desc=label):
        images, labels, _, image_names = dataset[idx]
        image = images.to(device)
        points = labels[0]
        image_name = image_names[0]
        gt_count = int(len(points))

        orig_h, orig_w = image.shape[-2], image.shape[-1]

        if downscale > 1:
            lr_h = max(32, int(round(orig_h / downscale)))
            lr_w = max(32, int(round(orig_w / downscale)))
            image = F.interpolate(image, size=(lr_h, lr_w), mode="bilinear", align_corners=False)

        if sr_fn is not None:
            image = sr_fn(image, orig_h, orig_w).to(device)

        with torch.no_grad():
            pred_density = sliding_window_predict(model, image, window_size, stride)
            pred_count = float(pred_density.sum(dim=(1, 2, 3)).item())

        rows.append({
            "image": image_name,
            "gt_count": gt_count,
            "pred_count": round(pred_count, 2),
            "abs_error": abs(pred_count - gt_count),
        })

    pred = np.array([r["pred_count"] for r in rows])
    gt = np.array([r["gt_count"] for r in rows])
    mae = float(np.mean(np.abs(pred - gt)))
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    return rows, mae, rmse


def save_outputs(rows, mae, rmse, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "summary.txt"), "w") as f:
        f.write(f"MAE:  {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"N:    {len(rows)}\n")
    with open(os.path.join(save_dir, "all_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "gt_count", "pred_count", "abs_error"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["abs_error"], reverse=True))


def main(args):
    device = torch.device(args.device)
    window_size = args.input_size
    stride = args.input_size

    dataset = Crowd(dataset="nwpu", split="val", return_filename=True)

    print("Loading teacher...")
    teacher = build_model(args, args.teacher_weight_path).to(device)

    print("Loading student...")
    student = build_model(args, args.student_weight_path).to(device)

    esrgan = None
    if args.esrgan_weight_path and os.path.exists(args.esrgan_weight_path):
        print("Loading Real-ESRGAN...")
        esrgan = build_esrgan(args.esrgan_weight_path, device)
    else:
        print("ESRGAN weights not found, skipping learned SR.")

    results = {}

    # 1. Teacher @ original
    rows, mae, rmse = evaluate(teacher, dataset, device, window_size, stride, downscale=1, label="Teacher @ 1x")
    save_outputs(rows, mae, rmse, os.path.join(args.save_dir, "teacher_1x"))
    results["Teacher @ 1x (original)"] = (mae, rmse)

    # 2. Teacher @ downscaled
    rows, mae, rmse = evaluate(teacher, dataset, device, window_size, stride, downscale=args.downscale, label=f"Teacher @ {args.downscale}x")
    save_outputs(rows, mae, rmse, os.path.join(args.save_dir, f"teacher_{args.downscale}x"))
    results[f"Teacher @ {args.downscale}x (degraded)"] = (mae, rmse)

    # 3. Teacher @ bicubic SR
    bicubic_fn = lambda img, h, w: apply_bicubic(img, h, w)
    rows, mae, rmse = evaluate(teacher, dataset, device, window_size, stride, downscale=args.downscale, sr_fn=bicubic_fn, label="Teacher @ bicubic SR")
    save_outputs(rows, mae, rmse, os.path.join(args.save_dir, "teacher_bicubic_sr"))
    results[f"Teacher @ bicubic SR ({args.downscale}x→1x)"] = (mae, rmse)

    # 4. Teacher @ Real-ESRGAN SR
    if esrgan is not None:
        esrgan_fn = lambda img, h, w: apply_esrgan(img, esrgan, h, w)
        rows, mae, rmse = evaluate(teacher, dataset, device, window_size, stride, downscale=args.downscale, sr_fn=esrgan_fn, label="Teacher @ ESRGAN SR")
        save_outputs(rows, mae, rmse, os.path.join(args.save_dir, "teacher_esrgan_sr"))
        results[f"Teacher @ Real-ESRGAN ({args.downscale}x→1x)"] = (mae, rmse)

    # 5. Student @ downscaled (distillation)
    rows, mae, rmse = evaluate(student, dataset, device, window_size, stride, downscale=args.downscale, label=f"Student @ {args.downscale}x")
    save_outputs(rows, mae, rmse, os.path.join(args.save_dir, f"student_{args.downscale}x"))
    results[f"Student @ {args.downscale}x (distilled)"] = (mae, rmse)

    # Comparison table
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "comparison.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "mae", "rmse"])
        writer.writeheader()
        writer.writerows([{"model": k, "mae": v[0], "rmse": v[1]} for k, v in results.items()])

    print("\n" + "=" * 60)
    print(f"{'Model':<40} {'MAE':>8} {'RMSE':>10}")
    print("-" * 60)
    for label, (mae, rmse) in results.items():
        print(f"{label:<40} {mae:>8.2f} {rmse:>10.2f}")
    print("=" * 60)
    print(f"\nResults saved to: {args.save_dir}")


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

    parser.add_argument("--teacher_weight_path", type=str, required=True)
    parser.add_argument("--student_weight_path", type=str, required=True)
    parser.add_argument("--esrgan_weight_path", type=str, default="./weights/RealESRGAN_x4plus.pth")

    parser.add_argument("--downscale", type=int, default=2, choices=[2, 4])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="./sr_eval_outputs")

    args = parser.parse_args()
    args.model = args.model.lower()
    main(args)
