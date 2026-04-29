import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import normalize, to_pil_image
import matplotlib.pyplot as plt
import json
from datasets.crowd import Crowd, Crowd_distilation
import numpy as np

parent_dir = os.path.abspath(os.path.pardir)
sys.path.append(parent_dir)

import datasets
from models import get_model
from utils import resize_density_map, sliding_window_predict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


truncation = 4
reduction = 8
granularity = "fine"
anchor_points = "average"

model_name = "clip_vit_l_14"  # Changed to match the checkpoint configuration
input_size = 224
window_size = 224
stride = 224
weight_count_loss = 1.0
count_loss = "dmcount"

# Comment the lines below to test non-CLIP models.
prompt_type = "word"
num_vpt = 32
vpt_drop = 0.
deep_vpt = True

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
alpha = 0.8

if truncation is None:  # regression, no truncation.
    bins, anchor_points = None, None
else:
    #with open(os.path.join(parent_dir, "configs", f"reduction_{reduction}.json"), "r") as f:
    with open(os.path.join("configs", f"reduction_{reduction}.json"), "r") as f:
        config = json.load(f)[str(truncation)]["nwpu"]
    bins = config["bins"][granularity]
    anchor_points = config["anchor_points"][granularity]["average"] if anchor_points == "average" else config["anchor_points"][granularity]["middle"]
    bins = [(float(b[0]), float(b[1])) for b in bins]
    anchor_points = [float(p) for p in anchor_points]


model = get_model(
    backbone=model_name,
    input_size=input_size,
    reduction=reduction,
    bins=bins,
    anchor_points=anchor_points,
    # CLIP parameters
    prompt_type=prompt_type,
    num_vpt=num_vpt,
    vpt_drop=vpt_drop,
    deep_vpt=deep_vpt
)

ckpt_dir_name = f"{model_name}_{prompt_type}_" if "clip" in model_name else f"{model_name}_"
ckpt_dir_name += f"{input_size}_{reduction}_{truncation}_{granularity}_"
ckpt_dir_name += f"{weight_count_loss}_{count_loss}"

ckpt_path = os.path.join(
    #parent_dir,
    "checkpoints",
    "nwpu",
    "best_rmse_0.pth"  # Use the available checkpoint file
)
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt)
model = model.to(device)
model.eval()

#dataset = datasets.NWPUTest(transforms=None, return_filename=True)
dataset = Crowd(
    dataset="nwpu",
    split="val",
    transforms=None,
    sigma=None,
    return_filename=True
)

import random

os.makedirs("output", exist_ok=True)

num_images = 15
random_ids = random.sample(range(len(dataset)), num_images)

for img_id in random_ids:
    sample = dataset[img_id]

    image = sample[0]
    points = sample[1]

    image_path = None
    for item in sample:
        if isinstance(item, (str, bytes, os.PathLike)):
            image_path = item
            break

    if image_path is None:
        image_path = f"sample_{img_id}"

    if torch.is_tensor(points):
        points_np = points.cpu().numpy()
    else:
        points_np = np.array(points)

    points_np = points_np.reshape(-1, 2)
    gt_count = points_np.shape[0]

    if image.dim() == 3:
        image = image.unsqueeze(0)
    elif image.dim() == 4:
        pass
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    image_height, image_width = image.shape[-2:]
    image = image.to(device)

    with torch.no_grad():
        if stride is not None:
            pred_density = sliding_window_predict(model, image, window_size, stride)
        else:
            pred_density = model(image)

        pred_count = pred_density.sum().item()
        resized_pred_density = resize_density_map(
            pred_density, (image_height, image_width)
        ).cpu()

    vis_image = normalize(
        image,
        mean=(0., 0., 0.),
        std=(1. / std[0], 1. / std[1], 1. / std[2])
    )
    vis_image = normalize(
        vis_image,
        mean=(-mean[0], -mean[1], -mean[2]),
        std=(1., 1., 1.)
    )
    vis_image = to_pil_image(vis_image.squeeze(0).cpu())

    resized_pred_density = resized_pred_density.squeeze().numpy()

    if torch.is_tensor(points):
        points_np = points.cpu().numpy()
    else:
        points_np = np.array(points)

    fig, axes = plt.subplots(1, 2, dpi=200, tight_layout=True, frameon=False)

    axes[0].imshow(vis_image)
    points_np = points_np.reshape(-1, 2)

    if len(points_np) > 0:
        axes[0].scatter(points_np[:, 0], points_np[:, 1], s=2, c="white")
    axes[0].axis("off")
    axes[0].set_title(f"GT: {gt_count}")

    axes[1].imshow(vis_image)
    axes[1].imshow(resized_pred_density, cmap="jet", alpha=alpha)
    axes[1].axis("off")
    axes[1].set_title(f"Pred: {pred_count:.2f}")

    base_name = os.path.basename(str(image_path))
    base_name = os.path.splitext(base_name)[0]

    save_path = os.path.join("output", f"{img_id}_{base_name}.png")
    fig.savefig(save_path)
    plt.close(fig)

    print(f"Saved: {save_path} | GT: {gt_count} | Pred: {pred_count:.2f}")