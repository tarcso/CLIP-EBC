import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Union


def calculate_errors(pred_counts: np.ndarray, gt_counts: np.ndarray) -> Dict[str, float]:
    assert isinstance(pred_counts, np.ndarray), f"Expected numpy.ndarray, got {type(pred_counts)}"
    assert isinstance(gt_counts, np.ndarray), f"Expected numpy.ndarray, got {type(gt_counts)}"
    assert len(pred_counts) == len(gt_counts), f"Length of predictions and ground truths should be equal, but got {len(pred_counts)} and {len(gt_counts)}"
    errors = {
        "mae": np.mean(np.abs(pred_counts - gt_counts)),
        "rmse": np.sqrt(np.mean((pred_counts - gt_counts) ** 2)),
    }
    return errors


def resize_density_map(x: Tensor, size: Tuple[int, int]) -> Tensor:
    x_sum = torch.sum(x, dim=(-1, -2))
    x = F.interpolate(x, size=size, mode="bilinear")
    scale_factor = torch.nan_to_num(torch.sum(x, dim=(-1, -2)) / x_sum, nan=0.0, posinf=0.0, neginf=0.0)
    return x * scale_factor


def sliding_window_predict(
    model: nn.Module,
    image: Tensor,
    window_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
) -> Tensor:
    """
    Generate the density map for an image using the sliding window method. Overlapping regions will be averaged.

    Args:
        model (nn.Module): The model to use.
        image (Tensor): The image (1, c, h, w) to generate the density map for. The batch size must be 1 due to varying image sizes.
        window_size (Union[int, Tuple[int, int]]): The size of the window.
        stride (Union[int, Tuple[int, int]]): The step size of the window.
    """
    assert len(image.shape) == 4, f"Image must be a 4D tensor (1, c, h, w), got {image.shape}"
    window_size = (int(window_size), int(window_size)) if isinstance(window_size, (int, float)) else window_size
    stride = (int(stride), int(stride)) if isinstance(stride, (int, float)) else stride
    window_size = tuple(window_size)
    stride = tuple(stride)
    assert isinstance(window_size, tuple) and len(window_size) == 2 and window_size[0] > 0 and window_size[1] > 0, f"Window size must be a positive integer tuple (h, w), got {window_size}"
    assert isinstance(stride, tuple) and len(stride) == 2 and stride[0] > 0 and stride[1] > 0, f"Stride must be a positive integer tuple (h, w), got {stride}"
    assert stride[0] <= window_size[0] and stride[1] <= window_size[1], f"Stride must be smaller than window size, got {stride} and {window_size}"

    # Pad image to at least window_size so every pixel is covered
    pad_h = max(0, window_size[0] - image.shape[-2])
    pad_w = max(0, window_size[1] - image.shape[-1])
    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="constant", value=0)

    image_height, image_width = image.shape[-2:]
    window_height, window_width = window_size
    stride_height, stride_width = stride

    num_rows = max(1, int(np.ceil((image_height - window_height) / stride_height) + 1)) if image_height >= window_height else 1
    num_cols = max(1, int(np.ceil((image_width - window_width) / stride_width) + 1)) if image_width >= window_width else 1

    reduction = model.reduction if hasattr(model, "reduction") else 1  # reduction factor of the model. For example, if reduction = 8, then the density map will be reduced by 8x.
    windows = []
    for i in range(num_rows):
        for j in range(num_cols):
            x_start, y_start = i * stride_height, j * stride_width
            x_end, y_end = x_start + window_height, y_start + window_width
            if x_end > image_height:
                x_start = max(0, image_height - window_height)
                x_end = image_height
            if y_end > image_width:
                y_start = max(0, image_width - window_width)
                y_end = image_width

            window = image[:, :, x_start:x_end, y_start:y_end]
            windows.append(window)

    windows = torch.cat(windows, dim=0).to(image.device)  # batched windows, shape: (num_windows, c, h, w)

    model.eval()
    with torch.no_grad():
        preds = model(windows)
    preds = preds.cpu().detach().numpy()

    # assemble the density map
    pred_map = np.zeros((preds.shape[1], image_height // reduction, image_width // reduction), dtype=np.float32)
    count_map = np.zeros((preds.shape[1], image_height // reduction, image_width // reduction), dtype=np.float32)
    idx = 0
    for i in range(num_rows):
        for j in range(num_cols):
            x_start, y_start = i * stride_height, j * stride_width
            x_end, y_end = x_start + window_height, y_start + window_width
            if x_end > image_height:
                x_start = max(0, image_height - window_height)
                x_end = image_height
            if y_end > image_width:
                y_start = max(0, image_width - window_width)
                y_end = image_width

            pred_h, pred_w = preds[idx].shape[-2], preds[idx].shape[-1]
            x_s = x_start // reduction
            y_s = y_start // reduction
            pred_map[:, x_s:x_s + pred_h, y_s:y_s + pred_w] += preds[idx, :, :, :]
            count_map[:, x_s:x_s + pred_h, y_s:y_s + pred_w] += 1
            idx += 1

    pred_map /= count_map  # average the overlapping regions
    return torch.tensor(pred_map).unsqueeze(0)  # shape: (1, 1, h // reduction, w // reduction)
