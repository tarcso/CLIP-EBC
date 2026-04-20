# 🚀 CLIP-EBC — DTU Fork

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip-ebc-clip-can-count-accurately-through/crowd-counting-on-ucf-qnrf)](https://paperswithcode.com/sota/crowd-counting-on-ucf-qnrf?p=clip-ebc-clip-can-count-accurately-through)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip-ebc-clip-can-count-accurately-through/crowd-counting-on-shanghaitech-a)](https://paperswithcode.com/sota/crowd-counting-on-shanghaitech-a?p=clip-ebc-clip-can-count-accurately-through)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip-ebc-clip-can-count-accurately-through/crowd-counting-on-shanghaitech-b)](https://paperswithcode.com/sota/crowd-counting-on-shanghaitech-b?p=clip-ebc-clip-can-count-accurately-through)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip-ebc-clip-can-count-accurately-through/crowd-counting-on-nwpu-crowd-val)](https://paperswithcode.com/sota/crowd-counting-on-nwpu-crowd-val?p=clip-ebc-clip-can-count-accurately-through)

This is a fork of the official [CLIP-EBC](https://github.com/Yiming-M/CLIP-EBC) repository, extended with:

- **Task 3** — baseline evaluation of CLIP-EBC (ViT-L/14) on the NWPU-Crowd validation set
- **Task 4** — downscaling study: evaluating the model at 1×, 2×, and 4× resolution reduction to study the effect of image resolution on counting accuracy
- Bug fixes to `utils/eval_utils.py` for robust sliding window prediction on small/downscaled images

Based on the paper [*CLIP-EBC: CLIP Can Count Accurately through Enhanced Blockwise Classification*](https://arxiv.org/abs/2403.09281v1).

---

## Results on NWPU Val — Downscaling Study

| **Downscale Factor** | **Eval Resolution** | **MAE** | **RMSE** |
|----------------------|---------------------|---------|----------|
| 1× (original)        | full                | ~61     | ~278     |
| 2×                   | half                | TBD     | TBD      |
| 4×                   | quarter             | 109.09  | 566.58   |

> The model was not retrained — only the input images are downscaled at inference time. Ground truth counts remain unchanged.

---

## Setup

> **DTU HPC users:** The NWPU-Crowd dataset is already available on the cluster at `/dtu/blackhole/02/137570/MultiRes/NWPU_crowd`. You do not need to download it.

### 1. Clone the repo

```bash
git clone git@github.com:tarcso/CLIP-EBC.git
cd CLIP-EBC
```

### 2. Create the environment

**Option A — Conda (recommended)**

```bash
conda create -n clip_ebc python=3.12.4 -y
conda activate clip_ebc
pip install -r requirements.txt
```

**Option B — venv**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Link the dataset (DTU HPC only)

```bash
ln -s /dtu/blackhole/02/137570/MultiRes/NWPU_crowd data/NWPU-Crowd
```

### 4. Preprocess the data

```bash
bash preprocess.sh
```

This populates `data/nwpu/` with train/val/test splits:

```
data/nwpu/
├── train/labels/
├── val/images/
├── val/labels/
└── test/images/
```

### 5. Download the checkpoint

Download the pretrained ViT-L/14 RMSE model from the [releases page](https://github.com/Yiming-M/CLIP-EBC/releases):

```bash
wget https://github.com/Yiming-M/CLIP-EBC/releases/download/v1.0.0/NWPU_CLIP_ViT_B_16_Word_rmse.tgz
tar -xzf NWPU_CLIP_ViT_B_16_Word_rmse.tgz.tar.gz
```

The checkpoint should end up at:

```
checkpoints/nwpu/best_rmse_0.pth
```

---

## Evaluation

### Task 3 — Baseline evaluation on NWPU Val

```bash
python -u task3_nwpu_val.py \
    --model clip_vit_l_14 \
    --input_size 224 \
    --reduction 8 \
    --truncation 4 \
    --anchor_points average \
    --prompt_type word \
    --num_vpt 32 \
    --vpt_drop 0.0 \
    --sliding_window \
    --stride 224 \
    --weight_path ./checkpoints/nwpu/best_rmse_0.pth \
    --device cuda \
    --save_dir ./task3_outputs
```

Results are saved to `task3_outputs/`.

### Task 4 — Downscaling study

```bash
python -u task4_nwpu_val.py \
    --model clip_vit_l_14 \
    --input_size 224 \
    --reduction 8 \
    --truncation 4 \
    --anchor_points average \
    --prompt_type word \
    --num_vpt 32 \
    --vpt_drop 0.0 \
    --sliding_window \
    --stride 224 \
    --weight_path ./checkpoints/nwpu/best_rmse_0.pth \
    --device cuda \
    --downscale_factors 1 2 4 \
    --save_dir ./task4_outputs_downscale
```

Output structure:

```
task4_outputs_downscale/
├── scale_1p0/
│   ├── summary.txt
│   ├── all_results.csv
│   ├── top_25_errors.csv
│   └── likely_resolution_failures.csv
├── scale_2p0/
├── scale_4p0/
└── scale_comparison.csv        ← MAE/RMSE across all scales side by side
```

---

## Bug Fixes in `utils/eval_utils.py`

The original sliding window implementation had three issues that surfaced when evaluating on small downscaled images. All fixes are in `utils/eval_utils.py`:

1. **Out-of-bounds accumulation** — slice bounds were derived from `x_end // reduction` instead of the actual prediction tensor shape, causing a shape mismatch. Fixed by using `pred_h, pred_w = preds[idx].shape[-2:]` to compute slice bounds.

2. **Window generation outside image bounds** — `num_rows`/`num_cols` could exceed 1 even when the image was smaller than the window, generating windows with negative `x_start`. Fixed by clamping `x_start` with `max(0, ...)`.

3. **NaN MAE/RMSE** — when a downscaled image was smaller than the window size, parts of `count_map` were never covered (remained zero), causing division by zero. Fixed by zero-padding the image to at least `window_size` before the sliding window loop:
    ```python
    pad_h = max(0, window_size[0] - image.shape[-2])
    pad_w = max(0, window_size[1] - image.shape[-1])
    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="constant", value=0)
    ```

---

## Citation

If you use this work, please cite the original paper:

```bibtex
@article{ma2024clip,
  title={CLIP-EBC: CLIP Can Count Accurately through Enhanced Blockwise Classification},
  author={Ma, Yiming and Sanchez, Victor and Guha, Tanaya},
  journal={arXiv preprint arXiv:2403.09281},
  year={2024}
}
```