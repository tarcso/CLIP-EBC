# 🚀 CLIP-EBC — DTU Fork

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip-ebc-clip-can-count-accurately-through/crowd-counting-on-ucf-qnrf)](https://paperswithcode.com/sota/crowd-counting-on-ucf-qnrf?p=clip-ebc-clip-can-count-accurately-through)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip-ebc-clip-can-count-accurately-through/crowd-counting-on-shanghaitech-a)](https://paperswithcode.com/sota/crowd-counting-on-shanghaitech-a?p=clip-ebc-clip-can-count-accurately-through)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip-ebc-clip-can-count-accurately-through/crowd-counting-on-shanghaitech-b)](https://paperswithcode.com/sota/crowd-counting-on-shanghaitech-b?p=clip-ebc-clip-can-count-accurately-through)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/clip-ebc-clip-can-count-accurately-through/crowd-counting-on-nwpu-crowd-val)](https://paperswithcode.com/sota/crowd-counting-on-nwpu-crowd-val?p=clip-ebc-clip-can-count-accurately-through)

This is a fork of the official [CLIP-EBC](https://github.com/Yiming-M/CLIP-EBC) repository, extended as part of the DTU course 02501 Advanced Deep Learning in Computer Vision.

- **Task 3** — baseline evaluation of CLIP-EBC (ViT-L/14) on the NWPU-Crowd validation set
- **Task 4** — downscaling study: evaluating the model at 1×, 2×, and 4× resolution reduction
- **Task 5** — teacher/student knowledge distillation: training student models on 2× and 4× downscaled images
- **Task 6** — real-world evaluation on zoom in/out image pairs (including king's crowning photos)
- Bug fixes to `utils/eval_utils.py` for robust sliding window prediction on small/downscaled images

Based on the paper [*CLIP-EBC: CLIP Can Count Accurately through Enhanced Blockwise Classification*](https://arxiv.org/abs/2403.09281v1).

---

## Results on NWPU Val

### Downscaling study (Task 4) — teacher model, no retraining

| **Downscale Factor** | **MAE** | **RMSE** |
|----------------------|---------|----------|
| 1× (original)        | 34.49   | 79.71    |
| 2×                   | 52.78   | 288.49   |
| 4×                   | 109.09  | 566.58   |

> Only the input images are downscaled at inference time. Ground truth counts remain unchanged.

### Knowledge distillation (Task 5) — 2× downscale

| **Model** | **MAE** | **RMSE** |
|-----------|---------|----------|
| Teacher @ 1× (upper bound) | 34.49 | 79.71 |
| Teacher @ 2× (baseline to beat) | 52.78 | 288.49 |
| Student @ 2× (50 epochs, lr=1e-5) | 102.65 | 230.52 |
| Student @ 2× (100 epochs, lr=3e-5) | 113.00 | 210.96 |

> The student starts from the teacher's pretrained weights and is fine-tuned on 2× downscaled images using the teacher's density maps as pseudo-labels (no extra annotations needed). RMSE improved significantly across both runs (288→210), showing distillation reduces worst-case failures on dense crowds. The 50-epoch run gives better MAE; the 100-epoch run overfit after epoch 44 but achieved the lowest RMSE.

### Knowledge distillation (Task 5) — 4× downscale

| **Model** | **MAE** | **RMSE** |
|-----------|---------|----------|
| Teacher @ 1× (upper bound) | 34.49 | 79.71 |
| Teacher @ 4× (baseline to beat) | 109.09 | 566.58 |
| Student @ 4× (50 epochs, lr=3e-5) | TBD | TBD |

### Real-world evaluation (Task 6) — zoom in/out pairs

61 image pairs with varying zoom ratios (~1.2× to ~7.5×). No ground truth — predictions only.

| **Folder** | **Zoom ratio** | **Teacher HR** | **Teacher LR** | **Student 2× LR** | **Student 4× LR** |
|-----------|---------------|---------------|---------------|------------------|------------------|
| 60 (king's crowning) | ~7.5× | TBD | TBD | TBD | TBD |
| Average (all 61 pairs) | — | TBD | TBD | TBD | TBD |

> Real-world images from `/dtu/blackhole/02/137570/MultiRes/test`. Each folder contains one HR and one LR image of the same scene taken at different focal lengths.

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

> `requirements.txt` includes `--extra-index-url https://download.pytorch.org/whl/cu121` so PyTorch with CUDA 12.1 support is installed automatically.

### 3. Link the dataset (DTU HPC only)

```bash
mkdir -p data
ln -s /dtu/blackhole/02/137570/MultiRes/NWPU_crowd data/NWPU-Crowd
```

### 4. Preprocess the data

```bash
bash preprocess.sh
```

This populates `data/nwpu/` with train/val/test splits. Errors about ShanghaiTech and UCF-QNRF are expected — those datasets are not available on HPC.

```
data/nwpu/
├── train/images/
├── train/labels/
├── val/images/
├── val/labels/
└── test/images/
```

> The test set has no ground truth labels and can be deleted after preprocessing to save ~3.4GB:
> ```bash
> rm -rf data/nwpu/test
> ```

### 5. Download the checkpoints

**Teacher (pretrained CLIP-EBC ViT-L/14):** download from the [releases page](https://github.com/Yiming-M/CLIP-EBC/releases):

```bash
wget https://github.com/Yiming-M/CLIP-EBC/releases/download/v1.0.0/NWPU_CLIP_ViT_B_16_Word_rmse.tgz
tar -xzf NWPU_CLIP_ViT_B_16_Word_rmse.tgz.tar.gz
```

The checkpoint should end up at `checkpoints/nwpu/best_rmse_0.pth`.

**Student (distilled):** available on HuggingFace at [dimos-stavaris/clip-ebc-student-teacher](https://huggingface.co/dimos-stavaris/clip-ebc-student-teacher):

```bash
hf download dimos-stavaris/clip-ebc-student-teacher best_student_e50_lr1e-5.pth --local-dir checkpoints/student/
```

Or in Python:
```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="dimos-stavaris/clip-ebc-student-teacher", filename="best_student_e50_lr1e-5.pth", local_dir="checkpoints/student/")
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

### Task 5 — Teacher/student distillation

Train the student (submit on DTU HPC):

```bash
bsub < train_student.sh
```

The student is trained with:
- Teacher frozen, providing density map pseudo-labels at 448×448
- Student fine-tuned on the same crops downscaled (2× → 224×224, 4× → 112×112)
- Loss: MSE on density maps + 0.1× L1 on total count
- Optimizer: AdamW, lr=3e-5, cosine LR decay, 50 epochs

Change `--downscale 2` to `--downscale 4` in `train_student.sh` for the 4× experiment.

Evaluate student vs teacher on full val images:

```bash
bsub < eval_student.sh
```

Results are saved to `student_eval_outputs/` (tagged by epochs, lr, and downscale factor).

### Task 6 — Real-world evaluation

```bash
bsub < eval_realworld.sh
```

Runs teacher and both students on all 61 zoom in/out pairs in `/dtu/blackhole/02/137570/MultiRes/test`. Results and visualizations saved to `realworld_outputs/`.

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