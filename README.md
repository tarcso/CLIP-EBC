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

> The student starts from the teacher's pretrained weights and is fine-tuned on 2× downscaled images using the teacher's density maps as pseudo-labels (no extra annotations needed).

**Per-image analysis (2×):** The student improves on only 97/500 images (19.4%). It overcounts on average — mean prediction 445 vs GT mean 392 — indicating a calibration problem. The RMSE improvement (288→230) is driven mainly by a handful of extreme cases where the teacher@2× fails catastrophically and the student corrects them (e.g. image 125.jpg: GT=12,924, teacher error 5,917 → student error 2,624). The student introduces its own large errors on other dense images, which is why MAE goes up even as RMSE goes down.

### Knowledge distillation (Task 5) — 4× downscale

| **Model** | **MAE** | **RMSE** |
|-----------|---------|----------|
| Teacher @ 1× (upper bound) | 34.49 | 79.71 |
| Teacher @ 4× (baseline to beat) | 109.09 | 566.58 |
| **Student @ 4× (50 epochs, lr=3e-5)** | **95.73** | **339.87** |

> The student outperforms the teacher at 4× downscale on both MAE and RMSE — the only case where distillation fully closes the gap.

**Per-image analysis (4×):** Student improves on 203/500 images (40.6%) — more than twice the 2× rate. Critically, the student is well-calibrated: mean prediction 369 vs GT 392 (−6% undercount), compared to teacher@4× which undercounts by 25% (pred 293 vs GT 392). On very dense images (GT≥500, n=90), the student wins 50/90 times with an average improvement of +138.8 — consistent and large. The 4× result is not driven by outliers but by genuine improvement across high-density scenes.

**Why 4× works better than 2×:** At 2×, the teacher@2× is already reasonable (MAE 52.78), so the student receives a weak learning signal and ends up miscalibrated. At 4×, the teacher@4× massively undercounts (MAE 109, −25% bias), giving the student a clear target to correct. The larger the degradation, the more the student benefits from high-resolution pseudo-labels.

### Knowledge distillation — partial backbone unfreezing

Same setup as 2× distillation but the last 3 CLIP transformer blocks are unfrozen with a lower LR (1e-6) to allow the backbone to adapt to low-resolution inputs. Trained with `train_distillation_unfreeze.py`.

| **Model** | **MAE** | **RMSE** |
|-----------|---------|----------|
| Student @ 2× frozen backbone (50 ep, lr=1e-5) | 102.65 | 230.52 |
| Student @ 2× unfrozen last 3 blocks (50 ep, lr=3e-5, backbone lr=1e-6) | 117.72 | 219.67 |

> Unfreezing the last 3 transformer blocks improved RMSE further (219 vs 230) but hurt MAE. The backbone adaptation reduces catastrophic errors but introduces more average error — the model becomes more conservative at low resolution.

### Knowledge distillation — higher count loss weight (λ=0.5)

Per-image analysis revealed the 2× student overcounts on average (mean pred 445 vs GT 392). This experiment increases λ from 0.1 to 0.5 to enforce count accuracy more strongly.

| **Model** | **MAE** | **RMSE** |
|-----------|---------|----------|
| Student @ 2× (λ=0.1, baseline) | 102.65 | 230.52 |
| Student @ 2× (λ=0.5) | 108.02 | 222.91 |

> Higher λ improved RMSE slightly (222 vs 230) but hurt MAE and made the overcounting bias worse (mean pred 458 vs GT 392). On its own, increasing the count loss weight does not fix calibration.

### Knowledge distillation — scale jitter (downscale 2×–4× random)

Instead of training at a fixed 2× downscale, the student sees a randomly sampled downscale factor between 2× and 4× per crop. The student input is always resized to 224×224, so varying amounts of blur/compression simulate different effective resolutions. Val uses fixed 2× for comparable metrics.

| **Model** | **MAE** | **RMSE** |
|-----------|---------|----------|
| Teacher @ 2× (baseline to beat) | 52.78 | 288.49 |
| Student @ 2× fixed (λ=0.1) | 102.65 | 230.52 |
| Student @ 2×–4× jitter (λ=0.1) | 84.78 | 166.48 |
| **Student @ 2×–4× jitter (λ=0.5)** | **80.54** | **166.70** |

> Scale jitter is by far the most effective improvement. MAE dropped from 102.65 → 80.54 and RMSE from 230 → 166 — a 42% RMSE reduction over the fixed-scale baseline. Interestingly, λ=0.5 works better *with* jitter (80.54) even though it was worse *without* it (108.02), suggesting the combination of scale jitter and stronger count supervision works as a regularizer.
>
> Per-image analysis explains why: on very dense images (GT≥500), the fixed-scale student averaged -93 improvement vs teacher@2× (i.e. was much worse). The jitter student averages -15 on the same images — nearly neutral. Image 125.jpg (GT=12,924): teacher@2× error 5,917, fixed student error 2,624, jitter student error **356**. Scale jitter forces the model to generalise across resolution levels, making it dramatically more robust to the extreme cases that previously dominated RMSE.

### Super-resolution as an alternative approach

An alternative to distillation is to use a super-resolution (SR) model as a preprocessing step: upscale the low-resolution image back to its original resolution before feeding it to the teacher. This requires no retraining of the counting model.

We compare two SR methods:
- **Bicubic** — classical interpolation, no learning, trivially fast
- **Real-ESRGAN** ([Wang et al., 2021](https://github.com/xinntao/Real-ESRGAN)) — a deep generative SR model trained to produce perceptually sharp images. Uses a residual-in-residual dense block (RRDB) architecture. We use the 4× upscaling variant.

The key question: does SR preprocessing outperform distillation, and is a learned SR model better than simple bicubic interpolation?

> Download the ESRGAN weights before running:
> ```bash
> mkdir -p weights
> wget -O weights/RealESRGAN_x4plus.pth \
>     https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
> ```
>
> Then run:
> ```bash
> bsub < eval_sr.sh
> ```

| **Method** | **MAE** | **RMSE** |
|-----------|---------|----------|
| Teacher @ 1× original | 34.49 | 79.71 |
| Teacher @ 2× downscaled (no recovery) | 52.78 | 288.49 |
| Teacher @ bicubic SR (2×→1×) | 45.55 | 168.94 |
| Teacher @ Real-ESRGAN (2×→1×) | 53.74 | 236.08 |
| Student @ 2× fixed distillation | 102.65 | 230.52 |
| **Student @ 2×–4× scale jitter (λ=0.5)** | **80.54** | **166.70** |

> **Finding:** Bicubic SR gives the best MAE (45.55) but the scale jitter student now matches it on RMSE (166.70 vs 168.94) without any preprocessing at inference time. Real-ESRGAN introduces hallucinated textures that confuse the counting model and gives no benefit over the degraded input. The scale jitter student is the best pure distillation approach and competitive with SR on RMSE.

### Real-world evaluation (Task 6) — zoom in/out pairs

61 image pairs with varying zoom ratios (~1.2× to ~7.5×). No ground truth — predictions only.

| **Folder** | **Zoom ratio** | **Teacher HR** | **Teacher LR** | **Student 2× LR** | **Student 4× LR** |
|-----------|---------------|---------------|---------------|------------------|------------------|
| 60 (king's crowning) | ~7.5× | 8204 | 4442 | 3584 | 1878 |
| Average (all 61 pairs) | ~2–4× | 545 | 373 | 571 | 581 |

> Real-world images from `/dtu/blackhole/02/137570/MultiRes/test`. Each folder contains one HR and one LR image of the same scene taken at different focal lengths. No ground truth — predictions only. The king's crowning (folder 60) has a 7.5× zoom ratio, well outside the 2×/4× training distribution of the students, which explains the large undercounting. At moderate zoom ratios (2–4×), the teacher@LR and students give comparable predictions.

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

**Student (distilled):** available on HuggingFace at [dimos-stavaris/clip-ebc-student-teacher](https://huggingface.co/dimos-stavaris/clip-ebc-student-teacher).

Best model (scale jitter λ=0.5, MAE 80.54):
```bash
hf download dimos-stavaris/clip-ebc-student-teacher best_student_e50_lr3e-05_clw0.5_ds2-4.0.pth --local-dir checkpoints/student/
```

Original fixed-scale model (MAE 102.65):
```bash
hf download dimos-stavaris/clip-ebc-student-teacher best_student_e50_lr1e-5.pth --local-dir checkpoints/student/
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

To train with partial backbone unfreezing (last 3 CLIP transformer blocks, differential LR):

```bash
bsub < train_student_unfreeze.sh
```

To train with scale jitter (random downscale 2×–4× per crop):

```bash
bsub < train_student_scalejitter.sh       # λ=0.1
bsub < train_student_scalejitter_clw05.sh # λ=0.5 (best overall)
```

Evaluate student vs teacher on full val images:

```bash
bsub < eval_student.sh
```

Analyse per-image errors and calibration bias for any eval folder:

```bash
python analyze_errors.py --eval_dir student_eval_outputs/<folder>
```

Results are saved to `student_eval_outputs/` (tagged by epochs, lr, count loss weight, and downscale).

### SR comparison

```bash
bsub < eval_sr.sh
```

Runs teacher + bicubic SR + Real-ESRGAN SR + student on NWPU val. Results saved to `sr_eval_outputs/`. Requires ESRGAN weights (see SR section above).

### Density map visualizations

Generate side-by-side density map comparisons (teacher@1×, teacher@2×, student@2×, teacher@4×, student@4×) for selected val images:

```bash
bsub < visualize_predictions.sh
```

Output PNGs saved to `assets/visualizations/`. Default images: `125.jpg` (GT=12,924, best student win), `047.jpg` (near-perfect 4× prediction), `299.jpg` (student loses), `244.jpg` (strong 4× win). Add or change images with `--images img1.jpg img2.jpg`.

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