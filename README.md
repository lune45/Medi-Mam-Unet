# Visual Mamba Teacher-Student

PyTorch research codebase for abdominal organ segmentation with a Teacher-Student setup and optional Mamba blocks.

## Highlights

- U-Net-like student and teacher branches
- Multiple teacher update modes: `ema`, `joint`, `frozen`
- Losses: supervised (`CE + Dice`), consistency (`MSE`/`KL`), pseudo-label
- Optional real Mamba backend via `mamba-ssm`
- 2D slice training from 3D volumes with foreground-aware sampling
- Per-organ Dice/NSD logging with explicit train/val/test splits

## Repository Structure

```text
visual_mamba_teacher_student/
  README.md
  requirements.txt
  train_demo.py
  scripts/
    download_flare22.sh
    prepare_flare22_layout.py
    check_flare22_ready.py
    run_flare22_train.sh
    run_unet_baseline.sh
  src/
    data/flare22.py
    losses.py
    metrics.py
    visualize.py
    models/
      blocks.py
      teacher_student_seg.py
```

## Installation

```bash
cd /hy-tmp/work/visual_mamba_teacher_student
python -m pip install -r requirements.txt
```

Optional (required only for `--vss_backend mamba`):

```bash
python -m pip install mamba-ssm
```

For this machine, set runtime library path before training:

```bash
export LD_PRELOAD=/hy-tmp/conda/lib/libstdc++.so.6
```

## Data Preparation (FLARE22)

Default data root:

```text
/hy-tmp/flare22/
  imagesTr/
  labelsTr/
```

If your download is nested, normalize layout:

```bash
python scripts/prepare_flare22_layout.py --raw_root /hy-tmp/flare22 --out_root /hy-tmp/flare22
python scripts/check_flare22_ready.py --data_root /hy-tmp/flare22 --min_pairs 50
```

## Quick Start

### Dummy smoke run

```bash
python train_demo.py \
  --dataset_type dummy \
  --epochs 5 \
  --batch_size 2 \
  --image_size 256 \
  --teacher_update ema \
  --teacher_arch student \
  --student_base_ch 32 \
  --teacher_base_ch 32 \
  --vss_backend vsslike
```

### FLARE22 training

```bash
python train_demo.py \
  --dataset_type flare22 \
  --data_root /hy-tmp/flare22 \
  --epochs 500 \
  --batch_size 2 \
  --image_size 256 \
  --num_workers 2 \
  --train_split 0.7 \
  --test_split 0.2 \
  --val_split 0.1 \
  --val_every 20 \
  --save_visual_every 20 \
  --max_vis_samples 4 \
  --ignore_bg \
  --auto_num_classes \
  --teacher_update ema \
  --teacher_arch student \
  --student_base_ch 32 \
  --teacher_base_ch 32 \
  --vss_backend mamba \
  --cons_mode mse \
  --sup_only_steps 1000 \
  --loss_warmup_steps 4000 \
  --lr_warmup_steps 1000 \
  --lr_min_ratio 0.05 \
  --foreground_sample_prob 0.8 \
  --min_foreground_pixels 30 \
  --save_dir ./outputs
```

## Resume Training

`last.pt` is updated on every validation cycle, and `best.pt` is updated on improved validation loss.

```bash
python train_demo.py \
  --dataset_type flare22 \
  --data_root /hy-tmp/flare22 \
  --epochs 500 \
  --save_dir ./outputs/your_run \
  --resume_ckpt ./outputs/your_run/last.pt
```

## Evaluation Protocol

- Split policy: `train/test/val` (default `7:2:1`)
- Per-organ metrics skip samples where the target organ is absent in GT
- Recommended reporting:
  - validation metrics for model selection
  - final test metrics for main results

## Baseline Script

Run supervised baseline with the same split and reporting setup:

```bash
bash scripts/run_unet_baseline.sh /hy-tmp/flare22 ./outputs/unet_baseline_run
```
