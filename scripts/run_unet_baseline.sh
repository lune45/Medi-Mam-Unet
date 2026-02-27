#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/hy-tmp/flare22}"
SAVE_DIR="${2:-./outputs/unet_baseline_$(date +%Y%m%d_%H%M)}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export LD_PRELOAD="${LD_PRELOAD:-/hy-tmp/conda/lib/libstdc++.so.6}"

cd "${PROJECT_ROOT}"

# Baseline setting: supervised student branch only.
/hy-tmp/conda/bin/python -u train_demo.py \
  --dataset_type flare22 \
  --data_root "${DATA_ROOT}" \
  --epochs 650 \
  --batch_size 2 \
  --image_size 256 \
  --num_workers 2 \
  --train_split 0.7 \
  --test_split 0.2 \
  --val_split 0.1 \
  --val_every 20 \
  --save_visual_every 40 \
  --max_vis_samples 2 \
  --ignore_bg \
  --nsd_tolerance_px 1.0 \
  --auto_num_classes \
  --teacher_update frozen \
  --teacher_arch student \
  --student_base_ch 32 \
  --teacher_base_ch 32 \
  --vss_backend vsslike \
  --cons_mode mse \
  --sup_only_steps 0 \
  --loss_warmup_steps 1 \
  --lambda_cons 0.0 \
  --lambda_pseudo 0.0 \
  --lambda_bottom_sup 0.0 \
  --lr 8e-4 \
  --lr_warmup_steps 500 \
  --lr_min_ratio 0.1 \
  --grad_clip_norm 1.0 \
  --foreground_sample_prob 0.9 \
  --min_foreground_pixels 20 \
  --save_dir "${SAVE_DIR}"

