#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/hy-tmp/flare22}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${PROJECT_ROOT}"

python scripts/prepare_flare22_layout.py --raw_root "${DATA_ROOT}" --out_root "${DATA_ROOT}" || true

if ! python scripts/check_flare22_ready.py --data_root "${DATA_ROOT}" --min_pairs 50; then
  echo "[INFO] Dataset is not ready yet. Complete download and retry."
  echo "[INFO] Try: bash scripts/download_flare22.sh ${DATA_ROOT}"
  exit 1
fi

python train_demo.py \
  --dataset_type flare22 \
  --data_root "${DATA_ROOT}" \
  --epochs 500 \
  --batch_size 2 \
  --image_size 256 \
  --num_workers 2 \
  --val_every 20 \
  --save_visual_every 20 \
  --max_vis_samples 4 \
  --ignore_bg \
  --nsd_tolerance_px 1.0 \
  --auto_num_classes \
  --teacher_update ema \
  --cons_mode mse \
  --vss_backend mamba \
  --sup_only_steps 1000 \
  --loss_warmup_steps 4000 \
  --lr_warmup_steps 1000 \
  --lr_min_ratio 0.05 \
  --foreground_sample_prob 0.8 \
  --min_foreground_pixels 30 \
  --save_dir ./outputs

