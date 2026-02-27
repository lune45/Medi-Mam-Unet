#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-/hy-tmp/flare22}"
BAIDU_URL="https://pan.baidu.com/s/1Md1KEZMqDCAa6hFg7hCyjw?pwd=2024"
GDRIVE_URL="https://drive.google.com/drive/folders/1x0l-bxte46QFn5K_ZJzBxp8HsscF8v6t?usp=sharing"

mkdir -p "${TARGET_DIR}"
echo "[INFO] target dir: ${TARGET_DIR}"

if command -v /hy-tmp/BaiduPCS-Go/BaiduPCS-Go >/dev/null 2>&1; then
  echo "[INFO] using BaiduPCS-Go route"
  /hy-tmp/BaiduPCS-Go/BaiduPCS-Go config set -savedir "${TARGET_DIR}" >/dev/null
  /hy-tmp/BaiduPCS-Go/BaiduPCS-Go transfer --download "${BAIDU_URL}"
  echo "[INFO] Baidu download done"
  exit 0
fi

if ! command -v gdown >/dev/null 2>&1; then
  echo "[INFO] installing gdown"
  python -m pip install gdown
fi

echo "[INFO] using Google Drive route"
gdown --folder "${GDRIVE_URL}" -O "${TARGET_DIR}"
echo "[INFO] Google Drive download done"

