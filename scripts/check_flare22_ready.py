#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def count_nii_gz(path: Path) -> int:
    if not path.exists():
        return 0
    return len(list(path.glob("*.nii.gz")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/hy-tmp/flare22")
    parser.add_argument("--min_pairs", type=int, default=50)
    args = parser.parse_args()

    root = Path(args.data_root)
    images = root / "imagesTr"
    labels = root / "labelsTr"

    n_img = count_nii_gz(images)
    n_lab = count_nii_gz(labels)
    n_pair = min(n_img, n_lab)

    print(f"data_root: {root}")
    print(f"imagesTr count: {n_img}")
    print(f"labelsTr count: {n_lab}")
    print(f"available pairs: {n_pair}")

    if n_pair >= args.min_pairs:
        print("READY: FLARE22 training data is available.")
        raise SystemExit(0)

    print(
        "NOT_READY: data is incomplete or not synchronized to imagesTr/labelsTr. "
        "Retry later or wait for the download to finish."
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()

