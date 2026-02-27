#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def find_labeled_case50(raw_root: Path) -> tuple[Path, Path]:
    for p in raw_root.rglob("FLARE_LabeledCase50"):
        img = p / "images"
        lab = p / "labels"
        if img.exists() and lab.exists():
            return img, lab
    raise FileNotFoundError(
        f"Could not find FLARE_LabeledCase50/images|labels under {raw_root}. Complete dataset download first."
    )


def make_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src, target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=str, default="/hy-tmp/flare22")
    parser.add_argument("--out_root", type=str, default="/hy-tmp/flare22")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    images, labels = find_labeled_case50(raw_root)
    images_tr = out_root / "imagesTr"
    labels_tr = out_root / "labelsTr"

    make_symlink(images.resolve(), images_tr)
    make_symlink(labels.resolve(), labels_tr)

    print(f"imagesTr -> {images_tr}")
    print(f"labelsTr -> {labels_tr}")
    print("FLARE22 standard layout is ready.")


if __name__ == "__main__":
    main()

