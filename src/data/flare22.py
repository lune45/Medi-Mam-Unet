from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SUPPORTED_EXTENSIONS = (".nii.gz", ".nii", ".png")


def _strip_known_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    if name.endswith(".png"):
        return name[: -len(".png")]
    return name


def _normalize_pair_key(name: str) -> str:
    """
    Normalize image/label pairing key:
    - remove known suffixes
    - support nnU-Net style `_0000` image suffix
    """
    key = _strip_known_suffix(name)
    if key.endswith("_0000"):
        key = key[:-5]
    return key


def _load_nii_as_numpy(path: Path) -> np.ndarray:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "Reading .nii/.nii.gz requires nibabel. Install it with: python -m pip install nibabel"
        ) from exc
    return np.asarray(nib.load(str(path)).get_fdata())


def _load_image(path: Path) -> np.ndarray:
    suffix = path.name.lower()
    if suffix.endswith(".nii.gz") or suffix.endswith(".nii"):
        return _load_nii_as_numpy(path)
    if suffix.endswith(".png"):
        return np.asarray(iio.imread(path))
    raise ValueError(f"Unsupported image suffix: {path}")


def _find_flare_train_dirs(root: Path) -> Tuple[Path, Path]:
    tr_images = root / "imagesTr"
    tr_labels = root / "labelsTr"
    if tr_images.exists() and tr_labels.exists():
        return tr_images, tr_labels

    # Support official FLARE22 release layout:
    # /.../FLARE/Training/FLARE_LabeledCase50/images|labels
    candidates = list(root.rglob("FLARE_LabeledCase50"))
    for c in candidates:
        c_img = c / "images"
        c_lab = c / "labels"
        if c_img.exists() and c_lab.exists():
            return c_img, c_lab

    raise FileNotFoundError(
        "Training directories not found. Use one of the following layouts:\n"
        f"1) {root}/imagesTr + {root}/labelsTr\n"
        f"2) {root}/.../FLARE/Training/FLARE_LabeledCase50/images|labels"
    )


def _ensure_2d_slice(arr: np.ndarray, slice_idx: Optional[int]) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape={arr.shape}")
    idx = arr.shape[-1] // 2 if slice_idx is None else int(slice_idx)
    idx = max(0, min(idx, arr.shape[-1] - 1))
    return arr[..., idx]


def _normalize_ct(image: np.ndarray, hu_min: float = -200.0, hu_max: float = 300.0) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    image = np.clip(image, hu_min, hu_max)
    image = (image - hu_min) / max(1e-6, hu_max - hu_min)
    return image


def _resize_image_label(
    image_2d: np.ndarray, label_2d: np.ndarray, image_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    image_2d = np.ascontiguousarray(image_2d)
    label_2d = np.ascontiguousarray(label_2d)
    image_t = torch.from_numpy(image_2d).float().unsqueeze(0).unsqueeze(0)
    label_t = torch.from_numpy(label_2d).long().unsqueeze(0).unsqueeze(0)

    image_t = F.interpolate(
        image_t, size=(image_size, image_size), mode="bilinear", align_corners=False
    )
    label_t = F.interpolate(label_t.float(), size=(image_size, image_size), mode="nearest").long()

    return image_t.squeeze(0), label_t.squeeze(0).squeeze(0)


def _augment_image_label(image_2d: np.ndarray, label_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Lightweight 2D augmentations commonly used in medical segmentation.
    if np.random.rand() < 0.5:
        image_2d = np.flip(image_2d, axis=0)
        label_2d = np.flip(label_2d, axis=0)
    if np.random.rand() < 0.5:
        image_2d = np.flip(image_2d, axis=1)
        label_2d = np.flip(label_2d, axis=1)
    if np.random.rand() < 0.5:
        k = int(np.random.randint(0, 4))
        image_2d = np.rot90(image_2d, k=k, axes=(0, 1))
        label_2d = np.rot90(label_2d, k=k, axes=(0, 1))

    if np.random.rand() < 0.7:
        scale = float(np.random.uniform(0.9, 1.1))
        shift = float(np.random.uniform(-0.08, 0.08))
        image_2d = image_2d * scale + shift
    if np.random.rand() < 0.3:
        noise_std = float(np.random.uniform(0.0, 0.03))
        image_2d = image_2d + np.random.normal(0.0, noise_std, size=image_2d.shape).astype(np.float32)

    image_2d = np.clip(image_2d, 0.0, 1.0)
    return image_2d, label_2d


def _collect_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    image_files: Dict[str, Path] = {}
    for p in images_dir.iterdir():
        if not p.is_file():
            continue
        if any(p.name.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
            image_files[_normalize_pair_key(p.name)] = p

    pairs: List[Tuple[Path, Path]] = []
    for p in labels_dir.iterdir():
        if not p.is_file():
            continue
        if not any(p.name.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
            continue
        key = _normalize_pair_key(p.name)
        if key in image_files:
            pairs.append((image_files[key], p))
    pairs.sort(key=lambda x: x[0].name)
    return pairs


class Flare22SliceDataset(Dataset):
    """
    FLARE22 2D slice dataset sampled from 3D CT volumes.
    - image: [1, H, W], float32
    - label: [H, W], int64
    """

    def __init__(
        self,
        pairs: Sequence[Tuple[Path, Path]],
        image_size: int = 256,
        use_random_slice: bool = True,
        foreground_sample_prob: float = 0.7,
        min_foreground_pixels: int = 20,
        enable_augment: bool = False,
    ) -> None:
        if len(pairs) == 0:
            raise ValueError("No valid image/label pairs found. Check directory layout and filenames.")
        self.pairs = list(pairs)
        self.image_size = int(image_size)
        self.use_random_slice = bool(use_random_slice)
        self.foreground_sample_prob = float(np.clip(foreground_sample_prob, 0.0, 1.0))
        self.min_foreground_pixels = int(min_foreground_pixels)
        self.enable_augment = bool(enable_augment)
        self._slice_stats = self._build_slice_stats()

    def _build_slice_stats(self) -> List[Tuple[int, np.ndarray]]:
        """
        Pre-compute slice candidates for foreground-aware sampling.
        """
        stats: List[Tuple[int, np.ndarray]] = []
        for _, label_path in self.pairs:
            label = _load_image(label_path)
            if label.ndim != 3:
                stats.append((1, np.array([0], dtype=np.int64)))
                continue
            fg_pixels = (label > 0).reshape(-1, label.shape[-1]).sum(axis=0)
            fg_slices = np.where(fg_pixels >= self.min_foreground_pixels)[0].astype(np.int64)
            stats.append((int(label.shape[-1]), fg_slices))
        return stats

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, label_path = self.pairs[idx]
        image = _load_image(image_path)
        label = _load_image(label_path)
        label = np.rint(label).astype(np.int64, copy=False)

        if image.ndim == 3:
            if self.use_random_slice:
                total_slices, fg_slices = self._slice_stats[idx]
                if (
                    fg_slices.size > 0
                    and np.random.rand() < self.foreground_sample_prob
                ):
                    chosen = int(np.random.choice(fg_slices))
                else:
                    chosen = int(np.random.randint(0, total_slices))
            else:
                chosen = image.shape[-1] // 2
        else:
            chosen = None

        image_2d = _ensure_2d_slice(image, chosen)
        label_2d = _ensure_2d_slice(label, chosen)
        image_2d = _normalize_ct(image_2d)
        if self.enable_augment and self.use_random_slice:
            image_2d, label_2d = _augment_image_label(image_2d, label_2d)
        image_t, label_t = _resize_image_label(image_2d, label_2d, self.image_size)
        return image_t, label_t


def infer_flare22_num_classes(data_root: str) -> int:
    """
    Infer FLARE22 number of classes from labels (max_label + 1).
    """
    root = Path(data_root)
    _, tr_labels = _find_flare_train_dirs(root)
    label_files = sorted(
        [
            p
            for p in tr_labels.iterdir()
            if p.is_file() and any(p.name.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS)
        ]
    )
    if len(label_files) == 0:
        raise FileNotFoundError(f"No label files found in {tr_labels}; cannot infer class count.")

    max_label = 0
    for p in label_files:
        arr = _load_image(p)
        arr = np.rint(arr).astype(np.int64, copy=False)
        max_label = max(max_label, int(arr.max()))
    return max_label + 1


def build_flare22_dataloaders(
    data_root: str,
    batch_size: int = 2,
    image_size: int = 256,
    num_workers: int = 4,
    train_split: float = 0.7,
    test_split: float = 0.2,
    val_split: float = 0.1,
    seed: int = 42,
    foreground_sample_prob: float = 0.7,
    min_foreground_pixels: int = 20,
    train_augment: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build FLARE22 train/test/val DataLoaders (default 7:2:1).

    Recommended directory layout:
    - data_root/imagesTr
    - data_root/labelsTr
    - current implementation splits from training pairs.
    """
    root = Path(data_root)
    tr_images, tr_labels = _find_flare_train_dirs(root)
    train_pairs = _collect_pairs(tr_images, tr_labels)
    if len(train_pairs) < 10:
        raise ValueError("Not enough samples to build stable train/test/val splits.")
    if train_split <= 0 or test_split <= 0 or val_split <= 0:
        raise ValueError("train_split/test_split/val_split must all be > 0.")
    split_sum = train_split + test_split + val_split
    if abs(split_sum - 1.0) > 1e-6:
        raise ValueError("train_split + test_split + val_split must equal 1.0.")

    n_total = len(train_pairs)
    n_train = max(1, int(round(n_total * train_split)))
    n_test = max(1, int(round(n_total * test_split)))
    n_val = n_total - n_train - n_test
    if n_val <= 0:
        n_val = 1
        if n_train >= n_test:
            n_train = max(1, n_train - 1)
        else:
            n_test = max(1, n_test - 1)
    if n_train <= 0 or n_test <= 0 or n_val <= 0:
        raise ValueError("One split is empty after partitioning; adjust split ratios.")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator).tolist()
    train_ids = set(perm[:n_train])
    test_ids = set(perm[n_train : n_train + n_test])
    val_ids = set(perm[n_train + n_test :])

    tr_pairs = [p for i, p in enumerate(train_pairs) if i in train_ids]
    test_pairs = [p for i, p in enumerate(train_pairs) if i in test_ids]
    val_pairs = [p for i, p in enumerate(train_pairs) if i in val_ids]

    train_ds = Flare22SliceDataset(
        tr_pairs,
        image_size=image_size,
        use_random_slice=True,
        foreground_sample_prob=foreground_sample_prob,
        min_foreground_pixels=min_foreground_pixels,
        enable_augment=train_augment,
    )
    val_ds = Flare22SliceDataset(
        val_pairs,
        image_size=image_size,
        use_random_slice=False,
        foreground_sample_prob=0.0,
        min_foreground_pixels=min_foreground_pixels,
        enable_augment=False,
    )
    test_ds = Flare22SliceDataset(
        test_pairs,
        image_size=image_size,
        use_random_slice=False,
        foreground_sample_prob=0.0,
        min_foreground_pixels=min_foreground_pixels,
        enable_augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader
