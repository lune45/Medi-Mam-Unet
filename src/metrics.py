from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def _safe_div(numerator: float, denominator: float, eps: float = 1e-8) -> float:
    return float(numerator) / float(denominator + eps)


def dice_per_class(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int | None = None
) -> torch.Tensor:
    scores = []
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        p = pred == c
        t = target == c
        # Exclude classes absent in GT for this sample/batch.
        if torch.sum(t).item() == 0:
            continue
        inter = torch.sum((p & t).float())
        denom = torch.sum(p.float()) + torch.sum(t.float())
        scores.append((2.0 * inter + 1e-8) / (denom + 1e-8))
    if not scores:
        return torch.tensor(0.0, device=pred.device)
    return torch.stack(scores)


def iou_per_class(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int | None = None
) -> torch.Tensor:
    scores = []
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        p = pred == c
        t = target == c
        # Exclude classes absent in GT for this sample/batch.
        if torch.sum(t).item() == 0:
            continue
        inter = torch.sum((p & t).float())
        union = torch.sum((p | t).float())
        scores.append((inter + 1e-8) / (union + 1e-8))
    if not scores:
        return torch.tensor(0.0, device=pred.device)
    return torch.stack(scores)


def _extract_boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if mask.ndim != 2:
        raise ValueError(f"NSD currently supports 2D masks only, got ndim={mask.ndim}")
    up = np.zeros_like(mask)
    up[1:, :] = mask[:-1, :]
    down = np.zeros_like(mask)
    down[:-1, :] = mask[1:, :]
    left = np.zeros_like(mask)
    left[:, 1:] = mask[:, :-1]
    right = np.zeros_like(mask)
    right[:, :-1] = mask[:, 1:]
    interior = mask & up & down & left & right
    boundary = mask & (~interior)
    return boundary


def _distance_to_foreground(binary_map: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:
        raise ImportError(
            "NSD computation requires scipy. Install it with: python -m pip install scipy"
        ) from exc
    # EDT returns distance to zeros, so invert to get distance to foreground.
    return distance_transform_edt(~binary_map)


def normalized_surface_dice_2d(
    pred_mask: np.ndarray, gt_mask: np.ndarray, tolerance_px: float = 1.0
) -> float:
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    if pred_mask.sum() == 0 and gt_mask.sum() == 0:
        return 1.0
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return 0.0

    pred_bd = _extract_boundary(pred_mask)
    gt_bd = _extract_boundary(gt_mask)

    if pred_bd.sum() == 0 and gt_bd.sum() == 0:
        return 1.0
    if pred_bd.sum() == 0 or gt_bd.sum() == 0:
        return 0.0

    dist_to_gt = _distance_to_foreground(gt_bd)
    dist_to_pred = _distance_to_foreground(pred_bd)

    pred_hit = np.sum(dist_to_gt[pred_bd] <= tolerance_px)
    gt_hit = np.sum(dist_to_pred[gt_bd] <= tolerance_px)
    denom = np.sum(pred_bd) + np.sum(gt_bd)
    return _safe_div(pred_hit + gt_hit, denom)


def segmentation_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = 0,
    nsd_tolerance_px: float = 1.0,
) -> Dict[str, float]:
    """
    Compute batch-level segmentation metrics:
    - dice: macro Dice
    - iou: macro IoU
    - nsd: macro NSD (2D slice based)
    """
    dice_values = dice_per_class(pred, target, num_classes, ignore_index=ignore_index)
    iou_values = iou_per_class(pred, target, num_classes, ignore_index=ignore_index)

    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    class_ids = [c for c in range(num_classes) if c != ignore_index]
    nsd_scores = []
    for b in range(pred_np.shape[0]):
        for c in class_ids:
            if np.sum(target_np[b] == c) == 0:
                continue
            nsd_scores.append(
                normalized_surface_dice_2d(
                    pred_np[b] == c, target_np[b] == c, tolerance_px=nsd_tolerance_px
                )
            )

    return {
        "dice": float(dice_values.mean().item()) if dice_values.numel() > 0 else 0.0,
        "iou": float(iou_values.mean().item()) if iou_values.numel() > 0 else 0.0,
        "nsd": float(np.mean(nsd_scores)) if len(nsd_scores) > 0 else 0.0,
    }


def segmentation_metrics_per_class(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = 0,
    nsd_tolerance_px: float = 1.0,
) -> Dict[int, Dict[str, float]]:
    """
    Compute batch-level per-class metrics (Dice/NSD).
    Returns:
    {
      class_id: {"dice": float, "nsd": float},
      ...
    }
    """
    class_ids = [c for c in range(num_classes) if c != ignore_index]
    results: Dict[int, Dict[str, float]] = {}

    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    for c in class_ids:
        dice_scores_c = []
        nsd_scores_c = []
        for b in range(pred_np.shape[0]):
            gt_mask = target_np[b] == c
            pred_mask = pred_np[b] == c
            if np.sum(gt_mask) == 0:
                continue
            inter = np.sum(pred_mask & gt_mask)
            denom = np.sum(pred_mask) + np.sum(gt_mask)
            dice_scores_c.append(_safe_div(2.0 * inter, denom))
            nsd_scores_c.append(
                normalized_surface_dice_2d(
                    pred_mask,
                    gt_mask,
                    tolerance_px=nsd_tolerance_px,
                )
            )
        dice_c = float(np.mean(dice_scores_c)) if len(dice_scores_c) > 0 else float("nan")
        nsd_c = float(np.mean(nsd_scores_c)) if len(nsd_scores_c) > 0 else float("nan")
        results[c] = {"dice": dice_c, "nsd": nsd_c}

    return results

