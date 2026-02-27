from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def _to_uint8_gray(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = (x - x.min()) / max(1e-8, x.max() - x.min())
    return (x * 255.0).clip(0, 255).astype(np.uint8)


def _label_to_color(label: np.ndarray) -> np.ndarray:
    palette = np.array(
        [
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [255, 128, 0],
            [128, 0, 255],
            [0, 128, 255],
        ],
        dtype=np.uint8,
    )
    idx = np.mod(label.astype(np.int64), len(palette))
    return palette[idx]


def _resize_panel(panel: np.ndarray, scale: int, is_label: bool) -> np.ndarray:
    if scale <= 1:
        return panel
    mode = Image.NEAREST if is_label else Image.BICUBIC
    img = Image.fromarray(panel)
    w, h = img.size
    img = img.resize((w * scale, h * scale), resample=mode)
    return np.asarray(img)


def _build_caption_canvas(
    img_rgb: np.ndarray,
    gt_rgb: np.ndarray,
    pd_rgb: np.ndarray,
    step: int,
    sample_idx: int,
) -> np.ndarray:
    gap = 12
    ph, pw = img_rgb.shape[0], img_rgb.shape[1]
    title_font = _load_font(max(18, int(ph * 0.05)), bold=True)
    meta_font = _load_font(max(14, int(ph * 0.032)), bold=False)
    header_h = max(56, int(ph * 0.12))
    footer_h = max(40, int(ph * 0.08))
    total_w = pw * 3 + gap * 4
    total_h = ph + header_h + footer_h + gap * 3
    canvas = np.full((total_h, total_w, 3), 255, dtype=np.uint8)

    x0 = gap
    x1 = x0 + pw + gap
    x2 = x1 + pw + gap
    y = header_h + gap
    canvas[y : y + ph, x0 : x0 + pw] = img_rgb
    canvas[y : y + ph, x1 : x1 + pw] = gt_rgb
    canvas[y : y + ph, x2 : x2 + pw] = pd_rgb

    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)
    # Center-aligned panel titles for publication-friendly figures.
    _draw_centered_text(draw, x0, x0 + pw, int(header_h * 0.48), "Input", title_font)
    _draw_centered_text(draw, x1, x1 + pw, int(header_h * 0.48), "Ground Truth", title_font)
    _draw_centered_text(draw, x2, x2 + pw, int(header_h * 0.48), "Prediction", title_font)
    footer_text = f"Step {step:06d}    Sample {sample_idx:02d}"
    _draw_centered_text(
        draw,
        gap,
        total_w - gap,
        header_h + gap + ph + gap + int(footer_h * 0.45),
        footer_text,
        meta_font,
        fill=(45, 45, 45),
    )
    # Thin separators keep panels readable after resizing in documents.
    draw.line([(gap, header_h), (total_w - gap, header_h)], fill=(200, 200, 200), width=2)
    draw.line(
        [(gap, header_h + gap + ph + gap), (total_w - gap, header_h + gap + ph + gap)],
        fill=(210, 210, 210),
        width=2,
    )
    return np.asarray(pil_img)


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    font_candidates = (
        ["DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf"]
        if bold
        else ["DejaVuSans.ttf", "Arial.ttf"]
    )
    for name in font_candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    x_left: int,
    x_right: int,
    y_center: int,
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = x_left + max(0, (x_right - x_left - text_w) // 2)
    y = y_center - text_h // 2
    draw.text((x, y), text, font=font, fill=fill)


def save_segmentation_visuals(
    images: torch.Tensor,
    labels: torch.Tensor,
    preds: torch.Tensor,
    save_dir: str | Path,
    step: int,
    max_samples: int = 4,
    upscale: int = 2,
) -> None:
    """
    Save concatenated input/GT/prediction figures for reports and papers.
    """
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images_np = images.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    preds_np = preds.detach().cpu().numpy()
    n = min(images_np.shape[0], max_samples)

    for i in range(n):
        img = images_np[i, 0]
        gt = labels_np[i]
        pd = preds_np[i]

        img_rgb = np.stack([_to_uint8_gray(img)] * 3, axis=-1)
        gt_rgb = _label_to_color(gt)
        pd_rgb = _label_to_color(pd)
        img_rgb = _resize_panel(img_rgb, scale=upscale, is_label=False)
        gt_rgb = _resize_panel(gt_rgb, scale=upscale, is_label=True)
        pd_rgb = _resize_panel(pd_rgb, scale=upscale, is_label=True)
        canvas = _build_caption_canvas(
            img_rgb=img_rgb,
            gt_rgb=gt_rgb,
            pd_rgb=pd_rgb,
            step=step,
            sample_idx=i,
        )

        file_path = out_dir / f"step_{step:06d}_sample_{i:02d}.png"
        iio.imwrite(file_path, canvas)

