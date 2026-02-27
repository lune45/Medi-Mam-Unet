#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import Dataset, DataLoader

from src.data import build_flare22_dataloaders, infer_flare22_num_classes
from src.losses import build_total_loss
from src.metrics import segmentation_metrics, segmentation_metrics_per_class
from src.models import TeacherStudentSegModel
from src.visualize import save_segmentation_visuals


class DummySegDataset(Dataset):
    """
    Random dataset for quick segmentation smoke tests:
    - image: [1, H, W]
    - label: [H, W], class index
    """

    def __init__(self, n: int = 64, image_size: int = 256, num_classes: int = 2):
        self.n = n
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        image = torch.randn(1, self.image_size, self.image_size)
        label = torch.randint(
            low=0,
            high=self.num_classes,
            size=(self.image_size, self.image_size),
            dtype=torch.long,
        )
        return image, label


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, default="dummy", choices=["dummy", "flare22"])
    parser.add_argument("--data_root", type=str, default="/hy-tmp/flare22")
    parser.add_argument("--steps", type=int, default=None, help="Total training steps (higher priority than epochs)")
    parser.add_argument("--epochs", type=int, default=None, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--auto_num_classes", dest="auto_num_classes", action="store_true")
    parser.add_argument("--no_auto_num_classes", dest="auto_num_classes", action="store_false")
    parser.set_defaults(auto_num_classes=True)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--train_split", type=float, default=0.7)
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--val_every", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Checkpoint path for resume training (e.g., outputs/xxx/last.pt or best.pt)",
    )
    parser.add_argument("--ignore_bg", action="store_true", help="Ignore background class (0) in metric computation")
    parser.add_argument("--nsd_tolerance_px", type=float, default=1.0)
    parser.add_argument("--save_visual_every", type=int, default=20)
    parser.add_argument("--max_vis_samples", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_cons", type=float, default=0.5)
    parser.add_argument("--lambda_pseudo", type=float, default=0.3)
    parser.add_argument("--lambda_bottom_sup", type=float, default=0.2)
    parser.add_argument("--pseudo_conf_thresh", type=float, default=0.6)
    parser.add_argument("--loss_warmup_steps", type=int, default=1000)
    parser.add_argument("--sup_only_steps", type=int, default=500)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_min_ratio", type=float, default=0.05)
    parser.add_argument("--foreground_sample_prob", type=float, default=0.8)
    parser.add_argument("--min_foreground_pixels", type=int, default=30)
    parser.add_argument("--no_train_augment", action="store_true")
    parser.add_argument("--cons_mode", type=str, default="mse", choices=["mse", "kl"])
    parser.add_argument("--teacher_update", type=str, default="ema", choices=["ema", "joint", "frozen"])
    parser.add_argument("--ema_momentum", type=float, default=0.99)
    parser.add_argument("--student_base_ch", type=int, default=32)
    parser.add_argument("--teacher_base_ch", type=int, default=32)
    parser.add_argument("--teacher_arch", type=str, default="lite", choices=["lite", "student"])
    parser.add_argument("--vss_backend", type=str, default="vsslike", choices=["vsslike", "mamba"])
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


FLARE22_CLASS_NAMES = {
    0: "Background",
    1: "Liver",
    2: "Right Kidney",
    3: "Spleen",
    4: "Pancreas",
    5: "Aorta",
    6: "Inferior Vena Cava",
    7: "Right Adrenal Gland",
    8: "Left Adrenal Gland",
    9: "Gall Bladder",
    10: "Esophagus",
    11: "Stomach",
    12: "Left Kidney",
    13: "Duodenum",
}


def _resolve_num_classes(args) -> int:
    if args.dataset_type == "dummy":
        return int(args.num_classes if args.num_classes is not None else 2)
    if args.auto_num_classes or args.num_classes is None:
        inferred = infer_flare22_num_classes(args.data_root)
        print(f"Auto inferred num_classes from labels: {inferred}")
        return int(inferred)
    return int(args.num_classes)


def _resolve_total_steps(args, steps_per_epoch: int) -> tuple[int, int]:
    if args.steps is not None:
        total_steps = int(args.steps)
        total_epochs = max(1, math.ceil(total_steps / max(1, steps_per_epoch)))
        return total_steps, total_epochs

    if args.epochs is not None:
        total_epochs = int(args.epochs)
    else:
        total_epochs = 5 if args.dataset_type == "dummy" else 500
    total_steps = total_epochs * max(1, steps_per_epoch)
    return total_steps, total_epochs


def _lr_scale(step_idx: int, total_steps: int, warmup_steps: int, min_ratio: float) -> float:
    warmup_steps = max(0, int(warmup_steps))
    min_ratio = float(min(1.0, max(0.0, min_ratio)))
    if warmup_steps > 0 and step_idx < warmup_steps:
        return float(step_idx + 1) / float(warmup_steps)

    remain = max(1, total_steps - warmup_steps)
    progress = float(step_idx - warmup_steps) / float(remain)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def _checkpoint_payload(model, optimizer, args, step: int, epoch: int, best_val_loss: float) -> dict:
    return {
        "step": int(step),
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_correct = 0.0
    total_pixels = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_nsd = 0.0
    class_ids = [c for c in range(args.num_classes) if c != (0 if args.ignore_bg else None)]
    per_class_dice_sum = {c: 0.0 for c in class_ids}
    per_class_nsd_sum = {c: 0.0 for c in class_ids}
    per_class_dice_cnt = {c: 0 for c in class_ids}
    per_class_nsd_cnt = {c: 0 for c in class_ids}
    cached_vis_batch = None

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        losses = build_total_loss(
            student_logits=outputs["student_logits"],
            teacher_logits=outputs["teacher_logits"],
            gt_labels=labels,
            lambda_cons=args.lambda_cons,
            lambda_pseudo=args.lambda_pseudo,
            lambda_bottom_sup=args.lambda_bottom_sup,
            cons_mode=args.cons_mode,
            pseudo_conf_thresh=args.pseudo_conf_thresh,
        )
        total_loss += losses["loss_total"].item()
        total_batches += 1

        pred = torch.argmax(outputs["student_logits"], dim=1)
        total_correct += (pred == labels).float().sum().item()
        total_pixels += labels.numel()
        m = segmentation_metrics(
            pred=pred,
            target=labels,
            num_classes=args.num_classes,
            ignore_index=0 if args.ignore_bg else None,
            nsd_tolerance_px=args.nsd_tolerance_px,
        )
        total_dice += m["dice"]
        total_iou += m["iou"]
        total_nsd += m["nsd"]
        m_pc = segmentation_metrics_per_class(
            pred=pred,
            target=labels,
            num_classes=args.num_classes,
            ignore_index=0 if args.ignore_bg else None,
            nsd_tolerance_px=args.nsd_tolerance_px,
        )
        for c in class_ids:
            d_c = m_pc[c]["dice"]
            n_c = m_pc[c]["nsd"]
            if not math.isnan(d_c):
                per_class_dice_sum[c] += d_c
                per_class_dice_cnt[c] += 1
            if not math.isnan(n_c):
                per_class_nsd_sum[c] += n_c
                per_class_nsd_cnt[c] += 1

        if cached_vis_batch is None:
            cached_vis_batch = (
                images.detach().cpu(),
                labels.detach().cpu(),
                pred.detach().cpu(),
            )

    model.train()
    avg_loss = total_loss / max(1, total_batches)
    pixel_acc = total_correct / max(1.0, total_pixels)
    avg_dice = total_dice / max(1, total_batches)
    avg_iou = total_iou / max(1, total_batches)
    avg_nsd = total_nsd / max(1, total_batches)
    per_class = {
        c: {
            "dice": (
                per_class_dice_sum[c] / per_class_dice_cnt[c]
                if per_class_dice_cnt[c] > 0
                else float("nan")
            ),
            "nsd": (
                per_class_nsd_sum[c] / per_class_nsd_cnt[c]
                if per_class_nsd_cnt[c] > 0
                else float("nan")
            ),
        }
        for c in class_ids
    }
    return {
        "val_loss": avg_loss,
        "val_pixel_acc": pixel_acc,
        "val_dice": avg_dice,
        "val_iou": avg_iou,
        "val_nsd": avg_nsd,
        "per_class": per_class,
        "vis_batch": cached_vis_batch,
    }


def _print_per_class_metrics(step: int, split: str, per_class: Dict[int, Dict[str, float]]) -> None:
    print(f"[{split} {step:05d}] per-organ metrics (DSC/NSD):")
    for c in sorted(per_class.keys()):
        name = FLARE22_CLASS_NAMES.get(c, f"class_{c}")
        d_txt = "N/A" if math.isnan(per_class[c]["dice"]) else f"{per_class[c]['dice']:.4f}"
        n_txt = "N/A" if math.isnan(per_class[c]["nsd"]) else f"{per_class[c]['nsd']:.4f}"
        print(
            f"  - c{c:02d} {name:<22} "
            f"DSC={d_txt} NSD={n_txt}"
        )


def main():
    args = parse_args()
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    args.num_classes = _resolve_num_classes(args)
    teacher_arch = args.teacher_arch

    if args.teacher_update == "ema":
        # Enforce isomorphic teacher/student under EMA for stable updates.
        teacher_arch = "student"
        if args.teacher_base_ch != args.student_base_ch:
            raise ValueError(
                "EMA mode requires --teacher_base_ch == --student_base_ch "
                "to ensure parameter-wise matching."
            )

    model = TeacherStudentSegModel(
        in_ch=1,
        num_classes=args.num_classes,
        student_base_ch=args.student_base_ch,
        teacher_base_ch=args.teacher_base_ch,
        teacher_arch=teacher_arch,
        vss_backend=args.vss_backend,
    ).to(device)

    if args.teacher_update == "ema":
        comp = model.get_ema_compatibility()
        if comp["matched"] == 0:
            raise ValueError(
                "EMA mode found no matching teacher/student parameters. "
                "Set --student_base_ch and --teacher_base_ch to the same value, "
                "or use --teacher_update joint/frozen."
            )
        init_stat = model.initialize_teacher_from_student()
        print(
            f"EMA compatible params: {comp['matched']}/{comp['total']}, "
            f"teacher init copied tensors: {init_stat['copied']}/{init_stat['total']}"
        )

    if args.teacher_update in ["ema", "frozen"]:
        for p in model.teacher.parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.dataset_type == "flare22":
        train_loader, val_loader, test_loader = build_flare22_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers,
            train_split=args.train_split,
            test_split=args.test_split,
            val_split=args.val_split,
            foreground_sample_prob=args.foreground_sample_prob,
            min_foreground_pixels=args.min_foreground_pixels,
            train_augment=not args.no_train_augment,
        )
        print(f"Using FLARE22 dataset from: {args.data_root}")
        print(
            "Split sizes (train/test/val): "
            f"{len(train_loader.dataset)}/{len(test_loader.dataset)}/{len(val_loader.dataset)}"
        )
    else:
        dummy_epochs = args.epochs if args.epochs is not None else 5
        dummy_steps = args.steps if args.steps is not None else dummy_epochs * 32
        dataset = DummySegDataset(
            n=max(dummy_steps * args.batch_size, 64),
            image_size=args.image_size,
            num_classes=args.num_classes,
        )
        train_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        test_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        print("Using DummySegDataset")

    steps_per_epoch = max(1, len(train_loader))
    total_steps, total_epochs = _resolve_total_steps(args, steps_per_epoch)
    print(
        f"Training schedule: total_steps={total_steps}, "
        f"total_epochs={total_epochs}, steps_per_epoch={steps_per_epoch}"
    )

    model.train()
    best_val_loss = float("inf")
    step = 0
    start_epoch = 0

    if args.resume_ckpt:
        resume_path = Path(args.resume_ckpt)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume_ckpt does not exist: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        step = int(ckpt.get("step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        start_epoch = min(total_epochs, step // steps_per_epoch)
        print(
            f"Resumed from: {resume_path} | "
            f"step={step}/{total_steps}, start_epoch={start_epoch + 1}/{total_epochs}, "
            f"best_val_loss={best_val_loss:.4f}"
        )
        if step >= total_steps:
            print(
                "Resume step has reached/exceeded total_steps. Nothing to continue under current config."
            )
            print(f"Training demo finished. total_steps={step}, total_epochs={total_epochs}")
            return

    for epoch in range(start_epoch, total_epochs):
        for images, labels in train_loader:
            if step >= total_steps:
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            lr_mult = _lr_scale(
                step_idx=step,
                total_steps=total_steps,
                warmup_steps=args.lr_warmup_steps,
                min_ratio=args.lr_min_ratio,
            )
            current_lr = args.lr * lr_mult
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            if step < args.sup_only_steps:
                warmup_ratio = 0.0
                lambda_cons_now = 0.0
                lambda_pseudo_now = 0.0
            else:
                aux_step = step - args.sup_only_steps + 1
                warmup_ratio = min(1.0, float(aux_step) / float(max(1, args.loss_warmup_steps)))
                lambda_cons_now = args.lambda_cons * warmup_ratio
                lambda_pseudo_now = args.lambda_pseudo * warmup_ratio

            outputs = model(images)
            student_logits = outputs["student_logits"]
            teacher_logits = outputs["teacher_logits"]
            losses = build_total_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                gt_labels=labels,
                lambda_cons=lambda_cons_now,
                lambda_pseudo=lambda_pseudo_now,
                lambda_bottom_sup=args.lambda_bottom_sup,
                cons_mode=args.cons_mode,
                pseudo_conf_thresh=args.pseudo_conf_thresh,
            )
            loss_total = losses["loss_total"]

            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
            optimizer.step()

            if args.teacher_update == "ema":
                model.ema_update_teacher(momentum=args.ema_momentum)

            step += 1
            print(
                f"[epoch {epoch + 1:03d}/{total_epochs:03d} step {step:05d}/{total_steps:05d}] "
                f"lr={current_lr:.6f} "
                f"loss_total={losses['loss_total'].item():.4f} "
                f"sup_top={losses['loss_sup_top'].item():.4f} "
                f"cons={losses['loss_cons'].item():.4f} "
                f"pseudo={losses['loss_pseudo'].item():.4f} "
                f"sup_bottom={losses['loss_sup_bottom'].item():.4f} "
                f"pseudo_keep={losses['pseudo_keep_ratio'].item():.3f} "
                f"warmup={warmup_ratio:.3f}"
            )

            if step % args.val_every == 0 or step == total_steps:
                metrics = evaluate(model, val_loader, device, args)
                print(
                    f"[val {step:05d}] "
                    f"val_loss={metrics['val_loss']:.4f} "
                    f"val_pixel_acc={metrics['val_pixel_acc']:.4f} "
                    f"val_dice={metrics['val_dice']:.4f} "
                    f"val_iou={metrics['val_iou']:.4f} "
                    f"val_nsd={metrics['val_nsd']:.4f}"
                )
                _print_per_class_metrics(step=step, split="val", per_class=metrics["per_class"])
                if step % args.save_visual_every == 0:
                    vis_batch = metrics["vis_batch"]
                    if vis_batch is not None:
                        vis_dir = Path(args.save_dir) / "visuals"
                        save_segmentation_visuals(
                            images=vis_batch[0],
                            labels=vis_batch[1],
                            preds=vis_batch[2],
                            save_dir=vis_dir,
                            step=step,
                            max_samples=args.max_vis_samples,
                        )
                        print(f"Saved visualization: {vis_dir}")
                last_ckpt_path = Path(args.save_dir) / "last.pt"
                torch.save(
                    _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        args=args,
                        step=step,
                        epoch=epoch + 1,
                        best_val_loss=best_val_loss,
                    ),
                    last_ckpt_path,
                )
                if metrics["val_loss"] < best_val_loss:
                    best_val_loss = metrics["val_loss"]
                    ckpt_path = Path(args.save_dir) / "best.pt"
                    torch.save(
                        _checkpoint_payload(
                            model=model,
                            optimizer=optimizer,
                            args=args,
                            step=step,
                            epoch=epoch + 1,
                            best_val_loss=best_val_loss,
                        ),
                        ckpt_path,
                    )
                    print(f"Saved best checkpoint: {ckpt_path}")
        if step >= total_steps:
            break

    test_metrics = evaluate(model, test_loader, device, args)
    print(
        "[test final] "
        f"test_loss={test_metrics['val_loss']:.4f} "
        f"test_pixel_acc={test_metrics['val_pixel_acc']:.4f} "
        f"test_dice={test_metrics['val_dice']:.4f} "
        f"test_iou={test_metrics['val_iou']:.4f} "
        f"test_nsd={test_metrics['val_nsd']:.4f}"
    )
    _print_per_class_metrics(step=step, split="test", per_class=test_metrics["per_class"])

    print(f"Training demo finished. total_steps={step}, total_epochs={total_epochs}")


if __name__ == "__main__":
    main()
