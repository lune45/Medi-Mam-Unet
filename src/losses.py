import torch
import torch.nn.functional as F


def segmentation_supervised_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Supervised segmentation loss: cross-entropy.
    - logits: [B, C, H, W]
    - labels: [B, H, W] (long)
    """
    return F.cross_entropy(logits, labels)


def dice_loss(logits: torch.Tensor, labels: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Multi-class soft Dice loss.
    - logits: [B, C, H, W]
    - labels: [B, H, W]
    """
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    labels_one_hot = F.one_hot(labels.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    intersection = torch.sum(probs * labels_one_hot, dim=(0, 2, 3))
    cardinality = torch.sum(probs + labels_one_hot, dim=(0, 2, 3))
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


def consistency_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mode: str = "mse",
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Consistency loss between student and teacher predictions:
    - mse: MSE on softmax probabilities
    - kl:  KL(student || teacher)
    """
    s_prob = F.softmax(student_logits / temperature, dim=1)
    t_prob = F.softmax(teacher_logits / temperature, dim=1).detach()

    if mode == "kl":
        s_log_prob = torch.log(s_prob.clamp(min=1e-8))
        return F.kl_div(s_log_prob, t_prob, reduction="batchmean")

    return F.mse_loss(s_prob, t_prob)


def build_total_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    gt_labels: torch.Tensor,
    lambda_cons: float = 0.5,
    lambda_pseudo: float = 0.3,
    lambda_bottom_sup: float = 0.2,
    cons_mode: str = "mse",
    pseudo_conf_thresh: float = 0.6,
) -> dict:
    """
    Composite loss used by the teacher-student setup:
    1) supervised top-branch loss (student vs GT)
    2) consistency loss (student vs teacher)
    3) pseudo-label loss (student vs teacher pseudo labels)
    4) optional bottom-branch supervision (teacher vs GT)
    """
    loss_sup_top = segmentation_supervised_loss(student_logits, gt_labels) + dice_loss(student_logits, gt_labels)
    loss_cons = consistency_loss(student_logits, teacher_logits, mode=cons_mode)

    teacher_prob = F.softmax(teacher_logits.detach(), dim=1)
    pseudo_conf, pseudo_labels = torch.max(teacher_prob, dim=1)
    pseudo_ce = F.cross_entropy(student_logits, pseudo_labels, reduction="none")
    pseudo_mask = (pseudo_conf >= float(pseudo_conf_thresh)).float()
    keep_count = pseudo_mask.sum()
    if keep_count.item() > 0:
        loss_pseudo = (pseudo_ce * pseudo_mask).sum() / keep_count
    else:
        # Skip noisy pseudo gradients when no confident pixels are available.
        loss_pseudo = pseudo_ce.mean() * 0.0

    loss_sup_bottom = segmentation_supervised_loss(teacher_logits, gt_labels)

    loss_total = (
        loss_sup_top
        + lambda_cons * loss_cons
        + lambda_pseudo * loss_pseudo
        + lambda_bottom_sup * loss_sup_bottom
    )
    return {
        "loss_total": loss_total,
        "loss_sup_top": loss_sup_top,
        "loss_cons": loss_cons,
        "loss_pseudo": loss_pseudo,
        "loss_sup_bottom": loss_sup_bottom,
        "pseudo_keep_ratio": pseudo_mask.mean().detach(),
    }
