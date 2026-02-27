from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBNAct, DownBlock, UpBlock, build_vss_block


class StudentUNetLike(nn.Module):
    """
    Student main branch: U-Net style with VSS/Mamba blocks.
    """

    def __init__(
        self,
        in_ch: int = 1,
        num_classes: int = 2,
        base_ch: int = 32,
        vss_backend: str = "vsslike",
    ):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_ch, base_ch),
            build_vss_block(base_ch, vss_backend=vss_backend),
            ConvBNAct(base_ch, base_ch),
        )
        self.down1 = DownBlock(base_ch, base_ch * 2, use_pool=True, vss_backend=vss_backend)
        self.down2 = DownBlock(base_ch * 2, base_ch * 4, use_pool=True, vss_backend=vss_backend)
        self.down3 = DownBlock(base_ch * 4, base_ch * 8, use_pool=True, vss_backend=vss_backend)

        self.bottleneck = nn.Sequential(
            ConvBNAct(base_ch * 8, base_ch * 8),
            build_vss_block(base_ch * 8, vss_backend=vss_backend),
            ConvBNAct(base_ch * 8, base_ch * 8),
        )

        self.up3 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4, vss_backend=vss_backend)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2, vss_backend=vss_backend)
        self.up1 = UpBlock(base_ch * 2, base_ch, base_ch, vss_backend=vss_backend)

        self.head = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.stem(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        b = self.bottleneck(s4)

        x = self.up3(b, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return self.head(x)


class TeacherUNetLike(nn.Module):
    """
    Teacher auxiliary branch: lightweight U-Net style network.
    """

    def __init__(
        self,
        in_ch: int = 1,
        num_classes: int = 2,
        base_ch: int = 24,
        vss_backend: str = "vsslike",
    ):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_ch, base_ch),
            ConvBNAct(base_ch, base_ch),
        )
        self.down1 = DownBlock(base_ch, base_ch * 2, use_pool=True, vss_backend=vss_backend)
        self.down2 = DownBlock(base_ch * 2, base_ch * 4, use_pool=True, vss_backend=vss_backend)

        self.bottleneck = nn.Sequential(
            ConvBNAct(base_ch * 4, base_ch * 4),
            build_vss_block(base_ch * 4, vss_backend=vss_backend),
        )

        self.up2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2, vss_backend=vss_backend)
        self.up1 = UpBlock(base_ch * 2, base_ch, base_ch, vss_backend=vss_backend)
        self.head = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.stem(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        b = self.bottleneck(s3)
        x = self.up2(b, s2)
        x = self.up1(x, s1)
        return self.head(x)


class TeacherStudentSegModel(nn.Module):
    """
    Teacher-student container.
    - student_logits: top branch output
    - teacher_logits: auxiliary branch output
    """

    def __init__(
        self,
        in_ch: int = 1,
        num_classes: int = 2,
        student_base_ch: int = 32,
        teacher_base_ch: int = 32,
        teacher_arch: str = "lite",
        vss_backend: str = "vsslike",
    ):
        super().__init__()
        self.student = StudentUNetLike(
            in_ch=in_ch,
            num_classes=num_classes,
            base_ch=student_base_ch,
            vss_backend=vss_backend,
        )
        if teacher_arch == "student":
            self.teacher = StudentUNetLike(
                in_ch=in_ch,
                num_classes=num_classes,
                base_ch=teacher_base_ch,
                vss_backend=vss_backend,
            )
        else:
            self.teacher = TeacherUNetLike(
                in_ch=in_ch,
                num_classes=num_classes,
                base_ch=teacher_base_ch,
                vss_backend=vss_backend,
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        student_logits = self.student(x)
        teacher_logits = self.teacher(x)
        student_prob = F.softmax(student_logits, dim=1)
        teacher_prob = F.softmax(teacher_logits, dim=1)
        teacher_pseudo = torch.argmax(teacher_prob.detach(), dim=1)
        return {
            "student_logits": student_logits,
            "teacher_logits": teacher_logits,
            "student_prob": student_prob,
            "teacher_prob": teacher_prob,
            "teacher_pseudo_labels": teacher_pseudo,
        }

    @torch.no_grad()
    def get_ema_compatibility(self) -> Dict[str, int]:
        """
        Count parameters that can be EMA-updated by name and shape.
        """
        student_named = dict(self.student.named_parameters())
        total = 0
        matched = 0
        for name, t_param in self.teacher.named_parameters():
            total += 1
            s_param = student_named.get(name, None)
            if s_param is not None and s_param.shape == t_param.shape:
                matched += 1
        return {"matched": matched, "total": total}

    @torch.no_grad()
    def initialize_teacher_from_student(self) -> Dict[str, int]:
        """
        Copy compatible student weights to teacher before EMA training.
        """
        student_state = self.student.state_dict()
        teacher_state = self.teacher.state_dict()
        copied = 0
        total = 0
        for name, t_tensor in teacher_state.items():
            total += 1
            s_tensor = student_state.get(name, None)
            if s_tensor is not None and s_tensor.shape == t_tensor.shape:
                teacher_state[name] = s_tensor.detach().clone()
                copied += 1
        self.teacher.load_state_dict(teacher_state, strict=False)
        return {"copied": copied, "total": total}

    @torch.no_grad()
    def ema_update_teacher(self, momentum: float = 0.99) -> None:
        """
        EMA update:
        teacher = m * teacher + (1 - m) * student
        """
        student_named = dict(self.student.named_parameters())
        updated = 0
        for name, t_param in self.teacher.named_parameters():
            s_param = student_named.get(name, None)
            if s_param is None or s_param.shape != t_param.shape:
                continue
            t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)
            updated += 1
        if updated == 0:
            raise RuntimeError(
                "EMA update failed: no matching teacher/student parameters were found. "
                "Set --student_base_ch and --teacher_base_ch to the same value, "
                "or switch to --teacher_update joint/frozen."
            )
