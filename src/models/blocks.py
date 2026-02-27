import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VSSLikeBlock(nn.Module):
    """
    Simplified VSS-like residual block:
    - normalization
    - two projection branches (1x1 conv)
    - depth-wise conv + activation
    - gated multiplication
    - projection + residual
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm_in = nn.GroupNorm(1, channels)
        self.linear_a = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.linear_b = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.dw = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, groups=channels
        )
        self.out_linear = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.norm_out = nn.GroupNorm(1, channels)
        self.act = nn.GELU()
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm_in(x)
        a = self.linear_a(x)
        b = self.linear_b(x)
        a = self.dw(a)
        a = self.act(a)
        x = a * self.gate(b)
        x = self.out_linear(x)
        x = self.norm_out(x)
        return x + residual


class Mamba2DBlock(nn.Module):
    """
    Flatten 2D features to token sequences and run mamba-ssm.
    Mamba backend is strict: missing dependencies raise an error.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.norm = nn.LayerNorm(channels)
        try:
            from mamba_ssm import Mamba
            self.mamba = Mamba(
                d_model=channels,
                d_state=16,
                d_conv=4,
                expand=2,
            )
        except Exception as exc:
            raise ImportError(
                "You selected --vss_backend mamba, but mamba-ssm is unavailable. "
                "Install it with: python -m pip install mamba-ssm"
            ) from exc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        b, c, h, w = x.shape
        seq = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        seq = self.norm(seq)
        seq = self.mamba(seq)
        seq = seq.transpose(1, 2).reshape(b, c, h, w)
        return seq + residual


def build_vss_block(channels: int, vss_backend: str = "vsslike") -> nn.Module:
    if vss_backend == "mamba":
        return Mamba2DBlock(channels)
    return VSSLikeBlock(channels)


class DownBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        use_pool: bool = True,
        vss_backend: str = "vsslike",
    ):
        super().__init__()
        self.use_pool = use_pool
        self.pool = nn.MaxPool2d(2) if use_pool else nn.Identity()
        self.conv1 = ConvBNAct(in_ch, out_ch)
        self.vss = build_vss_block(out_ch, vss_backend=vss_backend)
        self.conv2 = ConvBNAct(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv1(x)
        x = self.vss(x)
        x = self.conv2(x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        vss_backend: str = "vsslike",
    ):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = ConvBNAct(in_ch + skip_ch, out_ch)
        self.vss = build_vss_block(out_ch, vss_backend=vss_backend)
        self.conv2 = ConvBNAct(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.vss(x)
        x = self.conv2(x)
        return x
