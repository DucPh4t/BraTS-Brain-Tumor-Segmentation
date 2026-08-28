"""
Mô tả: Kiến trúc mô hình Attention U-Net 2D (Exp009 - Exp018). Tích hợp cơ chế Attention Gate trên các kết nối tắt (skip connections) để mô hình tập trung vào vùng khối u đích tốt hơn.
Đầu vào:
    x (torch.Tensor): Tensor ảnh MRI [Batch, Channels, Height, Width].
Đầu ra:
    Trả về dự đoán phân đoạn của mô hình.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# BUILDING BLOCK: Double Convolution (tái sử dụng từ UNet gốc)
# ──────────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    """Conv2d → BN → ReLU → Conv2d → BN → ReLU"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ──────────────────────────────────────────────────────────────
# ATTENTION GATE (Oktay et al., 2018 — "Attention U-Net")
#
# Công thức:
#   g  : gating signal từ decoder (thông tin ngữ cảnh cấp cao)
#   x  : skip connection từ encoder (thông tin chi tiết cấp thấp)
#
#   q_att = W_g(g) + W_x(x)           ← cộng 2 nhánh (1x1 conv)
#   alpha  = sigmoid(W_psi(ReLU(q_att)))  ← tạo attention map ∈ [0,1]
#   output = alpha ⊙ x                ← gate: nhấn chìm vùng nền, thắp sáng vùng u
# ──────────────────────────────────────────────────────────────
class AttentionGate(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        """
        Tham số:
            F_g   : số channels của gating signal (từ decoder)
            F_l   : số channels của skip connection (từ encoder)
            F_int : số channels trung gian (thường = F_l // 2)
        """
        super().__init__()
        # Nhánh Gating signal (context từ decoder)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        # Nhánh Skip connection (chi tiết từ encoder)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        # Lớp cuối tạo attention coefficient (scalar per spatial location)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        g: gating signal [B, F_g, H, W]   — từ decoder (có thể nhỏ hơn x)
        x: skip feature  [B, F_l, H', W'] — từ encoder (kích thước lớn hơn)
        """
        # Upsample g về cùng spatial size với x nếu khác nhau
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)

        g1  = self.W_g(g)               # [B, F_int, H, W]
        x1  = self.W_x(x)               # [B, F_int, H, W]
        psi = self.relu(g1 + x1)        # fused query
        psi = self.psi(psi)             # attention map [B, 1, H, W]
        return psi * x                  # attended skip connection


# ──────────────────────────────────────────────────────────────
# ATTENTION U-NET 2D
#
# Kiến trúc:    Encoder (Downsampling path)
#                   ↓  skip connections → ATTENTION GATE ↓
#               Bottleneck
#                   ↓  Decoder (Upsampling path)
#               Output
# ──────────────────────────────────────────────────────────────
class AttentionUNet2D(nn.Module):
    def __init__(self, n_channels: int = 4, n_classes: int = 3, init_features: int = 32):
        """
        Tham số:
            n_channels   : số kênh đầu vào (4 MRI modalities: T1, T1ce, T2, FLAIR)
            n_classes    : số lớp phân đoạn (3: WT, TC, ET)
            init_features: số filter tầng đầu tiên (thường 32 hoặc 64)
        """
        super().__init__()
        F = init_features  # alias cho gọn

        # ── Encoder ─────────────────────────
        self.enc1 = DoubleConv(n_channels, F)          # [B,  F, H,   W  ]
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(F,     F * 2)           # [B, 2F, H/2, W/2]
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(F * 2, F * 4)           # [B, 4F, H/4, W/4]
        self.pool3 = nn.MaxPool2d(2)

        # ── Bottleneck ───────────────────────
        self.bottleneck = DoubleConv(F * 4, F * 8)    # [B, 8F, H/8, W/8]

        # ── Decoder + Attention Gates ────────
        # Mức 3: bottleneck → + enc3
        self.up3      = nn.ConvTranspose2d(F * 8, F * 4, kernel_size=2, stride=2)
        self.attn3    = AttentionGate(F_g=F * 4, F_l=F * 4, F_int=F * 2)
        self.dec3     = DoubleConv(F * 8, F * 4)

        # Mức 2: → + enc2
        self.up2      = nn.ConvTranspose2d(F * 4, F * 2, kernel_size=2, stride=2)
        self.attn2    = AttentionGate(F_g=F * 2, F_l=F * 2, F_int=F)
        self.dec2     = DoubleConv(F * 4, F * 2)

        # Mức 1: → + enc1
        self.up1      = nn.ConvTranspose2d(F * 2, F,     kernel_size=2, stride=2)
        self.attn1    = AttentionGate(F_g=F,     F_l=F,     F_int=F // 2)
        self.dec1     = DoubleConv(F * 2, F)

        # ── Output ───────────────────────────
        self.outc = nn.Conv2d(F, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Encoder ─────────────────────────
        e1 = self.enc1(x)               # skip 1:  [B,  F, H,   W  ]
        e2 = self.enc2(self.pool1(e1))  # skip 2:  [B, 2F, H/2, W/2]
        e3 = self.enc3(self.pool2(e2))  # skip 3:  [B, 4F, H/4, W/4]
        bn = self.bottleneck(self.pool3(e3))  #    [B, 8F, H/8, W/8]

        # ── Decoder với Attention ────────────
        # Mức 3
        d3 = self.up3(bn)               # upsample: [B, 4F, H/4, W/4]
        e3 = self.attn3(g=d3, x=e3)    # ← ATTENTION GATE lọc skip conn
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        # Mức 2
        d2 = self.up2(d3)
        e2 = self.attn2(g=d2, x=e2)    # ← ATTENTION GATE
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        # Mức 1
        d1 = self.up1(d2)
        e1 = self.attn1(g=d1, x=e1)    # ← ATTENTION GATE
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.outc(d1)  # Logits thô — lớp sigmoid sẽ được áp dụng trong trainer (đồng bộ với UNet2D)
