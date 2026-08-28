"""
Mô tả: Kiến trúc mô hình U-Net đa phương thái (Multi-modal U-Net) cho ảnh MRI (Exp023 - Exp042). Thiết kế các nhánh encoder riêng biệt (disentangled stems) cho từng loại xung (FLAIR, T1, T1ce, T2) trước khi thực hiện fusion.
Đầu vào:
    x (torch.Tensor): Tensor ảnh MRI [Batch, Channels, Height, Width].
Đầu ra:
    Trả về dự đoán phân đoạn các vùng u tương ứng.
"""

import torch
import torch.nn as nn

from src.models.attention_unet import AttentionGate
from src.models.unet import DoubleConv


class MultiModalStemUNet2D(nn.Module):
    """
    Mô hình U-Net 2D với một nhánh encoder gọn nhẹ riêng biệt cho từng kênh xung MRI.

    Thứ tự kênh đầu vào theo tập dữ liệu: FLAIR, T1, T1ce, T2.
    Các nhánh giữ các đặc trưng phương thái tách biệt trước khi thực hiện fusion sớm.
    """

    def __init__(self, n_channels=4, n_classes=3, init_features=64, return_features=False):
        super().__init__()
        if n_channels != 4:
            raise ValueError("MultiModalStemUNet2D expects 4 MRI modalities.")

        self.return_features = return_features
        stem_features = max(init_features // 4, 8)

        self.modality_stems = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, stem_features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(stem_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_features, stem_features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(stem_features),
                nn.ReLU(inplace=True),
            )
            for _ in range(4)
        ])

        fused_channels = stem_features * 4
        self.inc = DoubleConv(fused_channels, init_features)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(init_features, init_features * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(init_features * 2, init_features * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(init_features * 4, init_features * 8))

        self.up1 = nn.ConvTranspose2d(init_features * 8, init_features * 4, 2, stride=2)
        self.conv_up1 = DoubleConv(init_features * 8, init_features * 4)
        self.up2 = nn.ConvTranspose2d(init_features * 4, init_features * 2, 2, stride=2)
        self.conv_up2 = DoubleConv(init_features * 4, init_features * 2)
        self.up3 = nn.ConvTranspose2d(init_features * 2, init_features, 2, stride=2)
        self.conv_up3 = DoubleConv(init_features * 2, init_features)

        self.outc = nn.Conv2d(init_features, n_classes, 1)
        self.feature_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        modality_features = [
            stem(x[:, i:i + 1])
            for i, stem in enumerate(self.modality_stems)
        ]

        if self.return_features:
            pooled = [
                self.feature_pool(feat).flatten(1)
                for feat in modality_features
            ]
            aux = {"modality_features": torch.stack(pooled, dim=1)}
        else:
            aux = None

        x1 = self.inc(torch.cat(modality_features, dim=1))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)

        logits = self.outc(x)
        if self.return_features:
            return logits, aux
        return logits


class DisentangledFusionUNet2D(nn.Module):
    """
    Cơ chế kết hợp shared/private (chung/riêng) gọn nhẹ lấy cảm hứng từ DFuse-Net.

    Một nhánh shared (chung) đọc tất cả các xung cùng lúc, trong khi các nhánh private (riêng) giữ các đặc trưng phương thái
    tách biệt. Một cổng học được (learned gate) kiểm soát mức độ thông tin từ nhánh riêng được bơm vào nhánh chung trước khi giải mã.
    """

    def __init__(self, n_channels=4, n_classes=3, init_features=64, return_features=False):
        super().__init__()
        if n_channels != 4:
            raise ValueError("DisentangledFusionUNet2D expects 4 MRI modalities.")

        self.return_features = return_features
        private_features = max(init_features // 4, 8)

        self.shared_stem = DoubleConv(n_channels, init_features)
        self.private_stems = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, private_features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(private_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(private_features, private_features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(private_features),
                nn.ReLU(inplace=True),
            )
            for _ in range(4)
        ])

        self.private_reduce = DoubleConv(private_features * 4, init_features)
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(init_features * 2, init_features, kernel_size=1),
            nn.Sigmoid(),
        )

        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(init_features, init_features * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(init_features * 2, init_features * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(init_features * 4, init_features * 8))

        self.up1 = nn.ConvTranspose2d(init_features * 8, init_features * 4, 2, stride=2)
        self.conv_up1 = DoubleConv(init_features * 8, init_features * 4)
        self.up2 = nn.ConvTranspose2d(init_features * 4, init_features * 2, 2, stride=2)
        self.conv_up2 = DoubleConv(init_features * 4, init_features * 2)
        self.up3 = nn.ConvTranspose2d(init_features * 2, init_features, 2, stride=2)
        self.conv_up3 = DoubleConv(init_features * 2, init_features)

        self.outc = nn.Conv2d(init_features, n_classes, 1)
        self.feature_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        shared = self.shared_stem(x)
        private_features = [
            stem(x[:, i:i + 1])
            for i, stem in enumerate(self.private_stems)
        ]
        private = self.private_reduce(torch.cat(private_features, dim=1))

        gate = self.fusion_gate(torch.cat([shared, private], dim=1))
        x1 = shared + gate * private

        if self.return_features:
            pooled = [
                self.feature_pool(feat).flatten(1)
                for feat in private_features
            ]
            aux = {"modality_features": torch.stack(pooled, dim=1)}
        else:
            aux = None

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)

        logits = self.outc(x)
        if self.return_features:
            return logits, aux
        return logits


class DisentangledFusionRegionHeadsUNet2D(nn.Module):
    """
    Khung xương kết hợp shared/private (disentangled fusion) với các nhánh dự đoán WT/TC/ET độc lập.

    Giữ lại phần cốt lõi của kết hợp shared/private ở Exp021, nhưng tránh việc bắt buộc tất cả các vùng u BraTS
    phải dùng chung một bộ phân loại 1x1 cuối cùng.
    """

    def __init__(self, n_channels=4, n_classes=3, init_features=64, return_features=False):
        super().__init__()
        if n_classes != 3:
            raise ValueError("DisentangledFusionRegionHeadsUNet2D expects 3 BraTS region outputs.")
        if n_channels != 4:
            raise ValueError("DisentangledFusionRegionHeadsUNet2D expects 4 MRI modalities.")

        self.return_features = return_features
        private_features = max(init_features // 4, 8)

        self.shared_stem = DoubleConv(n_channels, init_features)
        self.private_stems = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, private_features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(private_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(private_features, private_features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(private_features),
                nn.ReLU(inplace=True),
            )
            for _ in range(4)
        ])

        self.private_reduce = DoubleConv(private_features * 4, init_features)
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(init_features * 2, init_features, kernel_size=1),
            nn.Sigmoid(),
        )

        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(init_features, init_features * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(init_features * 2, init_features * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(init_features * 4, init_features * 8))

        self.up1 = nn.ConvTranspose2d(init_features * 8, init_features * 4, 2, stride=2)
        self.conv_up1 = DoubleConv(init_features * 8, init_features * 4)
        self.up2 = nn.ConvTranspose2d(init_features * 4, init_features * 2, 2, stride=2)
        self.conv_up2 = DoubleConv(init_features * 4, init_features * 2)
        self.up3 = nn.ConvTranspose2d(init_features * 2, init_features, 2, stride=2)
        self.conv_up3 = DoubleConv(init_features * 2, init_features)

        self.region_heads = nn.ModuleDict({
            "WT": nn.Conv2d(init_features, 1, kernel_size=1),
            "TC": nn.Conv2d(init_features, 1, kernel_size=1),
            "ET": nn.Conv2d(init_features, 1, kernel_size=1),
        })
        self.feature_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        shared = self.shared_stem(x)
        private_features = [
            stem(x[:, i:i + 1])
            for i, stem in enumerate(self.private_stems)
        ]
        private = self.private_reduce(torch.cat(private_features, dim=1))

        gate = self.fusion_gate(torch.cat([shared, private], dim=1))
        x1 = shared + gate * private

        if self.return_features:
            pooled = [
                self.feature_pool(feat).flatten(1)
                for feat in private_features
            ]
            aux = {"modality_features": torch.stack(pooled, dim=1)}
        else:
            aux = None

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)

        logits = torch.cat([
            self.region_heads["WT"](x),
            self.region_heads["TC"](x),
            self.region_heads["ET"](x),
        ], dim=1)
        if self.return_features:
            return logits, aux
        return logits


class DisentangledFusionRegionHeadsPresenceUNet2D(DisentangledFusionRegionHeadsUNet2D):
    """
    Kiến trúc Exp036 tích hợp thêm cổng dự đoán sự hiện diện của vùng ET (ET presence gate).

    Nhánh phụ trợ dự đoán xem lát cắt có chứa vùng ET hay không từ bottleneck.
    Logit của nó được cộng như một tiên nghiệm toàn cục vào logit phân đoạn ET, giúp mô hình
    loại bỏ các lát cắt nhận diện nhầm ET (false positive) và tăng cường các lát cắt chứa u ET siêu nhỏ.
    """

    def __init__(self, n_channels=4, n_classes=3, init_features=64, return_features=False):
        super().__init__(
            n_channels=n_channels,
            n_classes=n_classes,
            init_features=init_features,
            return_features=return_features,
        )
        self.et_presence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(init_features * 8, 1),
        )

    def forward(self, x):
        shared = self.shared_stem(x)
        private_features = [
            stem(x[:, i:i + 1])
            for i, stem in enumerate(self.private_stems)
        ]
        private = self.private_reduce(torch.cat(private_features, dim=1))

        gate = self.fusion_gate(torch.cat([shared, private], dim=1))
        x1 = shared + gate * private

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        et_presence_logit = self.et_presence_head(x4)

        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)

        wt_logit = self.region_heads["WT"](x)
        tc_logit = self.region_heads["TC"](x)
        et_logit = self.region_heads["ET"](x) + et_presence_logit.view(-1, 1, 1, 1)
        logits = torch.cat([wt_logit, tc_logit, et_logit], dim=1)

        aux = {"et_presence_logit": et_presence_logit}
        if self.return_features:
            pooled = [
                self.feature_pool(feat).flatten(1)
                for feat in private_features
            ]
            aux["modality_features"] = torch.stack(pooled, dim=1)
        return logits, aux


class DisentangledFusionMultiScaleRegionHeadsUNet2D(nn.Module):
    """
    Kết hợp shared/private đa tỷ lệ (Multi-scale fusion) kết hợp các nhánh dự đoán đặc thù theo vùng u.

    Khác với Exp021/036, quá trình fusion không bị giới hạn ở bản đồ đặc trưng độ phân giải cao đầu tiên.
    Các đặc trưng riêng của từng xung được truyền và lọc qua cổng ở mỗi mức tỷ lệ encoder trước khi giải mã.
    """

    def __init__(self, n_channels=4, n_classes=3, init_features=64, return_features=False):
        super().__init__()
        if n_classes != 3:
            raise ValueError("DisentangledFusionMultiScaleRegionHeadsUNet2D expects 3 BraTS region outputs.")
        if n_channels != 4:
            raise ValueError("DisentangledFusionMultiScaleRegionHeadsUNet2D expects 4 MRI modalities.")

        self.return_features = return_features
        private_features = max(init_features // 4, 8)

        self.shared_stem = DoubleConv(n_channels, init_features)
        self.private_stems = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, private_features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(private_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(private_features, private_features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(private_features),
                nn.ReLU(inplace=True),
            )
            for _ in range(4)
        ])

        self.private_reduce = DoubleConv(private_features * 4, init_features)
        self.gate1 = nn.Sequential(nn.Conv2d(init_features * 2, init_features, kernel_size=1), nn.Sigmoid())

        self.shared_down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(init_features, init_features * 2))
        self.private_down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(init_features, init_features * 2))
        self.gate2 = nn.Sequential(nn.Conv2d(init_features * 4, init_features * 2, kernel_size=1), nn.Sigmoid())

        self.shared_down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(init_features * 2, init_features * 4))
        self.private_down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(init_features * 2, init_features * 4))
        self.gate3 = nn.Sequential(nn.Conv2d(init_features * 8, init_features * 4, kernel_size=1), nn.Sigmoid())

        self.shared_down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(init_features * 4, init_features * 8))
        self.private_down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(init_features * 4, init_features * 8))
        self.gate4 = nn.Sequential(nn.Conv2d(init_features * 16, init_features * 8, kernel_size=1), nn.Sigmoid())

        self.up1 = nn.ConvTranspose2d(init_features * 8, init_features * 4, 2, stride=2)
        self.conv_up1 = DoubleConv(init_features * 8, init_features * 4)
        self.up2 = nn.ConvTranspose2d(init_features * 4, init_features * 2, 2, stride=2)
        self.conv_up2 = DoubleConv(init_features * 4, init_features * 2)
        self.up3 = nn.ConvTranspose2d(init_features * 2, init_features, 2, stride=2)
        self.conv_up3 = DoubleConv(init_features * 2, init_features)

        self.region_heads = nn.ModuleDict({
            "WT": nn.Conv2d(init_features, 1, kernel_size=1),
            "TC": nn.Conv2d(init_features, 1, kernel_size=1),
            "ET": nn.Conv2d(init_features, 1, kernel_size=1),
        })
        self.feature_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        shared1 = self.shared_stem(x)
        private_features = [
            stem(x[:, i:i + 1])
            for i, stem in enumerate(self.private_stems)
        ]
        private1 = self.private_reduce(torch.cat(private_features, dim=1))
        x1 = shared1 + self.gate1(torch.cat([shared1, private1], dim=1)) * private1

        shared2 = self.shared_down1(x1)
        private2 = self.private_down1(private1)
        x2 = shared2 + self.gate2(torch.cat([shared2, private2], dim=1)) * private2

        shared3 = self.shared_down2(x2)
        private3 = self.private_down2(private2)
        x3 = shared3 + self.gate3(torch.cat([shared3, private3], dim=1)) * private3

        shared4 = self.shared_down3(x3)
        private4 = self.private_down3(private3)
        x4 = shared4 + self.gate4(torch.cat([shared4, private4], dim=1)) * private4

        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)

        logits = torch.cat([
            self.region_heads["WT"](x),
            self.region_heads["TC"](x),
            self.region_heads["ET"](x),
        ], dim=1)
        if self.return_features:
            pooled = [
                self.feature_pool(feat).flatten(1)
                for feat in private_features
            ]
            return logits, {"modality_features": torch.stack(pooled, dim=1)}
        return logits


class DisentangledFusionMultiScalePresenceUNet2D(DisentangledFusionMultiScaleRegionHeadsUNet2D):
    """
    Kiến trúc đầy đủ: kết hợp xung đa tỷ lệ tích hợp cổng dự đoán sự hiện diện ET.
    """

    def __init__(self, n_channels=4, n_classes=3, init_features=64, return_features=False):
        super().__init__(
            n_channels=n_channels,
            n_classes=n_classes,
            init_features=init_features,
            return_features=return_features,
        )
        self.et_presence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(init_features * 8, 1),
        )

    def forward(self, x):
        shared1 = self.shared_stem(x)
        private_features = [
            stem(x[:, i:i + 1])
            for i, stem in enumerate(self.private_stems)
        ]
        private1 = self.private_reduce(torch.cat(private_features, dim=1))
        x1 = shared1 + self.gate1(torch.cat([shared1, private1], dim=1)) * private1

        shared2 = self.shared_down1(x1)
        private2 = self.private_down1(private1)
        x2 = shared2 + self.gate2(torch.cat([shared2, private2], dim=1)) * private2

        shared3 = self.shared_down2(x2)
        private3 = self.private_down2(private2)
        x3 = shared3 + self.gate3(torch.cat([shared3, private3], dim=1)) * private3

        shared4 = self.shared_down3(x3)
        private4 = self.private_down3(private3)
        x4 = shared4 + self.gate4(torch.cat([shared4, private4], dim=1)) * private4
        et_presence_logit = self.et_presence_head(x4)

        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)

        wt_logit = self.region_heads["WT"](x)
        tc_logit = self.region_heads["TC"](x)
        et_logit = self.region_heads["ET"](x) + et_presence_logit.view(-1, 1, 1, 1)
        logits = torch.cat([wt_logit, tc_logit, et_logit], dim=1)

        aux = {"et_presence_logit": et_presence_logit}
        if self.return_features:
            pooled = [
                self.feature_pool(feat).flatten(1)
                for feat in private_features
            ]
            aux["modality_features"] = torch.stack(pooled, dim=1)
        return logits, aux


class DisentangledFusionUNet2_5D(nn.Module):
    """
    Kiến trúc kết hợp shared/private phiên bản 2.5D.

    Các kênh đầu vào được gom nhóm theo xung phương thái, ví dụ: 3 lát cắt ngữ cảnh cho mỗi xung trong số
    FLAIR/T1/T1ce/T2 tạo thành 12 kênh. Mỗi nhánh riêng (private) xử lý ngăn xếp ngữ cảnh của một xung
    trong khi nhánh chung (shared) xử lý toàn bộ các kênh.
    """

    def __init__(self, n_channels=12, n_classes=3, init_features=64, return_features=False, context_slices=3):
        super().__init__()
        if n_channels % 4 != 0:
            raise ValueError("DisentangledFusionUNet2_5D expects channels grouped across 4 modalities.")

        self.return_features = return_features
        self.channels_per_modality = n_channels // 4
        if context_slices and self.channels_per_modality != context_slices:
            raise ValueError("context_slices must match in_channels / 4 for 2.5D fusion.")

        private_features = max(init_features // 4, 8)

        self.shared_stem = DoubleConv(n_channels, init_features)
        self.private_stems = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.channels_per_modality, private_features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(private_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(private_features, private_features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(private_features),
                nn.ReLU(inplace=True),
            )
            for _ in range(4)
        ])

        self.private_reduce = DoubleConv(private_features * 4, init_features)
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(init_features * 2, init_features, kernel_size=1),
            nn.Sigmoid(),
        )

        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(init_features, init_features * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(init_features * 2, init_features * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(init_features * 4, init_features * 8))

        self.up1 = nn.ConvTranspose2d(init_features * 8, init_features * 4, 2, stride=2)
        self.conv_up1 = DoubleConv(init_features * 8, init_features * 4)
        self.up2 = nn.ConvTranspose2d(init_features * 4, init_features * 2, 2, stride=2)
        self.conv_up2 = DoubleConv(init_features * 4, init_features * 2)
        self.up3 = nn.ConvTranspose2d(init_features * 2, init_features, 2, stride=2)
        self.conv_up3 = DoubleConv(init_features * 2, init_features)

        self.outc = nn.Conv2d(init_features, n_classes, 1)
        self.feature_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        shared = self.shared_stem(x)
        modality_inputs = torch.chunk(x, 4, dim=1)
        private_features = [
            stem(modality_input)
            for modality_input, stem in zip(modality_inputs, self.private_stems)
        ]
        private = self.private_reduce(torch.cat(private_features, dim=1))

        gate = self.fusion_gate(torch.cat([shared, private], dim=1))
        x1 = shared + gate * private

        if self.return_features:
            pooled = [
                self.feature_pool(feat).flatten(1)
                for feat in private_features
            ]
            aux = {"modality_features": torch.stack(pooled, dim=1)}
        else:
            aux = None

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)

        logits = self.outc(x)
        if self.return_features:
            return logits, aux
        return logits


class DisentangledFusionAttentionUNet2D(nn.Module):
    """
    Mô hình kết hợp shared/private tích hợp cổng chú ý (Attention Gates) trên các đường skip connection.

    Giữ cơ chế kết hợp shared/private của Exp021, sau đó lọc các kết nối tắt skip connection truyền sang decoder
    bằng cổng chú ý để giảm bớt sự nhầm lẫn giữa các phân vùng u phụ TC/ET.
    """

    def __init__(self, n_channels=4, n_classes=3, init_features=64, return_features=False):
        super().__init__()
        if n_channels != 4:
            raise ValueError("DisentangledFusionAttentionUNet2D expects 4 MRI modalities.")

        self.return_features = return_features
        private_features = max(init_features // 4, 8)

        self.shared_stem = DoubleConv(n_channels, init_features)
        self.private_stems = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, private_features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(private_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(private_features, private_features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(private_features),
                nn.ReLU(inplace=True),
            )
            for _ in range(4)
        ])

        self.private_reduce = DoubleConv(private_features * 4, init_features)
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(init_features * 2, init_features, kernel_size=1),
            nn.Sigmoid(),
        )

        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(init_features, init_features * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(init_features * 2, init_features * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(init_features * 4, init_features * 8))

        self.up1 = nn.ConvTranspose2d(init_features * 8, init_features * 4, 2, stride=2)
        self.attn1 = AttentionGate(F_g=init_features * 4, F_l=init_features * 4, F_int=init_features * 2)
        self.conv_up1 = DoubleConv(init_features * 8, init_features * 4)

        self.up2 = nn.ConvTranspose2d(init_features * 4, init_features * 2, 2, stride=2)
        self.attn2 = AttentionGate(F_g=init_features * 2, F_l=init_features * 2, F_int=init_features)
        self.conv_up2 = DoubleConv(init_features * 4, init_features * 2)

        self.up3 = nn.ConvTranspose2d(init_features * 2, init_features, 2, stride=2)
        self.attn3 = AttentionGate(F_g=init_features, F_l=init_features, F_int=max(init_features // 2, 1))
        self.conv_up3 = DoubleConv(init_features * 2, init_features)

        self.outc = nn.Conv2d(init_features, n_classes, 1)
        self.feature_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        shared = self.shared_stem(x)
        private_features = [
            stem(x[:, i:i + 1])
            for i, stem in enumerate(self.private_stems)
        ]
        private = self.private_reduce(torch.cat(private_features, dim=1))

        gate = self.fusion_gate(torch.cat([shared, private], dim=1))
        x1 = shared + gate * private

        if self.return_features:
            pooled = [
                self.feature_pool(feat).flatten(1)
                for feat in private_features
            ]
            aux = {"modality_features": torch.stack(pooled, dim=1)}
        else:
            aux = None

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4)
        x3_att = self.attn1(g=x, x=x3)
        x = self.conv_up1(torch.cat([x, x3_att], dim=1))

        x = self.up2(x)
        x2_att = self.attn2(g=x, x=x2)
        x = self.conv_up2(torch.cat([x, x2_att], dim=1))

        x = self.up3(x)
        x1_att = self.attn3(g=x, x=x1)
        x = self.conv_up3(torch.cat([x, x1_att], dim=1))

        logits = self.outc(x)
        if self.return_features:
            return logits, aux
        return logits
