"""
Mô tả: Kiến trúc mô hình ResNet34-UNet (Exp019 - Exp022) kết hợp khối Region Heads và kiểm soát phân cấp (Hierarchy Loss). Sử dụng mạng ResNet34 đã tiền huấn luyện làm encoder để tăng hiệu quả trích xuất đặc trưng.
Đầu vào:
    x (torch.Tensor): Tensor ảnh MRI [Batch, Channels, Height, Width].
Đầu ra:
    Trả về dự đoán phân đoạn của mô hình (WT, TC, ET).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet import DoubleConv


def _load_resnet34(encoder_weights, require_encoder_weights=False):
    from torchvision.models import resnet34

    if encoder_weights in (None, "none", False):
        return resnet34(weights=None)

    try:
        from torchvision.models import ResNet34_Weights

        return resnet34(weights=ResNet34_Weights.DEFAULT)
    except Exception as exc:
        if require_encoder_weights:
            raise RuntimeError(
                "ImageNet ResNet34 weights are required by this protocol but could not be loaded. "
                "Enable network access or populate the torchvision checkpoint cache."
            ) from exc
        print(f"[WARN] Could not load ResNet34 ImageNet weights ({exc}). Falling back to random weights.")
        try:
            return resnet34(weights=None)
        except TypeError:
            return resnet34(pretrained=False)


def _adapt_first_conv(model, in_channels):
    old_conv = model.conv1
    if old_conv.in_channels == in_channels:
        return

    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    with torch.no_grad():
        old_weight = old_conv.weight.data
        if in_channels == 1:
            new_conv.weight.copy_(old_weight.mean(dim=1, keepdim=True))
        elif in_channels > old_weight.shape[1]:
            new_conv.weight[:, : old_weight.shape[1]].copy_(old_weight)
            mean_weight = old_weight.mean(dim=1, keepdim=True)
            for channel_idx in range(old_weight.shape[1], in_channels):
                new_conv.weight[:, channel_idx:channel_idx + 1].copy_(mean_weight)
        else:
            new_conv.weight.copy_(old_weight[:, :in_channels])
    model.conv1 = new_conv


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ResNet34UNet2D(nn.Module):
    """
    Bộ giải mã U-Net (decoder) kết hợp bộ mã hóa ResNet34 (encoder).

    Đây là nhánh nâng cấp backbone: sử dụng bộ mã hóa residual sâu hơn trước, sau đó
    đối chiếu các nhánh region heads/phân cấp trên cùng một đặc trưng biểu diễn mạnh mẽ.
    """

    def __init__(
        self,
        n_channels=4,
        n_classes=3,
        init_features=64,
        encoder_weights="imagenet",
        region_heads=False,
        require_encoder_weights=False,
    ):
        super().__init__()
        self.region_heads_enabled = region_heads

        encoder = _load_resnet34(encoder_weights, require_encoder_weights=require_encoder_weights)
        _adapt_first_conv(encoder, n_channels)

        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, init_features)
        self.dec1 = DecoderBlock(init_features, 64, init_features)
        self.dec0 = DoubleConv(init_features, init_features)

        if region_heads:
            if n_classes != 3:
                raise ValueError("ResNet34UNet2D region heads expect 3 BraTS region outputs.")
            self.region_heads = nn.ModuleDict({
                "WT": nn.Conv2d(init_features, 1, kernel_size=1),
                "TC": nn.Conv2d(init_features, 1, kernel_size=1),
                "ET": nn.Conv2d(init_features, 1, kernel_size=1),
            })
        else:
            self.outc = nn.Conv2d(init_features, n_classes, kernel_size=1)

    def _decode_features(self, x):
        input_size = x.shape[-2:]
        x0 = self.stem(x)          # 120 x 120
        x1 = self.layer1(self.maxpool(x0))  # 60 x 60
        x2 = self.layer2(x1)       # 30 x 30
        x3 = self.layer3(x2)       # 15 x 15
        x4 = self.layer4(x3)       # Kích thước 8 x 8 cho đầu vào 240 x 240

        x = self.dec4(x4, x3)
        x = self.dec3(x, x2)
        x = self.dec2(x, x1)
        x = self.dec1(x, x0)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return self.dec0(x)

    def forward(self, x):
        x = self._decode_features(x)
        if not self.region_heads_enabled:
            return self.outc(x)

        return torch.cat([
            self.region_heads["WT"](x),
            self.region_heads["TC"](x),
            self.region_heads["ET"](x),
        ], dim=1)


class ResNet34RegionHeadsUNet2D(ResNet34UNet2D):
    def __init__(self, n_channels=4, n_classes=3, init_features=64, encoder_weights="imagenet", require_encoder_weights=False):
        super().__init__(
            n_channels=n_channels,
            n_classes=n_classes,
            init_features=init_features,
            encoder_weights=encoder_weights,
            region_heads=True,
            require_encoder_weights=require_encoder_weights,
        )


class RegionPredictionBranch(nn.Module):
    """Nhánh dự đoán đặc thù cho từng vùng u với khả năng học các bộ lọc không gian."""

    def __init__(self, in_channels, branch_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class ResNet34RegionBranchesUNet2D(ResNet34UNet2D):
    """ResNet34 U-Net với các nhánh dự đoán WT/TC/ET không tương đương (độc lập).

    Khác với việc sử dụng ba lớp tích chập 1x1 riêng biệt, mỗi nhánh có một phép biến đổi
    đặc trưng 3x3 riêng trước khi đưa ra logit vùng. Điều này giúp mỗi vùng u BraTS có các
    bộ lọc không gian độc lập trong khi vẫn chia sẻ encoder và decoder.
    """

    def __init__(
        self,
        n_channels=4,
        n_classes=3,
        init_features=64,
        encoder_weights="imagenet",
        branch_channels=32,
        require_encoder_weights=False,
    ):
        if n_classes != 3:
            raise ValueError("ResNet34RegionBranchesUNet2D expects 3 BraTS region outputs.")
        if branch_channels <= 0:
            raise ValueError("branch_channels must be positive.")

        super().__init__(
            n_channels=n_channels,
            n_classes=n_classes,
            init_features=init_features,
            encoder_weights=encoder_weights,
            region_heads=False,
            require_encoder_weights=require_encoder_weights,
        )
        del self.outc
        self.region_branches = nn.ModuleDict({
            region: RegionPredictionBranch(init_features, branch_channels)
            for region in ("WT", "TC", "ET")
        })

    def forward(self, x):
        features = self._decode_features(x)
        return torch.cat(
            [self.region_branches[region](features) for region in ("WT", "TC", "ET")],
            dim=1,
        )


class ResNet34RegionHeadsDeepSupervisionUNet2D(ResNet34UNet2D):
    """
    ResNet34 region-heads U-Net với các đầu ra decoder phụ trợ phục vụ giám sát sâu (Deep Supervision).

    Các logit cuối cùng giữ nguyên hình dạng giống như ResNet34RegionHeadsUNet2D.
    Trong quá trình huấn luyện, các đầu ra phụ trợ cung cấp sự giám sát trực tiếp tại các
    tỷ lệ giải mã trung gian, giúp các cấu trúc ET/TC nhỏ không bị bỏ sót khi chỉ học từ đầu ra phân giải cao cuối cùng.
    """

    def __init__(self, n_channels=4, n_classes=3, init_features=64, encoder_weights="imagenet", require_encoder_weights=False):
        super().__init__(
            n_channels=n_channels,
            n_classes=n_classes,
            init_features=init_features,
            encoder_weights=encoder_weights,
            region_heads=True,
            require_encoder_weights=require_encoder_weights,
        )
        if n_classes != 3:
            raise ValueError("ResNet34RegionHeadsDeepSupervisionUNet2D expects 3 BraTS region outputs.")

        self.ds3 = nn.Conv2d(128, n_classes, kernel_size=1)          # 30 x 30
        self.ds2 = nn.Conv2d(init_features, n_classes, kernel_size=1) # 60 x 60
        self.ds1 = nn.Conv2d(init_features, n_classes, kernel_size=1) # 120 x 120

    def forward(self, x):
        input_size = x.shape[-2:]
        x0 = self.stem(x)                  # 120 x 120
        x1 = self.layer1(self.maxpool(x0)) # 60 x 60
        x2 = self.layer2(x1)               # 30 x 30
        x3 = self.layer3(x2)               # 15 x 15
        x4 = self.layer4(x3)               # Kích thước 8 x 8 cho đầu vào 240 x 240

        x = self.dec4(x4, x3)
        dec3 = self.dec3(x, x2)
        dec2 = self.dec2(dec3, x1)
        dec1 = self.dec1(dec2, x0)

        x = F.interpolate(dec1, size=input_size, mode="bilinear", align_corners=False)
        x = self.dec0(x)
        logits = torch.cat([
            self.region_heads["WT"](x),
            self.region_heads["TC"](x),
            self.region_heads["ET"](x),
        ], dim=1)

        aux = {
            "deep_supervision": [
                self.ds3(dec3),
                self.ds2(dec2),
                self.ds1(dec1),
            ]
        }
        return logits, aux
