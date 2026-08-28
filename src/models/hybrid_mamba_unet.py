"""
Mô tả: Kiến trúc mô hình lai Hybrid 2.5D Mamba U-Net (Exp043 - Exp053). Kết hợp khả năng trích xuất đặc trưng không gian của ResNet34 và mô hình hóa chuỗi tuần tự bidirectional Mamba tại bottleneck cho đầu vào 2.5D (5 lát cắt).
Đầu vào:
    x (torch.Tensor): Tensor đầu vào kích thước [Batch, Channels * Slices, Height, Width].
Đầu ra:
    Trả về dự đoán phân đoạn 3 vùng u độc lập: WT, TC, ET.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.models.resnet_unet import ResNet34RegionHeadsUNet2D


def _group_count(channels, max_groups=8):
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )


class SharedContextEncoder(nn.Module):
    """Mã hóa từng xung phương thái/lát cắt ảnh bằng cùng một mạng CNN gọn nhẹ chia sẻ trọng số."""

    def __init__(self, channels=(8, 16, 32)):
        super().__init__()
        if len(channels) != 3 or any(int(value) <= 0 for value in channels):
            raise ValueError("context_channels must contain three positive values.")
        c0, c1, c2 = (int(value) for value in channels)
        self.out_channels = c2
        self.blocks = nn.Sequential(
            ConvNormAct(1, c0, stride=2),
            ConvNormAct(c0, c1, stride=2),
            ConvNormAct(c1, c2, stride=2),
        )

    def forward(self, x):
        return self.blocks(x)


class MultiOrderMambaMixer(nn.Module):
    """Trộn các token 2.5D sử dụng các lượt quét đảo ngược được ưu tiên theo lát cắt và không gian."""

    VALID_SCAN_COUNTS = (2, 4)

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        scan_orders=4,
        residual_scale_init=0.1,
        pack_scan_orders=True,
        activation_checkpointing=False,
        mamba_factory=None,
    ):
        super().__init__()
        self.scan_orders = int(scan_orders)
        self.pack_scan_orders = bool(pack_scan_orders)
        self.activation_checkpointing = bool(activation_checkpointing)
        if self.scan_orders not in self.VALID_SCAN_COUNTS:
            raise ValueError(
                f"scan_orders must be one of {self.VALID_SCAN_COUNTS}, got {self.scan_orders}."
            )

        if mamba_factory is None:
            try:
                from mamba_ssm import Mamba
            except ImportError as exc:
                raise ImportError(
                    "Exp052 requires mamba-ssm. Install PyTorch first, then run "
                    "`pip install mamba-ssm==2.3.2.post1 --no-build-isolation`."
                ) from exc
            mamba_factory = Mamba

        self.d_model = int(d_model)
        self.norm = nn.LayerNorm(self.d_model)
        self.mamba = mamba_factory(
            d_model=self.d_model,
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
        )
        self.out_projection = nn.Linear(self.d_model, self.d_model)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def _run_mamba(self, sequence):
        return self.mamba(self.norm(sequence))

    def _mix_sequence(self, sequence):
        if self.activation_checkpointing and self.training and sequence.requires_grad:
            return checkpoint(self._run_mamba, sequence, use_reentrant=False)
        return self._run_mamba(sequence)

    @staticmethod
    def _flatten_order(x, order):
        """Chuyển đổi đặc trưng dạng chuẩn [N, K, H, W, C] thành chuỗi token phẳng."""
        if order == "slice_major":
            return x.contiguous().view(x.shape[0], -1, x.shape[-1])
        if order == "spatial_major":
            return x.permute(0, 2, 3, 1, 4).contiguous().view(x.shape[0], -1, x.shape[-1])
        raise ValueError(f"Unknown Mamba token order: {order}")

    @staticmethod
    def _restore_order(sequence, order, shape):
        """Khôi phục chuỗi token phẳng trở lại tọa độ dạng chuẩn [N, K, H, W, C]."""
        n, context_slices, height, width, channels = shape
        if order == "slice_major":
            return sequence.contiguous().view(n, context_slices, height, width, channels)
        if order == "spatial_major":
            return sequence.contiguous().view(
                n, height, width, context_slices, channels
            ).permute(0, 3, 1, 2, 4).contiguous()
        raise ValueError(f"Unknown Mamba token order: {order}")

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"Mamba mixer expects [N, K, C, H, W], got {tuple(x.shape)}")
        if x.shape[2] != self.d_model:
            raise ValueError(
                f"Mamba d_model={self.d_model} does not match feature channels={x.shape[2]}."
            )

        canonical = x.permute(0, 1, 3, 4, 2).contiguous()
        scan_specs = [
            ("slice_major", False),
            ("slice_major", True),
        ]
        if self.scan_orders == 4:
            scan_specs.extend([
                ("spatial_major", False),
                ("spatial_major", True),
            ])

        if self.pack_scan_orders:
            sequences = []
            for order, reverse in scan_specs:
                sequence = self._flatten_order(canonical, order)
                sequences.append(torch.flip(sequence, dims=(1,)) if reverse else sequence)
            packed = self._mix_sequence(torch.cat(sequences, dim=0))
            chunks = packed.chunk(len(scan_specs), dim=0)
            restored = []
            for chunk, (order, reverse) in zip(chunks, scan_specs):
                if reverse:
                    chunk = torch.flip(chunk, dims=(1,))
                restored.append(self._restore_order(chunk, order, canonical.shape))
            mixed = torch.stack(restored, dim=0).mean(dim=0)
        else:
            # Các lượt quét độ phân giải cao được tích lũy tuần tự để tránh sự bùng nổ bộ nhớ token 
            # tăng lên gấp 4 lần ở mức 18,000 token cho mỗi phương thái.
            mixed = None
            for order, reverse in scan_specs:
                sequence = self._flatten_order(canonical, order)
                if reverse:
                    sequence = torch.flip(sequence, dims=(1,))
                sequence = self._mix_sequence(sequence)
                if reverse:
                    sequence = torch.flip(sequence, dims=(1,))
                restored = self._restore_order(sequence, order, canonical.shape)
                mixed = restored if mixed is None else mixed + restored
            mixed = mixed / len(scan_specs)
        mixed = canonical + self.residual_scale * self.out_projection(mixed)
        return mixed.permute(0, 1, 4, 2, 3).contiguous()


class ContextReconstructionHead(nn.Module):
    """Tái cấu trúc phần dư (residual) một kênh ở độ phân giải gốc của ảnh MRI."""

    def __init__(self, in_channels, channels=(16, 8, 8), zero_init_output=True):
        super().__init__()
        if len(channels) != 3 or any(int(value) <= 0 for value in channels):
            raise ValueError("reconstruction_channels must contain three positive values.")
        c0, c1, c2 = (int(value) for value in channels)
        self.refine0 = ConvNormAct(in_channels, c0)
        self.refine1 = ConvNormAct(c0, c1)
        self.refine2 = ConvNormAct(c1, c2)
        self.output_projection = nn.Conv2d(c2, 1, kernel_size=1)

        if zero_init_output:
            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)

    @staticmethod
    def _upsample(x):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x, output_size):
        x = self.refine0(self._upsample(x))
        x = self.refine1(self._upsample(x))
        x = self.refine2(self._upsample(x))
        if x.shape[-2:] != tuple(output_size):
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return self.output_projection(x)


class HighResolutionSharedContextEncoder(nn.Module):
    """Mã hóa từng phương thái/lát cắt ở kích thước 60x60 cho đầu vào kích thước 240x240."""

    def __init__(self, channels=(8, 32)):
        super().__init__()
        if len(channels) != 2 or any(int(value) <= 0 for value in channels):
            raise ValueError("highres_context_channels must contain two positive values.")
        c0, c1 = (int(value) for value in channels)
        self.out_channels = c1
        self.blocks = nn.Sequential(
            ConvNormAct(1, c0, stride=2),
            ConvNormAct(c0, c1, stride=2),
        )

    def forward(self, x):
        return self.blocks(x)


class CrossModalityReliabilityGate(nn.Module):
    """Ước lượng một trọng số tin cậy ngữ cảnh (context-reliability weight) cho mỗi xung phương thái và lát cắt trung tâm."""

    def __init__(
        self,
        num_modalities,
        feature_channels,
        hidden_channels=32,
        initial_value=0.1,
    ):
        super().__init__()
        self.num_modalities = int(num_modalities)
        self.feature_channels = int(feature_channels)
        hidden_channels = int(hidden_channels)
        initial_value = float(initial_value)
        if self.num_modalities < 2:
            raise ValueError("Cross-modality gating requires at least two modalities.")
        if self.feature_channels < 1 or hidden_channels < 1:
            raise ValueError("Gate feature and hidden channels must be positive.")
        if not 0.0 < initial_value < 1.0:
            raise ValueError("gate_initial_value must be strictly between 0 and 1.")

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.interaction = nn.Sequential(
            nn.Linear(self.num_modalities * self.feature_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, self.num_modalities),
        )

        # Bắt đầu gần với giá trị hiệu chỉnh an toàn 0.1 trong khi vẫn giữ lại một phần phụ thuộc 
        # nhỏ giữa các phương thái xung (cross-modal) để tất cả các lớp cổng đều có thể nhận gradient.
        final = self.interaction[-1]
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        nn.init.constant_(
            final.bias,
            math.log(initial_value / (1.0 - initial_value)),
        )

    def forward(self, modality_features):
        if modality_features.ndim != 5:
            raise ValueError(
                "Reliability gate expects [B, M, C, H, W], got "
                f"{tuple(modality_features.shape)}"
            )
        batch, modalities, channels, _, _ = modality_features.shape
        if modalities != self.num_modalities or channels != self.feature_channels:
            raise ValueError(
                "Reliability gate feature shape mismatch: expected "
                f"M={self.num_modalities}, C={self.feature_channels}; got "
                f"M={modalities}, C={channels}."
            )

        pooled = self.pool(
            modality_features.reshape(
                batch * modalities,
                channels,
                modality_features.shape[-2],
                modality_features.shape[-1],
            )
        ).view(batch, modalities * channels)
        return torch.sigmoid(self.interaction(pooled))


class HighResolutionSignedReconstructionHead(nn.Module):
    """Tái cấu trúc phần hiệu chỉnh có dấu bị giới hạn (bounded signed correction) từ kích thước 60x60 về kích thước đầu vào gốc."""

    def __init__(
        self,
        in_channels,
        channels=(16, 8),
        delta_scale=1.0,
        zero_init_output=True,
    ):
        super().__init__()
        if len(channels) != 2 or any(int(value) <= 0 for value in channels):
            raise ValueError("reconstruction_channels must contain two positive values.")
        if float(delta_scale) <= 0:
            raise ValueError("delta_scale must be positive.")
        c0, c1 = (int(value) for value in channels)
        self.delta_scale = float(delta_scale)
        self.refine0 = ConvNormAct(int(in_channels), c0)
        self.refine1 = ConvNormAct(c0, c1)
        self.output_projection = nn.Conv2d(c1, 1, kernel_size=1)
        if zero_init_output:
            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)

    @staticmethod
    def _upsample(x):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x, output_size):
        x = self.refine0(self._upsample(x))
        x = self.refine1(self._upsample(x))
        if x.shape[-2:] != tuple(output_size):
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return self.delta_scale * torch.tanh(self.output_projection(x))


class HighResolutionGatedMamba2_5DAdapter(nn.Module):
    """Bộ thích ứng Mamba tiền fusion độ phân giải cao tích hợp cổng lọc độ tin cậy an toàn."""

    def __init__(
        self,
        num_modalities=4,
        context_slices=5,
        context_channels=(8, 32),
        reconstruction_channels=(16, 8),
        gate_hidden_channels=32,
        gate_initial_value=0.1,
        delta_scale=1.0,
        zero_init_delta=True,
        mamba_layers=1,
        d_model=32,
        d_state=16,
        d_conv=4,
        expand=2,
        scan_orders=4,
        mamba_residual_scale_init=0.1,
        pack_scan_orders=False,
        activation_checkpointing=True,
        mamba_factory=None,
    ):
        super().__init__()
        self.num_modalities = int(num_modalities)
        self.context_slices = int(context_slices)
        self.center_index = self.context_slices // 2
        if self.num_modalities < 2:
            raise ValueError("num_modalities must be at least two.")
        if self.context_slices < 1 or self.context_slices % 2 == 0:
            raise ValueError("context_slices must be a positive odd number.")
        if int(mamba_layers) < 1:
            raise ValueError("mamba_layers must be at least 1.")

        self.context_encoder = HighResolutionSharedContextEncoder(context_channels)
        if self.context_encoder.out_channels != int(d_model):
            raise ValueError(
                "The final context channel count must match Mamba d_model: "
                f"{self.context_encoder.out_channels} != {int(d_model)}."
            )
        self.d_model = int(d_model)
        self.modality_embedding = nn.Parameter(
            torch.zeros(1, self.num_modalities, 1, self.d_model, 1, 1)
        )
        self.mixers = nn.ModuleList([
            MultiOrderMambaMixer(
                d_model=self.d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                scan_orders=scan_orders,
                residual_scale_init=mamba_residual_scale_init,
                pack_scan_orders=pack_scan_orders,
                activation_checkpointing=activation_checkpointing,
                mamba_factory=mamba_factory,
            )
            for _ in range(int(mamba_layers))
        ])
        self.reliability_gate = CrossModalityReliabilityGate(
            num_modalities=self.num_modalities,
            feature_channels=self.d_model,
            hidden_channels=gate_hidden_channels,
            initial_value=gate_initial_value,
        )
        self.reconstruction = HighResolutionSignedReconstructionHead(
            in_channels=self.d_model,
            channels=reconstruction_channels,
            delta_scale=delta_scale,
            zero_init_output=bool(zero_init_delta),
        )

    def _split_streams(self, x):
        if x.ndim != 4:
            raise ValueError(f"Exp053 expects [B, M*K, H, W], got {tuple(x.shape)}")
        batch, channels, height, width = x.shape
        expected_channels = self.num_modalities * self.context_slices
        if channels != expected_channels:
            raise ValueError(f"Exp053 expects {expected_channels} input channels, got {channels}.")
        if height % 4 != 0 or width % 4 != 0:
            raise ValueError("Input height and width must be divisible by 4.")
        # Thứ tự tập dữ liệu: FLAIR[K], T1[K], T1ce[K], T2[K].
        return x.view(batch, self.num_modalities, self.context_slices, 1, height, width)

    def encode_context(self, x):
        streams = self._split_streams(x)
        batch, modalities, slices, _, height, width = streams.shape
        encoded = self.context_encoder(
            streams.reshape(batch * modalities * slices, 1, height, width)
        )
        encoded = encoded.view(
            batch,
            modalities,
            slices,
            self.d_model,
            encoded.shape[-2],
            encoded.shape[-1],
        )
        encoded = encoded + self.modality_embedding
        mixed = encoded.view(
            batch * modalities,
            slices,
            self.d_model,
            encoded.shape[-2],
            encoded.shape[-1],
        )
        for mixer in self.mixers:
            mixed = mixer(mixed)
        center = mixed[:, self.center_index]
        return center.view(
            batch,
            modalities,
            self.d_model,
            center.shape[-2],
            center.shape[-1],
        )

    def enrich_center_slices(self, x, return_details=False):
        streams = self._split_streams(x)
        batch, modalities, _, _, height, width = streams.shape
        center_raw = streams[:, :, self.center_index]
        center_context = self.encode_context(x)
        gate = self.reliability_gate(center_context)
        delta = self.reconstruction(
            center_context.reshape(
                batch * modalities,
                self.d_model,
                center_context.shape[-2],
                center_context.shape[-1],
            ),
            output_size=(height, width),
        ).view(batch, modalities, 1, height, width)
        gated_delta = gate.view(batch, modalities, 1, 1, 1) * delta
        enriched = (center_raw + gated_delta).squeeze(2)
        if return_details:
            return enriched, {
                "modality_gate": gate,
                "delta": delta.squeeze(2),
                "gated_delta": gated_delta.squeeze(2),
            }
        return enriched

    def forward(self, x):
        return self.enrich_center_slices(x)


class Mamba2_5DResidualAdapter(nn.Module):
    """Tạo một lát cắt trung tâm được làm giàu đặc trưng cho mỗi phương thái mà không trộn lẫn giữa các phương thái khác nhau."""

    def __init__(
        self,
        num_modalities=4,
        context_slices=5,
        context_channels=(8, 16, 32),
        reconstruction_channels=(16, 8, 8),
        mamba_layers=1,
        d_model=32,
        d_state=16,
        d_conv=4,
        expand=2,
        scan_orders=4,
        mamba_residual_scale_init=0.1,
        adapter_scale_init=1.0,
        zero_init_delta=True,
        mamba_factory=None,
    ):
        super().__init__()
        self.num_modalities = int(num_modalities)
        self.context_slices = int(context_slices)
        self.center_index = self.context_slices // 2

        if self.num_modalities < 1:
            raise ValueError("num_modalities must be positive.")
        if self.context_slices < 1 or self.context_slices % 2 == 0:
            raise ValueError("context_slices must be a positive odd number.")
        if int(mamba_layers) < 1:
            raise ValueError("mamba_layers must be at least 1.")

        self.context_encoder = SharedContextEncoder(context_channels)
        if self.context_encoder.out_channels != int(d_model):
            raise ValueError(
                "The final context channel count must match Mamba d_model: "
                f"{self.context_encoder.out_channels} != {int(d_model)}."
            )
        self.d_model = int(d_model)

        # Một lượng định danh nhỏ được học giúp cho các trọng số shared CNN/Mamba vẫn giữ được đặc trưng riêng biệt của từng phương thái xung.
        self.modality_embedding = nn.Parameter(
            torch.zeros(1, self.num_modalities, 1, self.d_model, 1, 1)
        )
        self.mixers = nn.ModuleList([
            MultiOrderMambaMixer(
                d_model=self.d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                scan_orders=scan_orders,
                residual_scale_init=mamba_residual_scale_init,
                mamba_factory=mamba_factory,
            )
            for _ in range(int(mamba_layers))
        ])
        self.reconstruction = ContextReconstructionHead(
            in_channels=self.d_model,
            channels=reconstruction_channels,
            zero_init_output=bool(zero_init_delta),
        )
        self.adapter_scale = nn.Parameter(
            torch.full((self.num_modalities,), float(adapter_scale_init))
        )

    def _split_streams(self, x):
        if x.ndim != 4:
            raise ValueError(f"Exp052 expects [B, M*K, H, W], got {tuple(x.shape)}")
        batch, channels, height, width = x.shape
        expected_channels = self.num_modalities * self.context_slices
        if channels != expected_channels:
            raise ValueError(f"Exp052 expects {expected_channels} input channels, got {channels}.")
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("Input height and width must be divisible by 8.")

        # Thứ tự tập dữ liệu: FLAIR[K], T1[K], T1ce[K], T2[K].
        return x.view(batch, self.num_modalities, self.context_slices, 1, height, width)

    def encode_context(self, x):
        streams = self._split_streams(x)
        batch, modalities, slices, _, height, width = streams.shape
        encoded = self.context_encoder(
            streams.reshape(batch * modalities * slices, 1, height, width)
        )
        encoded = encoded.view(
            batch,
            modalities,
            slices,
            self.d_model,
            encoded.shape[-2],
            encoded.shape[-1],
        )
        encoded = encoded + self.modality_embedding

        mixed = encoded.view(
            batch * modalities,
            slices,
            self.d_model,
            encoded.shape[-2],
            encoded.shape[-1],
        )
        for mixer in self.mixers:
            mixed = mixer(mixed)
        return mixed[:, self.center_index]

    def enrich_center_slices(self, x, return_delta=False):
        streams = self._split_streams(x)
        batch, modalities, _, _, height, width = streams.shape
        center_raw = streams[:, :, self.center_index]
        center_context = self.encode_context(x)
        delta = self.reconstruction(center_context, output_size=(height, width))
        delta = delta.view(batch, modalities, 1, height, width)
        scale = self.adapter_scale.view(1, modalities, 1, 1, 1)
        enriched = center_raw + scale * delta
        enriched = enriched.squeeze(2)
        if return_delta:
            return enriched, delta.squeeze(2)
        return enriched

    def forward(self, x):
        return self.enrich_center_slices(x)


class Hybrid2_5DMambaRegionHeadsUNet(nn.Module):
    """Bộ thích ứng Mamba 2.5D tiền fusion gọn nhẹ kết hợp với phần thân Exp043."""

    def __init__(
        self,
        n_channels=20,
        n_classes=3,
        init_features=64,
        num_modalities=4,
        context_slices=5,
        context_channels=(8, 16, 32),
        reconstruction_channels=(16, 8, 8),
        mamba_layers=1,
        d_model=32,
        d_state=16,
        d_conv=4,
        expand=2,
        scan_orders=4,
        mamba_residual_scale_init=0.1,
        adapter_scale_init=1.0,
        zero_init_delta=True,
        encoder_weights="imagenet",
        require_encoder_weights=False,
        mamba_factory=None,
    ):
        super().__init__()
        expected_channels = int(num_modalities) * int(context_slices)
        if int(n_channels) != expected_channels:
            raise ValueError(
                f"Expected in_channels={expected_channels} ({num_modalities} modalities x "
                f"{context_slices} slices), got {n_channels}."
            )
        if int(n_classes) != 3:
            raise ValueError("Exp052 expects three WT/TC/ET output channels.")

        self.adapter = Mamba2_5DResidualAdapter(
            num_modalities=num_modalities,
            context_slices=context_slices,
            context_channels=context_channels,
            reconstruction_channels=reconstruction_channels,
            mamba_layers=mamba_layers,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            scan_orders=scan_orders,
            mamba_residual_scale_init=mamba_residual_scale_init,
            adapter_scale_init=adapter_scale_init,
            zero_init_delta=zero_init_delta,
            mamba_factory=mamba_factory,
        )
        self.backbone = ResNet34RegionHeadsUNet2D(
            n_channels=num_modalities,
            n_classes=n_classes,
            init_features=init_features,
            encoder_weights=encoder_weights,
            require_encoder_weights=require_encoder_weights,
        )

    def enrich_center_slices(self, x, return_delta=False):
        return self.adapter.enrich_center_slices(x, return_delta=return_delta)

    def forward(self, x):
        enriched_center = self.adapter(x)
        return self.backbone(enriched_center)


class HybridHighResolution2_5DMambaGatedRegionHeadsUNet(nn.Module):
    """Bộ thích ứng Mamba 2.5D kích thước 60x60 kết hợp với phần thân Exp043 giữ nguyên."""

    def __init__(
        self,
        n_channels=20,
        n_classes=3,
        init_features=64,
        num_modalities=4,
        context_slices=5,
        context_channels=(8, 32),
        reconstruction_channels=(16, 8),
        gate_hidden_channels=32,
        gate_initial_value=0.1,
        delta_scale=1.0,
        zero_init_delta=True,
        mamba_layers=1,
        d_model=32,
        d_state=16,
        d_conv=4,
        expand=2,
        scan_orders=4,
        mamba_residual_scale_init=0.1,
        pack_scan_orders=False,
        activation_checkpointing=True,
        encoder_weights="imagenet",
        require_encoder_weights=False,
        mamba_factory=None,
    ):
        super().__init__()
        expected_channels = int(num_modalities) * int(context_slices)
        if int(n_channels) != expected_channels:
            raise ValueError(
                f"Expected in_channels={expected_channels} ({num_modalities} modalities x "
                f"{context_slices} slices), got {n_channels}."
            )
        if int(n_classes) != 3:
            raise ValueError("Exp053 expects three WT/TC/ET output channels.")

        self.adapter = HighResolutionGatedMamba2_5DAdapter(
            num_modalities=num_modalities,
            context_slices=context_slices,
            context_channels=context_channels,
            reconstruction_channels=reconstruction_channels,
            gate_hidden_channels=gate_hidden_channels,
            gate_initial_value=gate_initial_value,
            delta_scale=delta_scale,
            zero_init_delta=zero_init_delta,
            mamba_layers=mamba_layers,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            scan_orders=scan_orders,
            mamba_residual_scale_init=mamba_residual_scale_init,
            pack_scan_orders=pack_scan_orders,
            activation_checkpointing=activation_checkpointing,
            mamba_factory=mamba_factory,
        )
        self.backbone = ResNet34RegionHeadsUNet2D(
            n_channels=num_modalities,
            n_classes=n_classes,
            init_features=init_features,
            encoder_weights=encoder_weights,
            require_encoder_weights=require_encoder_weights,
        )

    def enrich_center_slices(self, x, return_details=False):
        return self.adapter.enrich_center_slices(x, return_details=return_details)

    def forward(self, x):
        enriched_center, details = self.adapter.enrich_center_slices(
            x,
            return_details=True,
        )
        logits = self.backbone(enriched_center)
        # Chỉ giữ lại thông tin chẩn đoán nhẹ ở đầu ra huấn luyện. Bản đồ delta chi tiết đầy đủ 
        # vẫn khả dụng thông qua hàm enrich_center_slices(return_details=True).
        return logits, {"modality_gate": details["modality_gate"]}
