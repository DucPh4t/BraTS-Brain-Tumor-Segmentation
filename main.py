import argparse
import yaml
import torch
import sys
import os

from src.data.dataset import get_dataloaders, get_subject_splits
from src.models.unet import UNet2D
from src.models.attention_unet import AttentionUNet2D
from src.models.resnet_unet import (
    ResNet34UNet2D,
    ResNet34RegionHeadsUNet2D,
    ResNet34RegionBranchesUNet2D,
    ResNet34RegionHeadsDeepSupervisionUNet2D,
)
from src.models.multimodal_unet import (
    MultiModalStemUNet2D,
    DisentangledFusionUNet2D,
    DisentangledFusionRegionHeadsUNet2D,
    DisentangledFusionRegionHeadsPresenceUNet2D,
    DisentangledFusionMultiScaleRegionHeadsUNet2D,
    DisentangledFusionMultiScalePresenceUNet2D,
    DisentangledFusionUNet2_5D,
    DisentangledFusionAttentionUNet2D,
)
from src.models.hybrid_mamba_unet import (
    Hybrid2_5DMambaRegionHeadsUNet,
    HybridHighResolution2_5DMambaGatedRegionHeadsUNet,
)
from src.engine.trainer import Trainer
from src.engine.evaluator import Evaluator

import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    parser = argparse.ArgumentParser(description="BraTS MRI Segmentation Entrypoint")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train", help="Mode to run")
    parser.add_argument("--resume_path", type=str, default=None, help="Path to last_checkpoint.pth to resume training")
    parser.add_argument("--stop_epoch", type=int, default=None, help="Stop early at this epoch to prevent Kaggle timeout")
    args = parser.parse_args()

    # Tải cấu hình
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed = int(config.get("training", {}).get("seed", config.get("data", {}).get("split_seed", 42)))
    set_seed(seed)
        
    # --- IN RA CẤU HÌNH ĐỂ KIỂM TRA (ABLATION VERIFICATION) ---
    print("\n" + "="*50)
    print(f"🚀 RUNNING EXPERIMENT: {config.get('exp_name', 'Unknown')}")
    print(f"📝 Description: {config.get('description', '')}")
    print("-" * 50)
    print(f"📦 [DATA]")
    split_label = config["data"].get("split_protocol", config["data"].get("split_type", "sequential"))
    print(f"   - Split Protocol: {split_label}")
    if split_label == "stratified_kfold":
        print(
            f"   - Fold           : {config['data'].get('fold_index', 0)}/"
            f"{int(config['data'].get('num_folds', 5)) - 1}"
        )
    print(f"   - Sampling      : {config['data'].get('sampling', 'random')} (Slices: {config['data'].get('slice_range', 'ALL')})")
    print(f"   - Normalization : {config['data'].get('normalization', 'minmax')}")
    print(f"   - Augmentation  : {config['data'].get('augmentation', False)}")
    if config['data'].get('augmentation_intensity', False):
        print(f"   - Intensity Aug : True")
    print(f"🧠 [MODEL]")
    print(f"   - Architecture  : {config['model'].get('architecture', 'unet2d')}")
    print(f"   - Init Features : {config['model'].get('init_features', 32)}")
    print(f"⚙️  [TRAINING]")
    print(f"   - Optimizer     : {config['training'].get('optimizer', 'adam')}")
    print(f"   - Scheduler     : {config['training'].get('scheduler', 'none')}")
    print(f"   - Loss Function : {config['training'].get('loss', 'dice')}")
    print(f"   - Random Seed   : {seed}")
    accumulation_steps = int(config["training"].get("gradient_accumulation_steps", 1))
    if accumulation_steps > 1:
        effective_batch = int(config["training"]["batch_size"]) * accumulation_steps
        print(f"   - Grad Accum     : {accumulation_steps} (Effective Batch: {effective_batch})")
    if 'loss_params' in config['training']:
        print(f"   - Loss Params   : {config['training']['loss_params']}")
    print("="*50 + "\n")
        
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Khởi tạo mô hình
    init_features = config["model"].get("init_features", 32)
    arch = config["model"].get("architecture", "unet2d")
    # Các checkpoint đầy đủ đã chứa trọng số encoder; quá trình đánh giá (evaluation) không cần tải lại trọng số từ mạng.
    encoder_weights = (
        None if args.mode == "eval" else config["model"].get("encoder_weights", "imagenet")
    )
    require_encoder_weights = bool(
        args.mode == "train" and config["model"].get("require_encoder_weights", False)
    )
    if arch == "attention_unet2d":
        model = AttentionUNet2D(
            n_channels=config["model"].get("in_channels", 4),
            n_classes=config["model"].get("num_classes", 3),
            init_features=init_features
        ).to(device)
        arch_name = "Attention U-Net 2D"
    elif arch == "multimodal_stem_unet2d":
        model = MultiModalStemUNet2D(
            n_channels=config["model"].get("in_channels", 4),
            n_classes=config["model"].get("num_classes", 3),
            init_features=init_features,
            return_features=config["model"].get("return_features", False),
        ).to(device)
        arch_name = "Multi-modal Stem U-Net 2D"
    elif arch == "disentangled_fusion_unet2d":
        model = DisentangledFusionUNet2D(
            n_channels=config["model"].get("in_channels", 4),
            n_classes=config["model"].get("num_classes", 3),
            init_features=init_features,
            return_features=config["model"].get("return_features", False),
        ).to(device)
        arch_name = "Disentangled Fusion U-Net 2D"
    elif arch == "disentangled_fusion_region_heads_unet2d":
        model = DisentangledFusionRegionHeadsUNet2D(
            n_channels=config["model"].get("in_channels", 4),
            n_classes=config["model"].get("num_classes", 3),
            init_features=init_features,
            return_features=config["model"].get("return_features", False),
        ).to(device)
        arch_name = "Disentangled Fusion Region-Heads U-Net 2D"
    elif arch == "disentangled_fusion_region_heads_presence_unet2d":
        model = DisentangledFusionRegionHeadsPresenceUNet2D(
            n_channels=config["model"].get("in_channels", 4),
            n_classes=config["model"].get("num_classes", 3),
            init_features=init_features,
            return_features=config["model"].get("return_features", False),
        ).to(device)
        arch_name = "Disentangled Fusion Region-Heads Presence U-Net 2D"
    elif arch == "disentangled_fusion_multiscale_region_heads_unet2d":
        model = DisentangledFusionMultiScaleRegionHeadsUNet2D(
            n_channels=config["model"].get("in_channels", 4),
            n_classes=config["model"].get("num_classes", 3),
            init_features=init_features,
            return_features=config["model"].get("return_features", False),
        ).to(device)
        arch_name = "Multi-scale Disentangled Fusion Region-Heads U-Net 2D"
    elif arch == "disentangled_fusion_multiscale_presence_unet2d":
        model = DisentangledFusionMultiScalePresenceUNet2D(
            n_channels=config["model"].get("in_channels", 4),
            n_classes=config["model"].get("num_classes", 3),
            init_features=init_features,
            return_features=config["model"].get("return_features", False),
        ).to(device)
        arch_name = "Multi-scale Disentangled Fusion Presence U-Net 2D"
    elif arch == "disentangled_fusion_unet2_5d":
        model = DisentangledFusionUNet2_5D(
            n_channels=config["model"].get("in_channels", 12),
            n_classes=config["model"].get("num_classes", 3),
            init_features=init_features,
            return_features=config["model"].get("return_features", False),
            context_slices=config["data"].get("context_slices", 3),
        ).to(device)
        arch_name = "Disentangled Fusion U-Net 2.5D"
    elif arch == "disentangled_fusion_attention_unet2d":
        model = DisentangledFusionAttentionUNet2D(
            n_channels=config["model"].get("in_channels", 4),
            n_classes=config["model"].get("num_classes", 3),
            init_features=init_features,
            return_features=config["model"].get("return_features", False),
        ).to(device)
        arch_name = "Disentangled Fusion Attention U-Net 2D"
    elif arch == "hybrid_2_5d_mamba_region_heads_unet":
        adapter_cfg = config["model"].get("adapter", {})
        mamba_cfg = config["model"].get("mamba", {})
        model = Hybrid2_5DMambaRegionHeadsUNet(
            n_channels=config["model"].get("in_channels", 20),
            n_classes=config["model"].get("num_classes", 3),
            init_features=init_features,
            num_modalities=config["model"].get("num_modalities", 4),
            context_slices=config["data"].get("context_slices", 5),
            context_channels=adapter_cfg.get("context_channels", [8, 16, 32]),
            reconstruction_channels=adapter_cfg.get(
                "reconstruction_channels", [16, 8, 8]
            ),
            mamba_layers=mamba_cfg.get("num_layers", 1),
            d_model=mamba_cfg.get("d_model", 32),
            d_state=mamba_cfg.get("d_state", 16),
            d_conv=mamba_cfg.get("d_conv", 4),
            expand=mamba_cfg.get("expand", 2),
            scan_orders=mamba_cfg.get("scan_orders", 4),
            mamba_residual_scale_init=mamba_cfg.get("residual_scale_init", 0.1),
            adapter_scale_init=adapter_cfg.get("residual_scale_init", 1.0),
            zero_init_delta=adapter_cfg.get("zero_init_delta", True),
            encoder_weights=encoder_weights,
            require_encoder_weights=require_encoder_weights,
        ).to(device)
        arch_name = "2.5D Mamba Adapter + ResNet34 Region-Heads U-Net"
    elif arch == "highres_gated_2_5d_mamba_region_heads_unet":
        adapter_cfg = config["model"].get("adapter", {})
        mamba_cfg = config["model"].get("mamba", {})
        model = HybridHighResolution2_5DMambaGatedRegionHeadsUNet(
            n_channels=config["model"].get("in_channels", 20),
            n_classes=config["model"].get("num_classes", 3),
            init_features=init_features,
            num_modalities=config["model"].get("num_modalities", 4),
            context_slices=config["data"].get("context_slices", 5),
            context_channels=adapter_cfg.get("context_channels", [8, 32]),
            reconstruction_channels=adapter_cfg.get(
                "reconstruction_channels", [16, 8]
            ),
            gate_hidden_channels=adapter_cfg.get("gate_hidden_channels", 32),
            gate_initial_value=adapter_cfg.get("gate_initial_value", 0.1),
            delta_scale=adapter_cfg.get("delta_scale", 1.0),
            zero_init_delta=adapter_cfg.get("zero_init_delta", True),
            mamba_layers=mamba_cfg.get("num_layers", 1),
            d_model=mamba_cfg.get("d_model", 32),
            d_state=mamba_cfg.get("d_state", 16),
            d_conv=mamba_cfg.get("d_conv", 4),
            expand=mamba_cfg.get("expand", 2),
            scan_orders=mamba_cfg.get("scan_orders", 4),
            mamba_residual_scale_init=mamba_cfg.get("residual_scale_init", 0.1),
            pack_scan_orders=mamba_cfg.get("pack_scan_orders", False),
            activation_checkpointing=mamba_cfg.get("activation_checkpointing", True),
            encoder_weights=encoder_weights,
            require_encoder_weights=require_encoder_weights,
        ).to(device)
        arch_name = "High-Resolution Gated 2.5D Mamba + Exp043 Trunk"
    elif arch == "resnet34_unet2d":
        model = ResNet34UNet2D(
            n_channels=config["model"].get("in_channels", 4),
            n_classes=config["model"].get("num_classes", 3),
            init_features=init_features,
            encoder_weights=encoder_weights,
            require_encoder_weights=require_encoder_weights,
        ).to(device)
        arch_name = "ResNet34 U-Net 2D"
    elif arch == "resnet34_region_heads_unet2d":
        model = ResNet34RegionHeadsUNet2D(
            n_channels=config["model"].get("in_channels", 4),
            n_classes=config["model"].get("num_classes", 3),
            init_features=init_features,
            encoder_weights=encoder_weights,
            require_encoder_weights=require_encoder_weights,
        ).to(device)
        arch_name = "ResNet34 Region-Heads U-Net 2D"
    elif arch == "resnet34_region_branches_unet2d":
        model = ResNet34RegionBranchesUNet2D(
            n_channels=config["model"].get("in_channels", 4),
            n_classes=config["model"].get("num_classes", 3),
            init_features=init_features,
            encoder_weights=encoder_weights,
            branch_channels=config["model"].get("branch_channels", 32),
            require_encoder_weights=require_encoder_weights,
        ).to(device)
        arch_name = "ResNet34 True Region-Branches U-Net 2D"
    elif arch == "resnet34_region_heads_deep_supervision_unet2d":
        model = ResNet34RegionHeadsDeepSupervisionUNet2D(
            n_channels=config["model"].get("in_channels", 4),
            n_classes=config["model"].get("num_classes", 3),
            init_features=init_features,
            encoder_weights=encoder_weights,
            require_encoder_weights=require_encoder_weights,
        ).to(device)
        arch_name = "ResNet34 Region-Heads Deep-Supervision U-Net 2D"
    else:
        model = UNet2D(init_features=init_features).to(device)
        arch_name = "U-Net 2D"
        
    num_params = sum(p.numel() for p in model.parameters())
    print(f"==========================================")
    print(f"[MODEL] Architecture: {arch_name}")
    print(f"[MODEL] Init Features: {init_features}")
    print(f"[MODEL] Total Params: {num_params:,}")
    print(f"==========================================")
    
    # Bộ đánh giá Evaluator tải trực tiếp các ca bệnh và không cần tạo bộ đệm DataLoaders.
    # Giữ nguyên cấu hình đã phân giải để mã băm nguồn gốc (provenance hash) khớp với lúc huấn luyện.
    if args.mode == "train":
        train_loader, val_loader, test_ds, test_subjects = get_dataloaders(config)
        trainer = Trainer(model, config, device, train_loader, val_loader, 
                          resume_path=args.resume_path, stop_epoch=args.stop_epoch)
        trainer.fit()
        
    elif args.mode == "eval":
        train_subjects, val_subjects, test_subjects = get_subject_splits(config)
        evaluator = Evaluator(
            model,
            config,
            device,
            test_subjects,
            val_subjects=val_subjects,
            train_subjects=train_subjects,
        )
        evaluator.run()

if __name__ == "__main__":
    main()
