"""
Mô tả: Chạy đánh giá chéo (cross-dataset evaluation) của mô hình đã huấn luyện trên bộ dữ liệu ngoài BraTS 2023 GLI. Tự động xử lý ánh xạ nhãn ET (3 -> 4) để tương thích.
Đầu vào:
    --checkpoint: Đường dẫn đến file trọng số .pth của mô hình.
    --config: Đường dẫn đến file cấu hình YAML tương ứng.
    --data-root: Đường dẫn đến thư mục chứa dữ liệu BraTS 2023 GLI.
Đầu ra:
    In ra màn hình kết quả trung bình 3D Dice và khoảng cách Hausdorff HD95. Lưu báo cáo chi tiết và file metrics JSON.
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
import argparse

# Thêm project root vào PATH
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.engine.evaluator import Evaluator
from src.data.processors import get_preprocessor

class BraTS2023Evaluator(Evaluator):
    """
    Evaluator được tùy biến để chạy trên tập BraTS 2023 GLI.
    Ghi đè _load_subject để chuyển đổi modalities và ánh xạ nhãn ET (3 -> 4) cho tương thích.
    """
    def __init__(self, model, config, device, test_subjects, val_subjects=None, overlap_audit=None):
        super().__init__(model, config, device, test_subjects, val_subjects)
        self.overlap_audit = overlap_audit or {}
        # Thay đổi thư mục output để tránh đè kết quả của tập 2020
        self.out_dir = os.path.join("outputs", f"{self.exp_name}_eval_brats2023")
        os.makedirs(self.out_dir, exist_ok=True)
        # Đảm bảo best_model_path vẫn lấy từ thư mục gốc exp043
        self.best_model_path = self.eval_cfg.get(
            "checkpoint_path",
            os.path.join("outputs", self.exp_name, "best_model.pth"),
        )

    def run(self):
        checkpoint = self.best_model_path
        
        # Load weights
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True))
        self.model.eval()
        self.model.to(self.device)
        
        num_params = sum(p.numel() for p in self.model.parameters())
        
        print("\n" + "="*60)
        print("📥 [LOG] THÔNG TIN CHẠY ĐÁNH GIÁ (EVALUATION LOGS)")
        print("="*60)
        print(f"🧠 1. Model & Checkpoint:")
        print(f"   - Kiến trúc: {self.config['model'].get('architecture', 'N/A')}")
        print(f"   - Số lượng tham số (Params): {num_params:,}")
        print(f"   - Trọng số tải từ: {checkpoint}")
        print(f"   - Thiết bị chạy (Device): {self.device}")
        print("-" * 60)
        print(f"📂 2. Tập dữ liệu & Số lượng ca bệnh:")
        print(f"   - Thư mục Dataset: {self.config['data']['root_dir']}")
        print(f"   - Số lượng ca bệnh (Subjects): {len(self.test_subjects)} ca")
        print("="*60 + "\n")
        
        thresholds = self._load_or_default_thresholds()
        postprocess_cfg = self._base_postprocess_config()
        et_rescue_cfg = self._base_et_rescue_config()
        
        final_metrics = self._evaluate_subjects(
            self.test_subjects,
            thresholds,
            postprocess_cfg,
            et_rescue_cfg,
            desc="Đang xử lý các ca bệnh",
        )
        
        split_name = "external_nonoverlap"
        self._print_report(final_metrics, split_name)
        self._save_results(
            final_metrics,
            split_name,
            thresholds,
            postprocess_cfg,
            et_rescue_cfg,
            None,
            None,
            None
        )

    def _load_subject(self, subject_id):
        # Đường dẫn thư mục chứa ca bệnh
        data_dir = os.path.join(self.config["data"]["root_dir"], subject_id)
        
        # Ánh xạ từ Modality BraTS2020 sang hậu tố file BraTS2023 (không đuôi mở rộng)
        mod_suffixes = {
            "flair": "-t2f",
            "t1": "-t1n",
            "t1ce": "-t1c",
            "t2": "-t2w"
        }
        
        volumes = []
        for mod in ["flair", "t1", "t1ce", "t2"]:
            suffix = mod_suffixes[mod]
            path = None
            for f in os.listdir(data_dir):
                if f.startswith(subject_id + suffix) and (f.endswith('.nii') or f.endswith('.nii.gz')):
                    path = os.path.join(data_dir, f)
                    break
            if path is None:
                raise FileNotFoundError(f"Modality {mod} ({suffix}) không tìm thấy cho {subject_id} tại {data_dir}")
            
            vol_3d = nib.load(path).get_fdata()
            volumes.append(self.preprocessor(vol_3d, modality=mod))
            
        stack_4d = np.stack(volumes, axis=0)
        # Chuyển đổi shape từ (C, H, W, Z) thành (Z, C, H, W)
        stack_4d = np.transpose(stack_4d, (3, 0, 1, 2)).astype(np.float32)
        stack_4d = self._add_context_slices(stack_4d)
        
        # Tải nhãn phân vùng (Segmentation mask)
        mask_path = None
        for f in os.listdir(data_dir):
            if f.startswith(subject_id + "-seg") and (f.endswith('.nii') or f.endswith('.nii.gz')):
                mask_path = os.path.join(data_dir, f)
                break
        if mask_path is None:
            raise FileNotFoundError(f"Segmentation mask không tìm thấy cho {subject_id} tại {data_dir}")
                
        mask_img = nib.load(mask_path)
        mask_3d = np.transpose(mask_img.get_fdata(), (2, 0, 1))
        spacing = mask_img.header.get_zooms()[:3]
        spacing_zyx = (spacing[2], spacing[0], spacing[1])
        
        # Ánh xạ nhãn BraTS 2023: 1 = NCR, 2 = ED, 3 = ET
        # Biến đổi thành WT (Whole Tumor), TC (Tumor Core), ET (Enhancing Tumor)
        gt_volume = np.stack(
            [
                (mask_3d > 0).astype(np.float32),                               # WT
                np.logical_or(mask_3d == 1, mask_3d == 3).astype(np.float32),   # TC (NCR + ET)
                (mask_3d == 3).astype(np.float32),                              # ET
            ],
            axis=0,
        )
        
        return stack_4d, gt_volume, spacing_zyx

def get_model_architecture(config, device):
    from src.models.unet import UNet2D
    from src.models.attention_unet import AttentionUNet2D
    from src.models.resnet_unet import (
        ResNet34UNet2D,
        ResNet34RegionHeadsUNet2D,
        ResNet34RegionBranchesUNet2D,
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
    
    init_features = config["model"].get("init_features", 32)
    arch = config["model"].get("architecture", "unet2d")
    
    if arch == "attention_unet2d":
        model = AttentionUNet2D(
            n_channels=config["model"].get("in_channels", 4),
            n_classes=config["model"].get("num_classes", 3),
            init_features=init_features
        )
    elif arch == "multimodal_stem_unet2d":
        model = MultiModalStemUNet2D(
            n_channels=config["model"].get("in_channels", 4),
            n_classes=config["model"].get("num_classes", 3),
            init_features=init_features,
            return_features=config["model"].get("return_features", False),
        )
    elif arch == "disentangled_fusion_unet2d":
        model = DisentangledFusionUNet2D(
            n_channels=config["model"].get("in_channels", 4),
            n_classes=config["model"].get("num_classes", 3),
            init_features=init_features,
            return_features=config["model"].get("return_features", False),
        )
    elif arch == "disentangled_fusion_region_heads_unet2d":
        model = DisentangledFusionRegionHeadsUNet2D(
            n_channels=config["model"].get("in_channels", 4),
            n_classes=config["model"].get("num_classes", 3),
            init_features=init_features,
            return_features=config["model"].get("return_features", False),
        )
    elif arch == "disentangled_fusion_region_heads_presence_unet2d":
        model = DisentangledFusionRegionHeadsPresenceUNet2D(
            n_channels=config["model"].get("in_channels", 4),
            n_classes=config["model"].get("num_classes", 3),
            init_features=init_features,
            return_features=config["model"].get("return_features", False),
        )
    elif arch == "disentangled_fusion_multiscale_region_heads_unet2d":
        model = DisentangledFusionMultiScaleRegionHeadsUNet2D(
            n_channels=config["model"].get("in_channels", 4),
            n_classes=config["model"].get("num_classes", 3),
            init_features=init_features,
            return_features=config["model"].get("return_features", False),
        )
    elif arch == "disentangled_fusion_multiscale_presence_unet2d":
        model = DisentangledFusionMultiScalePresenceUNet2D(
            n_channels=config["model"].get("in_channels", 4),
            n_classes=config["model"].get("num_classes", 3),
            init_features=init_features,
            return_features=config["model"].get("return_features", False),
        )
    elif arch == "disentangled_fusion_unet2_5d":
        model = DisentangledFusionUNet2_5D(
            n_channels=config["model"].get("in_channels", 12),
            n_classes=config["model"].get("num_classes", 3),
            init_features=init_features,
            return_features=config["model"].get("return_features", False),
            context_slices=config["data"].get("context_slices", 3),
        )
    elif arch == "disentangled_fusion_attention_unet2d":
        model = DisentangledFusionAttentionUNet2D(
            n_channels=config["model"].get("in_channels", 4),
            n_classes=config["model"].get("num_classes", 3),
            init_features=init_features,
            return_features=config["model"].get("return_features", False),
        )
    elif arch == "resnet34_unet2d":
        model = ResNet34UNet2D(
            n_channels=config["model"].get("in_channels", 4),
            n_classes=config["model"].get("num_classes", 3),
            init_features=init_features,
            encoder_weights=None,
        )
    elif arch == "resnet34_region_heads_unet2d":
        model = ResNet34RegionHeadsUNet2D(
            n_channels=config["model"].get("in_channels", 4),
            n_classes=config["model"].get("num_classes", 3),
            init_features=init_features,
            encoder_weights=None,
        )
    elif arch == "resnet34_region_branches_unet2d":
        model = ResNet34RegionBranchesUNet2D(
            n_channels=config["model"].get("in_channels", 4),
            n_classes=config["model"].get("num_classes", 3),
            init_features=init_features,
            encoder_weights=None,
            branch_channels=config["model"].get("branch_channels", 32),
        )
    else:
        model = UNet2D(init_features=init_features)
        
    return model.to(device)

def get_completed_subjects(root_dir):
    """
    Lấy danh sách các bệnh nhân đã được tải hoàn tất (đủ cả 5 modalities + mask)
    Đề phòng trường hợp script tải vẫn đang chạy ngầm.
    """
    completed = []
    if not os.path.exists(root_dir):
        return completed
        
    required_suffixes = ["-t2f", "-t1n", "-t1c", "-t2w", "-seg"]
    for sid in sorted(os.listdir(root_dir)):
        sid_dir = os.path.join(root_dir, sid)
        if os.path.isdir(sid_dir) and sid.startswith("BraTS-GLI-"):
            # Kiểm tra xem folder có đủ 5 file không
            files = os.listdir(sid_dir)
            has_all = True
            for suffix in required_suffixes:
                if not any(f.startswith(sid + suffix) and (f.endswith('.nii') or f.endswith('.nii.gz')) for f in files):
                    has_all = False
                    break
            if has_all:
                completed.append(sid)
    return completed

def main():
    parser = argparse.ArgumentParser(description="Evaluate BraTS 2020 Model on BraTS 2023 GLI Dataset")
    parser.add_argument("--config", type=str, required=True, help="Đường dẫn đến file config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Đường dẫn đến model checkpoint .pth (nếu không khai báo sẽ tự lấy best_model của config)")
    parser.add_argument("--max_subjects", type=int, default=None, help="Giới hạn số lượng bệnh nhân tối đa để test nhanh")
    parser.add_argument("--dataset_root", type=str, default=None, help="Đường dẫn đến thư mục chứa ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
    parser.add_argument("--overlap-audit", type=str, default=None, help="JSON produced by audit_brats_overlap.py")
    parser.add_argument("--require-clean-audit", action="store_true", help="Refuse manuscript-style evaluation unless audit marks the target eligible")
    args = parser.parse_args()
    
    # Đọc config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    locked_external_refit = str(config.get("exp_name", "")).startswith("cmig_external_refit_")
    if locked_external_refit and not args.overlap_audit:
        raise SystemExit(
            "CMIG external refit evaluation requires --overlap-audit; unaudited BraTS2023 "
            "subjects are not allowed."
        )
        
    # Thiết lập thư mục dữ liệu BraTS 2023
    if args.dataset_root:
        brats23_root = args.dataset_root
    else:
        brats23_root = "/Users/nguyenducphat/Projects/ĐATN MRI/MRI dataset/BraTS2023_GLI/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    config["data"]["root_dir"] = brats23_root
    config["data"]["cache_volumes"] = False # Không cache tránh OOM
    config["data"]["dataset_name"] = "BraTS2023 GLI"
    config["data"]["dataset_version"] = "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    config["data"]["dataset_source"] = "BraTS/Synapse; local mirror must be recorded"
    
    # Checkpoint
    if args.checkpoint:
        if "evaluation" not in config:
            config["evaluation"] = {}
        config["evaluation"]["checkpoint_path"] = args.checkpoint
        
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device sử dụng: {device}")
    
    # Lấy danh sách bệnh nhân đã tải xong
    completed_subjects = get_completed_subjects(brats23_root)
    print(f"Tìm thấy {len(completed_subjects)} ca bệnh BraTS 2023 đã tải hoàn tất.")

    overlap_audit = None
    if args.overlap_audit:
        with open(args.overlap_audit, "r", encoding="utf-8") as f:
            overlap_audit = json.load(f)
        audited_ids = set(overlap_audit.get("target", {}).get("audited_subject_ids", []))
        candidate_nonoverlap = set(overlap_audit.get("candidate_nonoverlap_subjects", []))
        if not audited_ids:
            raise SystemExit("Overlap audit does not contain an audited BraTS2023 subject list.")
        completed_subjects = [
            sid for sid in completed_subjects
            if sid in audited_ids and sid in candidate_nonoverlap
        ]
        excluded_count = len(audited_ids - candidate_nonoverlap)
        config.setdefault("evaluation", {})["external_overlap_audit"] = {
            "path": args.overlap_audit,
            "independence_status": overlap_audit.get("independence_status"),
            "manuscript_eligible": bool(overlap_audit.get("manuscript_eligible", False)),
            "audit_complete": bool(overlap_audit.get("audit_complete", False)),
            "audited_subjects": len(audited_ids),
            "excluded_or_uncertain_subjects": excluded_count,
            "candidate_nonoverlap_subjects": len(completed_subjects),
        }
        print(
            f"Sau overlap audit còn {len(completed_subjects)} candidate non-overlap; "
            f"đã loại/giữ lại để review {excluded_count} ca."
        )
    if (args.require_clean_audit or locked_external_refit) and not (
        overlap_audit and overlap_audit.get("manuscript_eligible", False)
    ):
        raise SystemExit(
            "External evaluation blocked: overlap audit is missing or has not proven manuscript eligibility."
        )
    
    if len(completed_subjects) == 0:
        print("❌ Chưa có ca bệnh nào tải hoàn tất trong thư mục:")
        print(brats23_root)
        print("Vui lòng đợi script tải chạy thêm một lúc rồi thử lại.")
        sys.exit(1)
        
    # Giới hạn số lượng test nếu có cấu hình max_subjects
    if args.max_subjects is not None:
        completed_subjects = completed_subjects[:args.max_subjects]
        print(f"🔍 Chỉ đánh giá trên {args.max_subjects} ca đầu tiên theo yêu cầu.")
        
    # Khởi tạo model
    print("Initializing model...")
    model = get_model_architecture(config, device)
    
    # Khởi tạo custom Evaluator
    evaluator = BraTS2023Evaluator(
        model=model,
        config=config,
        device=device,
        test_subjects=completed_subjects,
        val_subjects=None,
        overlap_audit=overlap_audit,
    )
    
    # Chạy đánh giá
    # Kết quả sẽ được in ra console và lưu lại
    evaluator.run()

if __name__ == "__main__":
    main()
