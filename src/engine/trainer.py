"""
Mô tả: Lớp Trainer quản lý vòng lặp huấn luyện (training loop) và đánh giá trên tập validation của mô hình. Tự động xử lý tối ưu hóa (Optimizer), bộ lập lịch (Scheduler), Early Stopping, lưu trữ Checkpoint, ghi chép nhật ký và tính toán Loss tổng hợp.
Đầu vào:
    model (nn.Module): Kiến trúc mô hình deep learning.
    config (dict): Từ điển cấu hình thí nghiệm (YAML).
    device (torch.device): Thiết bị phần cứng để huấn luyện (CUDA/MPS/CPU).
Đầu ra:
    Thực hiện huấn luyện mô hình, ghi nhật ký history.json và lưu checkpoint mô hình tốt nhất (.pth).
"""

import os
import random
import time
import torch
from tqdm import tqdm
import json
import numpy as np
import nibabel as nib

from src.utils.losses import (
    dice_loss,
    focal_tversky_loss,
    dice_bce_loss,
    weighted_dice_bce_loss,
    region_adaptive_dice_bce_loss,
    et_positive_adaptive_dice_bce_loss,
    dice_focal_loss,
    boundary_consistency_loss,
    hierarchy_consistency_loss,
    modality_contrastive_loss,
)
import torch.nn.functional as F
from src.data.dataset import get_subject_splits
from src.utils.metrics import calc_dice_3d
from src.utils.postprocessing import REGION_NAMES
from src.utils.provenance import build_provenance, config_hash

class Trainer:
    def __init__(self, model, config, device, train_loader, val_loader, resume_path=None, stop_epoch=None):
        self.model = model
        self.config = config
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.resume_path = resume_path
        self.stop_epoch = stop_epoch
        
        train_cfg = self.config["training"]
        
        # Thiết lập Optimizer
        if train_cfg["optimizer"] == "adam":
            from torch.optim import Adam
            self.optimizer = Adam(self.model.parameters(), lr=train_cfg["lr"])
        elif train_cfg["optimizer"] == "adamw":
            from torch.optim import AdamW
            self.optimizer = AdamW(
                self.model.parameters(), 
                lr=train_cfg["lr"], 
                weight_decay=train_cfg.get("weight_decay", 1e-5)
            )
            
        self.epochs = train_cfg["epochs"]
        self.gradient_accumulation_steps = int(train_cfg.get("gradient_accumulation_steps", 1))
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1.")
        actual_effective_batch = int(train_cfg["batch_size"]) * self.gradient_accumulation_steps
        configured_effective_batch = train_cfg.get("effective_batch_size")
        if configured_effective_batch is not None and int(configured_effective_batch) != actual_effective_batch:
            raise ValueError(
                "effective_batch_size does not match batch_size * gradient_accumulation_steps: "
                f"expected {actual_effective_batch}, got {configured_effective_batch}."
            )
        self.effective_batch_size = actual_effective_batch
        
        # Thiết lập đường dẫn đầu ra
        self.exp_name = config["exp_name"]
        self.out_dir  = os.path.join("outputs", self.exp_name)
        os.makedirs(self.out_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.out_dir, "best_model.pth")
        self.best_model_3d_path = os.path.join(self.out_dir, "best_model_3d.pth")
        self.final_model_path = os.path.join(self.out_dir, "final_model.pth")
        
        # Thiết lập Scheduler
        self.scheduler = None
        if "scheduler" in train_cfg:
            if train_cfg["scheduler"] == "plateau":
                from torch.optim.lr_scheduler import ReduceLROnPlateau
                self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)
            elif train_cfg["scheduler"] == "cosine":
                from torch.optim.lr_scheduler import CosineAnnealingLR
                self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
            elif train_cfg["scheduler"] == "polynomial":
                from torch.optim.lr_scheduler import LambdaLR
                power = train_cfg.get("scheduler_power", 0.9)
                self.scheduler = LambdaLR(
                    self.optimizer,
                    lr_lambda=lambda e: (1 - e/self.epochs) ** power
                )
        # Thêm logic load Resume Checkpoint
        self.start_epoch = 0
        self.best_val_dice = 0.0
        self.best_val3d_score = None
        self.history = {"train_loss": [], "val_loss": [], "val_dice": []}
        self.history.update({
            "train_loss_components": [],
            "epoch_seconds": [],
            "peak_gpu_memory_mb": [],
            "optimizer_steps": [],
            "train_modality_gate": [],
            "val_modality_gate": [],
        })
        self.validation_3d_cfg = self.config.get("validation_3d", {})
        train_subjects, val_subjects, test_subjects = get_subject_splits(self.config)
        self.split_subjects = {
            "train": train_subjects,
            "validation": val_subjects,
            "test": test_subjects,
        }
        self.provenance = build_provenance(self.config, self.split_subjects)
        with open(os.path.join(self.out_dir, "run_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(self.provenance, f, ensure_ascii=False, indent=2)
        self.val3d_subjects = []
        if self.validation_3d_cfg.get("enabled", False):
            self.val3d_subjects = list(val_subjects)
            max_subjects = self.validation_3d_cfg.get("max_subjects")
            if max_subjects:
                self.val3d_subjects = self.val3d_subjects[: int(max_subjects)]
            self.history.setdefault("val3d", [])
        
        if self.resume_path and os.path.exists(self.resume_path):
            print(f"\n🔄 RESUMING FROM CHECKPOINT: {self.resume_path}")
            checkpoint = torch.load(self.resume_path, map_location=self.device, weights_only=False)
            stored_split_signature = checkpoint.get("split_sha256")
            if stored_split_signature and stored_split_signature != self.provenance["split_sha256"]:
                raise ValueError(
                    "Resume checkpoint split does not match the current fold/protocol. "
                    "Refusing to mix subjects across runs."
                )
            stored_config_hash = checkpoint.get("config_sha256")
            current_config_hash = config_hash(self.config)
            if stored_config_hash and stored_config_hash != current_config_hash:
                raise ValueError(
                    "Resume checkpoint config does not match the current resolved config. "
                    "Use the same fold, seed, optimizer, loss, and protocol settings."
                )
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            if self.scheduler and checkpoint.get('scheduler_state'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_dice = checkpoint.get('best_val_dice', 0.0)
            self.best_val3d_score = checkpoint.get('best_val3d_score')
            self.history = checkpoint.get('history', {"train_loss": [], "val_loss": [], "val_dice": []})
            for key in (
                "train_loss_components",
                "epoch_seconds",
                "peak_gpu_memory_mb",
                "optimizer_steps",
                "train_modality_gate",
                "val_modality_gate",
            ):
                self.history.setdefault(key, [])
            self._restore_rng_state(checkpoint.get("rng_state"))
            if self.validation_3d_cfg.get("enabled", False):
                self.history.setdefault("val3d", [])
            
            # Khôi phục luôn file best_model.pth cũ (đề phòng Hiệp 2 không phá được kỷ lục)
            import shutil
            old_best_model = os.path.join(os.path.dirname(self.resume_path), "best_model.pth")
            if os.path.exists(old_best_model) and os.path.realpath(old_best_model) != os.path.realpath(self.best_model_path):
                shutil.copy(old_best_model, self.best_model_path)
            old_best_model_3d = os.path.join(os.path.dirname(self.resume_path), "best_model_3d.pth")
            if (
                os.path.exists(old_best_model_3d)
                and os.path.realpath(old_best_model_3d) != os.path.realpath(self.best_model_3d_path)
            ):
                shutil.copy(old_best_model_3d, self.best_model_3d_path)
            for metadata_name in ("best_model_meta.json", "best_model_3d_meta.json"):
                old_metadata = os.path.join(os.path.dirname(self.resume_path), metadata_name)
                new_metadata = os.path.join(self.out_dir, metadata_name)
                if os.path.exists(old_metadata) and os.path.realpath(old_metadata) != os.path.realpath(new_metadata):
                    shutil.copy(old_metadata, new_metadata)
                
            print(f"   [+] Resumed successfully! Starting from Epoch {self.start_epoch+1}")
            print(f"   [+] Best Val Dice so far: {self.best_val_dice:.4f}\n")

    def _capture_rng_state(self):
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state):
        if not state:
            print("[WARN] Resume checkpoint has no RNG state; continuation is not bitwise reproducible.")
            return
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(self._as_cpu_rng_state(state["torch_cpu"]))
        if torch.cuda.is_available() and state.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all([
                self._as_cpu_rng_state(cuda_state)
                for cuda_state in state["torch_cuda"]
            ])

    @staticmethod
    def _as_cpu_rng_state(value):
        if torch.is_tensor(value):
            return value.detach().cpu().to(dtype=torch.uint8)
        return torch.as_tensor(value, dtype=torch.uint8, device="cpu")
            
    def _compute_loss(self, preds, masks, return_components=False):
        loss_type = self.config["training"]["loss"]
        loss_kwargs = dict(self.config["training"].get("loss_params", {}))
        hierarchy_weight = float(loss_kwargs.pop("hierarchy_weight", 0.0))
        boundary_weight = float(loss_kwargs.pop("boundary_weight", 0.0))
        boundary_kernel_size = int(loss_kwargs.pop("boundary_kernel_size", 3))
        boundary_channel = loss_kwargs.pop("boundary_channel", None)

        if loss_type == "bce":
            loss = F.binary_cross_entropy(preds, masks)
        elif loss_type == "focal_tversky":
            loss = focal_tversky_loss(preds, masks, **loss_kwargs)
        elif loss_type == "dice_bce":
            loss = dice_bce_loss(preds, masks, **loss_kwargs)
        elif loss_type == "weighted_dice_bce":
            loss = weighted_dice_bce_loss(preds, masks, **loss_kwargs)
        elif loss_type == "region_adaptive_dice_bce":
            loss = region_adaptive_dice_bce_loss(preds, masks, **loss_kwargs)
        elif loss_type == "et_positive_adaptive_dice_bce":
            loss = et_positive_adaptive_dice_bce_loss(preds, masks, **loss_kwargs)
        elif loss_type == "dice_focal":
            loss = dice_focal_loss(preds, masks, **loss_kwargs)
        else:  # mặc định: dice loss
            loss = dice_loss(preds, masks, **loss_kwargs)

        components = {"segmentation": float(loss.detach().item())}
        if hierarchy_weight > 0:
            hierarchy_raw = hierarchy_consistency_loss(preds)
            hierarchy_weighted = hierarchy_weight * hierarchy_raw
            loss = loss + hierarchy_weighted
            components["hierarchy_raw"] = float(hierarchy_raw.detach().item())
            components["hierarchy_weighted"] = float(hierarchy_weighted.detach().item())
        if boundary_weight > 0:
            boundary_preds, boundary_masks = preds, masks
            if boundary_channel is not None:
                channel_to_idx = {"WT": 0, "TC": 1, "ET": 2}
                if boundary_channel not in channel_to_idx:
                    raise ValueError(
                        f"boundary_channel must be one of {list(channel_to_idx)}, got {boundary_channel}"
                    )
                channel_idx = channel_to_idx[boundary_channel]
                boundary_preds = preds[:, channel_idx:channel_idx + 1]
                boundary_masks = masks[:, channel_idx:channel_idx + 1]
            boundary_raw = boundary_consistency_loss(
                boundary_preds,
                boundary_masks,
                kernel_size=boundary_kernel_size,
            )
            loss = loss + boundary_weight * boundary_raw
            components["boundary_raw"] = float(boundary_raw.detach().item())
            components["boundary_weighted"] = float((boundary_weight * boundary_raw).detach().item())
        components["total"] = float(loss.detach().item())
        return (loss, components) if return_components else loss

    def _compute_deep_supervision_loss(self, aux, masks):
        ds_cfg = self.config["training"].get("deep_supervision", {})
        if not ds_cfg.get("enabled", False):
            return None
        outputs = aux.get("deep_supervision") if isinstance(aux, dict) else None
        if not outputs:
            return None

        weights = ds_cfg.get("weights", [0.5, 0.25, 0.125])
        if len(weights) != len(outputs):
            raise ValueError(
                f"deep_supervision.weights must have {len(outputs)} values, got {len(weights)}"
            )

        total_weight = float(sum(weights))
        if total_weight <= 0:
            raise ValueError("deep_supervision.weights must sum to a positive value")

        ds_loss = 0.0
        for output, weight in zip(outputs, weights):
            aux_masks = F.interpolate(masks, size=output.shape[-2:], mode="nearest")
            aux_probs = torch.sigmoid(output)
            ds_loss = ds_loss + (float(weight) / total_weight) * self._compute_loss(aux_probs, aux_masks)
        return ds_loss

    def _unpack_model_output(self, output):
        if isinstance(output, tuple):
            return output
        return output, {}

    @staticmethod
    def _accumulate_modality_gate(accumulator, aux):
        gate = aux.get("modality_gate") if isinstance(aux, dict) else None
        if gate is None:
            return accumulator
        if gate.ndim != 2:
            raise ValueError(
                f"modality_gate must have shape [B, M], got {tuple(gate.shape)}"
            )
        values = gate.detach().float()
        if accumulator is None:
            return {
                "sum": values.sum(dim=0),
                "min": values.amin(dim=0),
                "max": values.amax(dim=0),
                "count": int(values.shape[0]),
            }
        if accumulator["sum"].shape[0] != values.shape[1]:
            raise ValueError("modality_gate changed modality count within one epoch.")
        accumulator["sum"] = accumulator["sum"] + values.sum(dim=0)
        accumulator["min"] = torch.minimum(accumulator["min"], values.amin(dim=0))
        accumulator["max"] = torch.maximum(accumulator["max"], values.amax(dim=0))
        accumulator["count"] += int(values.shape[0])
        return accumulator

    @staticmethod
    def _summarize_modality_gate(accumulator):
        if accumulator is None:
            return None
        count = max(int(accumulator["count"]), 1)
        means = (accumulator["sum"] / count).cpu().tolist()
        minima = accumulator["min"].cpu().tolist()
        maxima = accumulator["max"].cpu().tolist()
        default_names = ("FLAIR", "T1", "T1ce", "T2")
        names = (
            default_names
            if len(means) == len(default_names)
            else tuple(f"modality_{idx}" for idx in range(len(means)))
        )
        return {
            name: {
                "mean": float(mean),
                "min": float(minimum),
                "max": float(maximum),
            }
            for name, mean, minimum, maximum in zip(names, means, minima, maxima)
        }

    def _compute_contrastive_loss(self, aux):
        contrastive_cfg = self.config["training"].get("contrastive", {})
        if not contrastive_cfg.get("enabled", False):
            return None
        if "modality_features" not in aux:
            return None
        return modality_contrastive_loss(
            aux["modality_features"],
            temperature=contrastive_cfg.get("temperature", 0.07),
        )

    def _compute_et_presence_loss(self, aux, masks):
        presence_cfg = self.config["training"].get("et_presence", {})
        if not presence_cfg.get("enabled", False):
            return None
        if "et_presence_logit" not in aux:
            return None
        et_present = (masks[:, 2].amax(dim=(1, 2)) > 0).float().unsqueeze(1)
        return F.binary_cross_entropy_with_logits(aux["et_presence_logit"], et_present)

    def _should_run_3d_validation(self, epoch):
        if not self.validation_3d_cfg.get("enabled", False):
            return False
        interval = int(self.validation_3d_cfg.get("interval_epochs", 5))
        return (epoch + 1) % interval == 0 or (epoch + 1) == self.epochs

    def _val3d_score(self, metrics):
        monitor = self.validation_3d_cfg.get("monitor", "mean_dice")
        per_subject = metrics["per_subject"]
        if monitor == "mean_dice":
            return float(np.mean([
                np.mean([item["DICE"][region] for region in REGION_NAMES])
                for item in per_subject
            ]))
        if monitor == "et_hd95":
            if not per_subject or "ET" not in per_subject[0].get("HD95", {}):
                raise ValueError("validation_3d.monitor='et_hd95' requires validation_3d.include_hd95=true")
            return -float(np.mean([item["HD95"]["ET"] for item in per_subject]))
        raise ValueError(f"Unknown validation_3d.monitor: {monitor}")

    def _run_3d_validation(self, epoch, metrics=None):
        metrics = metrics or self._evaluate_val_loader_3d(epoch)
        summary = metrics["summary"]
        score = self._val3d_score(metrics)
        record = {
            "epoch": epoch + 1,
            "monitor": self.validation_3d_cfg.get("monitor", "mean_dice"),
            "score": score,
            "score_units": "fraction" if self.validation_3d_cfg.get("monitor", "mean_dice") == "mean_dice" else "negative_mm",
            "dice": summary["DICE"],
            "hd95": summary.get("HD95", {}),
            "num_subjects": metrics["num_subjects"],
        }
        self.history.setdefault("val3d", []).append(record)
        msg = f"   [3D Val] Mean Dice: {summary['DICE']['Mean']:.2f}%"
        if "ET" in summary.get("HD95", {}):
            msg += f" | ET-HD95: {summary['HD95']['ET']:.2f}"
        print(msg)

        if self.best_val3d_score is None or score > self.best_val3d_score:
            self.best_val3d_score = score
            torch.save(self.model.state_dict(), self.best_model_3d_path)
            self._write_checkpoint_metadata(
                "best_model_3d_meta.json",
                epoch=epoch + 1,
                criterion=self.validation_3d_cfg.get("monitor", "mean_dice"),
                score=score,
                summary=summary,
            )
            print(f"   [+] 3D validation improved! Saved best_model_3d.pth")

    def _new_val3d_accumulator(self):
        return {
            "current_sid": None,
            "pred_slices": [],
            "gt_slices": [],
            "per_subject": [],
            "subject_filter": set(self.val3d_subjects),
        }

    def _flush_val3d_subject(self, state):
        if state["current_sid"] is None or not state["pred_slices"]:
            return
        pred_volume = np.stack(state["pred_slices"], axis=0)
        gt_volume = np.stack(state["gt_slices"], axis=0)
        state["per_subject"].append(
            self._score_3d_subject(state["current_sid"], pred_volume, gt_volume)
        )
        state["pred_slices"] = []
        state["gt_slices"] = []

    def _accumulate_val3d_batch(self, state, sample_meta, binary, gt_batch):
        for item_idx, (subject_id, _) in enumerate(sample_meta):
            if state["subject_filter"] and subject_id not in state["subject_filter"]:
                continue
            if state["current_sid"] is None:
                state["current_sid"] = subject_id
            if subject_id != state["current_sid"]:
                self._flush_val3d_subject(state)
                state["current_sid"] = subject_id
            state["pred_slices"].append(binary[item_idx])
            state["gt_slices"].append(gt_batch[item_idx])

    def _finalize_val3d_accumulator(self, state):
        self._flush_val3d_subject(state)
        per_subject = state["per_subject"]
        return {
            "summary": self._summarize_3d_subjects(per_subject),
            "per_subject": per_subject,
            "num_subjects": len(per_subject),
        }

    def _evaluate_val_loader_3d(self, epoch):
        """Truyền các lát cắt validation vào thể tích ảnh (volumes) của ca bệnh mà không cần nạp lại các file NIfTI từ đĩa."""
        thresholds = self.validation_3d_cfg.get(
            "thresholds",
            {"WT": 0.5, "TC": 0.5, "ET": 0.5},
        )
        threshold_tensor = torch.tensor(
            [thresholds[region] for region in REGION_NAMES],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3, 1, 1)
        samples = self.val_loader.dataset.samples

        state = self._new_val3d_accumulator()
        sample_offset = 0

        self.model.eval()
        with torch.no_grad():
            iterator = tqdm(self.val_loader, desc=f"Epoch {epoch + 1} [3D Val]", leave=False)
            for imgs, masks in iterator:
                batch_size = imgs.shape[0]
                sample_meta = samples[sample_offset: sample_offset + batch_size]
                sample_offset += batch_size

                imgs = imgs.to(self.device)
                logits, _ = self._unpack_model_output(self.model(imgs))
                probs = torch.sigmoid(logits)
                binary = (probs > threshold_tensor).cpu().numpy().astype(np.uint8)
                gt_batch = masks.numpy().astype(np.uint8)

                self._accumulate_val3d_batch(state, sample_meta, binary, gt_batch)

        return self._finalize_val3d_accumulator(state)

    def _score_3d_subject(self, subject_id, pred_volume, gt_volume):
        scores = {"subject_id": subject_id, "DICE": {}, "HD95": {}}
        include_hd95 = bool(self.validation_3d_cfg.get("include_hd95", False))
        spacing_zyx = self._subject_spacing_zyx(subject_id)
        for channel, region in enumerate(REGION_NAMES):
            pred = pred_volume[:, channel]
            gt = gt_volume[:, channel]
            scores["DICE"][region] = float(calc_dice_3d(pred, gt))
            if include_hd95:
                from src.utils.metrics import calc_hd95_3d
                scores["HD95"][region] = float(calc_hd95_3d(pred, gt, voxelspacing=spacing_zyx))
        return scores

    def _subject_spacing_zyx(self, subject_id):
        root_dir = self.config["data"]["root_dir"]
        subject_dir = os.path.join(root_dir, subject_id)
        mask_path = os.path.join(subject_dir, f"{subject_id}_seg.nii")
        if not os.path.exists(mask_path):
            candidates = [
                os.path.join(subject_dir, filename)
                for filename in os.listdir(subject_dir)
                if "seg" in filename.lower() and filename.endswith((".nii", ".nii.gz"))
            ]
            if not candidates:
                raise FileNotFoundError(f"Segmentation mask not found for spacing: {subject_id}")
            mask_path = candidates[0]
        spacing_xyz = nib.load(mask_path).header.get_zooms()[:3]
        return (float(spacing_xyz[2]), float(spacing_xyz[0]), float(spacing_xyz[1]))

    def _write_checkpoint_metadata(self, filename, epoch, criterion, score, summary=None):
        metadata = {
            "epoch": int(epoch),
            "criterion": criterion,
            "score": None if score is None else float(score),
            "score_units": (
                "fraction" if criterion == "mean_dice"
                else "negative_mm" if criterion == "et_hd95"
                else None
            ),
            "summary": summary or {},
            "config_sha256": config_hash(self.config),
            "split_sha256": self.provenance["split_sha256"],
        }
        with open(os.path.join(self.out_dir, filename), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def _summarize_3d_subjects(self, per_subject):
        if not per_subject:
            raise ValueError("3D validation produced no subject volumes.")
        summary = {"DICE": {}, "HD95": {}}
        metrics = ("DICE", "HD95") if self.validation_3d_cfg.get("include_hd95", False) else ("DICE",)
        for metric in metrics:
            for region in REGION_NAMES:
                values = np.array([item[metric][region] for item in per_subject], dtype=np.float32)
                multiplier = 100.0 if metric == "DICE" else 1.0
                summary[metric][region] = round(float(values.mean() * multiplier), 2)
                summary[metric][f"{region}_std"] = round(float(values.std() * multiplier), 2)
                summary[metric][f"{region}_median"] = round(float(np.median(values) * multiplier), 2)
        summary["DICE"]["Mean"] = round(float(np.mean([summary["DICE"][r] for r in REGION_NAMES])), 2)
        if summary["HD95"]:
            summary["HD95"]["Mean"] = round(float(np.mean([summary["HD95"][r] for r in REGION_NAMES])), 2)
        return summary

    def fit(self):
        last_completed_epoch = self.start_epoch
        for epoch in range(self.start_epoch, self.epochs):
            epoch_start = time.perf_counter()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            # --- GIAI ĐOẠN HUẤN LUYỆN (TRAIN) ---
            self.model.train()
            train_loss = 0.0
            component_totals = {}
            optimizer_steps = 0
            train_gate_accumulator = None
            num_train_batches = len(self.train_loader)
            self.optimizer.zero_grad(set_to_none=True)
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
            for batch_idx, (imgs, masks) in enumerate(pbar):
                imgs, masks = imgs.to(self.device), masks.to(self.device)

                preds, aux = self._unpack_model_output(self.model(imgs))
                train_gate_accumulator = self._accumulate_modality_gate(
                    train_gate_accumulator,
                    aux,
                )
                probs = torch.sigmoid(preds) # Áp dụng sigmoid trước loss (hàm loss nhận xác suất probs, không nhận logits)
                loss, loss_components = self._compute_loss(probs, masks, return_components=True)
                for key, value in loss_components.items():
                    component_totals[key] = component_totals.get(key, 0.0) + value
                deep_supervision_loss = self._compute_deep_supervision_loss(aux, masks)
                if deep_supervision_loss is not None:
                    weight = self.config["training"]["deep_supervision"].get("weight", 0.3)
                    loss = loss + weight * deep_supervision_loss
                contrastive_loss = self._compute_contrastive_loss(aux)
                if contrastive_loss is not None:
                    weight = self.config["training"]["contrastive"].get("weight", 0.1)
                    loss = loss + weight * contrastive_loss
                et_presence_loss = self._compute_et_presence_loss(aux, masks)
                if et_presence_loss is not None:
                    weight = self.config["training"]["et_presence"].get("weight", 0.1)
                    loss = loss + weight * et_presence_loss

                window_start = (batch_idx // self.gradient_accumulation_steps) * self.gradient_accumulation_steps
                window_size = min(
                    self.gradient_accumulation_steps,
                    num_train_batches - window_start,
                )
                (loss / window_size).backward()

                should_step = (
                    (batch_idx + 1) % self.gradient_accumulation_steps == 0
                    or (batch_idx + 1) == num_train_batches
                )
                if should_step:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                
                train_loss += loss.item()
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                
            avg_train = train_loss / len(self.train_loader)
            avg_components = {
                key: value / len(self.train_loader)
                for key, value in component_totals.items()
            }
            
            # --- GIAI ĐOẠN ĐÁNH GIÁ (VAL) ---
            self.model.eval()
            val_loss = 0.0
            val_dice_wt, val_dice_tc, val_dice_et = [], [], []
            val_gate_accumulator = None
            run_3d_validation = self._should_run_3d_validation(epoch)
            val3d_state = self._new_val3d_accumulator() if run_3d_validation else None
            val_sample_offset = 0
            if run_3d_validation:
                val3d_thresholds = self.validation_3d_cfg.get(
                    "thresholds", {"WT": 0.5, "TC": 0.5, "ET": 0.5}
                )
                val3d_threshold_tensor = torch.tensor(
                    [val3d_thresholds[region] for region in REGION_NAMES],
                    dtype=torch.float32,
                    device=self.device,
                ).view(1, 3, 1, 1)

            with torch.no_grad():
                for imgs, masks in self.val_loader:
                    batch_size = imgs.shape[0]
                    sample_meta = self.val_loader.dataset.samples[
                        val_sample_offset: val_sample_offset + batch_size
                    ]
                    val_sample_offset += batch_size
                    imgs, masks = imgs.to(self.device), masks.to(self.device)
                    preds, aux = self._unpack_model_output(self.model(imgs))
                    val_gate_accumulator = self._accumulate_modality_gate(
                        val_gate_accumulator,
                        aux,
                    )
                    probs = torch.sigmoid(preds) # Áp dụng sigmoid trước loss
                    loss  = self._compute_loss(probs, masks)
                    deep_supervision_loss = self._compute_deep_supervision_loss(aux, masks)
                    if deep_supervision_loss is not None:
                        weight = self.config["training"]["deep_supervision"].get("weight", 0.3)
                        loss = loss + weight * deep_supervision_loss
                    contrastive_loss = self._compute_contrastive_loss(aux)
                    if contrastive_loss is not None:
                        weight = self.config["training"]["contrastive"].get("weight", 0.1)
                        loss = loss + weight * contrastive_loss
                    et_presence_loss = self._compute_et_presence_loss(aux, masks)
                    if et_presence_loss is not None:
                        weight = self.config["training"]["et_presence"].get("weight", 0.1)
                        loss = loss + weight * et_presence_loss
                    val_loss += loss.item()
                    
                    # Tính toán Dice ở mức lát cắt (slice-level) để theo dõi tiến độ — không phải đánh giá 3D volume
                    # Đây chỉ dùng để lưu mô hình tốt nhất tạm thời, không phải chỉ số đánh giá cuối cùng
                    binary = (probs > 0.5).float()
                    for i, dice_list in enumerate([val_dice_wt, val_dice_tc, val_dice_et]):
                        inter = (binary[:, i] * masks[:, i]).sum()
                        denom = binary[:, i].sum() + masks[:, i].sum()
                        dice  = (2 * inter / denom).item() if denom > 0 else 1.0
                        dice_list.append(dice)
                    if run_3d_validation:
                        binary_3d = (probs > val3d_threshold_tensor).cpu().numpy().astype(np.uint8)
                        self._accumulate_val3d_batch(
                            val3d_state,
                            sample_meta,
                            binary_3d,
                            masks.cpu().numpy().astype(np.uint8),
                        )
                        
            avg_val   = val_loss / len(self.val_loader)
            mean_dice = (np.mean(val_dice_wt) + np.mean(val_dice_tc) + np.mean(val_dice_et)) / 3

            train_gate_summary = self._summarize_modality_gate(train_gate_accumulator)
            val_gate_summary = self._summarize_modality_gate(val_gate_accumulator)

            print(f"-> Epoch {epoch+1:02d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Mean Dice: {mean_dice:.4f}")
            if val_gate_summary is not None:
                gate_means = ", ".join(
                    f"{name}={stats['mean']:.3f}"
                    for name, stats in val_gate_summary.items()
                )
                print(f"   [GATE] Validation mean reliability: {gate_means}")
            
            # Cập nhật Scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(mean_dice)
                else:
                    self.scheduler.step()
            
            self.history["train_loss"].append(avg_train)
            self.history["val_loss"].append(avg_val)
            self.history["val_dice"].append(mean_dice)
            self.history["train_modality_gate"].append(train_gate_summary)
            self.history["val_modality_gate"].append(val_gate_summary)
            
            # Lưu mô hình tốt nhất (best model) dựa trên chỉ số Mean Validation Dice, thay vì Validation Loss
            if mean_dice > self.best_val_dice:
                self.best_val_dice = mean_dice
                torch.save(self.model.state_dict(), self.best_model_path)
                self._write_checkpoint_metadata(
                    "best_model_meta.json",
                    epoch=epoch + 1,
                    criterion="slice_batch_mean_dice",
                    score=mean_dice,
                )
                print(f"   [+] Mean Dice improved to {mean_dice:.4f}! Saved best_model.pth")

            if run_3d_validation:
                self._run_3d_validation(
                    epoch,
                    metrics=self._finalize_val3d_accumulator(val3d_state),
                )

            epoch_seconds = time.perf_counter() - epoch_start
            peak_memory_mb = (
                torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                if self.device.type == "cuda"
                else None
            )
            self.history["train_loss_components"].append(avg_components)
            self.history["epoch_seconds"].append(epoch_seconds)
            self.history["peak_gpu_memory_mb"].append(peak_memory_mb)
            self.history["optimizer_steps"].append(optimizer_steps)
                
            # Lưu last_checkpoint sau mỗi epoch để có thể khôi phục tiến trình (tránh lỗi timeout trên Colab/Kaggle)
            checkpoint_state = {
                'epoch': epoch,
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
                'best_val_dice': self.best_val_dice,
                'best_val3d_score': self.best_val3d_score,
                'history': self.history,
                'config_sha256': config_hash(self.config),
                'split_sha256': self.provenance["split_sha256"],
                'rng_state': self._capture_rng_state(),
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'effective_batch_size': self.effective_batch_size,
            }
            torch.save(checkpoint_state, os.path.join(self.out_dir, "last_checkpoint.pth"))
            last_completed_epoch = epoch + 1
            
            # Kiểm tra Stop Epoch an toàn
            if self.stop_epoch and (epoch + 1) == self.stop_epoch:
                print(f"\n✋ CHỦ ĐỘNG DỪNG SỚM TẠI EPOCH {epoch+1} (Để tránh Kaggle Timeout).")
                print("   Dùng cờ --resume_path cho lần chạy sau để tiếp tục!")
                break
            
        # Lưu lịch sử huấn luyện
        with open(os.path.join(self.out_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=4)

        if last_completed_epoch >= self.epochs:
            torch.save(self.model.state_dict(), self.final_model_path)
            self._write_checkpoint_metadata(
                "final_model_meta.json",
                epoch=self.epochs,
                criterion=self.config["training"].get(
                    "final_checkpoint_criterion", "completed_final_epoch"
                ),
                score=None,
            )
            print("   [+] Saved final_model.pth at the completed final epoch.")
        
        print("\nTraining complete!")
