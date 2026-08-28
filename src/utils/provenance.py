"""
Mô tả: Lưu giữ thông tin lịch sử chạy thí nghiệm (provenance logging), lưu các tham số hệ thống, thông tin GPU, mã băm cấu hình (SHA256 hash) để phục vụ tính tái lặp (reproducibility) khoa học.
Đầu vào:
    config (dict): Từ điển lưu cấu hình thí nghiệm.
Đầu ra:
    Trả về chuỗi JSON chứa đầy đủ thông số môi trường phần cứng và phần mềm chạy mô hình.
"""

import hashlib
import json
import platform
import sys
from importlib import metadata

import numpy as np
import torch


def config_hash(config):
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def subject_split_signature(split_subjects):
    normalized = {
        key: sorted(str(subject_id) for subject_id in values)
        for key, values in split_subjects.items()
    }
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _package_version(package_name):
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def software_versions():
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "numpy": np.__version__,
        "torchvision": _package_version("torchvision"),
        "nibabel": _package_version("nibabel"),
        "medpy": _package_version("MedPy"),
        "scipy": _package_version("scipy"),
        "pyyaml": _package_version("PyYAML"),
    }


def build_provenance(config, split_subjects, checkpoint_metadata=None):
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    normalized_splits = {
        key: sorted(str(subject_id) for subject_id in values)
        for key, values in split_subjects.items()
    }
    return {
        "dataset": {
            "name": data_cfg.get("dataset_name", "BraTS2020"),
            "version": data_cfg.get("dataset_version", "BraTS2020 TrainingData"),
            "source": data_cfg.get("dataset_source", "BraTS/CBICA"),
        },
        "protocol": {
            "split_protocol": data_cfg.get("split_protocol", data_cfg.get("split_type", "sequential")),
            "num_folds": data_cfg.get("num_folds"),
            "fold_index": data_cfg.get("fold_index"),
            "inner_val_fraction": data_cfg.get("inner_val_fraction"),
            "split_seed": int(data_cfg.get("split_seed", 42)),
            "training_seed": int(train_cfg.get("seed", data_cfg.get("split_seed", 42))),
            "checkpoint_criterion": (
                config.get("validation_3d", {}).get("monitor")
                if config.get("validation_3d", {}).get("enabled", False)
                else train_cfg.get("monitor_metric", "mean_val_dice")
            ),
        },
        "config_sha256": config_hash(config),
        "split_sha256": subject_split_signature(normalized_splits),
        "subjects": normalized_splits,
        "counts": {key: len(values) for key, values in normalized_splits.items()},
        "checkpoint": checkpoint_metadata or {},
        "software": software_versions(),
    }
