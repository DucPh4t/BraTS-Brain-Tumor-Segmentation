"""
Mô tả: Quản lý dataset BraTS 2020, thực hiện cắt lát ảnh, lấy mẫu theo chiến lược (fixed, weighted, oversample) và chuẩn hóa dữ liệu.
Đầu vào:
    root_dir (str): Thư mục chứa các file NIfTI (.nii, .nii.gz) của BraTS 2020.
    subject_ids (list): Danh sách các ca bệnh.
    normalization (str): Phương thức chuẩn hóa ảnh.
    augmentation (bool): Có tăng cường dữ liệu hay không.
Đầu ra:
    Trả về batch tensor (images, labels) cho quá trình huấn luyện và đánh giá.
"""
import os
import random
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import csv

from src.data.processors import get_preprocessor


def normalize_context_slices(value):
    """Hỗ trợ các cấu hình cũ (0 tương ứng với single-slice 2D) đồng thời loại bỏ các cửa sổ chẵn không đối xứng."""
    context_slices = 1 if value is None or int(value) == 0 else int(value)
    if context_slices < 1 or context_slices % 2 == 0:
        raise ValueError(
            f"context_slices must be 0/1 for 2D or a positive odd number for 2.5D, got {value}."
        )
    return context_slices

class BraTSDataset(Dataset):
    """
    Lớp Dataset hợp nhất quản lý việc đọc dữ liệu và các chiến lược chuẩn hóa.
    Hỗ trợ các chế độ lấy mẫu: cố định (fixed), ngẫu nhiên (random), và lấy mẫu quanh tâm khối u (tumor-center-aware).
    Sử dụng bộ đệm thể tích ảnh trong bộ nhớ (cache_volumes) để loại bỏ nghẽn cổ chai đọc/ghi I/O (có thể tắt để tránh tràn RAM).
    """
    def __init__(self, root_dir, subject_ids, slice_range=(0, 155),
                 normalization="zscore_volume", augmentation=False,
                 augmentation_intensity=False,
                 sampling="fixed", context_slices=1, min_tumor_pixels=100,
                 cache_volumes=True, preprocess_config=None):
        self.root_dir = root_dir
        self.subject_ids = subject_ids
        self.slice_range = slice_range
        self.normalization = normalization
        self.augmentation = augmentation
        self.augmentation_intensity = augmentation_intensity
        self.sampling = sampling
        self.context_slices = normalize_context_slices(context_slices)
        self.context_radius = (self.context_slices - 1) // 2
        self.min_tumor_pixels = min_tumor_pixels
        self.cache_volumes = cache_volumes

        self.preprocessor = get_preprocessor(self.normalization, preprocess_config or {})

        # ── Tải trước toàn bộ các volume vào RAM (Pre-load) ──────────────────
        # Định dạng key: sid → {"flair": np.ndarray, "t1": ..., "t1ce": ..., "t2": ..., "seg": ...}
        self._cache = {}
        if self.cache_volumes:
            from tqdm import tqdm
            print(f"  [CACHE] Pre-loading {len(subject_ids)} subjects into RAM...")
            for sid in tqdm(subject_ids, desc="  Caching volumes", leave=False):
                self._cache[sid] = self._load_subject(sid)
        # ─────────────────────────────────────────────────────────────────────

        self.samples = []
        self.sample_weights = []  # luôn khởi tạo, dù sampling nào

        for sid in self.subject_ids:
            if self.sampling == "random":
                random_slice = np.random.randint(0, 155)
                self.samples.append((sid, random_slice))

            elif self.sampling == "weighted":
                seg_vol = self._get_seg_for_sampling(sid)
                
                sl = self.slice_range
                for slice_idx in range(sl[0], sl[1]):
                    self.samples.append((sid, slice_idx))
                    has_tumor = (seg_vol[:, :, slice_idx] > 0).sum() > 0
                    self.sample_weights.append(3.0 if has_tumor else 1.0)

            elif self.sampling == "oversample":
                seg_vol = self._get_seg_for_sampling(sid)
                
                sl = self.slice_range
                for slice_idx in range(sl[0], sl[1]):
                    self.samples.append((sid, slice_idx))
                    # Nhân đôi lát cắt nếu có chứa u (oversampling)
                    has_tumor = (seg_vol[:, :, slice_idx] > 0).sum() > 0
                    if has_tumor:
                        self.samples.append((sid, slice_idx))

            else:  # mặc định: lấy mẫu cố định "fixed"
                sl = self.slice_range
                for slice_idx in range(sl[0], sl[1]):
                    self.samples.append((sid, slice_idx))

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _find_seg_path(self, sid):
        """Tìm đường dẫn file mặt nạ phân đoạn cho ca bệnh."""
        default = os.path.join(self.root_dir, sid, f"{sid}_seg.nii")
        if os.path.exists(default):
            return default
        subdir = os.path.join(self.root_dir, sid)
        for f in os.listdir(subdir):
            if 'seg' in f.lower() and f.endswith('.nii'):
                return os.path.join(subdir, f)
        raise FileNotFoundError(f"Seg mask not found for {sid}")

    def _load_subject(self, sid):
        """Tải và chuẩn hóa tất cả 4 xung phương thái + mặt nạ phân đoạn cho một ca bệnh vào bộ nhớ RAM."""
        subdir = os.path.join(self.root_dir, sid)
        entry = {}
        for mod in ['flair', 't1', 't1ce', 't2']:
            path = os.path.join(subdir, f"{sid}_{mod}.nii")
            vol_3d = nib.load(path).get_fdata()
            # Ép kiểu float16 để cứu dung lượng RAM trên Kaggle (30GB)
            entry[mod] = self.preprocessor(vol_3d, modality=mod).astype(np.float16)
        seg_path = self._find_seg_path(sid)
        # Ép kiểu uint8 cho nhãn (0, 1, 2, 4) chỉ tốn 1 byte
        entry['seg'] = nib.load(seg_path).get_fdata().astype(np.uint8)
        return entry

    def _get_seg_for_sampling(self, sid):
        if self.cache_volumes:
            return self._cache[sid]["seg"]
        seg_path = self._find_seg_path(sid)
        return nib.load(seg_path).get_fdata().astype(np.uint8)
    # ─────────────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, slice_idx = self.samples[idx]

        # ── Nhánh nhanh: Lấy lát cắt từ bộ nhớ cache RAM ──────────────────────
        if self.cache_volumes:
            cached = self._cache[sid]
            imgs = []
            for mod in ['flair', 't1', 't1ce', 't2']:
                for ctx_idx in self._context_indices(slice_idx):
                    imgs.append(cached[mod][:, :, ctx_idx])
            mask  = cached['seg'][:, :, slice_idx]
        else:
            # ── Nhánh chậm: Đọc trực tiếp từ ổ đĩa (cho máy cấu hình RAM thấp) ──
            subdir = os.path.join(self.root_dir, sid)
            imgs = []
            for mod in ['flair', 't1', 't1ce', 't2']:
                path   = os.path.join(subdir, f"{sid}_{mod}.nii")
                vol_3d = nib.load(path).get_fdata()
                vol_norm = self.preprocessor(vol_3d, modality=mod)
                for ctx_idx in self._context_indices(slice_idx):
                    imgs.append(vol_norm[:, :, ctx_idx])
            mask = nib.load(self._find_seg_path(sid)).get_fdata()[:, :, slice_idx]

        stack = np.stack(imgs, axis=0).astype(np.float32)

        # Mã hóa mặt nạ phân đoạn đa lớp
        wt = (mask > 0).astype(np.float32)
        tc = np.logical_or(mask == 1, mask == 4).astype(np.float32)
        et = (mask == 4).astype(np.float32)
        masks = np.stack([wt, tc, et], axis=0)

        # Tăng cường dữ liệu (chỉ áp dụng khi training)
        if self.augmentation:
            if random.random() > 0.5:
                stack = np.flip(stack, axis=2).copy()
                masks = np.flip(masks, axis=2).copy()
            if random.random() > 0.5:
                stack = np.flip(stack, axis=1).copy()
                masks = np.flip(masks, axis=1).copy()
                
            # Intensity augmentation — chỉ apply lên image, không apply mask
            if getattr(self, 'augmentation_intensity', False):
                # Random brightness: cộng thêm offset nhỏ
                if random.random() > 0.5:
                    brightness_factor = random.uniform(-0.1, 0.1)
                    stack = stack + brightness_factor  # Dịch chuyển nhẹ độ sáng trong khoảng ±10%

                # Random contrast: nhân với factor gần 1.0
                if random.random() > 0.5:
                    contrast_factor = random.uniform(0.9, 1.1)
                    stack = stack * contrast_factor  # Thay đổi nhẹ độ tương phản trong khoảng ±10%

        return torch.from_numpy(stack), torch.from_numpy(masks)

    def _context_indices(self, slice_idx):
        if self.context_radius == 0:
            return [slice_idx]
        return [
            int(np.clip(slice_idx + offset, 0, 154))
            for offset in range(-self.context_radius, self.context_radius + 1)
        ]

def _load_grade_groups(root_dir, subjects, require_mapping=False):
    hgg, lgg = [], []
    csv_path = os.path.join(root_dir, "name_mapping.csv")
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                sid = row["BraTS_2020_subject_ID"]
                if sid not in subjects:
                    continue
                grade = row.get("Grade", "").strip().upper()
                if grade == "HGG":
                    hgg.append(sid)
                elif grade == "LGG":
                    lgg.append(sid)

    mapped = set(hgg) | set(lgg)
    missing = sorted(set(subjects) - mapped)
    if require_mapping and missing:
        raise ValueError(
            f"Stratified k-fold requires HGG/LGG labels for every subject; missing {len(missing)} subjects."
        )
    if not mapped:
        if require_mapping:
            raise FileNotFoundError(f"Missing valid grade mapping: {csv_path}")
        return subjects[:293], subjects[293:]
    return sorted(hgg), sorted(lgg)


def _stratified_kfold_splits(data_cfg, subjects):
    num_folds = int(data_cfg.get("num_folds", 5))
    fold_index = int(data_cfg.get("fold_index", 0))
    split_seed = int(data_cfg.get("split_seed", 42))
    inner_val_fraction = float(data_cfg.get("inner_val_fraction", 0.1))

    if num_folds < 2:
        raise ValueError("num_folds must be at least 2.")
    if not 0 <= fold_index < num_folds:
        raise ValueError(f"fold_index must be in [0, {num_folds - 1}], got {fold_index}.")
    if not 0.0 < inner_val_fraction < 0.5:
        raise ValueError("inner_val_fraction must be between 0 and 0.5.")

    hgg, lgg = _load_grade_groups(data_cfg["root_dir"], subjects, require_mapping=True)
    train_subjects, val_subjects, test_subjects = [], [], []

    for group_offset, group in enumerate((hgg, lgg)):
        shuffled = np.asarray(group, dtype=object)
        rng = np.random.default_rng(split_seed + group_offset)
        rng.shuffle(shuffled)
        folds = np.array_split(shuffled, num_folds)

        outer_test = list(folds[fold_index])
        remaining = np.concatenate([fold for idx, fold in enumerate(folds) if idx != fold_index])
        inner_rng = np.random.default_rng(split_seed + 1000 + fold_index * 10 + group_offset)
        inner_rng.shuffle(remaining)
        n_val = max(1, int(round(len(remaining) * inner_val_fraction)))

        test_subjects.extend(outer_test)
        val_subjects.extend(list(remaining[:n_val]))
        train_subjects.extend(list(remaining[n_val:]))

    return tuple(sorted(train_subjects)), tuple(sorted(val_subjects)), tuple(sorted(test_subjects))


def validate_subject_splits(train_subjects, val_subjects, test_subjects, expected_subjects=None):
    split_sets = [set(train_subjects), set(val_subjects), set(test_subjects)]
    if split_sets[0] & split_sets[1] or split_sets[0] & split_sets[2] or split_sets[1] & split_sets[2]:
        raise ValueError("Train, validation, and test subject splits must be disjoint.")
    combined = set().union(*split_sets)
    if expected_subjects is not None and combined != set(expected_subjects):
        missing = sorted(set(expected_subjects) - combined)
        extra = sorted(combined - set(expected_subjects))
        raise ValueError(f"Subject split coverage mismatch; missing={missing}, extra={extra}")


def _apply_debug_subject_limits(data_cfg, train_subjects, val_subjects, test_subjects):
    if not data_cfg.get("debug_mode", False):
        return train_subjects, val_subjects, test_subjects
    limits = data_cfg.get("debug_max_subjects_per_split")
    if not isinstance(limits, dict):
        raise ValueError("debug_mode requires debug_max_subjects_per_split.")
    limited = (
        list(train_subjects)[: int(limits.get("train", 2))],
        list(val_subjects)[: int(limits.get("validation", 1))],
        list(test_subjects)[: int(limits.get("test", 1))],
    )
    if any(len(values) == 0 for values in limited):
        raise ValueError("Debug subject limits produced an empty split.")
    validate_subject_splits(*limited)
    return limited


def get_subject_splits(config):
    data_cfg = config["data"]
    subjects = sorted([s for s in os.listdir(data_cfg["root_dir"]) if s.startswith("BraTS20_Training_")])
    split_protocol = data_cfg.get("split_protocol")
    split_type = data_cfg.get("split_type", "sequential")

    if split_protocol == "stratified_kfold":
        train_subjects, val_subjects, test_subjects = _stratified_kfold_splits(data_cfg, subjects)
        validate_subject_splits(train_subjects, val_subjects, test_subjects, expected_subjects=subjects)
        train_subjects, val_subjects, test_subjects = _apply_debug_subject_limits(
            data_cfg, train_subjects, val_subjects, test_subjects
        )
        return list(train_subjects), list(val_subjects), list(test_subjects)

    hgg, lgg = [], []
    
    if split_type == "stratified":
        hgg, lgg = _load_grade_groups(data_cfg["root_dir"], subjects, require_mapping=False)
            
        # Giữ quá trình sinh split độc lập để không ảnh hưởng đến bộ sinh số ngẫu nhiên huấn luyện (RNG).
        split_rng = np.random.RandomState(int(data_cfg.get("split_seed", 42)))
        split_rng.shuffle(hgg)
        split_rng.shuffle(lgg)
        
        n_hgg_tr, n_hgg_va = int(len(hgg)*0.8), int(len(hgg)*0.1)
        n_lgg_tr, n_lgg_va = int(len(lgg)*0.8), int(len(lgg)*0.1)
        
        train_subjects = hgg[:n_hgg_tr] + lgg[:n_lgg_tr]
        val_subjects   = hgg[n_hgg_tr:n_hgg_tr+n_hgg_va] + lgg[n_lgg_tr:n_lgg_tr+n_lgg_va]
        test_subjects  = hgg[n_hgg_tr+n_hgg_va:] + lgg[n_lgg_tr+n_lgg_va:]
    else:
        # Chia tách tuần tự (Sequential Split)
        train_subjects = subjects[:295]
        val_subjects   = subjects[295:332]
        test_subjects  = subjects[332:]

    validate_subject_splits(train_subjects, val_subjects, test_subjects, expected_subjects=subjects)
    train_subjects, val_subjects, test_subjects = _apply_debug_subject_limits(
        data_cfg, train_subjects, val_subjects, test_subjects
    )
    return train_subjects, val_subjects, test_subjects

def get_dataloaders(config):
    data_cfg = config["data"]
    train_cfg = config["training"]

    train_subjects, val_subjects, test_subjects = get_subject_splits(config)
    
    slice_range = data_cfg.get("slice_range", [0, 155])
    if isinstance(slice_range, list):
        slice_range = tuple(slice_range)
    norm_type       = data_cfg.get("normalization", "zscore_volume")
    augmentation    = data_cfg.get("augmentation", False)
    augmentation_intensity = data_cfg.get("augmentation_intensity", False)
    sampling        = data_cfg.get("sampling", "fixed")
    context_slices  = normalize_context_slices(data_cfg.get("context_slices", 1))
    min_tumor_pixels = data_cfg.get("min_tumor_pixels", 100)
    cache_volumes   = data_cfg.get("cache_volumes", True)   # Tải trước toàn bộ thể tích ảnh vào RAM

    train_ds = BraTSDataset(data_cfg["root_dir"], train_subjects, slice_range, norm_type,
                            augmentation=augmentation, augmentation_intensity=augmentation_intensity, sampling=sampling,
                            context_slices=context_slices, min_tumor_pixels=min_tumor_pixels,
                            cache_volumes=cache_volumes, preprocess_config=data_cfg)
    # Tập validation bắt buộc dùng fixed sampling, đủ 155 slices (không dùng lấy mẫu quanh tâm u để tránh rò rỉ thông tin - data leakage)
    val_ds   = BraTSDataset(data_cfg["root_dir"], val_subjects, (0, 155), norm_type,
                            augmentation=False, sampling="fixed", context_slices=context_slices,
                            cache_volumes=cache_volumes, preprocess_config=data_cfg)
    test_ds  = BraTSDataset(data_cfg["root_dir"], test_subjects, slice_range, norm_type,
                            augmentation=False, sampling="fixed", context_slices=context_slices,
                            cache_volumes=False, preprocess_config=data_cfg)

    
    print(f"[DATA] Train: {len(train_ds)} | Val: {len(val_ds)} | Test subjects: {len(test_subjects)}")
    
    from torch.utils.data import DataLoader, WeightedRandomSampler
    
    # Cho phép cấu hình qua YAML, mặc định tối ưu cho Kaggle (num_workers=2, pin_memory=True)
    num_workers = train_cfg.get("num_workers", 2)
    pin_memory  = train_cfg.get("pin_memory", True)
    
    train_sampler = None
    shuffle_train = True
    
    if sampling == "weighted" and hasattr(train_ds, "sample_weights") and len(train_ds.sample_weights) == len(train_ds.samples):
        train_sampler = WeightedRandomSampler(
            weights=train_ds.sample_weights,
            num_samples=len(train_ds.samples),
            replacement=True
        )
        shuffle_train = False # Bắt buộc tắt shuffle khi dùng sampler
        print("[DATA] WeightedRandomSampler enabled for train loader")
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=train_cfg["batch_size"], 
        shuffle=shuffle_train, 
        sampler=train_sampler,
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        persistent_workers=(num_workers > 0)
    )
    val_loader   = DataLoader(
        val_ds, 
        batch_size=train_cfg["batch_size"], 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        persistent_workers=(num_workers > 0)
    )
    
    return train_loader, val_loader, test_ds, test_subjects
