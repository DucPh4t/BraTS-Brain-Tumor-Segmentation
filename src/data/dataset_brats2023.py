"""
Mô tả: Lớp Dataset tải và xử lý dữ liệu ảnh MRI từ bộ dữ liệu BraTS 2023 GLI. Tự động chuẩn hóa tên kênh (t1n, t1c, t2w, t2f) và ánh xạ lại nhãn u (ET từ nhãn 3 sang 4) để tương thích với các mô hình huấn luyện trên tập BraTS 2020.
Đầu vào:
    root_dir (str): Thư mục gốc chứa dữ liệu BraTS 2023.
    subject_ids (list): Danh sách ID các ca bệnh.
    slice_range (tuple): Khoảng lát cắt lấy mẫu (mặc định từ 0 đến 155).
    normalization (str): Phương pháp chuẩn hóa ảnh.
    augmentation (bool): Có sử dụng tăng cường dữ liệu hay không.
    context_slices (int): Số lượng lát cắt ngữ cảnh lân cận (cho 2.5D).
Đầu ra:
    Trả về một mẫu dữ liệu (dict/tuple) gồm tensor ảnh MRI và nhãn phân đoạn (seg) tại lát cắt chỉ định.
"""

import os
import random
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np

from src.data.processors import get_preprocessor

class BraTS2023Dataset(Dataset):
    """
    Trình nạp dữ liệu (Dataset loader) cho bộ dữ liệu BraTS 2023 GLI.
    Tự động khớp tên các xung phương thái (t1n, t1c, t2w, t2f) và ánh xạ nhãn Enhancing Tumor (nhãn 3)
    để tương thích với định dạng đầu ra WT/TC/ET mong đợi của các mô hình huấn luyện trên BraTS 2020.
    """
    def __init__(self, root_dir, subject_ids, slice_range=(0, 155),
                 normalization="zscore_clip", augmentation=False,
                 context_slices=0, cache_volumes=False, preprocess_config=None):
        self.root_dir = root_dir
        self.subject_ids = subject_ids
        self.slice_range = slice_range
        self.normalization = normalization
        self.augmentation = augmentation
        self.context_slices = context_slices
        self.context_radius = max((int(context_slices) - 1) // 2, 0)
        self.cache_volumes = cache_volumes

        self.preprocessor = get_preprocessor(self.normalization, preprocess_config or {})

        # Lưu bộ nhớ cache: sid -> {"t2f": np.ndarray, "t1n": ..., "t1c": ..., "t2w": ..., "seg": ...}
        self._cache = {}
        if self.cache_volumes:
            from tqdm import tqdm
            print(f"  [CACHE] Đang nạp trước {len(subject_ids)} ca bệnh vào RAM...")
            for sid in tqdm(subject_ids, desc="Đang cache dữ liệu 2023", leave=False):
                self._cache[sid] = self._load_subject(sid)

        self.samples = []
        for sid in self.subject_ids:
            sl = self.slice_range
            for slice_idx in range(sl[0], sl[1]):
                self.samples.append((sid, slice_idx))

    def _find_file(self, sid, suffix):
        """Tìm file có hậu tố tương ứng (ví dụ: -t1c, -seg) của ca bệnh."""
        subdir = os.path.join(self.root_dir, sid)
        clean_suffix = suffix.replace('.nii.gz', '').replace('.nii', '')
        
        for f in os.listdir(subdir):
            if f.startswith(sid + clean_suffix) and (f.endswith('.nii') or f.endswith('.nii.gz')):
                return os.path.join(subdir, f)
        raise FileNotFoundError(f"File with suffix {clean_suffix} not found for {sid} under {subdir}")

    def _load_subject(self, sid):
        """Đọc và chuẩn hóa các kênh ảnh + mặt nạ phân đoạn cho một ca bệnh."""
        entry = {}
        # Các kênh ảnh BraTS 2023: t1n (T1), t1c (T1ce), t2w (T2), t2f (FLAIR)
        mod_suffixes = {
            'flair': '-t2f.nii.gz',
            't1': '-t1n.nii.gz',
            't1ce': '-t1c.nii.gz',
            't2': '-t2w.nii.gz'
        }
        for target_mod, suffix in mod_suffixes.items():
            path = self._find_file(sid, suffix)
            vol_3d = nib.load(path).get_fdata()
            entry[target_mod] = self.preprocessor(vol_3d, modality=target_mod).astype(np.float16)

        seg_path = self._find_file(sid, '-seg.nii.gz')
        entry['seg'] = nib.load(seg_path).get_fdata().astype(np.uint8)
        return entry

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, slice_idx = self.samples[idx]

        if self.cache_volumes:
            cached = self._cache[sid]
            imgs = []
            for mod in ['flair', 't1', 't1ce', 't2']:
                for ctx_idx in self._context_indices(slice_idx):
                    imgs.append(cached[mod][:, :, ctx_idx])
            mask = cached['seg'][:, :, slice_idx]
        else:
            imgs = []
            mod_suffixes = {
                'flair': '-t2f.nii.gz',
                't1': '-t1n.nii.gz',
                't1ce': '-t1c.nii.gz',
                't2': '-t2w.nii.gz'
            }
            for mod, suffix in mod_suffixes.items():
                path = self._find_file(sid, suffix)
                vol_3d = nib.load(path).get_fdata()
                vol_norm = self.preprocessor(vol_3d, modality=mod)
                for ctx_idx in self._context_indices(slice_idx):
                    imgs.append(vol_norm[:, :, ctx_idx])
            seg_path = self._find_file(sid, '-seg.nii.gz')
            mask = nib.load(seg_path).get_fdata()[:, :, slice_idx]

        stack = np.stack(imgs, axis=0).astype(np.float32)

        # Mã hóa mặt nạ phân đoạn đa lớp cho BraTS 2023:
        # Nhãn 1 = NCR, 2 = ED, 3 = ET (khác với nhãn 4 ở tập 2020)
        wt = (mask > 0).astype(np.float32)
        tc = np.logical_or(mask == 1, mask == 3).astype(np.float32)
        et = (mask == 3).astype(np.float32)
        masks = np.stack([wt, tc, et], axis=0)

        # Tăng cường dữ liệu (chỉ dùng khi train, mặc định False khi test/eval)
        if self.augmentation:
            if random.random() > 0.5:
                stack = np.flip(stack, axis=2).copy()
                masks = np.flip(masks, axis=2).copy()
            if random.random() > 0.5:
                stack = np.flip(stack, axis=1).copy()
                masks = np.flip(masks, axis=1).copy()

        return torch.from_numpy(stack), torch.from_numpy(masks)

    def _context_indices(self, slice_idx):
        if self.context_radius == 0:
            return [slice_idx]
        return [
            int(np.clip(slice_idx + offset, 0, 154))
            for offset in range(-self.context_radius, self.context_radius + 1)
        ]
