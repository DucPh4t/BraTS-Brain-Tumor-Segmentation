"""
Mô tả: Chứa các hàm toán học thực hiện chuẩn hóa cường độ sáng ảnh MRI (min_max, zscore_clip, zscore_clip_custom, zscore_raw, CLAHE).
Đầu vào:
    vol_3d (np.ndarray): Thể tích ảnh MRI 3D.
Đầu ra:
    Trả về mảng NumPy đã chuẩn hóa.
"""

import numpy as np

def _brain_mask(vol_3d):
    return vol_3d > 0

def _zscore_masked(vol_3d, brain_mask):
    out = np.zeros_like(vol_3d, dtype=np.float32)
    if brain_mask.sum() == 0:
        return out

    brain_voxels = vol_3d[brain_mask]
    mean = brain_voxels.mean()
    std = brain_voxels.std()
    out[brain_mask] = (brain_voxels - mean) / (std + 1e-8)
    return out

def _clip_to_unit(vol_3d, brain_mask, lower=0.5, upper=99.5):
    if brain_mask.sum() == 0:
        return np.zeros_like(vol_3d, dtype=np.float32)

    p_min = np.percentile(vol_3d[brain_mask], lower)
    p_max = np.percentile(vol_3d[brain_mask], upper)
    clipped = np.clip(vol_3d, p_min, p_max).astype(np.float32)
    clipped[~brain_mask] = 0

    out = np.zeros_like(clipped, dtype=np.float32)
    if p_max > p_min:
        out[brain_mask] = (clipped[brain_mask] - p_min) / (p_max - p_min)
    return out

def _normalize_modalities(modalities):
    if modalities in (None, "all"):
        return "all"
    return {str(mod).lower() for mod in modalities}

def min_max(vol_3d):
    """Chuẩn hóa Min-Max trên thể tích 3D được che (brain-masked) làm baseline cho ablation study.
    - Brain mask từ volume > 0
    - Min/Max chỉ trên brain voxels
    - Background giữ nguyên = 0
    """
    out = np.zeros_like(vol_3d, dtype=np.float32)
    brain_mask = vol_3d > 0
    if brain_mask.sum() == 0:
        return out
        
    min_val = vol_3d[brain_mask].min()
    max_val = vol_3d[brain_mask].max()
    
    if max_val > min_val:
        out[brain_mask] = (vol_3d[brain_mask] - min_val) / (max_val - min_val)
    else:
        out[brain_mask] = vol_3d[brain_mask]
    return out

def zscore_clip(vol_3d):
    """Z-Score (Chuẩn hóa thang đo độ sáng): 
    - Cắt ngưỡng phân vị (Percentile clip 0.5, 99.5) trước khi tính z-score
    - Masking (vol > 0) để chỉ chuẩn hóa vùng não, vùng nền background=0
    """
    out = np.zeros_like(vol_3d, dtype=np.float32)
    brain_mask = vol_3d > 0
    if brain_mask.sum() == 0:
        return out
        
    # Cắt ngưỡng phân vị (Percentile clipping 0.5, 99.5) để loại bỏ các điểm dị biệt (outliers) trước khi z-score
    p_min = np.percentile(vol_3d[brain_mask], 0.5)
    p_max = np.percentile(vol_3d[brain_mask], 99.5)
    
    clipped = np.clip(vol_3d, p_min, p_max)   # clip trên 3D volume
    clipped[~brain_mask] = 0                  # khôi phục lại vùng nền background
    brain_clipped = clipped[brain_mask]       # lấy brain voxels đã clip
    
    mean = brain_clipped.mean()
    std  = brain_clipped.std()
    
    out[brain_mask] = (brain_clipped - mean) / (std + 1e-8)
    return out

def zscore_clip_custom(vol_3d, config=None, modality=None):
    """Cắt ngưỡng phân vị có thể cấu hình trước khi chuẩn hóa z-score trên vùng não.

    Giữ nguyên họ tiền xử lý của Exp004 nhưng cho phép kiểm tra xem cửa sổ phân vị rộng hơn hay hẹp hơn sẽ giữ lại đặc trưng vùng ET tốt hơn.
    """
    config = config or {}
    percentiles = config.get("clip_percentiles", [0.5, 99.5])
    lower, upper = float(percentiles[0]), float(percentiles[1])

    brain_mask = _brain_mask(vol_3d)
    if brain_mask.sum() == 0:
        return np.zeros_like(vol_3d, dtype=np.float32)

    clipped_unit = _clip_to_unit(vol_3d, brain_mask, lower=lower, upper=upper)
    return _zscore_masked(clipped_unit, brain_mask)

def _apply_clahe_slicewise(vol_unit, brain_mask, config):
    try:
        from skimage import exposure
    except ImportError as exc:
        raise ImportError("CLAHE preprocessing requires scikit-image (`skimage`).") from exc

    clip_limit = float(config.get("clip_limit", 0.01))
    kernel_size = config.get("kernel_size", 32)
    nbins = int(config.get("nbins", 256))
    min_brain_pixels = int(config.get("min_brain_pixels", 100))

    out = np.zeros_like(vol_unit, dtype=np.float32)
    for z in range(vol_unit.shape[2]):
        slice_mask = brain_mask[:, :, z]
        if int(slice_mask.sum()) < min_brain_pixels:
            out[:, :, z] = vol_unit[:, :, z]
            continue

        enhanced = exposure.equalize_adapthist(
            vol_unit[:, :, z],
            kernel_size=kernel_size,
            clip_limit=clip_limit,
            nbins=nbins,
        ).astype(np.float32)
        enhanced[~slice_mask] = 0
        out[:, :, z] = enhanced
    return out

def zscore_clip_clahe(vol_3d, config=None, modality=None):
    """Cắt ngưỡng phân vị -> đưa về [0,1] -> áp dụng CLAHE tùy chọn trên từng lát cắt -> chuẩn hóa z-score.

    CLAHE chỉ được áp dụng cho các xung phương thái được cấu hình, giúp thử nghiệm riêng lẻ các xung (như T1ce-only hoặc FLAIR+T1ce).
    """
    config = config or {}
    modalities = _normalize_modalities(config.get("modalities", "all"))
    mod = str(modality).lower() if modality is not None else None
    apply_clahe = modalities == "all" or mod in modalities

    percentiles = config.get("clip_percentiles", [0.5, 99.5])
    lower, upper = float(percentiles[0]), float(percentiles[1])

    brain_mask = _brain_mask(vol_3d)
    if brain_mask.sum() == 0:
        return np.zeros_like(vol_3d, dtype=np.float32)

    clipped_unit = _clip_to_unit(vol_3d, brain_mask, lower=lower, upper=upper)
    if apply_clahe:
        clipped_unit = _apply_clahe_slicewise(clipped_unit, brain_mask, config)
    return _zscore_masked(clipped_unit, brain_mask)

def zscore_raw(vol_3d):
    """Z-score cơ bản KHÔNG có percentile clipping."""
    out = np.zeros_like(vol_3d, dtype=np.float32)
    brain_mask = vol_3d > 0
    if brain_mask.sum() == 0:
        return out
        
    brain_voxels = vol_3d[brain_mask]
    mean = brain_voxels.mean()
    std  = brain_voxels.std()
    
    out[brain_mask] = (brain_voxels - mean) / (std + 1e-8)
    return out
