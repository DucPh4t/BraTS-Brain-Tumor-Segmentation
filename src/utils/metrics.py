"""
Mô tả: Chứa các hàm tính toán chỉ số đánh giá chất lượng phân đoạn ảnh y tế: 3D Dice Score và khoảng cách Hausdorff 95% (HD95).
Đầu vào:
    pred_vol, target_vol (np.ndarray): Thể tích dự đoán và nhãn thực tế dạng nhị phân 3D.
Đầu ra:
    Trả về điểm số Dice (từ 0.0 đến 1.0) và khoảng cách HD95 (đơn vị mm).
"""

import numpy as np
from medpy.metric.binary import hd95

def calc_dice(pred, target):
    smooth = 1e-5
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def calc_dice_3d(pred_vol: np.ndarray, target_vol: np.ndarray):
    """Đánh giá 3D Dice (không dùng smooth). Xử lý trường hợp rỗng: cả hai rỗng -> 1.0, một bên rỗng -> 0.0."""
    intersection = (pred_vol * target_vol).sum()
    denom = pred_vol.sum() + target_vol.sum()
    if denom == 0:
        return 1.0  # Cả hai cùng rỗng = khớp hoàn hảo
    return float(2. * intersection / denom)

def calc_hd95_3d(pred_vol: np.ndarray, target_vol: np.ndarray, voxelspacing=None):
    """Tính HD95. Khi một trong hai vùng rỗng (empty-case), giá trị trả về mặc định là đường chéo của thể tích (volume diagonal) dựa trên voxel spacing."""
    if pred_vol.sum() == 0 and target_vol.sum() == 0:
        return 0.0
    if pred_vol.sum() == 0 or target_vol.sum() == 0:
        shape = np.asarray(pred_vol.shape, dtype=np.float32)
        spacing = np.ones(3, dtype=np.float32) if voxelspacing is None else np.asarray(voxelspacing, dtype=np.float32)
        return float(np.linalg.norm(shape * spacing))
    return hd95(pred_vol, target_vol, voxelspacing=voxelspacing)
