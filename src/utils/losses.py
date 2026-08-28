"""
Mô tả: Chứa các hàm hàm mất mát (Loss functions) được thiết kế đặc thù cho phân đoạn khối u não: Dice Loss, Focal Loss, Tversky Loss, và Hierarchy Loss (kiểm soát phân cấp giải phẫu giữa WT, TC, ET).
Đầu vào:
    pred (torch.Tensor): Đầu ra xác suất dự đoán của mô hình.
    target (torch.Tensor): Nhãn thực tế tương ứng (Ground Truth).
Đầu ra:
    Trả về giá trị hàm mất mát (loss value) là số thực torch.Tensor để thực hiện lan truyền ngược (backpropagation).
"""

import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1.):
    """Hàm mất mát nhận xác suất đầu ra probabilities (không nhận logits thô). smooth=1.0 để chống nổ gradient."""
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def focal_tversky_loss(pred, target, alpha=0.7, beta=0.3, gamma=4/3, smooth=1e-6):
    """
    Hàm mất mát Focal Tversky dành cho bài toán phân đoạn mất cân bằng nhãn.
    - alpha: trọng số phạt cho False Negatives (FN). Đặt cao hơn (ví dụ: 0.7) để tăng Recall.
    - beta: trọng số phạt cho False Positives (FP).
    - gamma: tham số focal (thường nằm trong khoảng [1, 3]).
    """
    pred = pred.contiguous()
    target = target.contiguous()

    # Tính toán các đại lượng TP, FN, FP
    tp = (pred * target).sum(dim=(2, 3))
    fn = ((1 - pred) * target).sum(dim=(2, 3))
    fp = (pred * (1 - target)).sum(dim=(2, 3))

    tversky_index = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    loss = (1 - tversky_index) ** gamma
    
    return loss.mean()

def dice_bce_loss(pred, target, dice_weight=0.5, smooth=1.):
    """
    Hàm mất mát kết hợp (Combo Loss) = dice_weight * DiceLoss + (1 - dice_weight) * BCELoss
    - Dice Loss: giữ vững chất lượng vùng lớn (WT)
    - BCE Loss: phạt ổn định từng pixel
    """
    d_loss = dice_loss(pred, target, smooth=smooth)
    bce_loss = F.binary_cross_entropy(pred, target)
    return dice_weight * d_loss + (1.0 - dice_weight) * bce_loss

def weighted_dice_bce_loss(pred, target, dice_weight=0.5, bce_channel_weights=None, smooth=1.):
    """
    Dice + BCE có phân bổ trọng số cho các kênh nhằm xử lý sự mất cân bằng giữa các vùng của BraTS.

    Thứ tự các kênh là [WT, TC, ET]. Phần Dice giữ nguyên trong khi phần BCE
    có thể tập trung hơn một chút vào các vùng u nhỏ như TC/ET.
    """
    d_loss = dice_loss(pred, target, smooth=smooth)

    bce = F.binary_cross_entropy(pred, target, reduction="none")
    if bce_channel_weights is not None:
        if len(bce_channel_weights) != pred.shape[1]:
            raise ValueError(
                f"bce_channel_weights must have {pred.shape[1]} values, got {len(bce_channel_weights)}"
            )
        weights = torch.tensor(
            bce_channel_weights,
            dtype=pred.dtype,
            device=pred.device,
        ).view(1, pred.shape[1], 1, 1)
        bce = bce * weights

    return dice_weight * d_loss + (1.0 - dice_weight) * bce.mean()

def region_adaptive_dice_bce_loss(
    pred,
    target,
    dice_weight=0.5,
    adaptive_channel="ET",
    adaptive_region="TC",
    region_multiplier=1.5,
    smooth=1.,
):
    """
    Dice + BCE thích ứng theo vùng (region-adaptive).

    Thứ tự kênh: [WT, TC, ET]. Khác với việc gán trọng số kênh toàn cục, phương pháp này chỉ
    tăng cường độ phạt BCE của kênh được chọn bên trong một vùng giải phẫu cụ thể từ ground-truth,
    ví dụ: phạt lỗi sai của ET bên trong vùng TC.
    """
    channel_to_idx = {"WT": 0, "TC": 1, "ET": 2}
    if adaptive_channel not in channel_to_idx:
        raise ValueError(f"adaptive_channel must be one of {list(channel_to_idx)}, got {adaptive_channel}")
    if adaptive_region not in channel_to_idx:
        raise ValueError(f"adaptive_region must be one of {list(channel_to_idx)}, got {adaptive_region}")

    d_loss = dice_loss(pred, target, smooth=smooth)
    bce = F.binary_cross_entropy(pred, target, reduction="none")

    channel_idx = channel_to_idx[adaptive_channel]
    region_idx = channel_to_idx[adaptive_region]
    region_mask = target[:, region_idx:region_idx + 1]

    weights = torch.ones_like(bce)
    weights[:, channel_idx:channel_idx + 1] = (
        1.0 + (float(region_multiplier) - 1.0) * region_mask
    )
    bce_loss = (bce * weights).mean()
    return dice_weight * d_loss + (1.0 - dice_weight) * bce_loss

def et_positive_adaptive_dice_bce_loss(
    pred,
    target,
    dice_weight=0.5,
    positive_multiplier=1.5,
    smooth=1.,
):
    """
    Dice + BCE tích hợp tăng trọng số phạt cho các ca dương tính/bỏ sót (False Negative) của ET.

    Thứ tự kênh là [WT, TC, ET]. Điều này tránh được lỗi của Exp046 là tăng BCE cho mọi voxel ET bên trong TC
    (kể cả những voxel thuộc TC nhưng không phải ET). Chỉ có phần hạng tử dương tính của ET
    là -GT_ET * log(Pred_ET) được nhân thêm trọng số.
    """
    d_loss = dice_loss(pred, target, smooth=smooth)

    eps = 1e-7
    pred_clamped = pred.clamp(eps, 1.0 - eps)
    positive_term = -target * torch.log(pred_clamped)
    negative_term = -(1.0 - target) * torch.log(1.0 - pred_clamped)

    weights = torch.ones_like(positive_term)
    weights[:, 2:3] = float(positive_multiplier)
    bce_loss = (positive_term * weights + negative_term).mean()

    return dice_weight * d_loss + (1.0 - dice_weight) * bce_loss

def hierarchy_consistency_loss(pred):
    """
    Hàm mất mát phạt các vi phạm phân cấp vùng u BraTS: ET phải nằm trong TC, TC phải nằm trong WT.

    Thứ tự kênh: [WT, TC, ET]. Hàm mất mát này khả vi vì hoạt động trực tiếp trên xác suất
    thay vì các mặt nạ nhị phân sau ngưỡng.
    """
    wt = pred[:, 0:1]
    tc = pred[:, 1:2]
    et = pred[:, 2:3]
    et_outside_tc = F.relu(et - tc).mean()
    tc_outside_wt = F.relu(tc - wt).mean()
    return et_outside_tc + tc_outside_wt

def boundary_consistency_loss(pred, target, kernel_size=3, smooth=1.):
    """
    Đồng bộ ranh giới dự đoán mềm (soft prediction boundaries) khớp với ranh giới nhãn thực tế.

    Bản đồ ranh giới được tính toán thông qua độ dốc hình thái học khả vi (differentiable morphological gradient):
    max_pool(x) - min_pool(x). Thứ tự kênh vẫn là [WT, TC, ET].
    """
    padding = kernel_size // 2
    pred_max = F.max_pool2d(pred, kernel_size=kernel_size, stride=1, padding=padding)
    pred_min = -F.max_pool2d(-pred, kernel_size=kernel_size, stride=1, padding=padding)
    target_max = F.max_pool2d(target, kernel_size=kernel_size, stride=1, padding=padding)
    target_min = -F.max_pool2d(-target, kernel_size=kernel_size, stride=1, padding=padding)

    pred_boundary = (pred_max - pred_min).clamp(0, 1)
    target_boundary = (target_max - target_min).clamp(0, 1)
    return dice_bce_loss(pred_boundary, target_boundary, dice_weight=0.5, smooth=smooth)

def dice_focal_loss(pred, target, dice_weight=0.5, alpha=0.7, beta=0.3, gamma=4/3, smooth=1e-6):
    """
    Kết hợp giữa Dice Loss và Focal Tversky Loss (Thử nghiệm Exp012)
    - Dice Loss: giữ vững chất lượng vùng lớn (WT)
    - Focal Tversky: ưu tiên vùng nhỏ khó tìm (ET, TC)
    """
    d_loss  = dice_loss(pred, target)
    ft_loss = focal_tversky_loss(pred, target, alpha=alpha, beta=beta, gamma=gamma, smooth=smooth)
    return dice_weight * d_loss + (1.0 - dice_weight) * ft_loss

def modality_contrastive_loss(modality_features, temperature=0.07):
    """
    Hàm mất mát tương phản (contrastive loss) dành cho các cặp xung MRI phương thái.

    Thứ tự đầu vào: FLAIR, T1, T1ce, T2.
    Các cặp tích cực (positive pairs) tuân theo DFuse-Net: FLAIR-T2 và T1-T1ce.
    """
    feats = F.normalize(modality_features, dim=-1)

    flair = feats[:, 0]
    t1 = feats[:, 1]
    t1ce = feats[:, 2]
    t2 = feats[:, 3]

    pairs = [
        (flair, t2, [t1, t1ce]),
        (t1, t1ce, [flair, t2]),
    ]

    losses = []
    for anchor, positive, negatives in pairs:
        pos = torch.exp(F.cosine_similarity(anchor, positive, dim=1) / temperature)
        neg = torch.stack([
            torch.exp(F.cosine_similarity(anchor, neg_feat, dim=1) / temperature)
            for neg_feat in negatives
        ], dim=0).sum(dim=0)
        losses.append(-torch.log(pos / (pos + neg + 1e-8)))

    return torch.stack(losses, dim=0).mean()
