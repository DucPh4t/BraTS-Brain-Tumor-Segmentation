"""
Mô tả: Chạy dự đoán và tạo ảnh động GIF mô phỏng quét lát cắt cho các ca phân đoạn dị thường và ca lỗi khó (như Training 236, 279, 326...).
Đầu vào:
    Đọc tự động dữ liệu các ca bệnh tương ứng từ BraTS dataset. Checkpoint mô hình được chỉ định cục bộ.
Đầu ra:
    Lưu các ảnh động GIF thể hiện mặt nạ dự đoán phủ lên ảnh gốc đối chiếu với Ground Truth.
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import nibabel as nib
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
import imageio

from src.models.multimodal_unet import DisentangledFusionUNet2D
from src.data.processors import get_preprocessor

# ─── BraTS Color Scheme ───────────────────────────────────────────────────────
WT_COLOR = np.array([0.2, 0.8, 0.2, 0.45])  # Green
TC_COLOR = np.array([0.9, 0.2, 0.2, 0.55])  # Red
ET_COLOR = np.array([1.0, 0.9, 0.0, 0.65])  # Yellow

OUTLIER_CASES = [
    "BraTS20_Training_236", 
    "BraTS20_Training_279", 
    "BraTS20_Training_326", 
    "BraTS20_Training_307", 
    "BraTS20_Training_267", 
    "BraTS20_Training_273", 
    "BraTS20_Training_188", 
    "BraTS20_Training_021"
]

def load_subject(data_dir, sid, preprocessor):
    subdir = os.path.join(data_dir, sid)
    vols = {}
    for mod in ["flair", "t1", "t1ce", "t2"]:
        path = os.path.join(subdir, f"{sid}_{mod}.nii")
        vols[mod] = preprocessor(nib.load(path).get_fdata())
    
    seg_path = os.path.join(subdir, f"{sid}_seg.nii")
    if not os.path.exists(seg_path):
        for f in os.listdir(subdir):
            if "seg" in f.lower() and f.endswith(".nii"):
                seg_path = os.path.join(subdir, f)
                break
    vols["seg"] = nib.load(seg_path).get_fdata().astype(np.uint8)
    return vols

def run_tta_inference_3d(model, vols, device, batch_size=16):
    stack_4d = np.stack([vols["flair"], vols["t1"], vols["t1ce"], vols["t2"]], axis=0)
    stack_4d = np.transpose(stack_4d, (3, 0, 1, 2))  # (155, 4, 240, 240)
    
    all_preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, 155, batch_size):
            batch = torch.from_numpy(stack_4d[i:i+batch_size].astype(np.float32)).to(device)
            # Original
            p_none = torch.sigmoid(model(batch))
            # Flip H (width)
            p_h = torch.sigmoid(model(torch.flip(batch, [3])))
            p_h = torch.flip(p_h, [3])
            # Flip W (height)
            p_w = torch.sigmoid(model(torch.flip(batch, [2])))
            p_w = torch.flip(p_w, [2])
            # Flip HW
            p_hw = torch.sigmoid(model(torch.flip(batch, [2, 3])))
            p_hw = torch.flip(p_hw, [2, 3])
            
            p_avg = (p_none + p_h + p_w + p_hw) / 4.0
            pred = (p_avg > 0.5).cpu().numpy().astype(np.uint8)
            all_preds.append(pred)
            
    return np.concatenate(all_preds, axis=0)  # (155, 3, 240, 240)

def calc_dice_2d(pred, gt):
    smooth = 1e-5
    intersection = (pred * gt).sum()
    return (2. * intersection + smooth) / (pred.sum() + gt.sum() + smooth)

def build_overlay(img_slice, wt_mask, tc_mask, et_mask):
    vmin, vmax = img_slice.min(), img_slice.max()
    img_norm = (img_slice - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(img_slice)
    canvas = np.stack([img_norm, img_norm, img_norm, np.ones_like(img_norm)], axis=-1)
    
    for mask, color in [(wt_mask, WT_COLOR), (tc_mask, TC_COLOR), (et_mask, ET_COLOR)]:
        loc = mask > 0
        for c in range(3):
            canvas[loc, c] = canvas[loc, c] * (1 - color[3]) + color[c] * color[3]
    return canvas

def make_frame(s_idx, vols, pred_vol, sid):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor("#0d1117")
    
    f_s = vols["flair"][:, :, s_idx]
    t2_s = vols["t2"][:, :, s_idx]
    seg_s = vols["seg"][:, :, s_idx]
    
    wt_g = (seg_s > 0).astype(np.float32)
    tc_g = np.logical_or(seg_s==1, seg_s==4).astype(np.float32)
    et_g = (seg_s==4).astype(np.float32)
    
    wt_p, tc_p, et_p = pred_vol[s_idx, 0].astype(np.float32), pred_vol[s_idx, 1].astype(np.float32), pred_vol[s_idx, 2].astype(np.float32)
    
    d_wt = calc_dice_2d(wt_p, wt_g) * 100
    d_tc = calc_dice_2d(tc_p, tc_g) * 100
    d_et = calc_dice_2d(et_p, et_g) * 100
    d_mean = (d_wt + d_tc + d_et) / 3.0
    
    gt_o = build_overlay(f_s, wt_g, tc_g, et_g)
    pr_o = build_overlay(f_s, wt_p, tc_p, et_p)
    
    imgs = [f_s, t2_s, gt_o, pr_o]
    titles = ["FLAIR", "T2", "Ground Truth", "Prediction (Exp023)"]
    
    for i in range(4):
        ax = axes[i]
        ax.set_facecolor("#0d1117")
        if i < 2:
            ax.imshow(imgs[i].T, cmap="gray", origin="lower")
        else:
            ax.imshow(imgs[i].transpose(1, 0, 2), origin="lower")
        ax.axis("off")
        ax.set_title(titles[i], color="white", fontsize=16, fontweight="bold")
    
    plt.suptitle(f"{sid} | Slice: {s_idx}/154 | Mean Dice: {d_mean:.1f}% (WT:{d_wt:.0f}% TC:{d_tc:.0f}% ET:{d_et:.0f}%)", color="white", y=0.98, fontsize=16, fontweight="bold")
    
    # Draw canvas to RGB array
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return image

def main():
    data_dir = "/Users/nguyenducphat/Projects/ĐATN MRI/MRI dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"  # Update with actual if needed
    if not os.path.exists(data_dir):
        print("Please check data directory path.")
        return
        
    out_dir = "outputs/figures/outliers_gifs"
    os.makedirs(out_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = DisentangledFusionUNet2D(n_channels=4, n_classes=3, init_features=64).to(device)
    ckpt_path = "outputs/exp021/best_model.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    
    preprocessor = get_preprocessor("zscore_clip")
    
    for sid in OUTLIER_CASES:
        print(f"Processing {sid}...")
        try:
            vols = load_subject(data_dir, sid, preprocessor)
            pred_vol = run_tta_inference_3d(model, vols, device)
            
            s_min, s_max = 0, 154
            
            frames = []
            sid_img_dir = os.path.join("outputs/figures/outliers_155slice", sid)
            os.makedirs(sid_img_dir, exist_ok=True)
            
            for s_idx in tqdm(range(s_min, s_max + 1), desc=f"Generating frames for {sid}"):
                frame = make_frame(s_idx, vols, pred_vol, sid)
                frames.append(frame)
                
                # Save individual slice
                img_path = os.path.join(sid_img_dir, f"slice_{s_idx:03d}.png")
                imageio.imwrite(img_path, frame)
                
            gif_path = os.path.join(out_dir, f"{sid}_155slices.gif")
            print(f"Saving GIF to {gif_path} (FPS=10)...")
            imageio.mimsave(gif_path, frames, fps=10, loop=0)
        except Exception as e:
            print(f"Failed to process {sid}: {e}")

if __name__ == "__main__":
    main()
