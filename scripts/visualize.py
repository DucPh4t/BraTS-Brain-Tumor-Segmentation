"""
Mô tả: Trực quan hóa thể tích ảnh MRI não 3D, chạy dự đoán các vùng u (WT, TC, ET) và phủ nhãn thực tế (Ground Truth) để đối chiếu. Xuất bố cục ảnh 2D lát cắt và thống kê Dice/HD95.
Đầu vào:
    --checkpoint: Đường dẫn checkpoint chứa trọng số mô hình.
    --config: Đường dẫn file cấu hình YAML tương ứng.
    --data-dir: Thư mục chứa dữ liệu tập BraTS training.
    --output-dir: Thư mục lưu kết quả ảnh trực quan hóa.
Đầu ra:
    Lưu các ảnh PNG biểu diễn lát cắt dự đoán so với nhãn thực tế.
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import argparse
import traceback
import numpy as np
import nibabel as nib
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

from src.models.attention_unet import AttentionUNet2D
from src.utils.metrics import calc_dice_3d, calc_hd95_3d
from src.data.processors import get_preprocessor

# ─── BraTS Color Scheme ───────────────────────────────────────────────────────
WT_COLOR = np.array([0.2, 0.8, 0.2, 0.45])  # Green
TC_COLOR = np.array([0.9, 0.2, 0.2, 0.55])  # Red
ET_COLOR = np.array([1.0, 0.9, 0.0, 0.65])  # Yellow

# ─── Helpers ──────────────────────────────────────────────────────────────────

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

def run_inference_3d(model, vols, device, batch_size=16):
    stack_4d = np.stack([vols["flair"], vols["t1"], vols["t1ce"], vols["t2"]], axis=0)
    stack_4d = np.transpose(stack_4d, (3, 0, 1, 2))  # (155, 4, 240, 240)
    
    all_preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, 155, batch_size):
            batch = torch.from_numpy(stack_4d[i:i+batch_size].astype(np.float32)).to(device)
            pred = (torch.sigmoid(model(batch)) > 0.5).cpu().numpy().astype(np.uint8)
            all_preds.append(pred)
            
    return np.concatenate(all_preds, axis=0)  # (155, 3, 240, 240)

def build_overlay(img_slice, wt_mask, tc_mask, et_mask):
    vmin, vmax = img_slice.min(), img_slice.max()
    img_norm = (img_slice - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(img_slice)
    canvas = np.stack([img_norm, img_norm, img_norm, np.ones_like(img_norm)], axis=-1)
    
    for mask, color in [(wt_mask, WT_COLOR), (tc_mask, TC_COLOR), (et_mask, ET_COLOR)]:
        loc = mask > 0
        for c in range(3):
            canvas[loc, c] = canvas[loc, c] * (1 - color[3]) + color[c] * color[3]
    return canvas

def find_best_slice(seg_3d):
    counts = np.array([(seg_3d[:, :, s] > 0).sum() for s in range(seg_3d.shape[2])])
    return int(np.argmax(counts))

# ─── Plotting Functions ───────────────────────────────────────────────────────

def save_subject_results(sid, vols, pred_vol, metrics, case_dir):
    os.makedirs(case_dir, exist_ok=True)
    seg = vols["seg"]
    s_idx = find_best_slice(seg)
    
    # 1. metrics.txt
    with open(os.path.join(case_dir, "metrics.txt"), "w") as f:
        f.write(f"Subject: {sid}\n")
        f.write(f"WT Dice: {metrics['WT']:.2f}%\n")
        f.write(f"TC Dice: {metrics['TC']:.2f}%\n")
        f.write(f"ET Dice: {metrics['ET']:.2f}%\n")
        f.write(f"Mean Dice: {metrics['Mean']:.2f}%\n")
        f.write(f"HD95 ET: {metrics['HD95_ET']:.2f}\n")

    # 2. overlay.png (1x4)
    flair_s = vols["flair"][:, :, s_idx]
    t1ce_s  = vols["t1ce"][:, :, s_idx]
    seg_s   = seg[:, :, s_idx]
    
    wt_gt = (seg_s > 0).astype(np.uint8)
    tc_gt = np.logical_or(seg_s==1, seg_s==4).astype(np.uint8)
    et_gt = (seg_s==4).astype(np.uint8)
    
    wt_p, tc_p, et_p = pred_vol[s_idx, 0], pred_vol[s_idx, 1], pred_vol[s_idx, 2]
    
    gt_overlay = build_overlay(flair_s, wt_gt, tc_gt, et_gt)
    p_overlay  = build_overlay(flair_s, wt_p, tc_p, et_p)
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor("#0d1117")
    
    imgs = [flair_s, t1ce_s, gt_overlay, p_overlay]
    titles = ["FLAIR", "T1ce", "Ground Truth", "Prediction"]
    
    for i in range(4):
        ax = axes[i]
        ax.set_facecolor("#0d1117")
        if i < 2:
            ax.imshow(imgs[i].T, cmap="gray", origin="lower")
        else:
            ax.imshow(imgs[i].transpose(1, 0, 2), origin="lower")
        ax.axis("off")
        ax.set_title(titles[i], color="white", fontsize=14)
    
    plt.suptitle(f"{sid} | Dice: WT={metrics['WT']:.2f}%, TC={metrics['TC']:.2f}%, ET={metrics['ET']:.2f}%", 
                 color="white", y=0.98, fontsize=16)
    plt.savefig(os.path.join(case_dir, "overlay.png"), facecolor="#0d1117", bbox_inches="tight")
    plt.close()

    # 3. slices.png (2x5)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.patch.set_facecolor("#0d1117")
    
    offsets = [-2, -1, 0, 1, 2]
    for i, offset in enumerate(offsets):
        idx = np.clip(s_idx + offset, 0, 154)
        f_s = vols["flair"][:, :, idx]
        s_s = seg[:, :, idx]
        
        wt_g = (s_s > 0).astype(np.uint8)
        tc_g = np.logical_or(s_s==1, s_s==4).astype(np.uint8)
        et_g = (s_s==4).astype(np.uint8)
        
        wt_m, tc_m, et_m = pred_vol[idx, 0], pred_vol[idx, 1], pred_vol[idx, 2]
        
        gt_o = build_overlay(f_s, wt_g, tc_g, et_g)
        pr_o = build_overlay(f_s, wt_m, tc_m, et_m)
        
        axes[0, i].imshow(gt_o.transpose(1, 0, 2), origin="lower")
        axes[1, i].imshow(pr_o.transpose(1, 0, 2), origin="lower")
        axes[0, i].set_title(f"GT Slice {idx}", color="white")
        axes[1, i].set_title(f"Pred Slice {idx}", color="white")
        axes[0, i].axis("off"); axes[1, i].axis("off")
        
    plt.savefig(os.path.join(case_dir, "slices.png"), facecolor="#0d1117", bbox_inches="tight")
    plt.close()

def make_qualitative_grid(cases, out_dir):
    fig, axes = plt.subplots(3, 4, figsize=(18, 14))
    fig.patch.set_facecolor("#0d1117")
    titles = ["FLAIR", "T1ce", "Ground Truth", "Prediction"]
    
    for row, (label, (sid, metrics, vols, pred_vol)) in enumerate(cases):
        s_idx = find_best_slice(vols["seg"])
        f_s = vols["flair"][:, :, s_idx]
        t_s = vols["t1ce"][:, :, s_idx]
        seg_s = vols["seg"][:, :, s_idx]
        
        wt_g = (seg_s > 0).astype(np.uint8)
        tc_g = np.logical_or(seg_s==1, seg_s==4).astype(np.uint8)
        et_g = (seg_s==4).astype(np.uint8)
        
        wt_p, tc_p, et_p = pred_vol[s_idx, 0], pred_vol[s_idx, 1], pred_vol[s_idx, 2]
        
        gt_o = build_overlay(f_s, wt_g, tc_g, et_g)
        pr_o = build_overlay(f_s, wt_p, tc_p, et_p)
        
        for col, img in enumerate([f_s, t_s, gt_o, pr_o]):
            ax = axes[row, col]
            ax.set_facecolor("#0d1117")
            if col < 2:
                ax.imshow(img.T, cmap="gray", origin="lower")
            else:
                ax.imshow(img.transpose(1, 0, 2), origin="lower")
            ax.axis("off")
            if row == 0:
                ax.set_title(titles[col], color="white", fontweight="bold", fontsize=14)
        
        axes[row, 0].text(-30, 120, f"{label}\n{sid[-6:]}\nMean Dice: {metrics['Mean']:.2f}%", 
                          color="white", ha="center", va="center", fontweight="bold", rotation=90, fontsize=12)

    # Legend
    legend_patches = [
        mpatches.Patch(color=WT_COLOR[:3], alpha=0.8, label="Edema / WT"),
        mpatches.Patch(color=TC_COLOR[:3], alpha=0.8, label="Necrotic / TC"),
        mpatches.Patch(color=ET_COLOR[:3], alpha=0.8, label="Enhancing / ET"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=14, 
               framealpha=0.3, labelcolor="white", facecolor="#21262d", edgecolor="gray", bbox_to_anchor=(0.5, 0.02))

    plt.suptitle("Qualitative Segmentation Grid — BraTS 2020", color="white", fontsize=18, fontweight="bold", y=0.96)
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    plt.savefig(os.path.join(out_dir, "fig1_qualitative_grid.png"), facecolor="#0d1117")
    plt.close()

def make_boxplot(all_metrics, out_dir):
    data = [ [m['WT'] for m in all_metrics], [m['TC'] for m in all_metrics], [m['ET'] for m in all_metrics] ]
    labels = ["Whole Tumor (WT)", "Tumor Core (TC)", "Enhancing (ET)"]
    colors = ["#2ea043", "#e85c47", "#d29922"]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    
    bp = ax.boxplot(data, patch_artist=True, notch=False, 
                    medianprops=dict(color="white", linewidth=2.5),
                    whiskerprops=dict(color="gray"), capprops=dict(color="gray"),
                    flierprops=dict(marker="o", color="gray", markersize=5, alpha=0.5))
    
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    
    for i, d in enumerate(data, 1):
        ax.scatter(i, np.mean(d), color="white", s=80, zorder=5, marker="D")
        ax.text(i+0.12, np.mean(d), f"{np.mean(d):.2f}%", color="white", fontsize=11, fontweight="bold")
        
    ax.set_xticklabels(labels, color="white", fontsize=12)
    ax.set_ylabel("Dice Score (%)", color="white", fontsize=12)
    ax.tick_params(colors="gray")
    ax.spines[:].set_color("#30363d")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", color="#30363d", linestyle="--", alpha=0.5)
    
    plt.title("Per-subject Dice Distribution (3D Volume)", color="white", fontsize=15, fontweight="bold", pad=15)
    plt.savefig(os.path.join(out_dir, "fig2_boxplot_dice.png"), facecolor="#0d1117", bbox_inches="tight")
    plt.close()

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to BraTS2020 Training Data root")
    parser.add_argument("--n_subjects", type=int, default=-1, help="Limit number of subjects for fast testing")
    parser.add_argument("--ckpt_path", type=str, default="outputs/exp018/best_model.pth", help="Path to best_model.pth")
    parser.add_argument("--results_path", type=str, default="outputs/exp018/results.json", help="Path to results.json")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    out_dir = "outputs/figures/exp018"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Model Fix
    model = AttentionUNet2D(n_channels=4, n_classes=3, init_features=64).to(device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device, weights_only=True))
    model.eval()
    print(f"[INFO] Loaded model successfully from {args.ckpt_path}")
    
    # 2. Setup Data
    preprocessor = get_preprocessor("zscore_clip")
    
    test_subjects = []
    if os.path.exists(args.results_path):
        with open(args.results_path, "r") as f:
            res = json.load(f)
        test_subjects = res.get("test_subjects", [])
    else:
        print(f"[WARNING] results.json not found at {args.results_path}. Falling back to auto-split.")
    
    if not test_subjects:
        import csv
        all_subjects = sorted([s for s in os.listdir(args.data_dir) if s.startswith("BraTS20_Training_")])
        hgg, lgg = [], []
        csv_path = os.path.join(args.data_dir, "name_mapping.csv")
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    sid = row["BraTS_2020_subject_ID"]
                    if sid in all_subjects:
                        (hgg if row["Grade"] == "HGG" else lgg).append(sid)
        else:
            hgg, lgg = all_subjects[:293], all_subjects[293:]
        
        np.random.seed(42)
        np.random.shuffle(hgg)
        np.random.shuffle(lgg)
        n_hgg_tr, n_hgg_va = int(len(hgg)*0.8), int(len(hgg)*0.1)
        n_lgg_tr, n_lgg_va = int(len(lgg)*0.8), int(len(lgg)*0.1)
        test_subjects = hgg[n_hgg_tr+n_hgg_va:] + lgg[n_lgg_tr+n_lgg_va:]
        print(f"[INFO] Auto-detected {len(test_subjects)} test subjects từ stratified split")
        
    if args.n_subjects > 0:
        test_subjects = test_subjects[:args.n_subjects]
    
    print(f"[INFO] Evaluating {len(test_subjects)} test subjects...")
    
    # 3. Inference
    subject_data = []
    
    for sid in tqdm(test_subjects, desc="Processing Cases"):
        try:
            vols = load_subject(args.data_dir, sid, preprocessor)
            pred_vol = run_inference_3d(model, vols, device)
            
            seg = np.transpose(vols["seg"], (2, 0, 1)) # (155, 240, 240)
            
            wt_g = (seg > 0).astype(np.float32)
            tc_g = np.logical_or(seg == 1, seg == 4).astype(np.float32)
            et_g = (seg == 4).astype(np.float32)
            
            wt_p = pred_vol[:, 0, :, :].astype(np.float32)
            tc_p = pred_vol[:, 1, :, :].astype(np.float32)
            et_p = pred_vol[:, 2, :, :].astype(np.float32)
            
            metrics = {
                "WT": calc_dice_3d(wt_p, wt_g) * 100,
                "TC": calc_dice_3d(tc_p, tc_g) * 100,
                "ET": calc_dice_3d(et_p, et_g) * 100,
                "HD95_ET": calc_hd95_3d(et_p, et_g)
            }
            metrics["Mean"] = (metrics["WT"] + metrics["TC"] + metrics["ET"]) / 3.0
            
            subject_data.append((sid, metrics, vols, pred_vol))
        except Exception as e:
            print(f"\n[WARNING] Error processing subject {sid}. Skipping... Details: {str(e)}")
            traceback.print_exc()
            continue
            
    if not subject_data:
        print("[ERROR] No subjects processed successfully. Exiting.")
        return
        
    # 4. Classify cases (sort by Mean Dice)
    subject_data.sort(key=lambda x: x[1]["Mean"], reverse=True) # High to Low
    n = len(subject_data)
    
    top_30_idx = max(1, int(n * 0.30))
    bottom_30_idx = n - top_30_idx
    
    good_cases = subject_data[:top_30_idx]
    medium_cases = subject_data[top_30_idx:bottom_30_idx]
    bad_cases = subject_data[bottom_30_idx:]
    
    # In case of small datasets
    if n < 3:
        good_cases = subject_data
        medium_cases, bad_cases = [], []
        
    # 5. Export individual subjects
    for group, label in [(good_cases, "good_cases"), (medium_cases, "medium_cases"), (bad_cases, "bad_cases")]:
        for sid, metrics, vols, pred_vol in tqdm(group, desc=f"Exporting {label}"):
            save_subject_results(sid, vols, pred_vol, metrics, os.path.join(out_dir, label, sid))

    # 6. Summary Figures
    summary_dir = os.path.join(out_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    if n >= 3:
        # Note: bad_cases are sorted high-to-low internally, so bad_cases[len//2] is the median of bad
        rep_cases = [
            ("Bad Case", bad_cases[len(bad_cases)//2]),
            ("Medium Case", medium_cases[len(medium_cases)//2]),
            ("Good Case", good_cases[len(good_cases)//2])
        ]
        make_qualitative_grid(rep_cases, summary_dir)
    
    make_boxplot([x[1] for x in subject_data], summary_dir)
    
    # 7. Final Summary
    overall_mean = np.mean([x[1]["Mean"] for x in subject_data])
    
    print("\n" + "="*50)
    print(" VISUALIZATION SUMMARY ")
    print("="*50)
    print(f" Total subjects processed: {n}")
    print(f" - Good cases   : {len(good_cases)}")
    print(f" - Medium cases : {len(medium_cases)}")
    print(f" - Bad cases    : {len(bad_cases)}")
    print(f" Overall Mean Dice : {overall_mean:.2f}%")
    print(f"\n✅ All outputs saved to: {out_dir}")
    print("="*50)

if __name__ == "__main__":
    main()
