"""
Mô tả: Vẽ biểu đồ so sánh đường cong huấn luyện (Training Curves) phục vụ Luận văn tốt nghiệp.
       Hỗ trợ 2 chế độ vẽ:
       1. So sánh phương pháp lấy mẫu lát cắt (Exp005 vs Exp007 vs Exp008).
       2. So sánh các mốc tiến hóa kiến trúc cốt lõi (Milestones: Exp005 vs Exp036 vs Exp043 vs Exp052).
Đầu vào:
    Đọc tự động file history.json từ thư mục outputs/exp*/ tương ứng.
Đầu ra:
    Lưu đồ thị tại outputs/figures/sampling_training_curves.png và outputs/figures/milestones_training_curves.png.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Định nghĩa tương đối theo project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Cấu hình giao diện biểu đồ chuyên nghiệp
plt.style.use("dark_background")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Inter", "Roboto", "Arial", "DejaVu Sans"]


def get_dice_data(history, n_epochs):
    """
    Trích xuất Mean Dice từ history:
    - Nếu có sẵn key 'val_dice', sử dụng trực tiếp.
    - Nếu không, tính trung bình cộng của 3 vùng: val_wt_dice, val_tc_dice, val_et_dice.
    """
    if "val_dice" in history:
        mean_dice = np.array(history["val_dice"])
    else:
        # Tính toán thủ công từ các vùng
        wt = np.array(history.get("val_wt_dice", [0.0] * n_epochs))
        tc = np.array(history.get("val_tc_dice", [0.0] * n_epochs))
        et = np.array(history.get("val_et_dice", [0.0] * n_epochs))
        mean_dice = (wt + tc + et) / 3
    
    # Đồng bộ thang đo về % (0-100) để khớp trục Y (50-95)
    if mean_dice.size > 0 and mean_dice.max() <= 1.0:
        mean_dice = mean_dice * 100
        
    return mean_dice


def plot_comparison(exps, labels, colors, title, output_name):
    """
    Vẽ 3 biểu đồ con nằm ngang: Training Loss, Validation Loss, và Validation Mean Dice.
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor="#0d1117")
    fig.suptitle(title, color="white", fontsize=18, fontweight="bold", y=1.02)
    
    valid_plots = 0
    
    for exp, label, color in zip(exps, labels, colors):
        history_path = os.path.join("outputs", exp, "history.json")
        if not os.path.exists(history_path):
            print(f"⚠️ Bỏ qua {exp}: Không tìm thấy {history_path}")
            continue
            
        with open(history_path, "r") as f:
            h = json.load(f)
            
        epochs = range(1, len(h["train_loss"]) + 1)
        dice_data = get_dice_data(h, len(epochs))
        
        # 1. Biểu đồ Training Loss
        axes[0].plot(epochs, h["train_loss"], label=label, color=color, linewidth=2.2, alpha=0.9)
        # 2. Biểu đồ Validation Loss
        axes[1].plot(epochs, h["val_loss"], label=label, color=color, linewidth=2.2, alpha=0.9)
        # 3. Biểu đồ Validation Mean Dice
        axes[2].plot(epochs, dice_data, label=label, color=color, linewidth=2.5, alpha=1.0)
        
        valid_plots += 1
        
    if valid_plots == 0:
        print(f"❌ Không có dữ liệu thí nghiệm nào để vẽ cho: {output_name}")
        plt.close()
        return

    # Định dạng các trục tọa độ
    sub_titles = [
        "Training Loss (Độ hội tụ)",
        "Validation Loss (Độ tổng quát)",
        "Validation Mean Dice (Độ chính xác %)"
    ]
    y_labels = ["Loss", "Loss", "Mean Dice Score (%)"]
    
    for i, (ax, sub_title, y_lab) in enumerate(zip(axes, sub_titles, y_labels)):
        ax.set_title(sub_title, color="white", fontsize=14, pad=15, fontweight="semibold")
        ax.set_xlabel("Epochs", color="gray", fontsize=11)
        ax.set_ylabel(y_lab, color="gray", fontsize=11)
        ax.legend(frameon=True, facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
        ax.grid(color="#30363d", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#30363d")
        ax.spines["bottom"].set_color("#30363d")
        ax.tick_params(colors="gray")
        
        # Giới hạn trục Y cho Dice để dễ quan sát từ 50% đến 100%
        if i == 2:
            ax.set_ylim(50, 95)
            
    plt.tight_layout()
    
    # Tạo thư mục lưu trữ nếu chưa có
    save_dir = "outputs/figures"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, output_name)
    
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"✅ Biểu đồ đã được lưu tại: {save_path}")


def main():
    # ─── PHẦN 1: SO SÁNH PHƯƠNG PHÁP LẤY MẪU (SLICE SAMPLING) ─────────────────
    print("🔄 Đang vẽ biểu đồ so sánh Slice Sampling...")
    plot_comparison(
        exps=["exp005", "exp007", "exp008"],
        labels=["Baseline (Fixed)", "Weighted 3:1", "Oversample x2"],
        colors=["#58a6ff", "#e85c47", "#d29922"], # Xanh dương, Đỏ, Vàng
        title="Slice Sampling Strategy Performance Comparison (Group D)",
        output_name="sampling_training_curves.png"
    )
    
    # ─── PHẦN 2: SO SÁNH CÁC MỐC TIẾN HÓA KIẾN TRÚC CHÍNH ───────────────────
    print("🔄 Đang vẽ biểu đồ so sánh các mốc tiến hóa kiến trúc (Milestones)...")
    plot_comparison(
        exps=["exp005", "exp036", "exp043", "exp052"],
        labels=[
            "Exp005: Baseline U-Net 2D",
            "Exp036: Disentangled + Hierarchy",
            "Exp043: ResNet34 + Region Heads + Hierarchy",
            "Exp052: Hybrid 2.5D Mamba Adapter"
        ],
        colors=["#8b949e", "#ff7b72", "#58a6ff", "#79c0ff"], # Xám, Đỏ nhạt, Xanh dương nhạt, Xanh cyan
        title="Key Architectural Milestones & Evolution Performance",
        output_name="milestones_training_curves.png"
    )


if __name__ == "__main__":
    main()