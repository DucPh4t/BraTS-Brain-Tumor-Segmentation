# 🏆 Cẩm Nang Cấu Hình Mô Hình Xuất Sắc Nhất (Best Configurations Guide)

Tài liệu này tổng hợp và phân loại các cấu hình **mạnh nhất và tiêu biểu nhất** của toàn bộ đề tài ĐATN MRI (53 thí nghiệm), phục vụ cho việc lựa chọn mô hình báo cáo trong luận văn, slide bảo vệ và ứng dụng thực tế.

---

## 🥇 1. Mô Hình Chủ Lực Toàn Diện (Primary Champion): `Exp043`

> **Đại diện chính cho Đồ Án Tốt Nghiệp — Đạt Mean Dice cao nhất trên toàn bộ 53 thí nghiệm.**

### 📋 Thông số kỹ thuật chi tiết:
| Thành phần | Cấu hình & Giá trị | Ý nghĩa kỹ thuật |
| :--- | :--- | :--- |
| **File cấu hình** | `configs/exp043.yaml` | Tái lập chính xác 100% bằng lệnh CLI |
| **File checkpoint** | `outputs/exp043/best_model.pth` | Checkpoint lưu lúc 3D validation Dice đạt cực đại |
| **Kiến trúc Encoder** | `ResNet34` (khởi tạo ImageNet) | Trích xuất đặc trưng sâu, độ phức tạp tính toán vừa phải |
| **Kiến trúc Decoder** | `U-Net Decoder` với Skip Connections | Khôi phục độ phân giải và bảo toàn ranh giới chi tiết |
| **Đầu ra dự đoán** | **3 Region-Specific Heads (1x1 Conv)** | Tách riêng 3 luồng dự đoán cho WT, TC, ET |
| **Hàm mất mát** | $0.5 \mathcal{L}_{Dice} + 0.5 \mathcal{L}_{BCE} + 0.1 \mathcal{L}_{hier}$ | Kết hợp phân đoạn chuẩn với ràng buộc phân cấp giải phẫu |
| **Tiền xử lý** | `Z-score Clip` (1% - 99%) | Chuẩn hóa cường độ ảnh MRI không phá hỏng phân phối gốc |
| **Tăng cường dữ liệu** | `Random Horizontal & Vertical Flip` | Tăng độ bền vững hình học mà không méo mó ảnh |
| **Lấy mẫu lát cắt** | `Fixed (All 155 Slices)` | Giữ nguyên trật tự không gian 3D của toàn bộ thể tích não |
| **Bộ tối ưu hóa** | `AdamW (lr=1e-4, weight_decay=1e-5)` | Huấn luyện ổn định và hội tụ mượt mà |

### 📊 Kết quả kiểm thử thực nghiệm (BraTS 2020 Test Set — 39 ca):
- **Mean Dice**: **85.51%** 🥇 *(Cao nhất toàn dự án)*
- **Whole Tumor (WT)**: **90.81%** (Std: $\pm 5.79$, Median: $93.33\%$)
- **Tumor Core (TC)**: **86.42%** (Std: $\pm 12.45$, Median: $91.87\%$)
- **Enhancing Tumor (ET)**: **79.29%** (Std: $\pm 21.25$, Median: $85.76\%$)
- **Mean HD95**: **10.84 mm**

---

## 🎯 2. Mô Hình Chuẩn Xác Vùng Biên Nhất (Best Boundary Precision): `Exp045`

> **Mô hình tối ưu hóa khoảng cách Hausdorff — Đường biên các vùng u mượt và sắc nét nhất.**

### 📋 Điểm khác biệt so với Exp043:
- Giữ nguyên toàn bộ kiến trúc và tiền xử lý của `Exp043`.
- Nâng cấp hàm mất mát với **Mild Channel-Weighted BCE**:
  $$\text{Trọng số BCE}: \quad w_{WT} = 1.0, \quad w_{TC} = 1.2, \quad w_{ET} = 1.5$$
- Tăng mức phạt lỗi cho các vùng u nhỏ khó phân đoạn (TC và ET).

### 📊 Kết quả kiểm thử thực nghiệm:
- **Mean HD95**: **7.42 mm** 🥇 *(Cải thiện vượt bậc từ 10.84 mm xuống 7.42 mm)*
- **HD95 Whole Tumor (WT)**: **4.01 mm** (so với 6.20 mm của Exp043)
- **HD95 Enhancing Tumor (ET)**: **13.62 mm** (so với 21.84 mm của Exp043)
- **Mean Dice**: **84.96%** *(Chỉ giảm nhẹ 0.55% so với Exp043 nhưng chất lượng biên hình học đẹp hơn rõ rệt)*

---

## 🔬 3. Mô Hình Đổi Mới Cấu Trúc Trộn Chuỗi Ảnh: `Exp036`

> **Minh chứng lý thuyết về Disentangled Multi-Modal Fusion + Hierarchy Consistency.**

- **Kiến trúc**: `DisentangledFusionRegionHeadsUNet2D`
- **Nguyên lý**: Tách biệt đặc trưng riêng biệt (*Private Features*) của 4 modality (FLAIR, T1, T1ce, T2) và đặc trưng tương đồng (*Shared Features*), sau đó mới chuyển sang 3 Region Heads có Hierarchy Loss.
- **Kết quả**:
  - **Mean Dice**: **85.25%**
  - **Mean HD95**: **8.06 mm** *(Rất cân bằng giữa cả Dice và HD95)*

---

## 🐍 4. Mô Hình Tiên Phong Ứng Dụng State Space Models (Mamba): `Exp052` / `Exp053`

> **Đóng góp học thuật về ứng dụng State Space Model (Mamba) trích xuất không gian 2.5D.**

- **Kiến trúc**: `Lightweight 2.5D Mamba Residual Adapter + ResNet34 Trunk`
- **Nguyên lý**: Nhận 5 lát cắt liên tiếp ($z-2, z-1, z, z+1, z+2$). Mamba quét 4 hướng đa chiều để học liên kết không gian và độ sâu, sau đó cộng phần bù $\Delta$ vào lát trung tâm trước khi đưa vào ResNet.
- **Kết quả**:
  - **Mean Dice**: **84.32%** (`Exp052`) và **84.22%** (`Exp053`).
  - Chứng minh mô hình State Space Model có thể chạy 2.5D nhẹ nhàng (~24.6M params) mà không bị nghẽn bộ nhớ GPU.

---

## 🌐 5. Đánh Giá Khả Năng Tổng Quát Hóa Ngoài (External Generalization on BraTS 2023)

Khi đem checkpoint **`Exp043`** (chỉ huấn luyện trên BraTS 2020) sang kiểm thử trực tiếp trên bộ dữ liệu **BraTS 2023 GLI** (không fine-tune):
- **Mean Dice**: **86.68%** 🚀 *(WT: 90.17%, TC: 87.25%, ET: 82.61%)*
- **Mean HD95**: **10.54 mm**
- **Kết luận**: Khung mô hình có tính tổng quát hóa cực kỳ vững chắc, hoạt động xuất sắc trên cả dữ liệu thế hệ mới.

---

## 📊 Bảng Đối Chiếu Quyết Định Lựa Chọn Mô Hình

| Mục tiêu sử dụng | Mô hình khuyến nghị | Lệnh chạy tương ứng |
| :--- | :--- | :--- |
| **Báo cáo kết quả chính trong đồ án / Slide** | **`Exp043`** | `python main.py --config configs/exp043.yaml --mode eval` |
| **Minh họa đường biên khối u sắc nét nhất** | **`Exp045`** | `python main.py --config configs/exp045.yaml --mode eval` |
| **Phân tích cơ chế Disentangled Fusion** | **`Exp036`** | `python main.py --config configs/exp036.yaml --mode eval` |
| **Phân tích đóng góp công nghệ mới (Mamba 2.5D)** | **`Exp052` / `Exp053`** | `python main.py --config configs/exp052.yaml --mode eval` |
| **Chứng minh khả năng tổng quát hóa** | **`Exp043` trên BraTS 2023**| `python evaluate_brats2023.py --config configs/exp043.yaml --checkpoint outputs/exp043/best_model.pth` |
