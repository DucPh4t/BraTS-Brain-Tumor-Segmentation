# 🧠 BraTS2020 & 2023 Systematic Ablation Study Summary (53 Experiments)

## 📌 Complete Pipeline Evolution Flow
```text
Phase 1: Baselines & Sampling (Exp001 - Exp008)
   │
   ▼
Phase 2: Loss, Optimization & Attention (Exp009 - Exp018)
   │
   ▼
Phase 3: Multi-Modal Stems & Disentangled Fusion (Exp019 - Exp034)
   │
   ▼
Phase 4: Region-Specific Decoding & Hierarchy Consistency (Exp035 - Exp041)
   │
   ▼
Phase 5: ResNet34 Trunk & Boundary Loss Calibration (Exp042 - Exp051)
   │
   ▼
Phase 6: State Space Models (2.5D Mamba Residual Adapters) (Exp052 - Exp053)
   │
   ▼
Phase 7: External Dataset Cross-Validation on BraTS 2023 GLI
```

---

## 📊 Bảng Tổng Hợp Kết Quả Toàn Diện (53 Thí Nghiệm)

| Nhóm / Exp | Chiến lược & Kiến trúc | Dice WT ↑ | Dice TC ↑ | Dice ET ↑ | **Mean Dice ↑** | HD95 WT ↓ | HD95 TC ↓ | HD95 ET ↓ | **Mean HD95 ↓** | Trạng thái / Đánh giá |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **A: Split** | | | | | | | | | | |
| `Exp001` | Sequential Split | 85.46 | 80.39 | 78.47 | 81.44 | 11.87 | 8.22 | 14.43 | 11.51 | ❌ Rò rỉ dữ liệu (ET inflate) |
| `Exp002` | **Stratified HGG/LGG Split** | **86.03** | **81.67** | **71.93** | **79.88** | **6.89** | **9.50** | **23.56** | **13.32** | ✅ Chuẩn hóa phân phối |
| **B: Normalization** | | | | | | | | | | |
| `Exp003` | Z-score Raw | 88.06 | 80.16 | 77.13 | 81.78 | 5.28 | 6.22 | 14.04 | 8.51 | Ổn định WT/ET, TC giảm |
| `Exp004` | **Z-score Clip (1% - 99%)** | **89.05** | **85.07** | **78.40** | **84.17** | **4.78** | **5.32** | **22.02** | **10.71** | ✅ Tiền xử lý tối ưu |
| `Exp033` | CLAHE T1ce only | 85.01 | 81.05 | 72.14 | 79.40 | 16.17 | 8.54 | 24.80 | 16.50 | ❌ Phá dải cường độ |
| `Exp034` | CLAHE FLAIR + T1ce | 86.70 | 78.67 | 74.63 | 80.00 | 8.68 | 6.84 | 15.50 | 10.34 | ❌ Giảm mạnh Mean Dice |
| **C: Augmentation** | | | | | | | | | | |
| `Exp005` | **Horizontal & Vertical Flip** | **90.00** | **82.25** | **77.78** | **83.34** | **4.32** | **5.02** | **13.80** | **7.71** | ✅ Giảm mạnh HD95-ET |
| `Exp006` | Flip + Random Intensity | 88.46 | 82.47 | 75.12 | 82.01 | 5.62 | 6.74 | 23.06 | 11.81 | ❌ Giảm độ chính xác |
| **D: Sampling** | | | | | | | | | | |
| `Exp007` | Weighted 3:1 (Tumor:Non-tumor) | 90.16 | 84.29 | 76.68 | 83.71 | 5.10 | 5.03 | 22.55 | 10.89 | Tăng Dice nhẹ, hỏng HD95 |
| `Exp008` | Oversample x2 | 88.26 | 82.89 | 76.03 | 82.39 | 5.23 | 6.02 | 23.12 | 11.46 | ❌ Kém hiệu quả |
| `Exp044` | Exp043 + Weighted 3:1 Sampling | 90.70 | 84.48 | 72.75 | 82.64 | 5.60 | 4.94 | 23.95 | 11.50 | ❌ Giảm mạnh TC/ET trên ResNet |
| **E: Loss Function** | | | | | | | | | | |
| `Exp009` | Binary Cross-Entropy (BCE) | 91.08 | 84.08 | 74.60 | 83.25 | 3.64 | 5.04 | 22.94 | 10.54 | WT rất cao, ET sụt giảm |
| `Exp010` | Focal Tversky Loss | 85.22 | 66.53 | 66.29 | 72.68 | 9.99 | 9.80 | 33.95 | 17.91 | ❌ Quá nặng cho U-Net nhỏ |
| `Exp011` | **Dice + BCE Loss** | **90.21** | **84.63** | **77.41** | **84.08** | **4.25** | **5.00** | **22.65** | **10.63** | ✅ Kết hợp tối ưu |
| `Exp012` | Dice + Focal Loss | 89.60 | 81.95 | 75.57 | 82.37 | 8.50 | 5.66 | 23.75 | 12.64 | Không vượt được Dice+BCE |
| `Exp049` | Exp043 + Dice-FocalTversky | 90.20 | 83.74 | 73.72 | 82.55 | 4.33 | 5.18 | 24.18 | 11.23 | ❌ Vẫn thất bại trên ResNet |
| **F: Optimizer & Scheduler** | | | | | | | | | | |
| `Exp013` | **AdamW (lr=1e-4)** | **90.21** | **84.63** | **77.41** | **84.08** | **4.25** | **5.00** | **22.65** | **10.63** | ✅ Chuẩn tối ưu ổn định |
| `Exp014` | AdamW + ReduceLROnPlateau | 89.70 | 82.80 | 77.82 | 83.44 | 4.21 | 7.06 | 22.34 | 11.20 | Giảm TC |
| `Exp015` | AdamW + CosineAnnealing | 90.62 | 84.69 | 75.43 | 83.58 | 5.91 | 5.08 | 32.02 | 14.34 | Làm tăng vọt HD95-ET |
| `Exp016` | AdamW + Poly Scheduler | 91.08 | 85.79 | 76.17 | 84.35 | 5.64 | 5.69 | 32.09 | 14.47 | HD95-ET kém |
| **G: Architecture** | | | | | | | | | | |
| `Exp017` | UNet64 (Baseline 64 filters) | 89.71 | 83.45 | 74.67 | 82.61 | 4.97 | 6.44 | 33.58 | 15.00 | Tăng params nhưng giảm ET |
| `Exp018` | Attention U-Net | 90.47 | 84.40 | 77.33 | 84.07 | 7.16 | 5.76 | 14.29 | 9.07 | Cải thiện HD95-ET rất tốt |
| `Exp019` | Multi-Modal Separate Stems | 90.37 | 84.40 | 78.64 | 84.47 | 4.50 | 6.12 | 22.51 | 11.04 | Tách biệt modality đầu vào |
| **H: Disentangled Fusion** | | | | | | | | | | |
| `Exp020` | Stems + Contrastive Loss | 90.65 | 84.24 | 75.55 | 83.48 | 4.40 | 5.51 | 23.50 | 11.14 | Contrastive làm giảm ET |
| `Exp021` | **Disentangled Fusion (Shared+Private)** | **91.00** | **85.62** | **78.21** | **84.94** | **6.12** | **4.84** | **22.72** | **11.23** | ✅ Best 2D Fusion |
| `Exp022` | Disentangled + Contrastive | 90.05 | 84.30 | 78.26 | 84.20 | 6.16 | 5.60 | 22.41 | 11.39 | Không cải thiện |
| `Exp028` | 2.5D Disentangled Fusion (3-slice) | 88.49 | 81.30 | 75.83 | 81.87 | 7.85 | 6.79 | 23.58 | 12.74 | ❌ Thất bại do nén quá sớm |
| `Exp032` | Disentangled + Attention Skip | 89.57 | 84.37 | 74.77 | 82.90 | 5.14 | 5.17 | 23.65 | 11.32 | ❌ Xung đột Attention |
| **I: Inference & Post-processing** | | | | | | | | | | |
| `Exp023` | **Exp021 + TTA Inference** | **91.21** | **85.92** | **78.32** | **85.15** | **4.51** | **4.81** | **22.67** | **10.66** | ✅ Best old baseline |
| `Exp024` | Validation Threshold Tuning | 91.02 | 85.63 | 78.13 | 84.93 | 6.13 | 4.85 | 22.73 | 11.24 | Overfitting ngưỡng |
| `Exp025` | TTA + Threshold Tuning | 91.29 | 85.84 | 78.11 | 85.08 | 6.08 | 4.78 | 22.69 | 11.18 | Không vượt Exp023 |
| `Exp026` | Exp025 + Connected Component (CC) | 91.29 | 85.89 | 78.11 | 85.10 | 6.08 | 4.78 | 22.69 | 11.18 | CC cleanup trung tính |
| `Exp027` | Exp025 + ET Empty Rescue | 91.29 | 85.84 | 78.11 | 85.08 | 6.08 | 4.78 | 22.69 | 11.18 | Không có tác động |
| `Exp029` | **Exp018 (Attention) + TTA** | **90.57** | **84.75** | **77.54** | **84.29** | **6.20** | **5.04** | **14.19** | **8.48** | ✅ Rất ổn định vùng biên |
| `Exp030` | Exp018 + TTA + Threshold Tuning | 90.50 | 84.71 | 74.82 | 83.34 | 6.18 | 5.05 | 23.89 | 11.71 | ❌ Ép ngưỡng làm chết ET |
| `Exp031` | Exp021 + 3D-Validation Selection | 90.88 | 84.56 | 77.38 | 84.27 | 5.77 | 6.00 | 23.05 | 11.61 | Không vượt chọn theo 2D |
| `Exp051` | Exp043 + Tuned ET Cleanup (min=20) | 90.81 | 86.42 | 79.32 | 85.52 | 6.20 | 4.49 | 21.79 | 10.83 | Gain +0.01% rất nhỏ |
| **N: Region-Hierarchy Breakdown** | | | | | | | | | | |
| `Exp035` | Region-Specific Heads Only | 89.69 | 82.84 | 76.78 | 83.10 | 5.16 | 5.91 | 22.83 | 11.30 | ❌ Tách head rời bị lệch |
| `Exp036` | **Region Heads + Hierarchy Loss** | **90.98** | **85.92** | **78.84** | **85.25** | **5.86** | **4.54** | **13.79** | **8.06** | 🌟 Đột phá lý thuyết chính |
| `Exp037` | Exp036 + ET Presence / Gate | 90.26 | 85.59 | 77.22 | 84.36 | 4.43 | 4.58 | 23.61 | 10.87 | Gate toàn cục bị cứng |
| `Exp038` | Exp036 + Multi-scale Disentangled | 90.42 | 85.42 | 78.45 | 84.76 | 7.94 | 6.20 | 23.00 | 12.38 | Inject nhiễu từ private |
| `Exp039` | Exp036 + Boundary-aware Loss | 90.08 | 84.08 | 78.22 | 84.13 | 9.13 | 5.97 | 22.69 | 12.60 | Xung đột hàm mục tiêu |
| `Exp041` | Exp036 + TTA Inference | 91.05 | 86.37 | 79.09 | 85.50 | 6.27 | 4.16 | 22.37 | 10.93 | Tăng Dice, HD95 tăng |
| **P/Q: Backbone & Loss Calibration** | | | | | | | | | | |
| `Exp042` | ResNet34 U-Net (Single Head) | 91.39 | 86.32 | 75.73 | 84.48 | 5.69 | 5.00 | 32.20 | 14.30 | ET bị giảm mạnh |
| 🥇 **`Exp043`** | **ResNet34 + Region Heads + Hierarchy** | **90.81** | **86.42** | **79.29** | **85.51** | **6.20** | **4.49** | **21.84** | **10.84** | 🏆 **BEST TRAINABLE MODEL** |
| 🎯 **`Exp045`** | **Exp043 + Mild Weighted BCE (1/1.2/1.5)** | **90.48** | **85.98** | **78.41** | **84.96** | **4.01** | **4.63** | **13.62** | **7.42** | 📐 **BEST HD95 BOUNDARY** |
| `Exp046` | Exp043 + Adaptive ET-BCE in GT-TC | 91.31 | 86.81 | 76.31 | 84.81 | 3.72 | 4.14 | 23.26 | 10.37 | Suppress nhầm ET âm tính |
| `Exp047` | Exp043 + ET-positive Adaptive BCE | 90.69 | 86.84 | 76.51 | 84.68 | 6.56 | 4.76 | 22.86 | 11.39 | Cải thiện nhẹ Exp046 |
| `Exp048` | Exp043 + ET-only Boundary Loss | 90.29 | 85.37 | 74.80 | 83.49 | 4.15 | 4.48 | 31.58 | 13.40 | Gây sụt giảm ET |
| `Exp050` | Exp043 + Multi-scale Deep Supervision | 91.45 | 86.39 | 76.42 | 84.75 | 6.04 | 5.11 | 23.42 | 11.52 | Tín hiệu tốt ở case khó |
| **X/Y: State Space Models (Mamba)** | | | | | | | | | | |
| 🐍 `Exp052` | **2.5D Mamba Residual Adapter (30x30)** | **90.17** | **85.14** | **77.66** | **84.32** | **6.30** | **4.95** | **23.32** | **11.52** | Đóng góp học thuật 2.5D Mamba |
| `Exp052_16b`| Exp052 với Batch Size 16 | 90.47 | 84.51 | 74.51 | 83.16 | 6.34 | 5.44 | 23.67 | 11.82 | Giảm ET khi batch lớn |
| 🐍 `Exp053` | **60x60 Mamba + Reliability Gate** | **90.35** | **86.39** | **75.92** | **84.22** | **6.28** | **4.70** | **22.60** | **11.19** | TC đạt 86.39% rất cao |
| **External Validation** | | | | | | | | | | |
| 🌐 `Exp043-2023`| **Exp043 Test trên BraTS 2023 GLI** | **90.17** | **87.25** | **82.61** | **86.68** | **10.00** | **9.15** | **12.47** | **10.54** | 🚀 **Khả năng tổng quát hóa vượt trội** |

---

## 🔍 Các Đúc Kết Khoa Học Quan Trọng Nhất

### 1. Khám phá cốt lõi: Region-Specific Heads bắt buộc phải đi kèm Hierarchy Loss
- **Hiện tượng**: `Exp035` tách 3 đầu ra riêng biệt cho WT, TC, ET nhưng hiệu năng sụt giảm nghiêm trọng (`Mean Dice 83.10%`).
- **Nguyên nhân**: 3 vùng u não có tính chất lồng ghép giải phẫu ($ET \subseteq TC \subseteq WT$). Khi tách head độc lập mà không ràng buộc, mô hình dự đoán rời rạc và sinh ra các pixel mâu thuẫn (ví dụ: dự đoán có ET nhưng không có TC/WT).
- **Giải pháp (`Exp036`, `Exp043`)**: Đưa vào hàm phạt vi phạm phân cấp mềm ở mức xác suất (**Hierarchy Consistency Loss**):
  $$\mathcal{L}_{hier} = \frac{1}{|V|} \sum_{v} \left( \text{ReLU}(P_{ET}(v) - P_{TC}(v)) + \text{ReLU}(P_{TC}(v) - P_{WT}(v)) \right)$$
- **Kết quả**: Kéo Mean Dice tăng vọt lên **85.51%** và giảm mạnh lỗi không nhất quán.

### 2. Sự đánh đổi giữa Thể tích (Dice) và Độ mịn đường biên (HD95)
- **`Exp043`** (Loss Dice+BCE chuẩn + $0.1 \mathcal{L}_{hier}$): Đạt **Mean Dice cao nhất (85.51%)**, phân đoạn khối thể tích u chính xác nhất.
- **`Exp045`** (Bổ sung Channel-Weighted BCE `1.0 / 1.2 / 1.5`): Tăng trọng số phạt cho các voxel TC và ET, giúp **Mean HD95 giảm ngoạn mục từ 10.84 mm xuống 7.42 mm** (HD95-WT đạt 4.01 mm, HD95-ET đạt 13.62 mm).

### 3. Đánh giá Mamba 2.5D Residual Adapter (`Exp052`, `Exp053`)
- Khác với U-Net 2.5D truyền thống nén kênh ngay từ đầu làm mất thông tin lát cắt (`Exp028` chỉ đạt 81.87%), cơ chế **Mamba Residual Adapter** xử lý độc lập từng modality qua 4 hướng quét không gian và lát cắt, sau đó cộng phần bù $\Delta$ vào lát trung tâm.
- Mô hình đạt **Mean Dice 84.32%** với số tham số bổ sung cực nhỏ (~24.6M params), chứng minh tiềm năng to lớn của State Space Models trong việc nắm bắt quan hệ liên lát cắt MRI.

---

## 🎯 Case-Level Diagnostic: Phân Tích Các Ca Siêu Khó

| Subject ID | Nhóm Ca | Vấn đề của Baseline | Kết quả với Exp043 / Exp045 |
| :--- | :--- | :--- | :--- |
| `BraTS20_Training_236` | U nhỏ, ranh giới mờ | Baseline Dice TC/ET cực thấp (`31.0% / 26.8%`) | `Exp043` tăng vọt TC/ET lên **`55.6% / 49.6%`**, HD95-ET giảm từ 12.88mm xuống 8.31mm. |
| `BraTS20_Training_273` | U vỏ não, hình dạng dị thường | ET Dice chỉ đạt 35.7% | Giữ vững WT/TC > 88%, giảm false-positive xung quanh màng não. |
| `BraTS20_Training_279` | LGG không có ET ($GT_{ET} = 0$) | Mô hình cũ bị False Positive ET $\rightarrow$ HD95 = 373 mm | `Exp045` và `Exp051` ức chế hoàn toàn các đốm nhiễu ET giả. |
| `BraTS20_Training_307` | ET siêu nhỏ (chỉ 32 voxels) | Bị bỏ sót hoàn toàn ($ET=0$) | `Exp043` nhận diện đúng vị trí lõi u, `Exp045` hạ HD95-ET xuống 19.13 mm. |
| `BraTS20_Training_280` | Lõi TC cực nhỏ ($GT_{TC} = 826$ voxels) | TC Dice bị phạt nặng do lệch tâm | Giữ được cấu trúc WT 85.80% và ET 80.09%. |
