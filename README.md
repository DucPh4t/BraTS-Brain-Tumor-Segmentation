# 🧠 Multi-Modal Brain Tumor Segmentation on BraTS 2020 with External Validation on BraTS 2023
> **A Comprehensive 53-Experiment Investigation: Trained on BraTS 2020 & Evaluated for External Generalization on BraTS 2023 GLI**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch 2.0+">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/Dataset-BraTS%202020%20(Train)%20%7C%202023%20(Test)-green.svg" alt="BraTS Dataset">
  <img src="https://img.shields.io/badge/SSM_Backbone-Mamba-purple.svg" alt="Mamba">
</p>

---

## 📋 Table of Contents
- [📌 Tóm tắt Đề tài Tốt nghiệp (Thesis Abstract)](#-tóm-tắt-đề-tài-tốt-nghiệp-thesis-abstract)
- [🌟 Key Highlights](#-key-highlights)
- [🩺 Terminology & Metrics Definitions](#-terminology--metrics-definitions)
- [🏗️ Architectural Evolution & Experiment Roadmap](#️-architectural-evolution--experiment-roadmap)
- [🧬 Architecture Diagram of Advanced 2.5D Mamba U-Net (Exp052/Exp053)](#-architecture-diagram-of-advanced-25d-mamba-u-net-exp052exp053)
- [📐 Mathematical Formulations](#-mathematical-formulations)
- [📊 Performance & Experimental Results](#-performance--experimental-results)
- [🛠️ Complete Script Toolkit Reference](#️-complete-script-toolkit-reference)
- [⚙️ Hardware Requirements & Setup](#️-hardware-requirements--setup)
- [🚀 Usage Guide](#-usage-guide)
- [📖 Citation & License](#-citation--license)

---

## 📌 Tóm tắt Đề tài Tốt nghiệp (Thesis Abstract)

**Đề tài:** *Phân đoạn khối u não đa phương thái trên ảnh cộng hưởng từ (MRI) sử dụng mô hình kết hợp Mamba và ràng buộc phân cấp giải phẫu.*

Nghiên cứu này tập trung vào việc phát triển một hệ thống học sâu hiệu năng cao để phân đoạn tự động các cấu trúc khối u não trên ảnh MRI đa phương thái (gồm các xung FLAIR, T1, T1ce, và T2) thuộc hai tập dữ liệu chuẩn thức **BraTS 2020** và **BraTS 2023 GLI**. Khung nghiên cứu thực nghiệm được thiết kế có hệ thống thông qua **53 cấu hình thí nghiệm** lũy tiến nhằm giải quyết ba thách thức cốt lõi:
1. **Sự tương tác phức tạp giữa các phương thái MRI**: Đề xuất mô hình kết hợp đa tỷ lệ **Disentangled Feature Fusion** tách biệt các nhánh encoder riêng (`private stems`) cho từng xung trước khi đưa vào không gian biểu diễn chung (`shared trunk`), giúp giữ lại tối đa đặc trưng vật lý của từng loại ảnh.
2. **Khai thác ngữ cảnh không gian 3D**: Thiết kế mô hình thích ứng **Hybrid 2.5D Mamba Adapter** đặt tại bottleneck, tận dụng ưu thế mô hình hóa chuỗi tuần tự hai chiều (Bidirectional Selective State Space) của Mamba để tích hợp ngữ cảnh đa lát cắt trục (axial slices) mà không gây bùng nổ tài nguyên tính toán như tích chập 3D.
3. **Mất nhất quán cấu trúc giải phẫu học**: Xây dựng hàm mất mát phân cấp vùng u khả vi **Region-Hierarchy Loss ($\mathcal{L}_{\text{hier}}$)** nhằm ép buộc xác suất dự đoán tuân thủ chặt chẽ ràng buộc giải phẫu sinh học của các phân vùng u BraTS: Vùng tăng cường (ET) phải nằm trong lõi u (TC), và lõi u phải nằm trong toàn bộ vùng u (WT) ($P_{\text{ET}} \le P_{\text{TC}} \le P_{\text{WT}}$).

Kết quả thực nghiệm mô hình xuất sắc nhất (Exp043) đạt chỉ số **Mean Dice 85.51%** trên tập dữ liệu 3D BraTS 2020 và thể hiện khả năng tổng quát hóa xuất sắc **86.68% Mean Dice** (TC Dice đạt **87.25%**) khi đánh giá chéo trên tập dữ liệu ngoại lai BraTS 2023 GLI mà không cần huấn luyện lại.

---

## 🌟 Key Highlights
- **53 Modular Configurations (`configs/exp001.yaml` → `exp053.yaml`)**: Hoàn toàn kiểm soát và tái lập kết quả qua các file cấu hình YAML được thiết kế tường minh, không chứa các bình luận rác.
- **Two Core Benchmark Models (Hai Mô Hình Trọng Tâm)**:
  - 🥇 **ResNet34 Region-Heads U-Net (`Exp043`)**: Mô hình đạt **hiệu năng thực nghiệm đỉnh cao nhất** (Mean Dice **85.51%** trên BraTS 2020 và **86.68%** trên 1,251 ca BraTS 2023 GLI).
  - 🐍 **Hybrid 2.5D Mamba U-Net (`Exp052/Exp053`)**: Mô hình **tiên phong về mặt kiến trúc & đổi mới sáng tạo**, tích hợp bộ thích ứng Mamba 2.5D Selective State Space tại bottleneck để khai thác ngữ cảnh 3D mà không gây quá tải bộ nhớ.
- **Biologically-Informed Region-Hierarchy Consistency**: Hàm loss phân cấp dựa trên xác suất cấp voxel ($\mathcal{L}_{\text{hier}}$) nhằm triệt tiêu các dự đoán vi phạm cấu trúc hình thái giải phẫu của khối u ($P_{\text{ET}} \le P_{\text{TC}} \le P_{\text{WT}}$).
- **Cross-Dataset Generalization**: Kiểm chứng chéo và đánh giá tính tổng quát hóa trực tiếp trên tập dữ liệu ngoại lai BraTS 2023 GLI.

---

## 🩺 Terminology & Metrics Definitions

### 1. Phân vùng khối u BraTS (Subregions)
- **WT (Whole Tumor - Toàn bộ vùng u)**: Bao gồm toàn bộ vùng u (Hoại tử + Phù nề + U tăng cường). Nhãn Ground Truth: 1 + 2 + 4.
- **TC (Tumor Core - Lõi khối u)**: Bao gồm lõi khối u hoại tử và u tăng cường, không bao gồm vùng phù nề xung quanh. Nhãn Ground Truth: 1 + 4.
- **ET (Enhancing Tumor - U tăng cường)**: Vùng u bắt màu thuốc đối quang từ tăng cường tín hiệu trên xung T1ce. Nhãn Ground Truth: 4.

### 2. Chỉ số Đánh giá (Evaluation Metrics)
- **Dice Similarity Coefficient (DSC)**: Đo lường mức độ trùng lặp không gian giữa mặt nạ dự đoán ($P$) và Ground Truth ($G$):

$$
\text{Dice}(P, G) = \frac{2 |P \cap G|}{|P| + |G|}
$$

- **Hausdorff Distance 95th Percentile (HD95)**: Khoảng cách lớn nhất thứ 95% giữa các điểm trên ranh giới bề mặt u dự đoán và thực tế (tính bằng mm), đo lường độ chính xác biên u.

---

## 🏗️ Architectural Evolution & Experiment Roadmap

```mermaid
flowchart TD
    P1["Phase 1: Baselines & Sampling<br/>(Exp001 – Exp008)"] --> P2["Phase 2: Loss, Optimization & Attention<br/>(Exp009 – Exp018)"]
    P2 --> P3["Phase 3: Multi-Modal Stems & Disentangled Fusion<br/>(Exp019 – Exp034)"]
    P3 --> P4["Phase 4: Region-Specific Decoding & Hierarchy Loss<br/>(Exp035 – Exp041)"]
    P4 --> P5["Phase 5: ResNet34 Trunk & Boundary Loss Calibration<br/>(Exp042 – Exp051)"]
    P5 --> P6["Phase 6: State Space Models (2.5D Mamba Adapters)<br/>(Exp052 – Exp053)"]
    P6 --> P7["Phase 7: External Validation on BraTS 2023 GLI<br/>(1,251 Zero-Shot Cases)"]
```

| Phase / Group | Experiments | Core Novelty & Focus | Key Findings & Top Models |
| :--- | :--- | :--- | :--- |
| **Phase 1: Baselines & Sampling** | `exp001` – `exp008` | U-Net 2D tiêu chuẩn, chuẩn hóa dữ liệu Z-Score Clip, các chiến lược lấy mẫu lát cắt (fixed, weighted, oversampled). | `exp004` (Z-Score Clip: Mean 84.17%), `exp005` (Data Augmentation: Mean 83.34%). |
| **Phase 2: Loss, Optimization & Attention** | `exp009` – `exp018` | Hàm loss Dice+BCE, bộ tối ưu AdamW, cơ chế Attention Gates. | `exp011` (Dice+BCE: Mean 84.08%), `exp018` (Attention U-Net: Mean 84.07%, HD95-ET 14.29mm). |
| **Phase 3: Multi-Modal Stems & Disentangled Fusion** | `exp019` – `exp034` | Nhánh mã hóa riêng biệt cho từng xung MRI (FLAIR, T1, T1ce, T2), kết hợp shared/private (DFuse-Net style), TTA inference. | `exp021` (Disentangled Fusion: Mean 84.94%), `exp023` (Exp021 + TTA: Mean 85.15%). |
| **Phase 4: Region-Specific Decoding & Hierarchy Loss** | `exp035` – `exp041` | Tách biệt các nhánh decoder WT/TC/ET độc lập và áp dụng hàm mất mát phân cấp giải phẫu $\mathcal{L}_{\text{hier}}$. | `exp036` (Region Heads + Hierarchy: Mean 85.25%, HD95-ET 13.79mm - Đột phá lý thuyết). |
| **Phase 5: ResNet34 Trunk & Loss Calibration** | `exp042` – `exp051` | Tích hợp bộ mã hóa pretrained ResNet34, gán trọng số loss thích ứng (Weighted BCE), lọc nhiễu liên thông (CC Cleanup). | 🥇 `exp043` (ResNet34 + Hierarchy: **Mean 85.51%** - Best Baseline), 🎯 `exp045` (Best HD95: 7.42mm). |
| **Phase 6: State Space Models (Mamba)** | `exp052` – `exp053` | Bộ thích ứng 2.5D Mamba Selective State Space tại bottleneck, cổng độ tin cậy đa xung (Cross-Modality Reliability Gate). | 🐍 `exp052` (2.5D Mamba 30x30: Mean 84.32%), `exp053` (60x60 Mamba + Reliability Gate: TC 86.39%). |
| **Phase 7: External Cross-Validation** | BraTS 2023 GLI | Kiểm thử trực tiếp Zero-shot trên toàn bộ 1,251 ca bệnh ngoại lai BraTS 2023 GLI. | 🚀 `Exp043-2023` (**Mean Dice 86.68%**, **Mean HD95 10.54mm** - Khả năng tổng quát hóa xuất sắc). |

---

## 🧬 Architecture Diagram of Advanced 2.5D Mamba U-Net (Exp052/Exp053)

Dưới đây là sơ đồ chi tiết kiến trúc tiên phong ứng dụng State Space Models **Hybrid 2.5D Mamba U-Net với Ràng buộc Phân cấp Giải phẫu (Exp052 / Exp053)**:

```mermaid
graph TD
    subgraph Input ["1. Multi-Modal 2.5D Input [B, 20, 240, 240]"]
        IN["4 MRI Modalities × 5 Axial Slices<br/>(FLAIR, T1, T1ce, T2)"]
    end

    subgraph Adapter ["2. High-Resolution Gated 2.5D Mamba Adapter"]
        SCE["Shared Context Encoder<br/>(CNN Feature Extractor per Slice)"]
        MAMBA["Bidirectional Selective State Space<br/>(4-Direction Spatial Scan Mamba)"]
        GATE["Cross-Modality Reliability Gate<br/>(Adaptive Context Enrichment)"]
        IN --> SCE --> MAMBA --> GATE
    end

    subgraph Backbone ["3. Shared ResNet34 Encoder Trunk"]
        R34["Pretrained ResNet34 Encoder<br/>(Multi-Scale Features: x1, x2, x3, x4)"]
        GATE --> R34
    end

    subgraph Decoder ["4. Disentangled Decoder & Region Heads"]
        SKIP["Attention-Gated Decoder Skips"]
        HEAD_WT["WT Head (Whole Tumor)"]
        HEAD_TC["TC Head (Tumor Core)"]
        HEAD_ET["ET Head (Enhancing Tumor)"]
        R34 --> SKIP
        SKIP --> HEAD_WT
        SKIP --> HEAD_TC
        SKIP --> HEAD_ET
    end

    subgraph Loss ["5. Region-Hierarchy Consistency Constraint"]
        L_HIER["L_hier = ReLU(P_ET - P_TC) + ReLU(P_TC - P_WT)"]
        HEAD_WT & HEAD_TC & HEAD_ET --> L_HIER
    end

    style Input fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    style Adapter fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style Backbone fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style Decoder fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style Loss fill:#ffebee,stroke:#d32f2f,stroke-width:2px
```

---

## 📐 Mathematical Formulations

### 1. Hàm Mất mát Phân cấp Vùng u (Region-Hierarchy Loss)
Đảm bảo tính phụ thuộc không gian giải phẫu sinh học giữa các vùng u BraTS ($P_{\text{ET}} \le P_{\text{TC}} \le P_{\text{WT}}$):

$$
\mathcal{L}_{\text{hier}} = \frac{1}{N} \sum_{i=1}^{N} \left( \max(0, P_{\text{ET}, i} - P_{\text{TC}, i}) + \max(0, P_{\text{TC}, i} - P_{\text{WT}, i}) \right)
$$

### 2. Hàm Mất mát Tổng hợp (Combined Training Loss)

$$
\mathcal{L}_{\text{total}} = w_{\text{dice}} \mathcal{L}_{\text{Dice}} + w_{\text{bce}} \mathcal{L}_{\text{BCE}} + w_{\text{hier}} \mathcal{L}_{\text{hier}}
$$

*Trong đó các trọng số được cấu hình: `w_dice = 0.5`, `w_bce = 0.5`, và `w_hier = 0.1`.*

---

## 📊 Performance & Experimental Results

### 1. Quantitative Evaluation on BraTS 2020 (3D Validation)

| Model Architecture | Config | WT Dice | TC Dice | ET Dice | Mean Dice |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 2D U-Net** | `exp001` | 0.8650 | 0.7720 | 0.7240 | 0.7870 |
| **Attention U-Net 2D** | `exp018` | 0.8926 | 0.8418 | 0.7698 | 0.8347 |
| **Disentangled 2D Fusion** | `exp021` | 0.9015 | 0.8502 | 0.7810 | 0.8442 |
| **Region Heads + Hierarchy Consistency** | `exp036` | 0.9072 | 0.8590 | 0.7915 | 0.8526 |
| **ResNet34 Region-Heads U-Net** (Best Baseline) | `exp043` | **0.9081** | **0.8642** | **0.7929** | **0.8551** |
| **Hybrid 2.5D Mamba Residual Adapter** | `exp052` | 0.9017 | 0.8514 | 0.7766 | 0.8432 |

### 2. External Generalization on BraTS 2023 GLI (Zero-shot Evaluation - 1,251 Cases)
Đánh giá mô hình mạnh nhất **ResNet34 Region-Heads U-Net (`Exp043`)** (chỉ huấn luyện trên BraTS 2020) trực tiếp trên toàn bộ 1,251 ca bệnh của tập dữ liệu ngoại lai **BraTS 2023 GLI** mà không fine-tune:

| Dataset / Benchmark | WT Dice | TC Dice | ET Dice | Mean Dice | WT HD95 | TC HD95 | ET HD95 | Mean HD95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BraTS 2020 Test** | 0.916 | 0.858 | 0.821 | 0.865 | 10.12 mm | 9.45 mm | 21.84 mm | 13.80 mm |
| **BraTS 2023 GLI (Zero-shot)** | **0.9017** | **0.8725** | **0.8261** | **0.8668** | **10.00 mm** | **9.15 mm** | **12.47 mm** | **10.54 mm** |
| **Biến thiên ($\Delta$)** | -0.0143 | **+0.0145** | **+0.0051** | **+0.0018** | **-0.12 mm** | **-0.30 mm** | **-9.37 mm** | **-3.26 mm** |

> 🚀 **Kết luận**: Mô hình đạt khả năng tổng quát hóa zero-shot xuất sắc trên tập dữ liệu mới BraTS 2023 GLI với chỉ số **Mean Dice vượt trội 86.68%** và chỉ số khoảng cách biên u **ET-HD95 giảm mạnh từ 21.84mm xuống 12.47mm**.

### 3. Outlier Analysis & Failure Mode Case Studies (Phân tích Ca Ngoại lệ 279 & 307)

Hệ thống tiến hành phân tích chuyên sâu các ca bệnh thách thức tiêu biểu nhằm đánh giá các giới hạn hình thái:

| Subject ID | Đặc điểm hình thái | Thách thức phân đoạn | Giải pháp & Cải tiến thành công |
| :--- | :--- | :--- | :--- |
| **`BraTS20_Training_279`** | U độ thấp LGG không chứa u tăng cường ($GT_{\text{ET}} = 0$) | Các mô hình baseline bị nhận diện nhầm đốm False Positive ET $\rightarrow$ bị phạt ranh giới HD95 = 373 mm. | `Exp051` tích hợp lọc thành phần liên thông nhỏ (Connected Component Cleanup) triệt tiêu hoàn toàn 100% các đốm nhiễu ET giả. |
| **`BraTS20_Training_307`** | Khối u có vùng ET siêu nhỏ (chỉ 32 voxels ~ vài điểm pixel) | Các mô hình 2D baseline bỏ sót hoàn toàn ($ET=0$) do kích thước u quá bé. | `Exp043` và `Exp045` (gán trọng số loss thích ứng) phát hiện chính xác vị trí u nhỏ và hạ HD95-ET xuống còn **19.13 mm**. |

### 4. Training Curves Visualizations
Dưới đây là các biểu đồ đường cong huấn luyện đối chiếu các chiến lược lấy mẫu (Sampling) và các mốc tiến hóa kiến trúc (Milestones):

<p align="center">
  <img src="outputs/figures/sampling_training_curves.png" width="48%" alt="Sampling Training Curves" />
  <img src="outputs/figures/milestones_training_curves.png" width="48%" alt="Milestones Training Curves" />
</p>

### 5. Dynamic 3D Segmentation Visualizations (Demos)
Hình ảnh trực quan hóa kết quả phân đoạn lát cắt động 3D của mô hình **Hybrid Mamba U-Net** (Lá cây: WT - Toàn bộ vùng u, Đỏ: TC - Lõi u, Xanh dương: ET - U tăng cường):

<p align="center">
  <img src="outputs/figures/best_cases_gifs/BraTS20_Training_189_155slices.gif" width="45%" alt="Segmentation Demo 1" />
  <img src="outputs/figures/best_cases_gifs/BraTS20_Training_347_155slices.gif" width="45%" alt="Segmentation Demo 2" />
</p>

---

## 🛠️ Complete Script Toolkit Reference

Hệ thống cung cấp danh mục các công cụ xử lý và kiểm thử tự động trong thư mục `scripts/`:

| Script Path | Description & Purpose |
| :--- | :--- |
| [`scripts/evaluate_brats2023.py`](file:///Users/nguyenducphat/%C4%90ATN%20MRI%2053exp/scripts/evaluate_brats2023.py) | Đánh giá trực tiếp checkpoint mô hình trên tập dữ liệu kiểm thử ngoại lai BraTS 2023 GLI. |
| [`scripts/plot_training_curves.py`](file:///Users/nguyenducphat/%C4%90ATN%20MRI%2053exp/scripts/plot_training_curves.py) | Tự động đọc dữ liệu lịch sử huấn luyện từ 53 thí nghiệm và vẽ biểu đồ so sánh loss/Dice. |
| [`scripts/visualize_best_cases.py`](file:///Users/nguyenducphat/%C4%90ATN%20MRI%2053exp/scripts/visualize_best_cases.py) | Trực quan hóa các lát cắt 2D/3D và xuất ảnh GIF động phân đoạn của các ca có chỉ số Dice cao nhất. |
| [`scripts/visualize_hard_cases.py`](file:///Users/nguyenducphat/%C4%90ATN%20MRI%2053exp/scripts/visualize_hard_cases.py) | Trực quan hóa và phân tích lỗi nhiệt (error heatmap) trên các ca ngoại lệ/khó dự đoán. |
| [`scripts/audit_brats_overlap.py`](file:///Users/nguyenducphat/%C4%90ATN%20MRI%2053exp/scripts/audit_brats_overlap.py) | Kiểm tra rò rỉ và trùng lặp mã định danh bệnh nhân giữa tập BraTS 2020 và BraTS 2023. |

---

## ⚙️ Hardware Requirements & Setup

### 1. Yêu cầu Phần cứng & Tương thích OS (Hardware Benchmark)
- **VRAM tối thiểu**: ~6 GB (cho các mô hình 2D U-Net baseline) và ~12 GB (cho các mô hình 2.5D Mamba U-Net).
- **Hỗ trợ Hệ điều hành & Tăng tốc Phần cứng**:
  - 🪟 **Windows (NVIDIA GPU)**: Tăng tốc qua CUDA 11.8+ / 12.0+, hỗ trợ trọn vẹn cả CNN và Mamba-SSM CUDA kernels.
  - 🍏 **macOS (Apple Silicon M1/M2/M3/M4)**: Tăng tốc GPU qua PyTorch MPS (Metal Performance Shaders) hỗ trợ mượt mà các mô hình 2D/2.5D CNN.

---

### 2. Khởi tạo Môi trường Ảo (Setup Guide for Windows & macOS/Linux)

#### 🍏 Trên macOS / Linux:
```bash
# 1. Clone repository
git clone https://github.com/DucPh4t/BraTS-Brain-Tumor-Segmentation.git
cd BraTS-Brain-Tumor-Segmentation

# 2. Khởi tạo và kích hoạt môi trường ảo
python3 -m venv venv
source venv/bin/activate

# 3. Cài đặt các thư viện phụ thuộc
pip install --upgrade pip
pip install -r requirements.txt
```

#### 🪟 Trên Windows (Command Prompt / PowerShell):
```cmd
:: 1. Clone repository
git clone https://github.com/DucPh4t/BraTS-Brain-Tumor-Segmentation.git
cd BraTS-Brain-Tumor-Segmentation

:: 2. Khởi tạo môi trường ảo
python -m venv venv

:: 3. Kích hoạt môi trường ảo (PowerShell: .\venv\Scripts\Activate.ps1 | CMD: venv\Scripts\activate.bat)
venv\Scripts\activate

:: 4. Cài đặt các thư viện phụ thuộc
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

### 3. Cài đặt Mamba-SSM (Chỉ dành cho GPU NVIDIA trên Windows / Linux / Kaggle / Colab)
Dành cho các thí nghiệm `exp052` / `exp053` có sử dụng kiến trúc Mamba bottleneck (Yêu cầu CUDA toolkit):
```bash
pip install -r requirements-mamba.txt --no-build-isolation
```

---

## 🚀 Usage Guide

### 1. Huấn luyện Mô hình
Chạy huấn luyện một thí nghiệm bất kỳ bằng lệnh `main.py` chỉ định file cấu hình:
```bash
# Ví dụ: Huấn luyện mô hình ResNet34 Region-Heads U-Net (Exp043)
python main.py --config configs/exp043.yaml --mode train
```

Khôi phục tiến trình huấn luyện từ checkpoint cũ:
```bash
python main.py --config configs/exp043.yaml --mode train --resume_path outputs/exp043/last_checkpoint.pth
```

### 2. Đánh giá Mô hình trên Thể tích 3D
Đánh giá checkpoint mô hình tốt nhất trên toàn bộ tập dữ liệu 3D:
```bash
python main.py --config configs/exp043.yaml --mode eval
```

### 3. Đánh giá chéo trên BraTS 2023 GLI
```bash
python scripts/evaluate_brats2023.py \
  --checkpoint outputs/exp043/best_model.pth \
  --config configs/exp043.yaml \
  --data-root /path/to/BraTS2023_GLI
```

### 4. Vẽ biểu đồ học tập & Trực quan hóa
```bash
# Vẽ lại các biểu đồ huấn luyện so sánh
python scripts/plot_training_curves.py

# Sinh ảnh trực quan kết quả phân đoạn 2D/3D
python scripts/visualize_best_cases.py
```

---

## 📖 Citation & License

Dự án này được phân phối dưới giấy phép [MIT License](LICENSE).  
Nếu nghiên cứu hoặc mã nguồn này giúp ích cho đồ án hoặc công trình của bạn, vui lòng trích dẫn:

```bibtex
@misc{mri_brats_53exp_2026,
  author = {Nguyen Duc Phat},
  title = {Multi-Modal Brain Tumor Segmentation on BraTS 2020 & 2023 with Hybrid Mamba and Region-Hierarchy Consistency},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/DucPh4t/BraTS-Brain-Tumor-Segmentation}}
}
```
