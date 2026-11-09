# BÁO CÁO ĐÁNH GIÁ BASELINE CHECKPOINT STATIC MLP V1 TRÊN DATASET SẠCH (TASK 10.1)

> **Dự án:** VSL Connect AI — Chuẩn hóa Nhận diện 29 Chữ cái Ngôn ngữ Ký hiệu Tiếng Việt  
> **Thời gian thực hiện:** 2026-11-07T12:59:10+0700  
> **Mục tiêu:** Thiết lập baseline đánh giá khách quan của checkpoint đang dùng trên camera runtime (`alphabet_static_head_v1.pt`) trên tập dữ liệu tĩnh sạch chuẩn miền Nam (`vsl_alphabet_static_south_clean_v1.npz`).

---

## 1. Thông tin Đối tượng Đánh giá

| Thành phần | Đường dẫn | Kích thước | Mã băm SHA-256 |
|---|---|---|---|
| **Checkpoint** | [`backend/models/alphabet_static_head_v1.pt`](file:///D:/SIGN_LANGUAGE/backend/models/alphabet_static_head_v1.pt) | 35,544 bytes | `cab8123e23ed41a6088ba1c6e80a8aaaa9a382663418cc669eacff834795c254` |
| **Dataset Sạch** | [`backend/data/alphabet_v2/datasets/vsl_alphabet_static_south_clean_v1.npz`](file:///D:/SIGN_LANGUAGE/backend/data/alphabet_v2/datasets/vsl_alphabet_static_south_clean_v1.npz) | 150,718 bytes | `78b10448b9d75c22d8338143c324d1fa235c702913c826ad744f25173a1d34c9` |

- **Chế độ thực thi:** `model.eval()`, `torch.no_grad()`.
- **Can thiệp kỹ thuật:** **KHÔNG CÓ** (Không cập nhật trọng số, không gradient, không geometric rules, không ensemble, không threshold gating).

---

## 2. Kết quả Tổng quan

- **Tổng số mẫu (Support):** 309
- **Dự đoán chính xác:** 271
- **Dự đoán sai:** 38
- **Độ chính xác tổng thể (Accuracy):** 87.7%
- **Macro-F1 (17 lớp sạch có mẫu):** 0.8896
- **Weighted-F1 (17 lớp sạch có mẫu):** 0.8886
- **Macro-F1 (22 lớp toàn bộ, zero_div=0):** 0.6874

---

## 3. Bảng Hiệu năng Chi tiết 22 Lớp

| Chữ cái | Support | Đúng (TP) | Sai (FN) | Precision | Recall | F1-Score | Nhầm sang | Trạng thái |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| **A** | 22 | 20 | 2 | 76.9% | 90.9% | 0.8333 | L (1), V (1) | Có mẫu |
| **B** | 22 | 18 | 4 | 100.0% | 81.8% | 0.9000 | A (1), E (1), I (1), N (1) | Có mẫu |
| **C** | 20 | 19 | 1 | 82.6% | 95.0% | 0.8837 | E (1) | Có mẫu |
| **D** | 20 | 18 | 2 | 100.0% | 90.0% | 0.9474 | C (1), Q (1) | Có mẫu |
| **E** | 15 | 12 | 3 | 75.0% | 80.0% | 0.7742 | A (2), I (1) | Có mẫu |
| **G** | 0 | 0 | 0 | 0.0% | Support=0 | 0.0000 | — | Thiếu (0 mẫu) |
| **H** | 0 | 0 | 0 | 0.0% | Support=0 | — | — | Thiếu (0 mẫu) |
| **I** | 18 | 18 | 0 | 81.8% | 100.0% | 0.9000 | — | Có mẫu |
| **K** | 17 | 14 | 3 | 100.0% | 82.3% | 0.9032 | C (1), G (1), Q (1) | Có mẫu |
| **L** | 16 | 15 | 1 | 83.3% | 93.8% | 0.8823 | A (1) | Có mẫu |
| **M** | 19 | 19 | 0 | 86.4% | 100.0% | 0.9268 | — | Có mẫu |
| **N** | 21 | 18 | 3 | 90.0% | 85.7% | 0.8780 | M (3) | Có mẫu |
| **O** | 20 | 17 | 3 | 100.0% | 85.0% | 0.9189 | C (2), G (1) | Có mẫu |
| **P** | 0 | 0 | 0 | 0.0% | Support=0 | — | — | Thiếu (0 mẫu) |
| **Q** | 19 | 12 | 7 | 80.0% | 63.2% | 0.7059 | A (1), G (1), X (5) | Có mẫu |
| **R** | 14 | 12 | 2 | 100.0% | 85.7% | 0.9231 | A (1), I (1) | Có mẫu |
| **S** | 15 | 13 | 2 | 100.0% | 86.7% | 0.9286 | E (1), Q (1) | Có mẫu |
| **T** | 0 | 0 | 0 | 0.0% | Support=0 | — | — | Thiếu (0 mẫu) |
| **U** | 18 | 15 | 3 | 100.0% | 83.3% | 0.9091 | I (1), L (2) | Có mẫu |
| **V** | 17 | 16 | 1 | 94.1% | 94.1% | 0.9412 | N (1) | Có mẫu |
| **X** | 0 | 0 | 0 | 0.0% | Support=0 | 0.0000 | — | Thiếu (0 mẫu) |
| **Y** | 16 | 15 | 1 | 100.0% | 93.8% | 0.9677 | E (1) | Có mẫu |

---

## 4. Phân tích Độ Tự tin (Confidence) và Nhầm lẫn

### 4.1. Mức độ tự tin trung bình
- **Mẫu dự đoán đúng:** Confidence trung bình đạt **0.8817**.
- **Mẫu dự đoán sai:** Confidence trung bình đạt **0.3235**.

### 4.2. Danh sách các mẫu sai có độ tự tin cao (Confidence $\ge 0.70$)
| Index | Chữ thật | Chữ dự đoán | Confidence | Xác suất lớp thật | Video ID | Frame |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `187` | **N** | **M** | 0.8877 | 0.0133 | `D0505` | f_52 |

---

## 5. Kết luận Nghiệm thu
Đã ghi nhận baseline thành công.
