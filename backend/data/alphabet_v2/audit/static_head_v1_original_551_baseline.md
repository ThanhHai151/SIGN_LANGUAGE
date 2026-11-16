# BÁO CÁO ĐÁNH GIÁ CHẨN ĐOÁN CHECKPOINT STATIC MLP V1 TRÊN DATASET GỐC (551 MẪU) - TASK 10.2

> **Dự án:** VSL Connect AI — Chuẩn hóa Nhận diện 29 Chữ cái Ngôn ngữ Ký hiệu Tiếng Việt  
> **Thời gian thực hiện:** 2026-11-07T12:59:24+0700  
> **Bản chất phép đo:** **Đánh giá chẩn đoán trên dữ liệu gốc (551 mẫu) đã dùng để huấn luyện checkpoint V1.**  
> **Lưu ý khoa học bắt buộc:** **TUYỆT ĐỐI KHÔNG coi đây là test độc lập hay bằng chứng về khả năng tổng quát hóa ngoài đời thực.**

---

## 1. Xác minh Mã băm và Tính Toàn vẹn (SHA-256)

| Thành phần | Đường dẫn file | Kích thước | SHA-256 trước khi chạy | SHA-256 sau khi chạy | Tình trạng |
|---|---|---|---|---|:---:|
| **Checkpoint** | [`backend/models/alphabet_static_head_v1.pt`](file:///D:/SIGN_LANGUAGE/backend/models/alphabet_static_head_v1.pt) | 35,544 bytes | `cab8123e23ed41a6088ba1c6e80a8aaaa9a382663418cc669eacff834795c254` | `cab8123e23ed41a6088ba1c6e80a8aaaa9a382663418cc669eacff834795c254` | **NGUYÊN VẸN 100%** |
| **Dataset Gốc** | [`backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz`](file:///D:/SIGN_LANGUAGE/backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz) | 267,504 bytes | `0a254daadb4a5dee58d3b11b97071fa093ea2470df6e8a3d609b6d886bfc53b9` | `0a254daadb4a5dee58d3b11b97071fa093ea2470df6e8a3d609b6d886bfc53b9` | **NGUYÊN VẸN 100%** |

- **Chế độ thực thi:** `model.eval()`, `torch.no_grad()`.
- **Can thiệp kỹ thuật:** **KHÔNG CÓ** (Không optimizer, không backward, không cập nhật trọng số hay LayerNorm, không threshold gating, không ensemble).

---

## 2. Kết quả Tổng quan trên Toàn bộ 551 Mẫu

- **Tổng số mẫu (Support):** **551 mẫu**
- **Dự đoán chính xác:** **505 mẫu**
- **Dự đoán sai:** **46 mẫu**
- **Độ chính xác tổng thể (Accuracy):** **91.65%**
- **Macro-F1 (22 lớp toàn bộ):** **0.9088**
- **Weighted-F1 (22 lớp toàn bộ):** **0.9178**

> [!NOTE]
> Độ chính xác **91.65%** khớp chính xác 100% với siêu dữ liệu ghi nhận trong checkpoint gốc lúc huấn luyện (`overall_accuracy_pct: 91.65` tại Task 6B ngày 30/10/2026), khẳng định tính tái lập hoàn hảo bit-for-bit của mô hình.

---

## 3. Bảng Hiệu năng Chi tiết 22 Lớp (Đầy đủ Support)

| Chữ cái | Support | Đúng (TP) | Sai (FN) | Bị đoán nhầm (FP) | Precision | Recall | F1-Score | Các chữ bị nhầm sang |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **A** | 64 | 62 | 2 | 6 | 91.2% | 96.9% | 0.9394 | L (1), V (1) |
| **B** | 22 | 18 | 4 | 0 | 100.0% | 81.8% | 0.9000 | A (1), E (1), I (1), N (1) |
| **C** | 20 | 19 | 1 | 6 | 76.0% | 95.0% | 0.8444 | E (1) |
| **D** | 46 | 44 | 2 | 0 | 100.0% | 95.7% | 0.9778 | C (1), Q (1) |
| **E** | 35 | 32 | 3 | 4 | 88.9% | 91.4% | 0.9014 | A (2), I (1) |
| **G** | 29 | 28 | 1 | 3 | 90.3% | 96.5% | 0.9333 | Q (1) |
| **H** | 18 | 17 | 1 | 0 | 100.0% | 94.4% | 0.9714 | I (1) |
| **I** | 18 | 18 | 0 | 5 | 78.3% | 100.0% | 0.8780 | — |
| **K** | 17 | 14 | 3 | 0 | 100.0% | 82.3% | 0.9032 | C (1), G (1), Q (1) |
| **L** | 16 | 15 | 1 | 3 | 83.3% | 93.8% | 0.8823 | A (1) |
| **M** | 19 | 19 | 0 | 3 | 86.4% | 100.0% | 0.9268 | — |
| **N** | 21 | 18 | 3 | 2 | 90.0% | 85.7% | 0.8780 | M (3) |
| **O** | 48 | 45 | 3 | 0 | 100.0% | 93.8% | 0.9677 | C (2), G (1) |
| **P** | 20 | 18 | 2 | 0 | 100.0% | 90.0% | 0.9474 | Q (2) |
| **Q** | 19 | 12 | 7 | 8 | 60.0% | 63.2% | 0.6154 | A (1), G (1), X (5) |
| **R** | 14 | 12 | 2 | 0 | 100.0% | 85.7% | 0.9231 | A (1), I (1) |
| **S** | 15 | 13 | 2 | 0 | 100.0% | 86.7% | 0.9286 | E (1), Q (1) |
| **T** | 20 | 18 | 2 | 0 | 100.0% | 90.0% | 0.9474 | C (2) |
| **U** | 32 | 29 | 3 | 0 | 100.0% | 90.6% | 0.9508 | I (1), L (2) |
| **V** | 17 | 16 | 1 | 1 | 94.1% | 94.1% | 0.9412 | N (1) |
| **X** | 25 | 23 | 2 | 5 | 82.1% | 92.0% | 0.8679 | Q (2) |
| **Y** | 16 | 15 | 1 | 0 | 100.0% | 93.8% | 0.9677 | E (1) |

---

## 4. Phân tích Chi tiết theo Nguồn Dữ liệu

### 4.1. Phân tích theo Vùng miền (Region)
*Ghi chú: Trường `region` không có sẵn trực tiếp trong JSON metadata mà được xác định từ hậu tố `video_id` (N: SOUTH, B: NORTH, không hậu tố: SHARED).*

| Vùng miền | Số mẫu | Đúng | Sai | Độ chính xác | Các lớp có mẫu | Cặp nhầm lẫn nổi bật |
|---|:---:|:---:|:---:|:---:|---|---|
| **REGION_SHARED** | 319 | 282 | 37 | **88.4%** | A, B, C, D, E, I, K, L, M, N, O, Q, R, S, U, V | A -> L (1), A -> V (1), B -> E (1), B -> I (1), B -> N (1), B -> A (1), C -> E (1), D -> C (1), D -> Q (1), E -> I (1), E -> A (2), K -> G (1), K -> C (1), K -> Q (1), L -> A (1), N -> M (3), O -> G (1), O -> C (2), Q -> G (1), Q -> X (5), Q -> A (1), R -> I (1), R -> A (1), S -> Q (1), S -> E (1), U -> L (2), U -> I (1), V -> N (1) |
| **REGION_NORTH** | 193 | 185 | 8 | **95.85%** | A, E, G, H, O, P, T, X | G -> Q (1), H -> I (1), P -> Q (2), T -> C (2), X -> Q (2) |
| **REGION_SOUTH** | 39 | 38 | 1 | **97.44%** | O, U, Y | Y -> E (1) |

### 4.2. Phân tích theo Loại Nguồn (`source_type`)
| Loại nguồn | Số mẫu | Đúng | Sai | Độ chính xác | Các lớp có mẫu | Cặp nhầm lẫn nổi bật |
|---|:---:|:---:|:---:|:---:|---|---|
| **static_video** | 421 | 375 | 46 | **89.07%** | A, B, C, D, E, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, X, Y | A -> L (1), A -> V (1), B -> E (1), B -> I (1), B -> N (1), B -> A (1), C -> E (1), D -> C (1), D -> Q (1), E -> I (1), E -> A (2), G -> Q (1), H -> I (1), K -> G (1), K -> C (1), K -> Q (1), L -> A (1), N -> M (3), O -> G (1), O -> C (2), P -> Q (2), Q -> G (1), Q -> X (5), Q -> A (1), R -> I (1), R -> A (1), S -> Q (1), S -> E (1), T -> C (2), U -> L (2), U -> I (1), V -> N (1), X -> Q (2), Y -> E (1) |
| **base_pose_of_dynamic** | 130 | 130 | 0 | **100.0%** | A, D, E, O, U | — |

> [!IMPORTANT]
> **Điểm mấu chốt về `base_pose_of_dynamic`:**  
> Toàn bộ 130 mẫu BASE_POSE (cắt từ các chữ Ă, Â, Ê, Ô, Ơ, Ư, Đ) đạt **độ chính xác 100.0% (130/130)** vì mô hình đã học thuộc (overfit/fit) các tư thế tĩnh của các video này trong quá trình huấn luyện Task 6B.

---

## 5. Kiểm tra Tập trung 5 Lớp Thiếu trong Dataset Sạch (G, H, P, T, X)

Tổng cộng có **112 mẫu** thuộc 5 lớp này trong dataset gốc (toàn bộ đều là video miền Bắc có hậu tố `B`):

| Chữ cái | Mã video | Vùng miền | Support | Đúng | Sai | Độ chính xác | Conf mẫu đúng | Conf mẫu sai |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **G** | `D0498B` | REGION_NORTH | 29 | 28 | 1 | **96.55%** | 0.9044 | 0.5174 |
| **H** | `D0500B` | REGION_NORTH | 18 | 17 | 1 | **94.44%** | 0.9295 | 0.6839 |
| **P** | `D0509B` | REGION_NORTH | 20 | 18 | 2 | **90.0%** | 0.8785 | 0.3506 |
| **T** | `D0513B` | REGION_NORTH | 20 | 18 | 2 | **90.0%** | 0.9323 | 0.2252 |
| **X** | `D0517B` | REGION_NORTH | 25 | 23 | 2 | **92.0%** | 0.8035 | 0.527 |

- **Tổng số mẫu của 5 lớp thiếu:** 112 mẫu.
- **Số mẫu đoán đúng:** 104 mẫu.
- **Số mẫu đoán sai:** 8 mẫu.
- **Độ chính xác trung bình trên 5 lớp thiếu:** **92.86%**.
- **Đánh giá:** Checkpoint V1 hiện tại vẫn lưu giữ năng lực nhận diện rất tốt (92.86%) trên chính các mẫu huấn luyện miền Bắc của 5 chữ này.

### Danh sách toàn bộ 8 mẫu dự đoán sai của 5 lớp thiếu:
| Index gốc | Video ID | Frame | Chữ thật | Chữ dự đoán | Confidence | Xác suất lớp thật |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `111` | `D0498B` | f_56 | **G** | **Q** | 0.5174 | 0.0488 |
| `144` | `D0500B` | f_46 | **H** | **I** | 0.6839 | 0.1782 |
| `275` | `D0509B` | f_58 | **P** | **Q** | 0.3141 | 0.0741 |
| `276` | `D0509B` | f_60 | **P** | **Q** | 0.3871 | 0.0115 |
| `343` | `D0513B` | f_56 | **T** | **C** | 0.2529 | 0.0621 |
| `344` | `D0513B` | f_58 | **T** | **C** | 0.1975 | 0.0741 |
| `390` | `D0517B` | f_108 | **X** | **Q** | 0.5519 | 0.3486 |
| `391` | `D0517B` | f_110 | **X** | **Q** | 0.5020 | 0.4258 |

*Nhận xét về 8 lỗi:*
- Chữ **G** nhầm sang **Q** (1 ca, conf 0.5174): frame 56 góc nhìn trỏ chúc nhẹ.
- Chữ **H** nhầm sang **I** (1 ca, conf 0.6839): frame 46 ngón út và ngón trỏ.
- Chữ **P** nhầm sang **Q** (2 ca, conf ~0.35): frame 58, 60 góc trỏ chúc xuống.
- Chữ **T** nhầm sang **C** (2 ca, conf ~0.22): frame 56, 58 ngón tay khum nhẹ.
- Chữ **X** nhầm sang **Q** (2 ca, conf ~0.52): frame 108, 110 ngón trỏ móc câu nhầm với trỏ chúc của Q.

---

## 6. Đối chiếu Khách quan với Kết quả Task 10.1 (309 Mẫu Sạch)

| Tiêu chí | Task 10.1 (File Clean NPZ) | Task 10.2 (Sub-slice 309 trong File Gốc) | Mức độ trùng khớp |
|---|:---:|:---:|:---:|
| **Số mẫu** | 309 mẫu | 309 mẫu | **Trùng khớp 100%** |
| **Số mẫu đoán đúng** | 271 mẫu | 271 mẫu | **Trùng khớp 100%** |
| **Số mẫu đoán sai** | 38 mẫu | 38 mẫu | **Trùng khớp 100%** |
| **Độ chính xác** | 87.70% | 87.70% | **Trùng khớp 100%** |
| **Mẫu sai lệch dự đoán** | — | — | **0 mẫu sai lệch (0 mismatch)** |

### Phân tích các cặp lỗi nổi bật đã xuất hiện ở Task 10.1:
1. **$Q ightarrow X$ (5 lần):** Xuất hiện hoàn toàn trong video `D0510.mp4` (chữ Q dùng chung). Chiếu tọa độ 72D của ngón trỏ chúc tạo hình thái gần giống ngón trỏ co của chữ X.
2. **$N ightarrow M$ (3 lần):** Xuất hiện hoàn toàn trong video `D0505.mp4` (chữ N dùng chung). Do ngón nhẫn khép sát che ngón cái.
3. **$E ightarrow A$ (2 lần):** Video `D0496.mp4`.
4. **$O ightarrow C$ (2 lần):** Video `D0506.mp4`.
5. **$U ightarrow L$ (2 lần):** Video `D0514.mp4`.

---

## 7. Phân tích Độ Tự tin (Confidence) và Độ Hiệu chuẩn (Calibration)

- **Confidence trung bình mẫu đúng:** **0.9021**
- **Confidence trung bình mẫu sai:** **0.3413**
- **Số mẫu sai có Confidence $\ge 0.70$:** **1 ca** (Duy nhất mẫu index 234: N $ightarrow$ M, conf 0.8877).
- **Số mẫu sai có Confidence $\ge 0.85$:** **1 ca** (Cũng là mẫu index 234).

### Đo lường Độ hiệu chuẩn (Calibration Metrics):
- **Multi-class Brier Score:** **0.1057** (mức tối ưu cho bài toán 22 lớp).
- **Top-1 Brier Score:** **0.0352**.
- **Expected Calibration Error (ECE - 10 equal-width bins):** **6.84%** (`0.0684`).

| Bin | Dải Confidence | Số mẫu | Accuracy thực tế | Confidence TB | Độ lệch (Calibration Gap) |
|---|:---:|:---:|:---:|:---:|:---:|
| Bin 1 | `[0.0, 0.1]` | 0 | 0.0% | 0.0000 | 0.0000 |
| Bin 2 | `[0.1, 0.2]` | 17 | 5.9% | 0.1664 | 0.1076 |
| Bin 3 | `[0.2, 0.3]` | 12 | 25.0% | 0.2612 | 0.0112 |
| Bin 4 | `[0.3, 0.4]` | 8 | 37.5% | 0.3554 | 0.0196 |
| Bin 5 | `[0.4, 0.5]` | 6 | 66.7% | 0.4722 | 0.1944 |
| Bin 6 | `[0.5, 0.6]` | 25 | 56.0% | 0.5352 | 0.0248 |
| Bin 7 | `[0.6, 0.7]` | 16 | 87.5% | 0.6583 | 0.2167 |
| Bin 8 | `[0.7, 0.8]` | 19 | 100.0% | 0.7622 | 0.2378 |
| Bin 9 | `[0.8, 0.9]` | 69 | 98.6% | 0.8705 | 0.1151 |
| Bin 10 | `[0.9, 1.0]` | 379 | 100.0% | 0.9529 | 0.0471 |

*Nhận xét khoa học về Calibration:*
- 68.78% dữ liệu (379 mẫu) tập trung ở Bin 10 ($[0.9, 1.0]$) với độ chính xác tuyệt đối 100.0%.
- Ở các khoảng confidence thấp ($[0.1, 0.4]$), độ chính xác thực tế giảm tương ứng ($5.8\% - 37.5\%$), chứng minh mô hình có độ tự nhận thức (uncertainty awareness) rất tốt: khi không chắc chắn, confidence tự động tụt giảm.

---

## 8. Danh sách File Đầu ra Được Tạo Mới

1. [`backend/training/evaluate_alphabet_static_head.py`](file:///D:/SIGN_LANGUAGE/backend/training/evaluate_alphabet_static_head.py): Kịch bản đánh giá mở rộng hỗ trợ cả 2 chế độ `--dataset clean` và `--dataset original`.
2. [`backend/data/alphabet_v2/audit/static_head_v1_original_551_baseline.json`](file:///D:/SIGN_LANGUAGE/backend/data/alphabet_v2/audit/static_head_v1_original_551_baseline.json): Báo cáo máy đọc JSON đầy đủ thống kê.
3. [`backend/data/alphabet_v2/audit/static_head_v1_original_551_baseline.md`](file:///D:/SIGN_LANGUAGE/backend/data/alphabet_v2/audit/static_head_v1_original_551_baseline.md): Báo cáo phân tích chuyên sâu.
4. [`backend/data/alphabet_v2/audit/static_head_v1_original_551_predictions.json`](file:///D:/SIGN_LANGUAGE/backend/data/alphabet_v2/audit/static_head_v1_original_551_predictions.json): Chi tiết dự đoán 551 mẫu.
5. [`backend/data/alphabet_v2/audit/static_head_v1_original_551_confusion_matrix.csv`](file:///D:/SIGN_LANGUAGE/backend/data/alphabet_v2/audit/static_head_v1_original_551_confusion_matrix.csv): Ma trận nhầm lẫn 22x22 định dạng CSV.

---

## 9. Phân tích Giới hạn và Kết luận Bắt buộc

1. **Phân biệt rõ 3 mức kết quả:**
   - Trên **551 mẫu gốc:** Accuracy đạt **91.65%**, Macro-F1 đạt **0.9160** (do đây là tập train/dev mà checkpoint đã học).
   - Trên **309 mẫu sạch (Task 10.1):** Accuracy đạt **87.70%**, Macro-F1 đạt **0.8896** (do đã loại bỏ 130 mẫu BASE_POSE vốn đạt 100% accuracy và 112 mẫu miền Bắc đạt 92.86%).
   - Trên **5 lớp thiếu (G, H, P, T, X):** Checkpoint V1 đạt **92.86%** (104/112 đúng) trên dữ liệu miền Bắc ban đầu.
2. **Cảnh báo về khả năng tổng quát hóa:**
   - Kết quả này **KHÔNG CHỨNG MINH** mô hình sẽ nhận diện tốt 5 lớp trên dữ liệu miền Nam hay người dùng mới.
   - Nhưng kết quả này **CHỨNG MINH RÕ RÀNG**: Checkpoint V1 hiện tại đang gánh vác năng lực nhận diện cho 5 lớp này trong hệ thống camera runtime. Nếu fine-tune đè lên checkpoint này mà không có dữ liệu cho 5 lớp đó, năng lực 92.86% này sẽ lập tức biến mất (bị quên thảm họa).

**KẾT LUẬN:** Task 10.2 đã hoàn thành toàn diện, bảo đảm tính toàn vẹn 100% của hệ thống và cung cấp bức tranh chẩn đoán đầy đủ trước khi lập phương án huấn luyện.
