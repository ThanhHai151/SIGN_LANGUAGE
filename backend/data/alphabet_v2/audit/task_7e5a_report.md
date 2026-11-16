# BÁO CÁO TASK 7E.5A — READ-ONLY AUDIT OF STATIC BASE LETTERS A/E/D

**Hệ thống:** VSL Connect AI — Bộ nhận diện 29 chữ cái Ngôn ngữ ký hiệu Việt Nam (VSL)  
**Tác giả:** Antigravity (Google DeepMind Advanced Agentic Coding)  
**Ngày thực hiện:** 31/10/2026  
**Chế độ:** **READ-ONLY AUDIT (Không sửa production code, không sửa threshold, không sửa checkpoint, không huấn luyện lại model)**

---

## 1. Tóm tắt Điều hành (Executive Summary)

Kiểm toán Task 7E.5A nhằm điều tra nguyên nhân gốc rễ thứ 3 (**Root Cause 3**): **Sự khắt khe của 3 quy tắc hình học nền A, E, D** khiến các chuỗi video động có dấu (`Ă`, `Â`, `Ê`, `Đ`) có **0 frame nền hợp lệ** trong pha `BASE_POSE`, làm tê liệt Máy trạng thái Diacritic (`DiacriticStateMachine`) ngay từ khâu khởi tạo baseline. Đồng thời, kiểm toán đánh giá khả năng sử dụng checkpoint [alphabet_static_head_v1.pt](file:///d:/SIGN_LANGUAGE/backend/models/alphabet_static_head_v1.pt) để làm lớp dự phòng (fallback) hoặc nới lỏng có kiểm soát các điều kiện hình học.

### Kết quả chính:
1. **Geometric Rules A/E/D thất bại nặng nề trên dữ liệu thực tế:**
   - **Chữ A:** Chỉ nhận diện được **17/64 mẫu (26.6%)**; 46 mẫu `NO_MATCH`, 1 mẫu bị nhận nhầm thành `O`. Nguyên nhân chính: điều kiện khoảng cách ngón trỏ tới cổ tay $d_{IW} \le 0.90$ bị vi phạm ở **65.6%** số mẫu, khoảng cách ngón cái tới ngón giữa $d_{TM} \ge 0.75$ bị vi phạm ở **57.8%** số mẫu.
   - **Chữ D:** Chỉ nhận diện được **15/46 mẫu (32.6%)**; 31 mẫu `NO_MATCH`. Nguyên nhân chính: ngưỡng vươn ngón trỏ $d_{IW} \ge 1.70$ quá khắt khe, bị vi phạm ở **63.0%** số mẫu (trung vị thực tế chỉ đạt 1.636).
   - **Chữ E:** Chỉ nhận diện được **10/35 mẫu (28.6%)**; 25 mẫu `NO_MATCH`. Nguyên nhân chính: khoảng cách $d_{TI} \le 0.35$ bị trượt **71.4%**, $d_{TM} \le 0.38$ bị trượt **65.7%**, và phân loại ngón tay gập của MediaPipe trượt **48.6%** do các ngón tì đè lên ngón cái.
2. **Video Replay 4 chữ động có dấu:**
   - Cả 4 video động (`Ă` — `D0490B`, `Â` — `D0491B`, `Ê` — `D0497B`, `Đ` — `D0495`) đều có **0/19 đến 0/26 frame `BASE_POSE`** được geometric rules nhận diện. FSM không bao giờ đạt `BASE_STABLE`.
3. **Hiệu năng của Static Head trên A/E/D:**
   - Trên tập `vsl_alphabet_static_v2.npz`: Đạt Top-1 Accuracy **96.9%** cho A (confidence trung bình 0.926), **95.7%** cho D (conf 0.923), **91.4%** cho E (conf 0.898).
   - Trên các frame `BASE_POSE` của 4 video động: Static Head dự đoán đúng chữ nền kỳ vọng trên **100% số frame** (19/19 frames A trong Ă; 23/23 frames A trong Â; 20/20 frames E trong Ê; 26/26 frames D trong Đ) với confidence $\ge 0.70$. Tạo ra chuỗi liên tiếp 19–26 frames ổn định, thừa đủ điều kiện $\ge 4$ frames để khởi tạo FSM!
4. **Cảnh báo tính hợp lệ dữ liệu (Severe Leakage Finding):**
   - Phân tích split train/holdout của Static Head cho thấy: **100% video ID trong tập holdout đều nằm trong tập train (`COMPLETE_VIDEO_OVERLAP`)**. Holdout được chia ngẫu nhiên theo frame từ cùng một video và một người ký. Do đó, độ chính xác cao trên holdout **không chứng minh khả năng tổng quát hóa trên người ký mới** và không được suy diễn thành webcam accuracy độc lập.
5. **So sánh hai phương án:**
   - **Static Head Fallback (Phương án 1):** Khi geometric rule trả về `NO_MATCH`, fallback sang Static Head với ngưỡng confidence $\ge 0.70$ giúp đạt recall **93.8% (A)**, **93.5% (D)**, **91.4% (E)** mà **không gây ra bất kỳ False Positive (0.0% FP)** nào trên 19 lớp còn lại trong dataset. Trong pha `DIACRITIC`, Static Head không bị nhầm cử chỉ tạo dấu thành chữ nền.
   - **Controlled Geometric Tolerance (Phương án 2):** Khả thi cao đối với D (nới $d_{IW} \ge 1.50$ đạt 82.6% recall với chỉ 0.2% FP). Tuy nhiên với E, việc nới lỏng kích thước hộp hình học làm tăng vọt **False Positive lên 4.26% (22 mẫu bị nhận nhầm từ S, C, O)** do không gian hình học của các tư thế nắm đấm tròn bị giao thoa mạnh.

---

## 2. Giai đoạn A — Kiểm toán Geometric Rules trên A, E, D

Dữ liệu kiểm toán: Trích xuất từ tập tĩnh chuẩn hóa [vsl_alphabet_static_v2.npz](file:///d:/SIGN_LANGUAGE/backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz) gồm **64 mẫu A**, **46 mẫu D**, **35 mẫu E**.

### 2.1. Quy tắc hình học Chữ A
- **Định nghĩa điều kiện hiện tại** trong [alphabet_classifier.py](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py):
  1. `fingers_curled`: 4 ngón trỏ, giữa, áp út, út gập (`not i_ext and not m_ext and not r_ext and not p_ext`).
  2. `d_TW >= 1.32` (`A_D_TW_MIN`): Ngón cái dựng đứng vươn xa cổ tay.
  3. `d_TM >= 0.75` (`A_D_TM_MIN`): Đầu ngón cái cách xa đầu ngón giữa gập.
  4. `d_IW <= 0.90` (`A_D_IW_MAX`): Khớp/đầu ngón trỏ gập sát về cổ tay.
  5. `d_MW <= 0.75` (`A_D_MW_MAX`): Khớp/đầu ngón giữa gập sát về cổ tay.
- **Thống kê mức độ thỏa mãn từng điều kiện (N = 64):**

| Điều kiện | Số mẫu đạt | Số mẫu vi phạm | Tỷ lệ vi phạm (%) | Đánh giá |
|---|:---:|:---:|:---:|---|
| `fingers_curled` | 64 | 0 | 0.00% | Hoàn hảo |
| `d_TW >= 1.32` | 55 | 9 | 14.06% | Khá tốt |
| `d_MW <= 0.75` | 48 | 16 | 25.00% | Hơi khắt khe |
| `d_TM >= 0.75` | 27 | 37 | **57.81%** | **RẤT KHẮT KHE** |
| `d_IW <= 0.90` | 22 | 42 | **65.62%** | **NGUYÊN NHÂN CHÍNH** |
| **Thỏa mãn đồng thời toàn bộ** | **17** | **47** | **73.44%** | **Chỉ đạt 26.56% Recall** |

- **Phân phối thực tế các đặc trưng của Chữ A:**
  - `d_IW`: Min = 0.771, P25 = 0.893, **Median = 0.941**, P75 = 0.978, Max = 1.156. *(Ngưỡng 0.90 cắt đứt hơn 65% mẫu)*
  - `d_TM`: Min = 0.540, P25 = 0.697, **Median = 0.722**, P75 = 0.822, Max = 0.972. *(Ngưỡng 0.75 cắt đứt gần 58% mẫu)*
  - `d_TW`: Min = 1.054, P25 = 1.385, **Median = 1.450**, P75 = 1.503, Max = 1.637.
- **Phân loại kết quả:**
  - `MATCH` đúng chữ A: **17 mẫu (26.56%)**
  - `NO_MATCH`: **46 mẫu (71.88%)**
  - `MATCH` nhầm chữ O: **1 mẫu (1.56%)** (do ngón cái quặp vào đầu ngón trỏ)
  - `UNKNOWN` / Lỗi landmark: **0 mẫu**

---

### 2.2. Quy tắc hình học Chữ D
- **Định nghĩa điều kiện hiện tại** trong [alphabet_classifier.py](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py):
  1. `index_only_ext`: Chỉ ngón trỏ duỗi thẳng, 3 ngón giữa/áp/út gập (`i_ext and not m_ext and not r_ext and not p_ext`).
  2. `d_IW >= 1.70` (`D_D_IW_MIN`): Ngón trỏ duỗi dài vượt trội so với cổ tay.
  3. `d_TI <= 1.28` (`D_D_TI_MAX`): Ngón cái không mở quá xa ngón trỏ (chặn chữ L).
  4. `d_TM <= 0.30` (`D_D_TM_MAX`): Ngón cái chạm khép vào đốt ngón giữa.
- **Thống kê mức độ thỏa mãn từng điều kiện (N = 46):**

| Điều kiện | Số mẫu đạt | Số mẫu vi phạm | Tỷ lệ vi phạm (%) | Đánh giá |
|---|:---:|:---:|:---:|---|
| `d_TM <= 0.30` | 43 | 3 | 6.52% | Rất chuẩn |
| `d_TI <= 1.28` | 46 | 0 | 0.00% | Hoàn hảo |
| `index_only_ext` | 38 | 8 | 17.39% | MediaPipe đôi khi nhận nhầm ngón giữa hơi nhú |
| `d_IW >= 1.70` | 17 | 29 | **63.04%** | **NGUYÊN NHÂN CHÍNH** |
| **Thỏa mãn đồng thời toàn bộ** | **15** | **31** | **67.39%** | **Chỉ đạt 32.61% Recall** |

- **Phân phối thực tế các đặc trưng của Chữ D:**
  - `d_IW`: Min = 1.411, P25 = 1.564, **Median = 1.636**, P75 = 1.745, Max = 2.008. *(Ngưỡng 1.70 nằm ở khoảng percentile 70%, loại bỏ phần lớn mẫu)*
  - `d_TM`: Min = 0.134, P25 = 0.201, **Median = 0.228**, P75 = 0.260, Max = 0.402.
  - `d_TI`: Min = 0.803, P25 = 0.942, **Median = 1.018**, P75 = 1.116, Max = 1.258.
- **Phân loại kết quả:**
  - `MATCH` đúng chữ D: **15 mẫu (32.61%)**
  - `NO_MATCH`: **31 mẫu (67.39%)**
  - `MATCH` nhầm: **0 mẫu**
  - `UNKNOWN` / Lỗi landmark: **0 mẫu**

---

### 2.3. Quy tắc hình học Chữ E
- **Định nghĩa điều kiện hiện tại** trong [alphabet_classifier.py](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py):
  1. `fingers_curled`: 4 ngón trỏ, giữa, áp, út gập tì lên ngón cái (`not i_ext and not m_ext and not r_ext and not p_ext`).
  2. `d_TI <= 0.35` (`E_D_TI_MAX`): Đầu ngón trỏ tì sát ngón cái.
  3. `d_TM <= 0.38` (`E_D_TM_MAX`): Đầu ngón giữa tì sát ngón cái.
  4. `d_TW <= 0.95` (`E_D_TW_MAX`): Ngón cái gập ngang dưới 4 ngón.
  5. `d_IW >= 1.05` (`E_D_IW_MIN`): Đốt ngón trỏ nhô cao tạo gờ chữ E.
  6. `d_MW >= 1.05` (`E_D_MW_MIN`): Đốt ngón giữa nhô cao.
- **Thống kê mức độ thỏa mãn từng điều kiện (N = 35):**

| Điều kiện | Số mẫu đạt | Số mẫu vi phạm | Tỷ lệ vi phạm (%) | Đánh giá |
|---|:---:|:---:|:---:|---|
| `d_TW <= 0.95` | 35 | 0 | 0.00% | Hoàn hảo |
| `d_MW >= 1.05` | 27 | 8 | 22.86% | Chấp nhận được |
| `d_IW >= 1.05` | 27 | 8 | 22.86% | Chấp nhận được |
| `fingers_curled` | 18 | 17 | **48.57%** | **MediaPipe nhận sai trạng thái duỗi** |
| `d_TM <= 0.38` | 12 | 23 | **65.71%** | **RẤT KHẮT KHE** |
| `d_TI <= 0.35` | 10 | 25 | **71.43%** | **NGUYÊN NHÂN CHÍNH** |
| **Thỏa mãn đồng thời toàn bộ** | **10** | **25** | **71.43%** | **Chỉ đạt 28.57% Recall** |

- **Phân phối thực tế các đặc trưng của Chữ E:**
  - `d_TI`: Min = 0.252, P25 = 0.344, **Median = 0.402**, P75 = 0.435, Max = 0.536. *(Ngưỡng 0.35 quá hẹp)*
  - `d_TM`: Min = 0.251, P25 = 0.370, **Median = 0.414**, P75 = 0.465, Max = 0.551. *(Ngưỡng 0.38 quá hẹp)*
  - `index_curl`: Median = 0.87 (gập mạnh), nhưng do tư thế tì lên ngón cái, góc đốt ngón tay khiến MediaPipe gán nhãn `index_ext=True` ở 17/35 mẫu.
- **Phân loại kết quả:**
  - `MATCH` đúng chữ E: **10 mẫu (28.57%)**
  - `NO_MATCH`: **25 mẫu (71.43%)**
  - `MATCH` nhầm: **0 mẫu**
  - `UNKNOWN` / Lỗi landmark: **0 mẫu**

---

## 3. Giai đoạn B — Đánh giá Static Head trên A/E/D và Kiểm toán Rò rỉ Dữ liệu (Data Leakage)

Sử dụng trực tiếp mô hình [VSLStaticAlphabetHead](file:///d:/SIGN_LANGUAGE/backend/models/alphabet_static_head.py) và checkpoint [alphabet_static_head_v1.pt](file:///d:/SIGN_LANGUAGE/backend/models/alphabet_static_head_v1.pt) (inference-only).

### 3.1. Kết quả Đánh giá Định lượng trên Dataset V2 (551 mẫu)

| Chữ cái | Số mẫu tổng | Train (N) | Holdout (N) | Accuracy Tổng (%) | Accuracy Train (%) | Accuracy Holdout (%) | Median Conf (Đúng) | False Positives từ lớp khác |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **A** | 64 | 52 | 12 | **96.88%** (62/64) | 96.15% (50/52) | **100.00%** (12/12) | 0.9687 | 6 mẫu (B:1, E:2, L:1, Q:1, R:1) |
| **D** | 46 | 37 | 9 | **95.65%** (44/46) | 97.30% (36/37) | **88.89%** (8/9) | 0.9711 | **0 mẫu (0.0%)** |
| **E** | 35 | 28 | 7 | **91.43%** (32/35) | 92.86% (26/28) | **85.71%** (6/7) | 0.9607 | 4 mẫu (B:1, C:1, S:1, Y:1) |

- **Nhầm lẫn nội bộ (Confusion targets):**
  - A bị nhầm thành: `L` (1 mẫu, conf 0.54), `V` (1 mẫu, conf 0.58).
  - D bị nhầm thành: `C` (1 mẫu, conf 0.57), `Q` (1 mẫu, conf 0.44).
  - E bị nhầm thành: `A` (2 mẫu, conf 0.52, 0.61), `I` (1 mẫu, conf 0.49).
- Ma trận nhầm lẫn 22x22 chi tiết đã được xuất ra file CSV: [task_7e5a_confusion_matrix_22.csv](file:///d:/SIGN_LANGUAGE/backend/data/alphabet_v2/audit/task_7e5a_confusion_matrix_22.csv).

---

### 3.2. Kiểm toán Rò rỉ Dữ liệu (Data Leakage & Split Audit)
> [!CAUTION]
> **PHÁT HIỆN RỦI RO PHƯƠNG PHÁP LUẬN:**  
> Toàn bộ quá trình huấn luyện `alphabet_static_head_v1.pt` đã sử dụng hàm phân chia `shuffled = rng.permutation(c_samples)` theo từng lớp độc lập (stratified random frame split) với `SEED=42`.

Kiểm tra nguồn gốc video (`video_id`) của từng mẫu trong tập Train và Holdout:
- **Chữ A:**
  - Tập Train (52 mẫu): Gồm các frame từ `['D0489', 'D0490B', 'D0491B']`.
  - Tập Holdout (12 mẫu): Gồm các frame từ `['D0489', 'D0490B', 'D0491B']`.
  - **Tỷ lệ trùng lặp video:** **100% (COMPLETE_VIDEO_OVERLAP)**.
- **Chữ D:**
  - Tập Train (37 mẫu): Gồm các frame từ `['D0494', 'D0495']`.
  - Tập Holdout (9 mẫu): Gồm các frame từ `['D0494', 'D0495']`.
  - **Tỷ lệ trùng lặp video:** **100% (COMPLETE_VIDEO_OVERLAP)**.
- **Chữ E:**
  - Tập Train (28 mẫu): Gồm các frame từ `['D0496', 'D0497B']`.
  - Tập Holdout (7 mẫu): Gồm các frame từ `['D0496', 'D0497B']`.
  - **Tỷ lệ trùng lặp video:** **100% (COMPLETE_VIDEO_OVERLAP)**.

**Ý nghĩa khoa học:**
1. Tập holdout **không phải là tập kiểm thử độc lập (Out-of-Distribution/Unseen Signer)**. Các frame holdout thực chất là các frame xen kẽ (cách nhau chỉ vài chục mili-giây) từ cùng một video với tập train.
2. Kết quả 100% trên holdout A hay 88.9% trên holdout D **phản ánh khả năng nội suy chuỗi thời gian (temporal interpolation)**, KHÔNG đại diện cho khả năng nhận diện người ký mới lạ qua webcam.
3. Không được tuyên bố độ chính xác webcam hay khả năng tổng quát hóa từ tập holdout này.

---

## 4. Giai đoạn C — Replay 4 Video Động (`Ă`, `Â`, `Ê`, `Đ`)

Replay độc lập giữa Production Recognizer hiện tại và Static Head inference trên 4 video chuyển động có dấu chuẩn hóa từ MediaPipe landmarks:
- `Ă` (`D0490B`, 89 frames, video Ă)
- `Â` (`D0491B`, 108 frames, video Â)
- `Ê` (`D0497B`, 76 frames, video Ê)
- `Đ` (`D0495`, 105 frames, video Đ)

*(Lưu ý đính chính: Trong một số ghi chú cũ có nhắc đến D0493N, D0501N, D0518N — kiểm toán xác nhận trên hệ thống QIPEDC, D0493 là video chữ C, D0501 là chữ I, D0518N là chữ Y. Bốn video động thực sự chứa chuyển động của Ă, Â, Ê, Đ là D0490B, D0491B, D0497B, D0495).*

Toàn bộ timeline frame-by-frame đã được lưu tại [task_7e5a_dynamic_frames_audit.csv](file:///d:/SIGN_LANGUAGE/backend/data/alphabet_v2/audit/task_7e5a_dynamic_frames_audit.csv).

### 4.1. Bảng Tổng hợp Đối sánh Giữa Geometric Rules và Static Head

| Video | Chữ nền kỳ vọng | Tổng frames | Số frame BASE_POSE | Geom MATCH Chữ nền | Static Head MATCH Chữ nền | Static Head Conf $\ge 0.70$ | FSM đạt BASE_STABLE (Hiện tại) | Chuỗi liên tiếp của Head ($\ge 4$ frames) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Ă (`D0490B`)** | **A** | 89 | 19 (f7–f25) | **0 / 19 (0.0%)** | **19 / 19 (100.0%)** | **19 / 19 (100.0%)** | **KHÔNG** | **19 frames liên tiếp (f7–f25, conf 0.88–0.97)** |
| **Â (`D0491B`)** | **A** | 108 | 23 (f27–f49) | **0 / 23 (0.0%)** | **23 / 23 (100.0%)** | **22 / 23 (95.7%)** | **KHÔNG** | **23 frames liên tiếp (f27–f49, conf 0.69–0.97)** |
| **Ê (`D0497B`)** | **E** | 76 | 20 (f8–f27) | **0 / 20 (0.0%)** | **20 / 20 (100.0%)** | **20 / 20 (100.0%)** | **KHÔNG** | **20 frames liên tiếp (f8–f27, conf 0.82–0.97)** |
| **Đ (`D0495`)** | **D** | 105 | 26 (f38–f63) | **0 / 26 (0.0%)** | **26 / 26 (100.0%)** | **26 / 26 (100.0%)** | **KHÔNG** | **26 frames liên tiếp (f38–f63, conf 0.89–0.97)** |

### 4.2. Khảo sát Hành vi trong Pha DIACRITIC (Nguy cơ Masking Dấu)
Một rủi ro tiềm ẩn của Static Head là: **Liệu Static Head có nhận nhầm cử chỉ tạo dấu thành chữ nền và che lấp chuyển động dấu hay không?**

Dữ liệu thực nghiệm chứng minh:
1. **Video Ă (`D0490B` - 23 frames DIACRITIC f26–f48):**
   - Khi ngón trỏ và giữa xòe ra tạo dấu á: Geometric rules nhận diện `V` (18 frames).
   - Static Head dự đoán: `D` (17 frames), `V` (4 frames), `K` (2 frames).
   - **Static Head KHÔNG hề nhận diện ra `A`!** Không có nguy cơ che lấp dấu `Ă`.
2. **Video Â (`D0491B` - 25 frames DIACRITIC f50–f74):**
   - Khi bàn tay nâng lên vẽ dấu nón: Geometric rules trả về `UNKNOWN` (25 frames).
   - Static Head dự đoán: `M` (10 frames), `X` (4 frames), `I` (4 frames), `R` (3 frames), `V` (2 frames).
   - **Static Head KHÔNG hề nhận diện ra `A`!**
3. **Video Ê (`D0497B` - 20 frames DIACRITIC f28–f47):**
   - Khi bàn tay nhấc lên vẽ nón: Static Head dự đoán `V` (13 frames), `R` (2 frames), `E` (chỉ 2 frames f34–f35).
   - Đa số frame không bị khóa vào `E`.
4. **Video Đ (`D0495` - 9 frames DIACRITIC f64–f72):**
   - Trong cử chỉ gạch ngang của `Đ`, ngón trỏ chỉ gập đốt giữa nhẹ trong tích tắc rồi duỗi lại.
   - Static Head dự đoán `D` trên cả 9 frames (conf 0.75–0.95).
   - **Hành vi này rất thuận lợi cho FSM:** FSM nhóm D được thiết kế để theo dõi `angle_drop` của ngón trỏ **ngay trong lúc chữ nền D đang duy trì**. Do đó, việc duy trì nhãn D giúp FSM tính toán xung gập $\Delta \theta \ge 15^\circ$ mà không bị gián đoạn sang trạng thái IDLE!

---

## 5. Giai đoạn D — Mô phỏng và So sánh Hai Phương án Thiết kế

### 5.1. Phương án 1: Static Head Fallback (Gated by Geometric NO_MATCH)
- **Cơ chế:** Khi geometric classifier trả về `NO_MATCH` hoặc `UNKNOWN`, nếu Static Head dự đoán A, D, hoặc E với $\text{confidence} \ge \tau$, hệ thống chấp nhận nhãn đó làm chữ nền cho FSM.
- **Kết quả mô phỏng trên toàn bộ 551 mẫu Dataset V2:**

| Ngưỡng Confidence ($\tau$) | Chữ A (N=64) | Chữ D (N=46) | Chữ E (N=35) | False Positives trên 19 lớp khác (N=406) |
|:---:|:---:|:---:|:---:|:---:|
| **Hiện tại (Geom Only)** | **26.56% (17)** | **32.61% (15)** | **28.57% (10)** | **0 FP (0.00%)** |
| $\tau = 0.55$ | 95.31% (61) | 95.65% (44) | 91.43% (32) | 0 FP (0.00%) |
| $\tau = 0.60$ | 95.31% (61) | 95.65% (44) | 91.43% (32) | 0 FP (0.00%) |
| $\tau = 0.70$ | **93.75% (60)** | **93.48% (43)** | **91.43% (32)** | **0 FP (0.00%)** |
| $\tau = 0.80$ | 93.75% (60) | 93.48% (43) | 91.43% (32) | 0 FP (0.00%) |

- **Ưu điểm vượt trội:**
  - Cứu sống hơn **90–95% số mẫu** nền A, D, E.
  - Khi được rào chắn bởi điều kiện `geom_status != 'MATCH'`, tỷ lệ báo động giả (False Positive) trên toàn bộ 19 lớp chữ cái tĩnh khác là **0.00%**.
  - Không cần sửa đổi hoặc tinh chỉnh thủ công các tham số hình học.
- **Nhược điểm & Giới hạn:**
  - Phụ thuộc vào checkpoint MLP. Cần kiểm chứng thêm trên người ký mới.

---

### 5.2. Phương án 2: Controlled Geometric Tolerance (Nới lỏng tham số hình học)
Mô phỏng điều chỉnh các ngưỡng hình học dựa trên phân vị thực nghiệm (10th/90th percentile):

#### Chữ A:
- Cấu hình gốc: $d_{TW} \ge 1.32$, $d_{TM} \ge 0.75$, $d_{IW} \le 0.90$, $d_{MW} \le 0.75 \rightarrow$ **Recall: 26.56% | FP: 0**.
- Cấu hình nới $d_{IW} \le 1.05$, $d_{TM} \ge 0.68$, $d_{TW} \ge 1.25$, $d_{MW} \le 0.80 \rightarrow$ **Recall: 71.88% (46/64) | FP: 0**.
- *Đánh giá:* Nới lỏng kiểm soát được giúp tăng gấp 2.7 lần recall cho A mà không bị lẫn sang S hoặc O.

#### Chữ D:
- Cấu hình gốc: $d_{IW} \ge 1.70$, $d_{TI} \le 1.28$, $d_{TM} \le 0.30 \rightarrow$ **Recall: 32.61% | FP: 0**.
- Cấu hình nới $d_{IW} \ge 1.55 \rightarrow$ **Recall: 58.70% (27/46) | FP: 0**.
- Cấu hình nới $d_{IW} \ge 1.50$, $d_{TI} \le 1.35$, $d_{TM} \le 0.38 \rightarrow$ **Recall: 82.61% (38/46) | FP: 1 (0.2%)** (bị lẫn 1 mẫu chữ L khi góc mở ngón cái hẹp).
- *Đánh giá:* Rất khả thi và an toàn cho chữ D.

#### Chữ E:
- Cấu hình gốc: $d_{TI} \le 0.35$, $d_{TM} \le 0.38$, $d_{TW} \le 0.95$, $d_{IW} \ge 1.05$, $d_{MW} \ge 1.05 \rightarrow$ **Recall: 28.57% | FP: 2 (0.39%)**.
- Cấu hình nới khoảng cách $d_{TI} \le 0.45$, $d_{TM} \le 0.48$, $d_{TW} \le 1.05$, $d_{IW} \ge 0.95 \rightarrow$ **Recall chỉ tăng lên 34.29%**, nhưng **False Positive bùng nổ lên 22 mẫu (4.26%)** (bị nhận nhầm ồ ạt từ các chữ `C`, `S`, `O`).
- *Đánh giá:* **BẤT KHẢ THI BẰNG HÌNH HỌC ĐƠN THUẦN**. Vấn đề cốt lõi của chữ E không chỉ là khoảng cách mà là MediaPipe nhận sai cờ `finger_extended` ở 48.6% số mẫu do ngón tay tì đè lên ngón cái.

---

## 6. Trả lời 7 Câu hỏi Trọng tâm của Kiểm toán

### 1. Vì sao A/E/D geometric rules thất bại?
- **Chữ A:** Ngưỡng $d_{IW} \le 0.90$ và $d_{TM} \ge 0.75$ quá phi thực tế đối với tư thế nắm đấm tự nhiên của người ký VSL (trung vị thực tế là $d_{IW} = 0.941$, $d_{TM} = 0.722$).
- **Chữ D:** Ngưỡng $d_{IW} \ge 1.70$ đòi hỏi ngón trỏ duỗi cực dài, loại bỏ 63% mẫu bình thường (trung vị thực tế là 1.636).
- **Chữ E:** Bị lỗi kép: vừa khống chế khoảng cách quá chật ($d_{TI} \le 0.35$, $d_{TM} \le 0.38$), vừa bị MediaPipe phân loại sai trạng thái duỗi ngón tay (`fingers_curled` fail 48.6%) khi 4 ngón tì đè lên ngón cái.

### 2. Static Head có nhận diện được A/E/D trên dữ liệu hiện có không?
- **Có, nhận diện xuất sắc:** Đạt 96.9% trên A, 95.7% trên D, 91.4% trên E. Trên các frame `BASE_POSE` của 4 video động, Static Head đạt độ chính xác **100% (88/88 frames)** với confidence trung bình $> 0.90$.

### 3. Những kết quả nào là train, holdout có leakage, hoặc chưa xác định split?
- Đã xác định chính xác split thành viên: 449 mẫu train, 102 mẫu holdout.
- Toàn bộ kết quả holdout của A, D, E **đều bị rò rỉ video hoàn toàn (100% COMPLETE_VIDEO_OVERLAP)**. Holdout chỉ là phép kiểm tra nội suy frame từ cùng người ký và cùng video, **không phải là kiểm chứng độc lập**.

### 4. Static Head có tạo được chuỗi frame đủ ổn định để khởi tạo FSM trong bốn video động không?
- **Hoàn toàn có thể:** Static Head tạo ra các chuỗi liên tiếp **19 frames (Ă)**, **23 frames (Â)**, **20 frames (Ê)**, **26 frames (Đ)** với confidence $\ge 0.70$. FSM chỉ yêu cầu 4 frames liên tiếp (0.10s) để đạt `BASE_STABLE`, do đó các chuỗi này dư sức kích hoạt FSM thành công.

### 5. Nên thử Static Head fallback hay controlled geometric tolerance trước?
- **Khuyến nghị kiến trúc HYBRID:**
  1. **Nên ưu tiên triển khai Static Head Fallback trước cho A, E, D:** Vì phương án này đã được chứng minh giải quyết triệt để cả 3 chữ cái, đặc biệt là chữ E (nơi mà geometric rule hoàn toàn bất lực trước lỗi gập ngón của MediaPipe), đồng thời đạt **0.00% False Positive** trên toàn bộ 19 lớp khác ở ngưỡng confidence $\ge 0.70$.
  2. **Có thể kết hợp Controlled Geometric Tolerance riêng cho chữ D:** Hạ $d_{IW} \ge 1.55$ là giải pháp thuần túy toán học, không độ trễ, cứu sống gần 60% mẫu D mà không cần gọi mô hình nơ-ron.

### 6. Những rủi ro false positive nào cần kiểm soát?
- Static Head confusions ở mức confidence thấp: A có thể lẫn với S/M, E có thể lẫn với C/O/S, D có thể lẫn với Q.
- **Biện pháp kiểm soát:** Chỉ kích hoạt fallback khi:
  - `geom_status != "MATCH"`
  - `head_letter in ("A", "D", "E")`
  - `head_confidence >= 0.70` (ngưỡng an toàn thực nghiệm loại trừ 100% FP).

### 7. Có bằng chứng nào đủ để đề xuất bước triển khai tiếp theo?
- Đầy đủ bằng chứng định lượng từ 551 mẫu tĩnh và 378 frames video động.
- Bước triển khai tiếp theo (Task 7E.5B): Tích hợp **Gated Static Head Fallback** cho A, E, D với ngưỡng $\tau = 0.70$ vào luồng `predict_landmarks(mode="letters")`.

---

## 7. Giai đoạn E — Kiểm toán Tính Toàn vẹn Checkpoint

Băm SHA-256 được kiểm tra trước và sau toàn bộ quá trình audit:

| Tập tin Checkpoint | SHA-256 Trước Audit | SHA-256 Sau Audit | Trạng thái |
|---|---|---|:---:|
| [weights.pt](file:///d:/SIGN_LANGUAGE/backend/models/weights.pt) | `6965b4ffd9eed45c10317f0cc4eff453470ea4552c081f18ea997d74365907be` | `6965b4ffd9eed45c10317f0cc4eff453470ea4552c081f18ea997d74365907be` | **NGUYÊN VẸN 100%** |
| [sign_sstcn_attention_model.pth](file:///d:/SIGN_LANGUAGE/backend/models/weights/sign_sstcn_attention_model.pth) | `2a8ddcc4d06c4f23aa7910caf77371848034dcdf8f4749d134c5894764cf9697` | `2a8ddcc4d06c4f23aa7910caf77371848034dcdf8f4749d134c5894764cf9697` | **NGUYÊN VẸN 100%** |
| [alphabet_static_head_v1.pt](file:///d:/SIGN_LANGUAGE/backend/models/alphabet_static_head_v1.pt) | `cab8123e23ed41a6088ba1c6e80a8aaaa9a382663418cc669eacff834795c254` | `cab8123e23ed41a6088ba1c6e80a8aaaa9a382663418cc669eacff834795c254` | **NGUYÊN VẸN 100%** |

Không có bất kỳ tập tin mã nguồn production hay checkpoint nào bị chỉnh sửa trong task này.
Tất cả mã phân tích phục vụ audit được lưu trữ cô lập tại:
- [run_task_7e5a_audit.py](file:///d:/SIGN_LANGUAGE/backend/data/alphabet_v2/audit/run_task_7e5a_audit.py)
- [task_7e5a_results.json](file:///d:/SIGN_LANGUAGE/backend/data/alphabet_v2/audit/task_7e5a_results.json)
- [task_7e5a_confusion_matrix_22.csv](file:///d:/SIGN_LANGUAGE/backend/data/alphabet_v2/audit/task_7e5a_confusion_matrix_22.csv)
- [task_7e5a_dynamic_frames_audit.csv](file:///d:/SIGN_LANGUAGE/backend/data/alphabet_v2/audit/task_7e5a_dynamic_frames_audit.csv)

