# BÁO CÁO KIỂM TOÁN VÀ ĐỐI CHIẾU SỐ LIỆU ĐỘC LẬP (TASK 12.1.1.1)

> **Mã nhiệm vụ:** TASK 12.1.1.1 — Statistical Reconciliation Audit  
> **Tệp dữ liệu:** `backend/data/diagnostics/pilot_pilot_20261107_2141.jsonl` (SHA-256: `fb4f725757bcbc73b96748896c4149c21927461314c72e68a49310b7cd76429f`)  
> **Nguyên tắc thực hiện:** Chỉ đọc (STRICTLY READ-ONLY) — Không sửa code, không đổi rule hình học, không sửa threshold, không train lại model, không đổi checkpoint, không xóa hoặc sửa JSONL, không đưa ra kết luận nhân quả vượt quá bằng chứng thực tế.

---

## 1. XÁC MINH DANH TÍNH CHECKPOINT (CHECKPOINT IDENTITY AUDIT)

### A. Kiểm tra SHA-256 của Checkpoint trên đĩa
- **Đường dẫn tệp:** `backend/models/alphabet_static_head_v1.pt`
- **Kích thước:** `35,544` bytes
- **Mã băm SHA-256 đầy đủ:**
  ```text
  cab8123e23ed41a6088ba1c6e80a8aaaa9a382663418cc669eacff834795c254
  ```
- **Đối chiếu với hash đã xác nhận ở TASK 11.2.4:**
  `cab8123e23ed41a6088ba1c6e80a8aaaa9a382663418cc669eacff834795c254`  
  $ightarrow$ **KHỚP TUYỆT ĐỐI 100% (EXACT MATCH)**.

### B. Nguồn gốc chuỗi `ca4d35ebff...` trong báo cáo TASK 12.1.1
- **Kết quả rà soát toàn bộ kho mã nguồn và dữ liệu:**
  Quét toàn bộ thư mục `backend/`, `scratch/` và lịch sử chat cho thấy chuỗi `ca4d` **chỉ xuất hiện duy nhất** trong văn bản báo cáo Markdown `vsl_task12_1_1_deep_recomputation_report.md` được sinh ra ở phiên trước.
- **Kết luận kiểm toán:** Không hề tồn tại bất kỳ tệp mô hình nào có hash bắt đầu bằng `ca4d...`. Chuỗi `ca4d35ebff...` là một **lỗi gõ nhầm (typographical error/hallucination)** khi biên soạn báo cáo TASK 12.1.1 thay vì ghi đúng `cab8123e23...`. Checkpoint thực tế duy nhất hiện diện và được tải là `alphabet_static_head_v1.pt` với hash `cab8123e...`.

### C. Cấu trúc mạng (Architecture), Chiều dữ liệu và Danh sách nhãn
Trích xuất trực tiếp từ siêu dữ liệu lưu trong checkpoint bằng PyTorch:
- **Kiến trúc:** `VSLStaticAlphabetHead (Linear 72->64, LayerNorm, LeakyReLU, Dropout 0.15; Linear 64->32, LayerNorm, LeakyReLU, Dropout 0.1; Linear 32->22)`
- **Input dimension:** 72 (tương ứng 24 đặc trưng không gian landmark bàn tay sau chuẩn hóa)
- **Hidden dimensions:** `[64, 32]`
- **Output dimension (num_classes):** 22
- **Danh sách nhãn (classes):**
  `['A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y']`
- **Bộ trích xuất đặc trưng:** `backend.vision.feature_extractor.normalize_hand_landmarks`
- **Tệp dữ liệu huấn luyện gốc:** `backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz` (551 mẫu, 60 epochs)
- **Thời điểm tạo checkpoint:** `2026-10-30T22:16:41+0700`

### D. Xác minh Runtime môi trường thực tế
- Trước khi thu thập dữ liệu webcam (TASK 11.2.4 & 11.2.5), backend Global Python (`Python 3.12`, PyTorch `2.2.2+cpu`) đã kiểm tra hash và xác nhận `cab8123e...`.
- Trong từng bản ghi JSONL của tệp `pilot_pilot_20261107_2141.jsonl`, các trường hệ thống ghi nhận:
  - `has_static_head: True` (13.341 / 13.341 frames)
  - `load_status: loaded` (13.341 / 13.341 frames)
  - `pytorch_version: 2.2.2+cpu` (13.341 / 13.341 frames)
  - `load_error: null` (13.341 / 13.341 frames)
- *Ghi chú tính chặt chẽ:* Do cấu trúc payload của `pilot_logger.py` không lưu trường SHA-256 hash của checkpoint trên từng frame, việc runtime sử dụng đúng checkpoint `cab8123e...` được chứng minh thông qua trạng thái nạp `alphabet_static_head_v1.pt` duy nhất trong tiến trình backend và nhật ký khởi động.

---

## 2. ĐỐI CHIẾU VÀ GIẢI TRÌNH LẦN GỌI MLP (MLP INVOCATION RECONCILIATION)

### A. Bảng đối chiếu số lần gọi MLP và trạng thái hình học
Để giải thích sự khác biệt giữa hai con số **7.390** (trong báo cáo TASK 11.2.6) và **3.641** (trong báo cáo TASK 12.1.1), dữ liệu được tách bạch tuyệt đối giữa **Active Trial Frames** và **Transition Frames**:

| Phân nhóm dữ liệu | Tổng số Frame | Geometric MATCH | Geometric NO_MATCH | Geometric UNKNOWN / Khác | MLP được gọi (Predict Feature) | MLP KHÔNG gọi (Bypassed) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Active Trial Frames** | **7.456** | 3.815 | 3.641 | 0 | **3.641** | **3.815** |
| **Transition Frames** | **5.885** | 2.135 | 3.749 | 1 (CONFLICT) | **3.749** | **2.136** |
| **TOÀN BỘ FILE JSONL** | **13.341** | **5.950** | **7.390** | **1** | **7.390** | **5.951** |

### B. Giải trình căn nguyên kỹ thuật
1. **Con số 7.390:**
   - Trong `backend/vision/recognizer.py`, khi `status == "MATCH"`, hệ thống ưu tiên luật hình học và không gọi MLP (`static_head` không có dự đoán). Khi `status == "NO_MATCH"`, hệ thống kích hoạt `static_alphabet_head.predict_feature()`.
   - Báo cáo TASK 11.2.6 đã đếm toàn bộ tệp (13.341 dòng). Số frame có `NO_MATCH` trên toàn bộ tệp là:
     $$3.641 	ext{ (trial)} + 3.749 	ext{ (transition)} = 7.390 	ext{ frames}$$
   - Do đó con số 7.390 là **tổng số lần gọi MLP trên toàn bộ tệp (bao gồm cả lúc người dùng nghỉ giữa các lượt)**.
2. **Con số 3.641:**
   - Báo cáo TASK 12.1.1 đã lọc sạch 5.885 transition frames, chỉ phân tích trong 7.456 active trial frames có người thực hiện động tác. Trong 7.456 frames này, có đúng **3.641 frames bị trượt luật hình học (`NO_MATCH`) và chuyển sang gọi MLP**.
3. **Phép cộng kiểm toán bảo toàn:**
   $$3.641 	ext{ (Trial MLP Invocations)} + 3.749 	ext{ (Transition MLP Invocations)} = 7.390 	ext{ (Total MLP Invocations)}$$
   Không hề có sự mâu thuẫn hay mất mát dữ liệu; khác biệt hoàn toàn xuất phát từ phạm vi mẫu (Active Trials vs Toàn bộ Stream).

---

## 3. MA TRẬN GÁC CỔNG GATING ĐÃ XÁC MINH (RECONCILED GATING MATRIX)

Với toàn bộ **3.641 frames** thực sự gọi MLP trong các lượt thử nghiệm active trial, mỗi frame được phân loại vào đúng 1 trong 4 nhóm:
- **True Accept (TA):** Raw prediction đoán đúng nhãn mục tiêu VÀ được Gating chấp nhận (`passed_supplemental` hoặc `passed_fallback`).
- **False Accept (FA):** Raw prediction đoán SAI nhãn mục tiêu NHƯNG lại được Gating chấp nhận.
- **False Reject (FR):** Raw prediction đoán ĐÚNG nhãn mục tiêu NHƯNG bị Gating từ chối đưa về `NEUTRAL`.
- **True Reject (TR):** Raw prediction đoán SAI nhãn mục tiêu VÀ được Gating từ chối chính xác về `NEUTRAL`.

### A. Bảng số lượng và Tỷ lệ chính xác

| Nhóm Gating | Số lượng Frame | Mẫu số | Tỷ lệ (%) | Các trường hợp cụ thể |
|:---|:---:|:---:|:---:|:---|
| **True Accept (TA)** | **11** | 3.641 | **0.30%** | G $ightarrow$ G (5), N $ightarrow$ N (4), M $ightarrow$ M (2) |
| **False Accept (FA)** | **28** | 3.641 | **0.77%** | O $ightarrow$ A via fallback (25), V $ightarrow$ R via supplemental (3) |
| **False Reject (FR)** | **435** | 3.641 | **11.95%** | N (291), M (97), V (17), I (16), R (11), B (3) |
| **True Reject (TR)** | **3.167** | 3.641 | **86.98%** | V (2789), A (195), D (122), R (23), N (18), B (16), K (3), L (1) |
| **TỔNG KIỂM TOÁN** | **3.641** | **3.641** | **100.00%** | **Bảo toàn 100% mẫu** |

$$	ext{TA} + 	ext{FA} + 	ext{FR} + 	ext{TR} = 11 + 28 + 435 + 3.167 = 3.641$$

### B. Danh sách các Frame tiêu biểu cho từng nhóm

#### 1. True Accept (TA) — 11 frames
| Trial ID | Frame Idx | Expected | Raw Pred | Confidence | Threshold | Gating Status | Final Sign |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `pilot_20261107_2141_N_rep01_att01` | 27 | **N** | N | 0.8122 | 0.70 | passed_supplemental | **N** |
| `pilot_20261107_2141_N_rep01_att01` | 28 | **N** | N | 0.8068 | 0.70 | passed_supplemental | **N** |
| `pilot_20261107_2141_N_rep01_att01` | 29 | **N** | N | 0.8784 | 0.70 | passed_supplemental | **N** |
| `pilot_20261107_2141_N_rep04_att01` | 24 | **N** | N | 0.7501 | 0.70 | passed_supplemental | **N** |
| `pilot_20261107_2141_M_rep03_att01` | 13 | **M** | M | 0.7678 | 0.75 | passed_supplemental | **M** |
| `pilot_20261107_2141_M_rep03_att01` | 14 | **M** | M | 0.7533 | 0.75 | passed_supplemental | **M** |
| `pilot_20261107_2141_G_rep03_att01` | 18 | **G** | G | 0.7241 | 0.70 | passed_supplemental | **G** |

#### 2. False Accept (FA) — 28 frames
| Trial ID | Frame Idx | Expected | Raw Pred | Confidence | Threshold | Gating Status | Final Sign | Reject Reason |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `pilot_20261107_2141_O_rep01_att01` | 18 | **O** | A | 0.8204 | 0.80 | passed_fallback | **A** | *None (Leaked)* |
| `pilot_20261107_2141_O_rep01_att01` | 19 | **O** | A | 0.8228 | 0.80 | passed_fallback | **A** | *None (Leaked)* |
| `pilot_20261107_2141_O_rep02_att01` | 23 | **O** | A | 0.8219 | 0.80 | passed_fallback | **A** | *None (Leaked)* |
| `pilot_20261107_2141_V_rep01_att01` | 31 | **V** | R | 0.7182 | 0.70 | passed_supplemental | **R** | *None (Leaked)* |

#### 3. False Reject (FR) — 435 frames
| Trial ID | Frame Idx | Expected | Raw Pred | Confidence | Threshold | Gating Status | Final Sign | Reject Reason |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `pilot_20261107_2141_N_rep01_att01` | 14 | **N** | N | 0.5186 | 0.70 | rejected | **NEUTRAL** | Confidence 0.52 < threshold 0.70 |
| `pilot_20261107_2141_N_rep01_att01` | 15 | **N** | N | 0.5187 | 0.70 | rejected | **NEUTRAL** | Confidence 0.52 < threshold 0.70 |
| `pilot_20261107_2141_M_rep01_att01` | 10 | **M** | M | 0.4320 | 0.75 | rejected | **NEUTRAL** | Confidence 0.43 < threshold 0.75 |
| `pilot_20261107_2141_V_rep07_att01` | 12 | **V** | V | 0.6541 | None | rejected | **NEUTRAL** | Letter 'V' not in supplemental/fallback set |
| `pilot_20261107_2141_I_rep08_att01` | 19 | **I** | I | 0.5842 | None | rejected | **NEUTRAL** | Letter 'I' not in supplemental/fallback set |

#### 4. True Reject (TR) — 3.167 frames
| Trial ID | Frame Idx | Expected | Raw Pred | Confidence | Threshold | Gating Status | Final Sign | Reject Reason |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `pilot_20261107_2141_P_rep01_att01` | 5 | **P** | V | 0.4812 | None | rejected | **NEUTRAL** | Letter 'V' not in supplemental/fallback set |
| `pilot_20261107_2141_X_rep01_att01` | 15 | **X** | V | 0.5421 | None | rejected | **NEUTRAL** | Letter 'V' not in supplemental/fallback set |
| `pilot_20261107_2141_E_rep01_att01` | 11 | **E** | V | 0.4632 | None | rejected | **NEUTRAL** | Letter 'V' not in supplemental/fallback set |
| `pilot_20261107_2141_Q_rep01_att01` | 8 | **Q** | V | 0.5119 | None | rejected | **NEUTRAL** | Letter 'V' not in supplemental/fallback set |

---

## 4. KIỂM TOÁN CHUYÊN SÂU SỐ LIỆU CHỮ M VÀ N (M & N AUDIT)

### A. Bảng kiểm toán số liệu M và N
Cả M và N đều là các chữ cái không có luật hình học hoàn chỉnh (100% frames đều trả về `NO_MATCH` và chuyển sang gọi MLP):

| Chỉ số kiểm toán | Chữ cái M | Chữ cái N | Ghi chú kỹ thuật |
|:---|:---:|:---:|:---|
| **Tổng số frame kỳ vọng (Expected Frames)** | **323** | **328** | Tổng frame trong 10 lượt thử nghiệm |
| **Số frame gọi MLP (Invocation Count)** | **323 (100.0%)** | **328 (100.0%)** | 100% frame hình học NO_MATCH |
| **Số frame MLP đoán thô ĐÚNG (Raw Correct)** | **99 (30.65%)** | **295 (89.94%)** | MLP thực sự nhận diện được đặc trưng ngón cái kẹp |
| **Số frame MLP đoán thô SAI (Raw Incorrect)** | **224 (69.35%)** | **33 (10.06%)** | MLP đoán thiên lệch thành V (164), B (16),... |
| **Số frame chấp nhận đúng (Accepted Correct - TA)** | **2 (0.62%)** | **4 (1.22%)** | Vượt qua ngưỡng Gating |
| **Số frame chấp nhận sai (Accepted Incorrect - FA)** | **0 (0.00%)** | **0 (0.00%)** | Gating không để rò rỉ bất kỳ lỗi nào |
| **Số frame từ chối đúng (Rejected Correct - FR)** | **97 (30.03%)** | **291 (88.72%)** | **Đoán đúng nhưng bị Gating chặn** |
| **Số frame từ chối sai (Rejected Incorrect - TR)** | **224 (69.35%)** | **33 (10.06%)** | Gating chặn chính xác các dự đoán sai |
| **Số frame đầu ra cuối cùng (Final Correct)** | **2 (0.62%)** | **4 (1.22%)** | Khớp 100% với ma trận nhầm lẫn |
| **Số frame đầu ra NEUTRAL** | **321 (99.38%)** | **324 (98.78%)** | Khớp 100% với ma trận nhầm lẫn |

### B. Phân phối độ tin cậy (Confidence Distribution)
- **Chữ N:**
  - *Raw Correct (295 frames):* Min = $0.2591$, Mean = $0.5084$, Median = $0.5054$, Max = $0.8784$.
    - Phân vị 10% = $0.3701$, 25% = $0.4360$, 50% = $0.5054$, 75% = $0.5796$, 90% = $0.6420$.
  - *Raw Incorrect (33 frames):* Min = $0.2558$, Mean = $0.4525$, Median = $0.4763$, Max = $0.5956$.
- **Chữ M:**
  - *Raw Correct (99 frames):* Min = $0.1948$, Mean = $0.2964$, Median = $0.2706$, Max = $0.7678$.
    - Phân vị 10% = $0.2091$, 25% = $0.2405$, 50% = $0.2706$, 75% = $0.3126$, 90% = $0.4122$.
  - *Raw Incorrect (224 frames):* Min = $0.1925$, Mean = $0.3559$, Median = $0.3367$, Max = $0.6068$.

### C. Giải trình sự chênh lệch giữa Raw Prediction và Final Prediction
- **Đối với chữ N:**
  MLP đoán đúng thô tới **295 frames** (tỷ lệ thô đạt **$89.94\%$**). Tuy nhiên, ngưỡng Gating trong `recognizer.py` yêu cầu `threshold = 0.70`. Vì $98.6\%$ số frame đúng có độ tin cậy nằm trong khoảng $0.37 - 0.64$ ($< 0.70$), bộ gác cổng Gating đã từ chối **291 frames**, chỉ cho phép đúng **4 frames** vượt qua. Do đó, kết quả cuối cùng ghi nhận 4 frame đúng và 324 frame NEUTRAL.
- **Đối với chữ M:**
  MLP đoán đúng thô **99 frames** ($30.65\%$). Ngưỡng Gating trong `recognizer.py` yêu cầu `threshold = 0.75`. Chỉ có duy nhất **2 frames** đạt độ tin cậy $\ge 0.75$ ($0.7678$ và $0.7533$), còn lại **97 frames** đúng bị Gating từ chối đưa về NEUTRAL.

---

## 5. TÍNH LẠI VÀ ĐỐI CHIẾU TRIAL ACCURACY (TRIAL ADJUDICATION)

### A. Quy tắc phán quyết đồng nhất (Uniform Adjudication Rule)
Áp dụng nhất quán một quy tắc toán học duy nhất cho toàn bộ 224 trials:
1. Đặt $C$ = `correct_frames`, $I$ = `incorrect_frames`, $U$ = `neutral_frames`, $NN = C + I$ = `non_neutral_frames`.
2. Nếu $NN == 0$ (100% frames là `NEUTRAL`) $\implies$ **NEUTRAL**.
3. Nếu $NN > 0$:
   - Nếu $C > I$ (đa số frame kích hoạt là đúng) $\implies$ **CORRECT**.
   - Nếu $I > C$ (đa số frame kích hoạt là sai) $\implies$ **INCORRECT**.
   - Nếu $C == I$ (tranh chấp bằng nhau) $\implies$ **MIXED**.

### B. Bảng kết quả tổng hợp theo 3 chiến lược tính

| Chiến lược tính toán | Tổng số Trial | CORRECT | NEUTRAL | INCORRECT | MIXED | Trial Accuracy (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Toàn bộ 224 trials (gộp cả att01 & att02)** | **224** | **132** | **66** | **26** | **0** | **58.93%** |
| **2. Chỉ tính 220 lượt cơ sở (Only att01)** | **220** | **129** | **65** | **26** | **0** | **58.64%** |
| **3. Chỉ tính 4 lượt kiểm thử lại (Only att02)** | **4** | **3** | **1** | **0** | **0** | **75.00%** |
| **4. Chuẩn hóa 220 lượt (att02 thay thế att01)** | **220** | **131** | **65** | **24** | **0** | **59.55%** |

### C. Chi tiết 4 cặp Retest
1. **Lượt `A_rep03`:** `att01` (C=28, I=0, U=4 $ightarrow$ CORRECT); `att02` (C=34, I=0, U=6 $ightarrow$ CORRECT). Kết quả không đổi.
2. **Lượt `P_rep01`:** `att01` (C=0, I=0, U=9 $ightarrow$ NEUTRAL); `att02` (C=0, I=0, U=32 $ightarrow$ NEUTRAL). Kết quả không đổi.
3. **Lượt `S_rep01`:** `att01` bị dừng sớm sau 8 frame nhiễu (C=1, I=2, U=5 $ightarrow$ INCORRECT); `att02` thử lại hoàn hảo (C=31, I=0, U=0 $ightarrow$ CORRECT). Khi thay thế, số INCORRECT giảm 1, CORRECT tăng 1.
4. **Lượt `K_rep01`:** `att01` có 17 frame đầu giơ tay rung (C=3, I=4, U=10 $ightarrow$ INCORRECT vì $I > C$); `att02` thử lại chuẩn (C=30, I=0, U=3 $ightarrow$ CORRECT). Khi thay thế, số INCORRECT giảm 1, CORRECT tăng 1.

---

## 6. BẢNG XẾP HẠNG 22 CHỮ CÁI TỰ ĐỘNG (RECONCILED RANKINGS)

Toàn bộ các bảng xếp hạng dưới đây được trích xuất và sắp xếp tự động 100% bằng code từ dữ liệu đã kiểm toán, không phân nhóm thủ công:

### A. Xếp hạng theo Trial Accuracy (Chuẩn hóa 220 Lượt — att02 thay thế att01)
1. **Y:** 100.00% (10/10) | Frame Acc: 100.00% | Neutral: 0.00% | FP: 0.00% | Confirmed: 100.00%
2. **U:** 100.00% (10/10) | Frame Acc: 99.49% | Neutral: 0.51% | FP: 0.00% | Confirmed: 100.00%
3. **B:** 100.00% (10/10) | Frame Acc: 98.91% | Neutral: 1.09% | FP: 0.00% | Confirmed: 100.00%
4. **S:** 100.00% (10/10) | Frame Acc: 93.20% | Neutral: 6.15% | FP: 0.65% | Confirmed: 100.00%
5. **V:** 100.00% (10/10) | Frame Acc: 90.50% | Neutral: 8.75% | FP: 0.75% | Confirmed: 100.00%
6. **A:** 100.00% (10/10) | Frame Acc: 89.43% | Neutral: 10.57% | FP: 0.00% | Confirmed: 100.00%
7. **G:** 100.00% (10/10) | Frame Acc: 80.94% | Neutral: 19.06% | FP: 0.00% | Confirmed: 100.00%
8. **K:** 100.00% (10/10) | Frame Acc: 80.27% | Neutral: 18.06% | FP: 1.67% | Confirmed: 100.00%
9. **C:** 100.00% (10/10) | Frame Acc: 43.80% | Neutral: 56.20% | FP: 0.00% | Confirmed: 80.00%
10. **L:** 90.00% (9/10) | Frame Acc: 61.89% | Neutral: 38.11% | FP: 0.00% | Confirmed: 80.00%
11. **T:** 80.00% (8/10) | Frame Acc: 54.90% | Neutral: 44.77% | FP: 0.33% | Confirmed: 80.00%
12. **D:** 60.00% (6/10) | Frame Acc: 55.56% | Neutral: 37.14% | FP: 7.30% | Confirmed: 70.00%
13. **I:** 60.00% (6/10) | Frame Acc: 48.62% | Neutral: 46.13% | FP: 5.25% | Confirmed: 70.00%
14. **R:** 50.00% (5/10) | Frame Acc: 25.63% | Neutral: 64.87% | FP: 9.49% | Confirmed: 70.00%
15. **E:** 20.00% (2/10) | Frame Acc: 3.04% | Neutral: 93.62% | FP: 3.34% | Confirmed: 20.00%
16. **N:** 20.00% (2/10) | Frame Acc: 1.22% | Neutral: 98.78% | FP: 0.00% | Confirmed: 0.00%
17. **M:** 10.00% (1/10) | Frame Acc: 0.62% | Neutral: 99.38% | FP: 0.00% | Confirmed: 0.00%
18. **Q:** 10.00% (1/10) | Frame Acc: 0.33% | Neutral: 99.01% | FP: 0.66% | Confirmed: 0.00%
19. **O:** 10.00% (1/10) | Frame Acc: 0.31% | Neutral: 91.98% | FP: 7.72% | Confirmed: 30.00%
20. **H:** 0.00% (0/10) | Frame Acc: 0.00% | Neutral: 63.96% | FP: 36.04% | Confirmed: 80.00%
21. **P:** 0.00% (0/10) | Frame Acc: 0.00% | Neutral: 100.00% | FP: 0.00% | Confirmed: 0.00%
22. **X:** 0.00% (0/10) | Frame Acc: 0.00% | Neutral: 100.00% | FP: 0.00% | Confirmed: 0.00%

### B. Xếp hạng theo Tỷ lệ Nhầm lẫn Sai (False Positive Rate — Cao xuống thấp)
1. **H:** **36.04%** (164 frames sai: 140 V, 22 K, 2 R)
2. **R:** **9.49%** (30 frames sai: 19 H, 9 D, 1 U, 1 L)
3. **O:** **7.72%** (25 frames sai: 25 A do fallback leak)
4. **D:** **7.30%** (23 frames sai: 22 P, 1 G)
5. **I:** **5.25%** (19 frames sai: 19 Y)
6. **E:** **3.34%** (11 frames sai: 11 C)
7. **K:** **1.67%** (5 frames sai: 2 I, 1 V, 1 H, 1 S)
8. **V:** **0.75%** (3 frames sai: 3 R)
9. **Q:** **0.66%** (2 frames sai: 2 G)
10. **S:** **0.65%** (2 frames sai: 2 L)
11. **T:** **0.33%** (1 frame sai: 1 B)
12. **11 chữ cái còn lại (Y, U, B, A, G, L, C, N, M, P, X):** **0.00%** (Hoàn toàn không có bất kỳ frame nhầm lẫn sai nào!)

### C. Xếp hạng theo Tỷ lệ Bị từ chối (Neutral Rate — Cao xuống thấp)
1. **P:** **100.00%** (337/337 frames)
2. **X:** **100.00%** (310/310 frames)
3. **M:** **99.38%** (321/323 frames)
4. **Q:** **99.01%** (300/303 frames)
5. **N:** **98.78%** (324/328 frames)
6. **E:** **93.62%** (308/329 frames)
7. **O:** **91.98%** (298/324 frames)
8. **R:** **64.87%** (205/316 frames)
9. **H:** **63.96%** (291/455 frames)
10. **C:** **56.20%** (145/258 frames)
11. **I:** **46.13%** (167/362 frames)
12. **T:** **44.77%** (137/306 frames)
13. **L:** **38.11%** (117/307 frames)
14. **D:** **37.14%** (117/315 frames)
15. **G:** **19.06%** (57/299 frames)
16. **K:** **18.06%** (54/299 frames)
17. **A:** **10.57%** (54/511 frames)
18. **V:** **8.75%** (35/400 frames)
19. **S:** **6.15%** (19/309 frames)
20. **B:** **1.09%** (4/366 frames)
21. **U:** **0.51%** (2/396 frames)
22. **Y:** **0.00%** (0/303 frames)

---

## 7. DANH SÁCH CÁC SỐ LIỆU TASK 12.1.1 CẦN ĐÍNH CHÍNH (ERRATA)

Qua quá trình kiểm toán đối chiếu, các điểm chưa chuẩn hóa trong báo cáo TASK 12.1.1 được đính chính chính thức như sau:

1. **Đính chính SHA-256 Checkpoint:**
   - *Số liệu cũ (TASK 12.1.1 Phần 6.B):* Ghi `ca4d35ebff...`.
   - *Số liệu đính chính:* `cab8123e23ed41a6088ba1c6e80a8aaaa9a382663418cc669eacff834795c254` (Khớp 100% với TASK 11.2.4).
2. **Đính chính Phán quyết Lượt `K_rep01_att01`:**
   - *Số liệu cũ:* Ghi 1 lượt MIXED cho chữ K.
   - *Số liệu đính chính:* Lượt `K_rep01_att01` có C=3, I=4 (out of 7 active frames). Theo quy tắc toán học nhất quán $I > C$, lượt này được phán quyết chính xác là **INCORRECT** (đưa tổng số lượt INCORRECT của 224 trials lên 26, MIXED về 0). Khi tester thực hiện `att02`, lượt này đạt C=30, I=0, chuyển thành **CORRECT** hoàn toàn.
3. **Làm rõ ranh giới Mẫu số MLP Invocation:**
   - Số lần gọi MLP trong Active Trials là **3.641**; số lần gọi trong Transition là **3.749**; tổng toàn tệp là **7.390**. Báo cáo TASK 12.1.1 đúng khi dùng 3.641 cho Active Trials nhưng cần làm rõ quan hệ cộng gộp với con số 7.390 của TASK 11.2.6.
4. **Phân rã chuẩn mực Ma trận Gating:**
   - Khẳng định đẳng thức bảo toàn mẫu: $TA (11) + FA (28) + FR (435) + TR (3.167) = 3.641$.

---

## 8. NHỮNG ĐIỂM VẪN CHƯA THỂ KẾT LUẬN (UNRESOLVED ITEMS)

Để tuân thủ nghiêm ngặt nguyên tắc khoa học và không suy đoán vượt quá bằng chứng thực tế:

1. **Khả năng xác minh mã băm Checkpoint nội tại từng bản ghi JSONL:**
   - Dù ta đã chứng minh tệp checkpoint trên đĩa có hash `cab8123e...` và tiến trình backend nạp thành công, cấu trúc JSONL hiện thời của `pilot_logger.py` **không lưu trữ trường SHA-256 của checkpoint trong từng frame payload**. Do đó, về mặt hình thức dữ liệu độc lập, JSONL chỉ ghi nhận `has_static_head: True, load_status: loaded, pytorch_version: 2.2.2+cpu` chứ không tự mang chứng chỉ băm của chính model.
2. **Nguyên nhân sâu xa của hiện tượng MLP thiên lệch (Bias) về nhãn V ($77.07\%$):**
   - Dữ liệu chẩn đoán cho thấy $2.806 / 3.641$ frame `NO_MATCH` bị MLP đoán thô thành V. Tuy nhiên, nếu chỉ dựa vào tệp JSONL và checkpoint đã biên dịch, không thể kết luận nguyên nhân thiên lệch này bắt nguồn từ: (a) Mất cân bằng mẫu trong tập dữ liệu huấn luyện `vsl_alphabet_static_v2.npz`, (b) Hàm mất mát CrossEntropy không có trọng số, hay (c) Bản chất không gian vector của tư thế bàn tay khi ở trạng thái nghỉ. Vấn đề này cần kiểm toán riêng quy trình huấn luyện ở nhiệm vụ khác.

---

## 9. PHỤ LỤC: BẢNG CHI TIẾT TOÀN BỘ 224 TRIALS (TRIAL ADJUDICATION TABLE)

| Trial ID | Exp | Att | Total | Correct | Incor | Neutral | Non-Neut | Non-Neut Acc | Confirmed | Adjudication |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `pilot_20261107_2141_A_rep01_att01` | **A** | att01 | 38 | 21 | 0 | 17 | 21 | 100.0% | 2 | **CORRECT** |
| `pilot_20261107_2141_A_rep02_att01` | **A** | att01 | 27 | 25 | 0 | 2 | 25 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_A_rep03_att01` | **A** | att01 | 32 | 28 | 0 | 4 | 28 | 100.0% | 3 | **CORRECT** |
| `pilot_20261107_2141_A_rep03_att02` | **A** | att02 | 40 | 34 | 0 | 6 | 34 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_A_rep04_att01` | **A** | att01 | 55 | 55 | 0 | 0 | 55 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_A_rep05_att01` | **A** | att01 | 55 | 55 | 0 | 0 | 55 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_A_rep06_att01` | **A** | att01 | 55 | 43 | 0 | 12 | 43 | 100.0% | 3 | **CORRECT** |
| `pilot_20261107_2141_A_rep07_att01` | **A** | att01 | 47 | 38 | 0 | 9 | 38 | 100.0% | 3 | **CORRECT** |
| `pilot_20261107_2141_A_rep08_att01` | **A** | att01 | 56 | 56 | 0 | 0 | 56 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_A_rep09_att01` | **A** | att01 | 53 | 49 | 0 | 4 | 49 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_A_rep10_att01` | **A** | att01 | 53 | 53 | 0 | 0 | 53 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_B_rep01_att01` | **B** | att01 | 52 | 52 | 0 | 0 | 52 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_B_rep02_att01` | **B** | att01 | 33 | 33 | 0 | 0 | 33 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_B_rep03_att01` | **B** | att01 | 32 | 32 | 0 | 0 | 32 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_B_rep04_att01` | **B** | att01 | 38 | 38 | 0 | 0 | 38 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_B_rep05_att01` | **B** | att01 | 38 | 37 | 0 | 1 | 37 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_B_rep06_att01` | **B** | att01 | 34 | 32 | 0 | 2 | 32 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_B_rep07_att01` | **B** | att01 | 34 | 34 | 0 | 0 | 34 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_B_rep08_att01` | **B** | att01 | 33 | 32 | 0 | 1 | 32 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_B_rep09_att01` | **B** | att01 | 36 | 36 | 0 | 0 | 36 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_B_rep10_att01` | **B** | att01 | 36 | 36 | 0 | 0 | 36 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_C_rep01_att01` | **C** | att01 | 32 | 2 | 0 | 30 | 2 | 100.0% | 0 | **CORRECT** |
| `pilot_20261107_2141_C_rep02_att01` | **C** | att01 | 26 | 2 | 0 | 24 | 2 | 100.0% | 0 | **CORRECT** |
| `pilot_20261107_2141_C_rep03_att01` | **C** | att01 | 25 | 16 | 0 | 9 | 16 | 100.0% | 3 | **CORRECT** |
| `pilot_20261107_2141_C_rep04_att01` | **C** | att01 | 24 | 14 | 0 | 10 | 14 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_C_rep05_att01` | **C** | att01 | 25 | 10 | 0 | 15 | 10 | 100.0% | 2 | **CORRECT** |
| `pilot_20261107_2141_C_rep06_att01` | **C** | att01 | 27 | 25 | 0 | 2 | 25 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_C_rep07_att01` | **C** | att01 | 26 | 11 | 0 | 15 | 11 | 100.0% | 2 | **CORRECT** |
| `pilot_20261107_2141_C_rep08_att01` | **C** | att01 | 24 | 10 | 0 | 14 | 10 | 100.0% | 2 | **CORRECT** |
| `pilot_20261107_2141_C_rep09_att01` | **C** | att01 | 22 | 6 | 0 | 16 | 6 | 100.0% | 2 | **CORRECT** |
| `pilot_20261107_2141_C_rep10_att01` | **C** | att01 | 27 | 17 | 0 | 10 | 17 | 100.0% | 3 | **CORRECT** |
| `pilot_20261107_2141_D_rep01_att01` | **D** | att01 | 24 | 0 | 0 | 24 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_D_rep02_att01` | **D** | att01 | 32 | 6 | 22 | 4 | 28 | 21.4% | 4 | **INCORRECT** |
| `pilot_20261107_2141_D_rep03_att01` | **D** | att01 | 29 | 29 | 0 | 0 | 29 | 100.0% | 3 | **CORRECT** |
| `pilot_20261107_2141_D_rep04_att01` | **D** | att01 | 28 | 24 | 0 | 4 | 24 | 100.0% | 3 | **CORRECT** |
| `pilot_20261107_2141_D_rep05_att01` | **D** | att01 | 31 | 31 | 0 | 0 | 31 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_D_rep06_att01` | **D** | att01 | 29 | 29 | 0 | 0 | 29 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_D_rep07_att01` | **D** | att01 | 52 | 0 | 1 | 51 | 1 | 0.0% | 0 | **INCORRECT** |
| `pilot_20261107_2141_D_rep08_att01` | **D** | att01 | 30 | 0 | 0 | 30 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_D_rep09_att01` | **D** | att01 | 30 | 30 | 0 | 0 | 30 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_D_rep10_att01` | **D** | att01 | 30 | 26 | 0 | 4 | 26 | 100.0% | 3 | **CORRECT** |
| `pilot_20261107_2141_E_rep01_att01` | **E** | att01 | 32 | 0 | 0 | 32 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_E_rep02_att01` | **E** | att01 | 32 | 0 | 0 | 32 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_E_rep03_att01` | **E** | att01 | 39 | 0 | 0 | 39 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_E_rep04_att01` | **E** | att01 | 42 | 0 | 0 | 42 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_E_rep05_att01` | **E** | att01 | 30 | 0 | 0 | 30 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_E_rep06_att01` | **E** | att01 | 34 | 0 | 11 | 23 | 11 | 0.0% | 1 | **INCORRECT** |
| `pilot_20261107_2141_E_rep07_att01` | **E** | att01 | 32 | 4 | 0 | 28 | 4 | 100.0% | 0 | **CORRECT** |
| `pilot_20261107_2141_E_rep08_att01` | **E** | att01 | 31 | 6 | 0 | 25 | 6 | 100.0% | 1 | **CORRECT** |
| `pilot_20261107_2141_E_rep09_att01` | **E** | att01 | 27 | 0 | 0 | 27 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_E_rep10_att01` | **E** | att01 | 30 | 0 | 0 | 30 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_G_rep01_att01` | **G** | att01 | 30 | 30 | 0 | 0 | 30 | 100.0% | 3 | **CORRECT** |
| `pilot_20261107_2141_G_rep02_att01` | **G** | att01 | 30 | 30 | 0 | 0 | 30 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_G_rep03_att01` | **G** | att01 | 31 | 31 | 0 | 0 | 31 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_G_rep04_att01` | **G** | att01 | 27 | 27 | 0 | 0 | 27 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_G_rep05_att01` | **G** | att01 | 33 | 24 | 0 | 9 | 24 | 100.0% | 3 | **CORRECT** |
| `pilot_20261107_2141_G_rep06_att01` | **G** | att01 | 29 | 18 | 0 | 11 | 18 | 100.0% | 3 | **CORRECT** |
| `pilot_20261107_2141_G_rep07_att01` | **G** | att01 | 29 | 22 | 0 | 7 | 22 | 100.0% | 3 | **CORRECT** |
| `pilot_20261107_2141_G_rep08_att01` | **G** | att01 | 29 | 21 | 0 | 8 | 21 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_G_rep09_att01` | **G** | att01 | 29 | 16 | 0 | 13 | 16 | 100.0% | 3 | **CORRECT** |
| `pilot_20261107_2141_G_rep10_att01` | **G** | att01 | 32 | 23 | 0 | 9 | 23 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_H_rep01_att01` | **H** | att01 | 30 | 0 | 13 | 17 | 13 | 0.0% | 2 | **INCORRECT** |
| `pilot_20261107_2141_H_rep02_att01` | **H** | att01 | 36 | 0 | 12 | 24 | 12 | 0.0% | 2 | **INCORRECT** |
| `pilot_20261107_2141_H_rep03_att01` | **H** | att01 | 44 | 0 | 5 | 39 | 5 | 0.0% | 0 | **INCORRECT** |
| `pilot_20261107_2141_H_rep04_att01` | **H** | att01 | 58 | 0 | 5 | 53 | 5 | 0.0% | 0 | **INCORRECT** |
| `pilot_20261107_2141_H_rep05_att01` | **H** | att01 | 38 | 0 | 10 | 28 | 10 | 0.0% | 1 | **INCORRECT** |
| `pilot_20261107_2141_H_rep06_att01` | **H** | att01 | 51 | 0 | 10 | 41 | 10 | 0.0% | 1 | **INCORRECT** |
| `pilot_20261107_2141_H_rep07_att01` | **H** | att01 | 53 | 0 | 24 | 29 | 24 | 0.0% | 3 | **INCORRECT** |
| `pilot_20261107_2141_H_rep08_att01` | **H** | att01 | 59 | 0 | 19 | 40 | 19 | 0.0% | 2 | **INCORRECT** |
| `pilot_20261107_2141_H_rep09_att01` | **H** | att01 | 49 | 0 | 46 | 3 | 46 | 0.0% | 4 | **INCORRECT** |
| `pilot_20261107_2141_H_rep10_att01` | **H** | att01 | 37 | 0 | 20 | 17 | 20 | 0.0% | 4 | **INCORRECT** |
| `pilot_20261107_2141_I_rep01_att01` | **I** | att01 | 34 | 0 | 18 | 16 | 18 | 0.0% | 2 | **INCORRECT** |
| `pilot_20261107_2141_I_rep02_att01` | **I** | att01 | 34 | 19 | 1 | 14 | 20 | 95.0% | 3 | **CORRECT** |
| `pilot_20261107_2141_I_rep03_att01` | **I** | att01 | 40 | 6 | 0 | 34 | 6 | 100.0% | 0 | **CORRECT** |
| `pilot_20261107_2141_I_rep04_att01` | **I** | att01 | 28 | 14 | 0 | 14 | 14 | 100.0% | 3 | **CORRECT** |
| `pilot_20261107_2141_I_rep05_att01` | **I** | att01 | 49 | 45 | 0 | 4 | 45 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_I_rep06_att01` | **I** | att01 | 47 | 47 | 0 | 0 | 47 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_I_rep07_att01` | **I** | att01 | 51 | 45 | 0 | 6 | 45 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_I_rep08_att01` | **I** | att01 | 26 | 0 | 0 | 26 | 0 | 0.0% | 1 | **NEUTRAL** |
| `pilot_20261107_2141_I_rep09_att01` | **I** | att01 | 28 | 0 | 0 | 28 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_I_rep10_att01` | **I** | att01 | 25 | 0 | 0 | 25 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_K_rep01_att01` | **K** | att01 | 17 | 3 | 4 | 10 | 7 | 42.9% | 0 | **INCORRECT** |
| `pilot_20261107_2141_K_rep01_att02` | **K** | att02 | 33 | 30 | 0 | 3 | 30 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_K_rep02_att01` | **K** | att01 | 23 | 21 | 0 | 2 | 21 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_K_rep03_att01` | **K** | att01 | 29 | 6 | 0 | 23 | 6 | 100.0% | 1 | **CORRECT** |
| `pilot_20261107_2141_K_rep04_att01` | **K** | att01 | 28 | 28 | 0 | 0 | 28 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_K_rep05_att01` | **K** | att01 | 31 | 31 | 0 | 0 | 31 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_K_rep06_att01` | **K** | att01 | 27 | 17 | 0 | 10 | 17 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_K_rep07_att01` | **K** | att01 | 27 | 26 | 0 | 1 | 26 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_K_rep08_att01` | **K** | att01 | 26 | 25 | 0 | 1 | 25 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_K_rep09_att01` | **K** | att01 | 28 | 26 | 0 | 2 | 26 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_K_rep10_att01` | **K** | att01 | 30 | 27 | 1 | 2 | 28 | 96.4% | 4 | **CORRECT** |
| `pilot_20261107_2141_L_rep01_att01` | **L** | att01 | 27 | 13 | 0 | 14 | 13 | 100.0% | 2 | **CORRECT** |
| `pilot_20261107_2141_L_rep02_att01` | **L** | att01 | 43 | 37 | 0 | 6 | 37 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_L_rep03_att01` | **L** | att01 | 32 | 2 | 0 | 30 | 2 | 100.0% | 0 | **CORRECT** |
| `pilot_20261107_2141_L_rep04_att01` | **L** | att01 | 32 | 17 | 0 | 15 | 17 | 100.0% | 2 | **CORRECT** |
| `pilot_20261107_2141_L_rep05_att01` | **L** | att01 | 29 | 27 | 0 | 2 | 27 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_L_rep06_att01` | **L** | att01 | 32 | 32 | 0 | 0 | 32 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_L_rep07_att01` | **L** | att01 | 29 | 22 | 0 | 7 | 22 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_L_rep08_att01` | **L** | att01 | 27 | 0 | 0 | 27 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_L_rep09_att01` | **L** | att01 | 29 | 13 | 0 | 16 | 13 | 100.0% | 2 | **CORRECT** |
| `pilot_20261107_2141_L_rep10_att01` | **L** | att01 | 27 | 27 | 0 | 0 | 27 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_M_rep01_att01` | **M** | att01 | 28 | 0 | 0 | 28 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_M_rep02_att01` | **M** | att01 | 31 | 0 | 0 | 31 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_M_rep03_att01` | **M** | att01 | 37 | 2 | 0 | 35 | 2 | 100.0% | 0 | **CORRECT** |
| `pilot_20261107_2141_M_rep04_att01` | **M** | att01 | 36 | 0 | 0 | 36 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_M_rep05_att01` | **M** | att01 | 35 | 0 | 0 | 35 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_M_rep06_att01` | **M** | att01 | 31 | 0 | 0 | 31 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_M_rep07_att01` | **M** | att01 | 29 | 0 | 0 | 29 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_M_rep08_att01` | **M** | att01 | 34 | 0 | 0 | 34 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_M_rep09_att01` | **M** | att01 | 34 | 0 | 0 | 34 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_M_rep10_att01` | **M** | att01 | 28 | 0 | 0 | 28 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_N_rep01_att01` | **N** | att01 | 31 | 3 | 0 | 28 | 3 | 100.0% | 0 | **CORRECT** |
| `pilot_20261107_2141_N_rep02_att01` | **N** | att01 | 28 | 0 | 0 | 28 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_N_rep03_att01` | **N** | att01 | 29 | 0 | 0 | 29 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_N_rep04_att01` | **N** | att01 | 28 | 1 | 0 | 27 | 1 | 100.0% | 0 | **CORRECT** |
| `pilot_20261107_2141_N_rep05_att01` | **N** | att01 | 33 | 0 | 0 | 33 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_N_rep06_att01` | **N** | att01 | 29 | 0 | 0 | 29 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_N_rep07_att01` | **N** | att01 | 30 | 0 | 0 | 30 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_N_rep08_att01` | **N** | att01 | 45 | 0 | 0 | 45 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_N_rep09_att01` | **N** | att01 | 31 | 0 | 0 | 31 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_N_rep10_att01` | **N** | att01 | 44 | 0 | 0 | 44 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_O_rep01_att01` | **O** | att01 | 33 | 0 | 0 | 33 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_O_rep02_att01` | **O** | att01 | 37 | 0 | 6 | 31 | 6 | 0.0% | 1 | **INCORRECT** |
| `pilot_20261107_2141_O_rep03_att01` | **O** | att01 | 36 | 0 | 1 | 35 | 1 | 0.0% | 0 | **INCORRECT** |
| `pilot_20261107_2141_O_rep04_att01` | **O** | att01 | 31 | 0 | 6 | 25 | 6 | 0.0% | 1 | **INCORRECT** |
| `pilot_20261107_2141_O_rep05_att01` | **O** | att01 | 31 | 0 | 2 | 29 | 2 | 0.0% | 0 | **INCORRECT** |
| `pilot_20261107_2141_O_rep06_att01` | **O** | att01 | 31 | 0 | 4 | 27 | 4 | 0.0% | 0 | **INCORRECT** |
| `pilot_20261107_2141_O_rep07_att01` | **O** | att01 | 32 | 1 | 0 | 31 | 1 | 100.0% | 0 | **CORRECT** |
| `pilot_20261107_2141_O_rep08_att01` | **O** | att01 | 31 | 0 | 0 | 31 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_O_rep09_att01` | **O** | att01 | 30 | 0 | 0 | 30 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_O_rep10_att01` | **O** | att01 | 32 | 0 | 6 | 26 | 6 | 0.0% | 1 | **INCORRECT** |
| `pilot_20261107_2141_P_rep01_att01` | **P** | att01 | 9 | 0 | 0 | 9 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_P_rep01_att02` | **P** | att02 | 32 | 0 | 0 | 32 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_P_rep02_att01` | **P** | att01 | 32 | 0 | 0 | 32 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_P_rep03_att01` | **P** | att01 | 29 | 0 | 0 | 29 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_P_rep04_att01` | **P** | att01 | 31 | 0 | 0 | 31 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_P_rep05_att01` | **P** | att01 | 32 | 0 | 0 | 32 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_P_rep06_att01` | **P** | att01 | 41 | 0 | 0 | 41 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_P_rep07_att01` | **P** | att01 | 37 | 0 | 0 | 37 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_P_rep08_att01` | **P** | att01 | 34 | 0 | 0 | 34 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_P_rep09_att01` | **P** | att01 | 31 | 0 | 0 | 31 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_P_rep10_att01` | **P** | att01 | 29 | 0 | 0 | 29 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_Q_rep01_att01` | **Q** | att01 | 30 | 0 | 0 | 30 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_Q_rep02_att01` | **Q** | att01 | 29 | 0 | 0 | 29 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_Q_rep03_att01` | **Q** | att01 | 26 | 0 | 0 | 26 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_Q_rep04_att01` | **Q** | att01 | 28 | 0 | 0 | 28 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_Q_rep05_att01` | **Q** | att01 | 32 | 0 | 0 | 32 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_Q_rep06_att01` | **Q** | att01 | 32 | 0 | 2 | 30 | 2 | 0.0% | 0 | **INCORRECT** |
| `pilot_20261107_2141_Q_rep07_att01` | **Q** | att01 | 29 | 0 | 0 | 29 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_Q_rep08_att01` | **Q** | att01 | 31 | 0 | 0 | 31 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_Q_rep09_att01` | **Q** | att01 | 31 | 1 | 0 | 30 | 1 | 100.0% | 0 | **CORRECT** |
| `pilot_20261107_2141_Q_rep10_att01` | **Q** | att01 | 35 | 0 | 0 | 35 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_R_rep01_att01` | **R** | att01 | 27 | 0 | 0 | 27 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_R_rep02_att01` | **R** | att01 | 29 | 2 | 7 | 20 | 9 | 22.2% | 1 | **INCORRECT** |
| `pilot_20261107_2141_R_rep03_att01` | **R** | att01 | 30 | 8 | 0 | 22 | 8 | 100.0% | 2 | **CORRECT** |
| `pilot_20261107_2141_R_rep04_att01` | **R** | att01 | 33 | 31 | 0 | 2 | 31 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_R_rep05_att01` | **R** | att01 | 37 | 11 | 4 | 22 | 15 | 73.3% | 2 | **CORRECT** |
| `pilot_20261107_2141_R_rep06_att01` | **R** | att01 | 38 | 0 | 14 | 24 | 14 | 0.0% | 2 | **INCORRECT** |
| `pilot_20261107_2141_R_rep07_att01` | **R** | att01 | 34 | 11 | 1 | 22 | 12 | 91.7% | 1 | **CORRECT** |
| `pilot_20261107_2141_R_rep08_att01` | **R** | att01 | 29 | 0 | 0 | 29 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_R_rep09_att01` | **R** | att01 | 31 | 2 | 4 | 25 | 6 | 33.3% | 0 | **INCORRECT** |
| `pilot_20261107_2141_R_rep10_att01` | **R** | att01 | 28 | 16 | 0 | 12 | 16 | 100.0% | 2 | **CORRECT** |
| `pilot_20261107_2141_S_rep01_att01` | **S** | att01 | 8 | 1 | 2 | 5 | 3 | 33.3% | 0 | **INCORRECT** |
| `pilot_20261107_2141_S_rep01_att02` | **S** | att02 | 31 | 31 | 0 | 0 | 31 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_S_rep02_att01` | **S** | att01 | 28 | 21 | 0 | 7 | 21 | 100.0% | 3 | **CORRECT** |
| `pilot_20261107_2141_S_rep03_att01` | **S** | att01 | 26 | 26 | 0 | 0 | 26 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_S_rep04_att01` | **S** | att01 | 29 | 22 | 0 | 7 | 22 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_S_rep05_att01` | **S** | att01 | 32 | 32 | 0 | 0 | 32 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_S_rep06_att01` | **S** | att01 | 29 | 29 | 0 | 0 | 29 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_S_rep07_att01` | **S** | att01 | 25 | 25 | 0 | 0 | 25 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_S_rep08_att01` | **S** | att01 | 30 | 30 | 0 | 0 | 30 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_S_rep09_att01` | **S** | att01 | 40 | 40 | 0 | 0 | 40 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_S_rep10_att01` | **S** | att01 | 31 | 31 | 0 | 0 | 31 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_T_rep01_att01` | **T** | att01 | 33 | 12 | 1 | 20 | 13 | 92.3% | 2 | **CORRECT** |
| `pilot_20261107_2141_T_rep02_att01` | **T** | att01 | 29 | 29 | 0 | 0 | 29 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_T_rep03_att01` | **T** | att01 | 28 | 28 | 0 | 0 | 28 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_T_rep04_att01` | **T** | att01 | 29 | 8 | 0 | 21 | 8 | 100.0% | 1 | **CORRECT** |
| `pilot_20261107_2141_T_rep05_att01` | **T** | att01 | 27 | 22 | 0 | 5 | 22 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_T_rep06_att01` | **T** | att01 | 32 | 27 | 0 | 5 | 27 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_T_rep07_att01` | **T** | att01 | 27 | 25 | 0 | 2 | 25 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_T_rep08_att01` | **T** | att01 | 29 | 17 | 0 | 12 | 17 | 100.0% | 3 | **CORRECT** |
| `pilot_20261107_2141_T_rep09_att01` | **T** | att01 | 29 | 0 | 0 | 29 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_T_rep10_att01` | **T** | att01 | 43 | 0 | 0 | 43 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_U_rep01_att01` | **U** | att01 | 31 | 31 | 0 | 0 | 31 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_U_rep02_att01` | **U** | att01 | 29 | 29 | 0 | 0 | 29 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_U_rep03_att01` | **U** | att01 | 30 | 29 | 0 | 1 | 29 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_U_rep04_att01` | **U** | att01 | 39 | 38 | 0 | 1 | 38 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_U_rep05_att01` | **U** | att01 | 44 | 44 | 0 | 0 | 44 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_U_rep06_att01` | **U** | att01 | 40 | 40 | 0 | 0 | 40 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_U_rep07_att01` | **U** | att01 | 48 | 48 | 0 | 0 | 48 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_U_rep08_att01` | **U** | att01 | 47 | 47 | 0 | 0 | 47 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_U_rep09_att01` | **U** | att01 | 46 | 46 | 0 | 0 | 46 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_U_rep10_att01` | **U** | att01 | 42 | 42 | 0 | 0 | 42 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_V_rep01_att01` | **V** | att01 | 37 | 35 | 2 | 0 | 37 | 94.6% | 3 | **CORRECT** |
| `pilot_20261107_2141_V_rep02_att01` | **V** | att01 | 43 | 42 | 1 | 0 | 43 | 97.7% | 4 | **CORRECT** |
| `pilot_20261107_2141_V_rep03_att01` | **V** | att01 | 39 | 39 | 0 | 0 | 39 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_V_rep04_att01` | **V** | att01 | 41 | 38 | 0 | 3 | 38 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_V_rep05_att01` | **V** | att01 | 47 | 44 | 0 | 3 | 44 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_V_rep06_att01` | **V** | att01 | 43 | 37 | 0 | 6 | 37 | 100.0% | 3 | **CORRECT** |
| `pilot_20261107_2141_V_rep07_att01` | **V** | att01 | 40 | 35 | 0 | 5 | 35 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_V_rep08_att01` | **V** | att01 | 43 | 25 | 0 | 18 | 25 | 100.0% | 3 | **CORRECT** |
| `pilot_20261107_2141_V_rep09_att01` | **V** | att01 | 32 | 32 | 0 | 0 | 32 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_V_rep10_att01` | **V** | att01 | 35 | 35 | 0 | 0 | 35 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_X_rep01_att01` | **X** | att01 | 46 | 0 | 0 | 46 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_X_rep02_att01` | **X** | att01 | 29 | 0 | 0 | 29 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_X_rep03_att01` | **X** | att01 | 29 | 0 | 0 | 29 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_X_rep04_att01` | **X** | att01 | 28 | 0 | 0 | 28 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_X_rep05_att01` | **X** | att01 | 27 | 0 | 0 | 27 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_X_rep06_att01` | **X** | att01 | 30 | 0 | 0 | 30 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_X_rep07_att01` | **X** | att01 | 30 | 0 | 0 | 30 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_X_rep08_att01` | **X** | att01 | 30 | 0 | 0 | 30 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_X_rep09_att01` | **X** | att01 | 31 | 0 | 0 | 31 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_X_rep10_att01` | **X** | att01 | 30 | 0 | 0 | 30 | 0 | 0.0% | 0 | **NEUTRAL** |
| `pilot_20261107_2141_Y_rep01_att01` | **Y** | att01 | 31 | 31 | 0 | 0 | 31 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_Y_rep02_att01` | **Y** | att01 | 36 | 36 | 0 | 0 | 36 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_Y_rep03_att01` | **Y** | att01 | 27 | 27 | 0 | 0 | 27 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_Y_rep04_att01` | **Y** | att01 | 29 | 29 | 0 | 0 | 29 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_Y_rep05_att01` | **Y** | att01 | 26 | 26 | 0 | 0 | 26 | 100.0% | 3 | **CORRECT** |
| `pilot_20261107_2141_Y_rep06_att01` | **Y** | att01 | 27 | 27 | 0 | 0 | 27 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_Y_rep07_att01` | **Y** | att01 | 28 | 28 | 0 | 0 | 28 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_Y_rep08_att01` | **Y** | att01 | 33 | 33 | 0 | 0 | 33 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_Y_rep09_att01` | **Y** | att01 | 28 | 28 | 0 | 0 | 28 | 100.0% | 4 | **CORRECT** |
| `pilot_20261107_2141_Y_rep10_att01` | **Y** | att01 | 38 | 38 | 0 | 0 | 38 | 100.0% | 4 | **CORRECT** |

---
*Báo cáo kiểm toán số liệu độc lập TASK 12.1.1.1 hoàn thành.*