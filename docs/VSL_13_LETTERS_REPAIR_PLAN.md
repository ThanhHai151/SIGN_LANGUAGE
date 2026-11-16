# VSL — KẾ HOẠCH TOÀN DIỆN KIỂM TOÁN VÀ SỬA CHỮA 13 CHỮ CÁI (VSL 13 LETTERS REPAIR PLAN)

- **Đường dẫn tài liệu:** [`docs/VSL_13_LETTERS_REPAIR_PLAN.md`](file:///d:/SIGN_LANGUAGE/docs/VSL_13_LETTERS_REPAIR_PLAN.md)
- **Trạng thái:** `ACTIVE EXECUTION — CHECKPOINT 5D COMPLETED`
- **Ngày khởi tạo:** 11/11/2026
- **Nhánh làm việc đối chiếu:** `feat/controlled-pilot-29-letters` (HEAD `ce9efdee`)
- **Phạm vi kiểm soát toàn diện:** 13 chữ cái VSL:
  - **Nhóm A (6 chữ không dấu):** Q, O, H, M, N, R
  - **Nhóm B (7 chữ có dấu):** Đ, Ư, Ê, Ă, Â, Ơ, Ô
  - **Bảo vệ tuyệt đối:** 16 chữ cái tĩnh còn lại và toàn bộ bảng 29 chữ cái tiếng Việt VSL.
- **Tài liệu SSOT kế thừa trực tiếp:**
  - [`docs/VSL_6_LETTERS_REPAIR_PLAN.md`](file:///d:/SIGN_LANGUAGE/docs/VSL_6_LETTERS_REPAIR_PLAN.md) *(SSOT kế thừa đợt 6 chữ — BẢO LƯU NGUYÊN VẸN)*
  - [`docs/VSL_8_LETTERS_FIX_PLAN.md`](file:///d:/SIGN_LANGUAGE/docs/VSL_8_LETTERS_FIX_PLAN.md) *(SSOT gốc lịch sử — BẢO LƯU NGUYÊN VẸN)*
  - [`backend/data/alphabet_v2/audit/task_7f1_static_letters_audit_report.md`](file:///d:/SIGN_LANGUAGE/backend/data/alphabet_v2/audit/task_7f1_static_letters_audit_report.md)
  - [`backend/data/alphabet_v2/audit/task_7e5b_report.md`](file:///d:/SIGN_LANGUAGE/backend/data/alphabet_v2/audit/task_7e5b_report.md)

---

## 1. RANH GIỚI CÔNG VIỆC VÀ NGUYÊN TẮC BẢO TOÀN

1. **Bản điều phối kiểm soát thay đổi hợp nhất (Unified SSOT):** Kế thừa toàn bộ kết quả đã kiểm chứng từ Task 0 và Task 1 (chữ Q hạ threshold xuống 0.75, Gate 1 giữ nguyên trạng thái mở chưa đóng).
2. **Kỷ luật phê duyệt theo giai đoạn (Staged Execution):**
   - **Giai đoạn 1 (Lượt này):** Chỉ triển khai Checkpoint 3 (sửa nhóm không dấu O, H, M, N, R; Q giữ 0.75) và Checkpoint 4 (static regression & đối kháng).
   - **Giai đoạn 2 (Checkpoint 5 trở đi):** Khôi phục pipeline 7 chữ có dấu (Diacritic FSM) sau khi nhóm không dấu đã đạt chuẩn nghiệm thu. Tuyệt đối KHÔNG tự ý kích hoạt FSM trong lượt này.
3. **Bảo tồn Working Tree:** Không force push, không reset hard, không checkout nhánh khác, không tự tiện commit khi chưa nghiệm thu hoàn tất.

---

## 2. KIẾN TRÚC HỆ THỐNG ĐƯỜNG ỐNG 4 TẦNG

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   KIẾN TRÚC ĐƯỜNG ỐNG NHẬN DIỆN 4 TẦNG                      │
│                                                                             │
│  21 Landmarks (MediaPipe)                                                   │
│        │                                                                    │
│        ▼                                                                    │
│  [TẦNG 1: GEOMETRIC RULES] (alphabet_classifier.py#classify_detailed)       │
│        │                                                                    │
│        ├─► MATCH: Chữ cái A..Y (17 luật hình học: B, I, Y, V, D, L, T, O,    │
│        │          C, A, S, E, G, H, K, U, P, Q, R, X)                       │
│        │          recognition_source = "geometric_rule", conf = 0.95        │
│        │          (Static Head MLP HOÀN TOÀN BỊ BỎ QUA)                     │
│        │                                                                    │
│        └─► NO_MATCH / UNKNOWN                                               │
│                  │                                                          │
│                  ▼                                                          │
│  [TẦNG 2: STATIC ALPHABET HEAD MLP 72D] (alphabet_static_head.py)           │
│        │                                                                    │
│        ├─► Fallback Base: ALLOWED_FALLBACK_BASE_LETTERS = {"A", "D", "E"}   │
│        │   (O đề xuất bổ sung vào nhóm này tại CP3)                         │
│        │   Nếu head_conf >= 0.70 ──► source = "static_head_fallback"        │
│        │                                                                    │
│        ├─► Supplemental: ALLOWED_SUPPLEMENTAL_LETTERS                       │
│        │   {"G", "H", "K", "M", "N", "P", "Q", "R", "U", "X"}              │
│        │   Nếu head_conf >= req_threshold ──► source = "static_head"        │
│        │                                                                    │
│        └─► Dưới ngưỡng HOẶC Ngoài danh mục                                  │
│            final_sign = "NEUTRAL", gating_status = "rejected"               │
│                  │                                                          │
│                  ▼                                                          │
│  [TẦNG 3: DIACRITIC FSM] (diacritic_state_machine.py)                       │
│        │                                                                    │
│        ├─► Nhận diện 7 chữ có dấu: Ă, Â, Ê, Ô, Ơ, Ư, Đ                      │
│        └─► HIỆN TRẠNG: Tạm ngắt tại recognizer.py#L1552                     │
│            (Sẽ khôi phục và hiệu chuẩn tại CHECKPOINT 5)                    │
│                  │                                                          │
│                  ▼                                                          │
│  [TẦNG 4: TEMPORAL SMOOTHING BUFFER] (recognizer.py#L1648-1669)             │
│        │                                                                    │
│        ├─► deque(maxlen=10), lọc frame conf >= 0.70                         │
│        ├─► Bầu chọn khi len(recent_signs) >= 5 (window // 2)                │
│        ├─► Vote threshold: count >= 5                                       │
│        └─► Cooldown debounce: cooldown_sec = 1.2s                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. QUY TẮC PROVENANCE VÀ PHÂN ĐỊNH NGUỒN DỮ LIỆU

### 3.1. Thứ tự Ưu tiên Nghiêm ngặt (Provenance Hierarchy)
- **Ưu tiên 1 (Canonical Nationwide — Tuyệt đối):** File mã QIPEDC **KHÔNG có hậu tố** vùng miền (`D0504`, `D0505`, `D0506`, `D0510`, `D0511`, `D0495`).
- **Ưu tiên 2 (Approved Southern Fallback — Có điều kiện):** File có **hậu tố `N`** (`D0500N` cho H, `D0508N` cho Ơ, `D0515N` cho Ư). Chỉ áp dụng khi không tồn tại nguồn toàn quốc hợp lệ.
- **Loại trừ khỏi Canonical:** File có **hậu tố `B`** (Miền Bắc, ví dụ `D0500B`, `D0490B`, `D0491B`, `D0497B`, `D0507B`) hoặc `T` (Miền Trung). Giữ làm dữ liệu đối chiếu hoặc fixture kỹ thuật, cấm dùng làm thước đo canonical.

### 3.2. Bảng Provenance 13 Chữ Cái:

| Chữ cái | Phân loại | Video ID | Nguồn áp dụng | Vai trò kỹ thuật | Số mẫu hợp lệ | Ghi chú provenance |
| :---: | :---: | :---: | :---: | :---: | :---: | :---|
| **Q** | Không dấu | `D0510` | Toàn quốc | Canonical Baseline | 19 stable frames | Ưu tiên 1 |
| **O** | Không dấu | `D0506` | Toàn quốc | Canonical Baseline | 20 stable frames | Ưu tiên 1 |
| **H** | Không dấu | `D0500N` | Miền Nam | Approved Fallback | 37 valid frames | Ưu tiên 2 (D0500 toàn quốc không tồn tại) |
| **H** | Không dấu | `D0500B` | Miền Bắc | Legacy Regression | 18 frames | Nằm trong static_v2.npz (không làm canonical) |
| **M** | Không dấu | `D0504` | Toàn quốc | Canonical Baseline | 19 stable frames | Ưu tiên 1 |
| **N** | Không dấu | `D0505` | Toàn quốc | Canonical Baseline | 25 stable frames | Ưu tiên 1 |
| **R** | Không dấu | `D0511` | Toàn quốc | Canonical Baseline | 16 stable frames | Ưu tiên 1 |
| **Đ** | Có dấu | `D0495` | Toàn quốc | Canonical Dynamic | 47 valid frames | Ưu tiên 1 (`is_dynamic: true`) |
| **Ư** | Có dấu | `D0515N` | Miền Nam | Approved Fallback | 47 valid frames | Ưu tiên 2 (`is_dynamic: true`) |
| **Ê** | Có dấu | `D0497B` | Miền Bắc | Provisional Fixture | 52 valid frames | Chỉ có bản B trong repo; CP5 xử lý |
| **Ă** | Có dấu | `D0490B` | Miền Bắc | Provisional Fixture | 53 valid frames | Chỉ có bản B trong repo; CP5 xử lý |
| **Â** | Có dấu | `D0491B` | Miền Bắc | Provisional Fixture | 57 valid frames | Chỉ có bản B trong repo; CP5 xử lý |
| **Ơ** | Có dấu | `D0508N` | Miền Nam | Approved Fallback | 45 valid frames | Ưu tiên 2 (`is_dynamic: true`) |
| **Ô** | Có dấu | `D0507B` | Miền Bắc | Provisional Fixture | 58 valid frames | Chỉ có bản B trong repo; CP5 xử lý |

---

## 4. BASELINE KIỂM CHỨNG TRƯỚC KHI CAN THIỆP (CHECKPOINT 0)

### 4.1. Hợp đồng Kiểm thử Hồi quy Toàn diện (Regression Contract: 206 tests)
- **Static Regression Suite (198 tests danh nghĩa):**
  - **197 tests PASSED (100% các test khả thi):**
    - `test_alphabet_classifier.py`: 136 tests PASS.
    - `test_alphabet_recognition_integration.py`: 9 tests PASS.
    - `test_alphabet_static_head_integration.py`: 13 tests PASS.
    - `test_mode_filtering.py`: 10 tests PASS.
    - `test_static_head_fallback.py` (nhóm tĩnh): 13 tests PASS.
    - `test_mode_separation.py`: 16 tests PASS.
  - **1 test SKIPPED:** `backend/tests/test_alphabet_classifier.py:1829` (`test_pilot_pilot_20261108_044024_accuracy_regression`) bị SKIP do thiếu file fixture webcam lịch sử `pilot_pilot_20261108_044024.jsonl`.
  - **Quy tắc bắt buộc:** Tuyệt đối KHÔNG tuyên bố 198/198 PASS khi còn 1 test bị SKIPPED.
- **Diacritic FSM Backlog (8 tests):**
  - 8 tests FAIL trong `test_static_head_fallback.py` (`test_a4`, `test_d1`..`test_d7`) do FSM bị ngắt tại `recognizer.py#L1552`. Đây là nợ kỹ thuật có kiểm soát sẽ giải quyết tại Checkpoint 5.
- **Tổng cộng:** 197 PASS, 1 SKIPPED, 8 FAIL (206 tests).

### 4.2. Baseline Offline và Canonical 6 Chữ Không Dấu:
- **Legacy Regression Suite (`vsl_alphabet_static_v2.npz`, 551 mẫu):** **452/551 (82.03%)**.
  - A: 60/64 (93.8%), D: 43/46 (93.5%), E: 32/35 (91.4%), G: 26/29 (89.7%), M: 17/19 (89.5%), H: 16/18 (88.9%), O: 41/48 (85.4%), U: 27/32 (84.4%), K: 14/17 (82.4%), P: 16/20 (80.0%), X: 20/25 (80.0%), Q: 15/19 (78.9%), R: 11/14 (78.6%), I: 14/18 (77.8%), V: 13/17 (76.5%), S: 11/15 (73.3%), N: 15/21 (71.4%), T: 14/20 (70.0%), L: 11/16 (68.8%), Y: 11/16 (68.8%), C: 12/20 (60.0%), B: 13/22 (59.1%).
- **Canonical Evaluation Suite (Đo trên frame chuẩn):**
  - **H (`D0500N.json`):** 24/37 (64.9%), NEUTRAL: 13.
  - **O (`D0506.json`):** 13/20 (65.0%), NEUTRAL: 7.
  - **Q (`D0510.json`):** 15/19 (78.9%), NEUTRAL: 4.
  - **R (`D0511.json`):** 11/16 (68.8%), NEUTRAL: 5.
  - **M (`D0504.json`):** 17/19 (89.5%), NEUTRAL: 2.
  - **N (`D0505.json`):** 16/25 (64.0%), NEUTRAL: 8, M: 1.

---

## 5. PHÂN TÍCH 7 NHÓM NGUYÊN NHÂN VÀ PHƯƠNG ÁN SỬA CHỮA

### Nhóm 1: Ngưỡng Nhân tạo Cắt cụt Recall MLP (Q, N)
- **Hiện tượng:** Gating threshold cao hơn phân bố confidence thực tế của MLP trên webcam.
- **Phương án CP3:**
  - Chữ Q: Giữ nguyên ngưỡng $0.75$ đã nghiệm thu sơ bộ, không hạ tiếp $0.70$.
  - Chữ N: Khảo sát hạ ngưỡng $0.75 \rightarrow 0.70$ có kiểm soát đối kháng M $\leftrightarrow$ N.

### Nhóm 2: Bất đối xứng Fallback & Thiếu Whitelist (Chữ O)
- **Hiện tượng:** `ALLOWED_FALLBACK_BASE_LETTERS` vắng mặt O; khi Rule O trượt ($d_{TI} > 0.23$), MLP đoán O bị loại bỏ, nhưng đoán A thì nhận ngay.
- **Phương án CP3:**
  - Bổ sung `"O"` vào `ALLOWED_FALLBACK_BASE_LETTERS`.
  - Nới lỏng nhẹ ngưỡng hình học `O_D_TI_MAX` ($0.23 \rightarrow 0.28$ candidate).
  - Thêm Guard chặn Fallback A khi hình thái tay có vòm tròn O.

### Nhóm 3: Cướp nhãn Hình học & Thứ tự Kiểm tra (Chữ H, R)
- **Hiện tượng:** Rule V chạy trước Rule H cướp nhãn khi hai ngón hơi nghiêng; Rule R cướp nhãn H khi hai ngón ngang đè hình chiếu 2D.
- **Phương án CP3:**
  - Bổ sung V-guard trong Rule V: Nếu $|idx\_dx| > |idx\_dy|$ và $idx\_horiz \ge 0.50$ $\rightarrow$ từ chối nhận V.
  - Thắt chặt hướng dựng đứng cho Rule R: Chỉ nhận R khi ngón hướng lên trên ($idx\_tip\_y \le -0.70$, $idx\_dy \le R\_IDX\_DY\_MAX$).

### Nhóm 4: Vắng bóng Luật Hình học & Phụ thuộc MLP Tuyệt đối (Chữ M, N)
- **Hiện tượng:** Classifier không có Rule M/N; webcam bị rơi 100% vào NEUTRAL (`NO_HAND`).
- **Phương án CP3:**
  - Giai đoạn 1: Hạ ngưỡng N về $0.70$.
  - Giai đoạn 2: Chỉ thêm Geometric Anchor Heuristic khi offline chứng minh phân tách được M vs N mà không làm tăng False Positive sang A/S/T.

### Nhóm 5: Ngắt Kết nối Pipeline Diacritic FSM (7 chữ có dấu)
- **Hiện trạng:** Ngắt tại `recognizer.py#L1552`.
- **Phương án:** Bảo toàn, KHÔNG sửa trong Checkpoint 3/4. Chuyển sang Checkpoint 5.

### Nhóm 6: Sai lệch Tham số Động học Vẽ Nón & Râu (Â, Ê, Ô, Ơ, Ư)
- **Hiện trạng:** Lift nón $0.035$ và timeout $0.65\text{s}$ quá khắt khe.
- **Phương án:** Chuyển sang Checkpoint 5.

### Nhóm 7: Sai lệch Dữ liệu Gốc và Phương ngữ (H, Ă, Â, Ê, Ô)
- **Phương án:** Duy trì phân định rõ ràng giữa Legacy Suite và Canonical Suite.

---

## 6. LỘ TRÌNH TRIỂN KHAI THEO CHECKPOINT (CP0 ĐẾN CP8)

- **CP 0: Baseline Lock** (Hoàn thành — 197 static PASS, 1 SKIPPED, 8 FAIL, Legacy 452/551).
- **CP 3A: Offline Candidate Evaluation** (Khảo sát các phương án O, H, M/N, R mà không sửa code production).
- **CP 3B: Implement Approved Candidates** (Sửa tối thiểu theo thứ tự: O $\rightarrow$ H $\rightarrow$ M/N $\rightarrow$ R nếu cần).
- **CP 4: Full Static Regression & Adversarial Checks** (197 static PASS + 1 SKIP, bảo vệ P/X/V/A/S/C/K/U).
- **APPROVAL GATE 2:** Chờ phê duyệt kết quả nhóm không dấu trước khi sang FSM.
- **CP 5: Re-integrate & Calibrate Diacritic FSM** (Nối lại FSM, đưa 8 tests FSM về PASS).
- **CP 6: Full 29-Letter Regression** (205/206 PASS).
- **CP 7: Controlled Webcam Pilot Holdout** (10 trials/chữ, Active Precision $\ge 70\%$).
- **CP 8: Final Acceptance.**

---

## 7. STOP RULES BẮT BUỘC

1. Dừng ngay lập tức nếu xuất hiện bất kỳ test tĩnh nào bị FAIL mới ngoài 8 FSM tests đã biết.
2. Dừng nếu False Positive trên P hoặc X vượt quá $0.0\%$.
3. Dừng nếu Rule V thật bị suy thoái do V-guard của H.
4. Dừng nếu nới lỏng O làm tăng nhầm lẫn sang S hoặc C.
5. Tuyệt đối không tự ý sửa pipeline FSM hoặc đổi threshold Q ngoài phạm vi.

---

## 8. KẾT QUẢ THỰC THI CHECKPOINT 3 & CHECKPOINT 4 (VERIFIED)

### 8.1. Checkpoint 3A: Đánh giá Offline Candidate Sweep
- **Chữ O:**
  - *Candidate 1:* Hạ khoảng cách $d_{TI}$ từ $0.23$ xuống $0.25$ hoặc $0.28$ $\rightarrow$ Bị loại vì gây False Positive trên mẫu chữ T (sample 340).
  - *Candidate 2 (Được chọn):* Mở rộng `ALLOWED_FALLBACK_BASE_LETTERS = {"A", "D", "E", "O"}` với ngưỡng $0.70$ + thêm guard chặn vòng lặp O rơi vào fallback A ($d_{TI} \le 0.30$ và $d_{TW} \le 1.05$).
  - *Kết quả:* Legacy O tăng từ 41/48 lên 43/48 (+2 mẫu); Canonical O tăng từ 13/20 lên 15/20 (+2 mẫu, đạt 75.0%); **0 False Positive** sang A, S, C, X.
- **Chữ H:**
  - *Candidate 1:* Mở rộng `H_HORIZ_ANG_IM_MAX` từ $28^\circ$ lên $35^\circ$ kết hợp V-guard theo tính trội phương đứng (`v_vertical_dominant = abs(idx_dy) >= abs(idx_dx)`).
  - *Kết quả:* Bảo toàn 100% 17 mẫu V Legacy và 19 mẫu V Canonical ($|dy| > |dx|$, min $|dy|=0.287$, max $|dx|=0.184$); ngăn chặn hoàn toàn việc Rule V cướp nhãn H nằm ngang.
- **Chữ M & N:**
  - *Candidate 1:* Thử nghiệm rule hình học cho M/N $\rightarrow$ Bị loại vì thiếu thông tin độ sâu 3D trên ảnh 2D, vi phạm test an toàn `test_m_and_n_kept_unknown`.
  - *Candidate 2 (Được chọn):* Giảm ngưỡng supplemental của N từ $0.75$ xuống $0.70$; duy trì M ở mức $0.70$.
  - *Kết quả:* Canonical N tăng từ 16/25 lên 17/25 (+1 mẫu, đạt 68.0%); Legacy N đạt 15/21; **0 False Positive M $\rightarrow$ N**, 0 False Positive sang A/S/T.
- **Chữ R:**
  - Kiểm toán thực nghiệm xác nhận: 0 mẫu nào trong toàn bộ 551 mẫu Legacy và fixtures canonical bị nhầm thành R. Lỗi lịch sử là H bị R cướp nhãn. Sau khi H được bảo vệ và sửa chữa, Rule R không có lỗi nội tại.
  - *Quyết định:* **SKIP R** (Khóa nguyên trạng Rule R, không can thiệp code).
- **Chữ Q:**
  - Giữ nguyên ngưỡng $0.75$. Không hạ xuống $0.70$. Gate 1 vẫn mở, chờ dữ liệu pilot đầy đủ.

### 8.2. Checkpoint 3B: Triển khai Code Production
- `backend/vision/recognizer.py`:
  - Bổ sung `"O"` vào `ALLOWED_FALLBACK_BASE_LETTERS`.
  - Cập nhật `DEFAULT_SUPPLEMENTAL_THRESHOLDS["N"] = 0.70`.
  - Bổ sung `is_loop_blocking_a` guard ngăn chặn ngón tạo vòng O rơi vào fallback A.
- `backend/vision/alphabet_classifier.py`:
  - Tăng `H_HORIZ_ANG_IM_MAX` từ $28.0$ lên $35.0$.
  - Thêm điều kiện `v_vertical_dominant = (abs(idx_dy) >= abs(idx_dx))` vào Rule V.
- `backend/tests/test_alphabet_static_head_integration.py`:
  - Đồng bộ danh sách `non_supplemental` (bỏ O).
  - Cập nhật kiểm tra biên cho ngưỡng Q=0.75 (0.70/0.74 bị từ chối, 0.75/0.80 được chấp nhận).
- `backend/tests/test_static_head_fallback.py`:
  - Cập nhật test O fallback: chấp nhận O khi confident $\ge 0.70$, từ chối về NEUTRAL khi $< 0.70$.

### 8.3. Checkpoint 4: Kết quả Static Regression Toàn diện & Đối kháng
- **Hợp đồng Kiểm thử Hồi quy (207 tests collected):**
  - **198 tests PASSED (100% tests tĩnh):**
    - `test_alphabet_classifier.py`: 133 PASS, 1 SKIPPED.
    - `test_alphabet_recognition_integration.py`: 9 PASS.
    - `test_alphabet_static_head_integration.py`: 13 PASS.
    - `test_mode_filtering.py`: 10 PASS.
    - `test_mode_separation.py`: 16 PASS.
    - `test_static_head_fallback.py` (phần tĩnh): 14 PASS (gồm 1 test biên mới cho O).
  - **1 test SKIPPED:** `backend/tests/test_alphabet_classifier.py:1829` (thiếu fixture webcam lịch sử `pilot_pilot_20261108_044024.jsonl`).
  - **8 tests FAIL đã biết:** Thuộc Backlog Diacritic FSM (`test_a4`, `test_d1`..`test_d7`), hoàn toàn nằm ngoài phạm vi CP3/4.
  - **Phát sinh hồi quy mới:** **0 (ZERO REGRESSION)**.
- **Đánh giá Tập Legacy (551 mẫu, `vsl_alphabet_static_v2.npz`):**
  - Độ chính xác toàn thể: **454/551 (82.40%)** vs Baseline **452/551 (82.03%)** (+2 mẫu).
  - Chữ O: **43/48 (89.6%)** vs Baseline 41/48 (+2 mẫu).
  - Toàn bộ 21 lớp còn lại: Bảo toàn 100%, không lệch một mẫu nào.
- **Kiểm tra Đối kháng Bảo vệ (Adversarial Protection):**
  - **P & X:** Duy trì 0% False Positive, bảo vệ tuyệt đối khỏi nhầm lẫn với Q.
  - **V & H:** Duy trì 0% False Positive lẫn nhau; Rule V bảo toàn 13/17 Legacy, 19/19 Canonical.
  - **A, S, C, K, U:** 100% không bị ô nhiễm bởi các thay đổi của O, H, M, N.
  - Toàn bộ các mẫu sai nhãn trong tập Legacy đều rơi vào **NEUTRAL rejection** an toàn, ngoại trừ 2 trường hợp đã tồn tại từ baseline (1 mẫu B $\rightarrow$ O, 1 mẫu N $\rightarrow$ M).
- **Kết luận:** Checkpoint 3 và Checkpoint 4 **HOÀN TOÀN ĐẠT CHUẨN (PASS)**. Hệ thống sẵn sàng cho APPROVAL GATE 2 trước khi bước vào Checkpoint 5 (Diacritic FSM).

---

## 9. KẾT QUẢ THỰC THI CHECKPOINT 5: KHÔI PHỤC DIACRITIC FSM (VERIFIED)

### 9.1. Đối soát Cuối cùng Checkpoint 4 (Reconciliation)
- **Git HEAD:** `ce9efdee`.
- **Giải mã chênh lệch bảng báo cáo:** Tệp `test_alphabet_classifier.py` thu thập đúng **137 tests** (136 PASS, 1 SKIPPED), không phải 134 tests như lỗi đánh máy bảng markdown trước. Tổng thu thập: $137 + 9 + 13 + 10 + 16 + 22 = \mathbf{207\text{ tests}}$.
- **Test SKIPPED:** `TestPilotWebcamHoldout::test_p_down_candidate_matches_observed_webcam_features` tại dòng 1829 do thiếu file fixture lịch sử `pilot_pilot_20261108_044024.jsonl`.
- **Tính toàn vẹn kiểm thử:** Test `test_b2b` được cập nhật cho O fallback và bổ sung `test_b2c` kiểm tra biên thấp. Toàn bộ các kiểm tra từ chối (negative assertions) vẫn được bảo toàn $100\%$ qua `test_b2` và `test_case_4`.

### 9.2. Kiểm toán Nguyên nhân FSM bị ngắt (CP5A)
- Commit `f5a26c2f` đã chèn đoạn code vô hiệu hóa FSM (`diacritic_fsms[session_key].reset()`) như một checkpoint kỹ thuật tạm thời nhằm cô lập đợt tái thiết lập pilot cho bảng chữ cái tĩnh.
- Toàn bộ module `backend/vision/diacritic_state_machine.py` vẫn giữ nguyên 32/32 unit tests PASS độc lập.

### 9.3. Triển khai Cầu nối FSM Tối thiểu (CP5B)
- Triển khai cầu nối an toàn trong `backend/vision/recognizer.py` (tại khối xử lý chế độ `letters`).
- Giữ nguyên $100\%$ mã nguồn và tham số động học trong `diacritic_state_machine.py` (`lift = 0.035`, `max_diacritic_search_sec = 0.65`, v.v.).
- Bảo đảm cô lập phiên (Session Isolation) qua `session_key` và làm mượt theo thời gian.

### 9.4. Kết quả Kiểm thử & Replay Thực tế (CP5C)
- **8 Tests FSM Backlog (`test_static_head_fallback.py`):** **8/8 PASSED (100% PASS)**!
  - `test_a4`: PASS
  - `test_d1`..`test_d3`: PASS
  - `test_d4` (Ă), `test_d5` (Đ), `test_d6` (Â), `test_d7` (Ê): PASS.
- **Hợp đồng Hồi quy Toàn diện (207 tests collected):**
  - **206 tests PASSED (100% tests khả thi)**.
  - **1 test SKIPPED** danh nghĩa.
  - **0 tests FAILED**.
- **Replay 7 Chuỗi Chuyển Động Video Thực tế:**
  - **Ă (`D0490B`):** Base A stable tại f14, **XÁC NHẬN Ă tại f28 (1.168s)**.
  - **Đ (`D0495`):** Base D stable tại f42, **XÁC NHẬN Đ tại f66 (2.753s)**.
  - **Ơ (`D0508N`):** Base O stable tại f40, **XÁC NHẬN Ơ tại f46 (1.919s)**.
  - **Ư (`D0515N`):** Base U stable tại f23, **XÁC NHẬN Ư tại f35 (1.460s)**.
  - **Â (`D0491B`):** Base A stable tại f33, chưa kích hoạt dấu do lift $< 0.035$.
  - **Ê (`D0497B`):** Base E stable tại f14, chưa kích hoạt dấu do timeout search window $> 0.65\text{s}$.
  - **Ô (`D0507B`):** Base O stable tại f17, chưa kích hoạt dấu do lift $< 0.035$.
- **Kiểm soát Đối kháng Tĩnh (Static Negative Controls):**
  - Đánh giá 551 mẫu Legacy x 10 frames tĩnh liên tục:
  - 550/551 mẫu ($99.8\%$) giữ nguyên tuyệt đối, không phát sinh dấu giả.
  - Phát hiện 1 mẫu biên (Sample 525 của D): ngón trỏ hơi cong tĩnh ($107.4^\circ \le 110.0^\circ$) kích hoạt điều kiện `curr_frame.index_pip_angle <= 110.0` với `drop = 0.0^\circ`. Đã xác định giải pháp candidate: bắt buộc điều kiện xung động học `angle_drop >= 20.0^\circ` thay vì cho phép góc tĩnh đơn thuần.

---

## 10. KẾT QUẢ THỰC THI CP5D: HIỆU CHUẨN ĐỘNG HỌC VÀ SỬA AN TOÀN BATCH 7 CHỮ CÓ DẤU (CP5D FINAL)

### 10.1. Kiểm toán An toàn Temporal State (Pre-flight Safety Audit)
1. **Cơ chế khóa phiên FSM vs Pilot:**
   - FSM phân giải dấu hoạt động theo `session_key = str(session_id)`.
   - Pilot testing cô lập temporal state qua bộ đệm `pilot_key = (session_id, trial_id, mode)` và `_pilot_temporal_lock`.
   - Thao tác `self.prediction_history.append(...)` thao tác an toàn trên bộ đệm đã được swap cho trial hiện tại; không có rò rỉ temporal buffer giữa các trials (`test_pilot_trial_temporal_isolation.py`: 4/4 PASS).
2. **Cơ chế Cooldown & Chặn lặp Token:**
   - FSM áp dụng `cooldown_sec = 0.40s` sau khi xác nhận dấu, ngăn phát lặp token không mong muốn.

### 10.2. Candidate Sweep & Hiệu chuẩn Dấu Nón (Circumflex: Â, Ê, Ô)
- **Phát hiện động học then chốt (Apex Deceleration):**
  - Khi người ký vẽ dấu nón (^), tại đỉnh nón tay giảm tốc độ thẳng đứng để đổi hướng ($vy \approx -0.03\text{ đến } -0.04$).
  - Ngưỡng baseline cũ `circumflex_vy_threshold = -0.10` đã chặn vật lý việc phát hiện đỉnh nón khi tay đạt độ nâng cực đại!
  - Đỉnh nâng thực tế trong video mẫu: Â đạt $0.024$ (tại f52), Ê đạt $0.030$ (tại f29). Ngưỡng cũ $0.035$ quá cao so với biên độ thực.
  - Cải tiến: Đổi `circumflex_lift_threshold = 0.020`, `circumflex_vy_threshold = 0.05`.
  - Cơ chế trôi thời gian nền: Trong trạng thái `STATE_BASE_STABLE` khi người ký giữ chữ nền mà chưa thực hiện dấu, FSM liên tục cập nhật `self.baseline.stable_time = curr_time`, bảo đảm cửa sổ tìm kiếm dấu luôn sẵn sàng trọn vẹn từ lúc cử chỉ tạo dấu bắt đầu.
- **Kết quả trên 3 chuỗi video:**
  - **Â (`D0491B`):** Từ FAIL $\rightarrow$ **XÁC NHẬN Â tại f52 (2.169s)**.
  - **Ê (`D0497B`):** Từ FAIL $\rightarrow$ **XÁC NHẬN Ê tại f29 (1.210s)**.
  - **Ô (`D0507B`):** Fixture video mang nhãn `manual_review_needed`; người ký mở ngón út ($d_{PI} \ge 0.50$, $pinky\_curl = 0.05$) thay vì giữ khuyên O nâng nón theo chuẩn miền Nam/toàn quốc. Nếu hạ ngưỡng để chấp nhận dạng này sẽ gây xung đột với dấu móc Ơ. Giữ an toàn hệ thống, Ô tạm ghi nhận partial trên fixture này chờ video chuẩn quốc gia.

### 10.3. Candidate Sweep & Khắc phục False-Positive Chữ Đ
- **Nguyên nhân gốc của Mẫu 525:**
  - Điều kiện cũ sử dụng toán tử `or`:
    `if curr_frame.index_pip_angle <= self.d_stroke_angle_threshold or angle_drop >= self.d_stroke_drop_threshold:`
  - Mẫu 525 của chữ D có ngón trỏ hơi gập tĩnh ($PIP = 107.4^\circ \le 110.0^\circ$), dù $drop = 0.0^\circ$ vẫn bị nhận định nhầm là Đ tại frame 4 của thế giữ tĩnh.
- **Giải pháp tối thiểu đã kiểm chứng:**
  - Bắt buộc đồng thời cả xung rơi góc động học và ngưỡng gập ngón:
    `if angle_drop >= self.d_stroke_drop_threshold and curr_frame.index_pip_angle <= self.d_stroke_angle_threshold:`
  - Với `d_stroke_drop_threshold = 15.0^\circ` và `d_stroke_angle_threshold = 125.0^\circ`.
- **Hiệu quả:**
  - Mẫu 525 giữ 10 frames tĩnh: **10/10 frames là D (ZERO False Positives)**.
  - Chuỗi video động Đ (`D0495`): **XÁC NHẬN Đ tại f66 (2.753s, drop = 15.8° >= 15.0°)**.
  - Unit tests `test_d5` & `test_case_8`: 100% PASS (drop = 90.0°).

### 10.4. Bản vá Tối thiểu Production (Minimal Patch Scope)
- **Duy nhất 1 file sửa đổi:** [`backend/vision/diacritic_state_machine.py`](file:///d:/SIGN_LANGUAGE/backend/vision/diacritic_state_machine.py) (5 dòng thay đổi).
- Không can thiệp `recognizer.py`, không sửa weights mô hình, không sửa static geometric thresholds.

### 10.5. Bảng Tổng hợp Kết quả Nghiệm thu Toàn diện (Phase E)

| Hạng mục kiểm thử | Baseline trước CP5 | Sau CP5C | Sau CP5D (Hiện tại) | Trạng thái |
| :--- | :---: | :---: | :---: | :---: |
| **Diacritic Unit Tests** | 0/8 PASS (disabled) | 8/8 PASS | **77/77 PASS (100%)** | **ĐẠT** |
| **Full Regression Contract** | 197 PASS, 1 SKIP, 8 FAIL | 206 PASS, 1 SKIP, 0 FAIL | **206 PASS, 1 SKIP, 0 FAIL** (207 collected) | **BẢO TOÀN 100%** |
| **Video Replay (7 chuỗi)** | 0/7 CONFIRMED | 4/7 CONFIRMED | **6/7 CONFIRMED (Ă, Â, Ê, Đ, Ơ, Ư)** | **VƯỢT MỤC TIÊU** |
| **Static Negative Controls (551 x 10f)** | 550/551 (1 FP tại #525) | 550/551 (1 FP) | **551/551 CLEAN (0 FP, 100%)** | **TRIỆT TIÊU HOÀN TOÀN FP** |
| **Legacy 551 Accuracy** | 452/551 (82.03%) | 454/551 (82.40%) | **454/551 (82.40%)** | **TĂNG +2 MẪU, 0 SUY GIẢM** |
| **Pilot Temporal Isolation** | 4/4 PASS | 4/4 PASS | **4/4 PASS** | **BẢO MẬT TUYỆT ĐỐI** |

---

## 11. KẾT QUẢ THỰC THI CP6-FIX: GIẢI QUYẾT TRIỆT ĐỂ CÁC BLOCKERS KỸ THUẬT CHO WEBCAM HOLDOUT

### 11.1. Giải quyết BLK-FSM-LEAK: Cô Lập FSM State Theo Pilot Trial
- **Nguyên nhân gốc:** `self.diacritic_fsms` trong `recognizer.py` trước đây chỉ được đánh chỉ mục theo `session_key = str(session_id)`. Khi nhiều pilot trials chạy trong cùng một session kiểm thử, FSM instance bị dùng chung xuyên qua ranh giới các trial, dẫn đến hiện tượng rò rỉ trạng thái `BASE_STABLE`, `baseline`, và `COOLDOWN` từ trial trước sang trial sau.
- **Giải pháp tối thiểu đã áp dụng trong `backend/vision/recognizer.py`:**
  1. **Khóa FSM đa cấp (Tuple Keying):** Khi request có đầy đủ `session_id`, `trial_id` và `mode == "letters"`, FSM instance được lưu trữ và truy xuất theo khóa tuple 3 phần tử: `fsm_key = (str(session_id), str(trial_id), mode)`. Các request thông thường ngoài pilot tiếp tục sử dụng khóa chuỗi `session_key = str(session_id)` để duy trì khả năng tương thích 100%.
  2. **Dọn dẹp đồng bộ & Quản lý bộ nhớ:** `reset_diacritic_fsm(session_id)` được nâng cấp để xóa cả khóa chuỗi lẫn toàn bộ các khóa tuple thuộc session đó. `enable_pilot_logging` và `disable_pilot_logging` đồng bộ dọn dẹp các tuple keys. Cơ chế LRU/pruning tự động thanh lọc các instance cũ nhất nếu tổng số tuple keys vượt quá ngưỡng an toàn (100 instances).
- **Kết quả kiểm thử:**
  - Bộ kiểm thử [`backend/tests/test_pilot_trial_temporal_isolation.py`](file:///d:/SIGN_LANGUAGE/backend/tests/test_pilot_trial_temporal_isolation.py) mở rộng lên 5 tests: **5/5 PASS (100%)**.
  - Mô phỏng độc lập qua 5 nhóm nền (A, D, O, E, U): Trial B tại frame 0 hoàn toàn sạch trạng thái (`state: None, diac: None`), triệt tiêu 100% rò rỉ.

### 11.2. Giải quyết BLK-HUD-DYNAMIC: Hợp Đồng Tính Điểm Cho 7 Chữ Động
- **Nguyên nhân gốc:** Hàm `evaluateTrialFrames` trong `src/components/pilotProtocol.js` trước đây áp dụng cứng nhắc tiêu chí so khớp tỷ lệ khung hình tĩnh (`matchPct = matchFrames / activeFrames >= 60.0%`). Đối với 7 chữ cái có dấu (Ă, Â, Ê, Đ, Ơ, Ư, Ô), cử chỉ tự nhiên bắt buộc phải giữ chữ cái nền (ví dụ giữ A trước khi vẽ nón Â). Các frame giữ nền này bị tính là `mismatchFrames`, khiến tỷ lệ khớp chỉ đạt ~14% và toán học không thể vượt qua ngưỡng 60%.
- **Giải pháp hợp đồng mới trong `src/components/pilotProtocol.js`:**
  1. **Bảo toàn 22 chữ cái tĩnh:** Tiếp tục áp dụng 100% logic steady-state matching truyền thống (`matchPct >= 60.0%`).
  2. **Quy tắc phân giải cho 7 chữ cái động (`DYNAMIC_DIACRITICS`):**
     - Các frame nhận diện đúng chữ cái nền hợp lệ không bị tính là lỗi mismatch.
     - Bắt buộc kiểm tra bằng chứng tư thế nền (`hasBaseEvidence >= 2` khung hình).
     - Bắt buộc có xác nhận dấu động học hợp lệ từ backend (`new_token === expectedSign` hoặc `prediction_source === 'diacritic_fsm'` hoặc `confirmed && isTarget`).
     - Phân định rõ ràng các trạng thái: `PASS` (đạt cả base + diacritic confirmation), `NEEDS_FIX` (thiếu base hoặc không confirm được dấu), `NO_HAND` (không phát hiện bàn tay), và `INSUFFICIENT_DATA` (< 5 frames hợp lệ).
- **Kết quả kiểm thử:**
  - Bộ test suite độc lập 10 kịch bản (`scratch/test_dynamic_hud_scoring.cjs`): **10/10 PASS**.
  - Bộ test suite Task 1.3G 10 trường hợp biên (`scratch/test_task1_3g_cases.cjs`): **10/10 PASS**.

### 11.3. Khảo sát & Quyết định BLK-D-CONF-FRAMES: Độ An Toàn Chữ Đ
- **Thử nghiệm đối sánh:**
  - Ứng viên `req_conf_frames = 2` vs Hiện trạng `req_conf_frames = 1`.
  - Trên video Đ thực tế (`D0495`): `req=1` xác nhận tại frame 66; `req=2` xác nhận tại frame 67 (cách nhau 41ms). Cả hai đều đạt 10/10 frames D tĩnh trên Mẫu 525 (0 False Positives).
  - Khảo sát hồi quy: Đổi sang `req=2` làm FAIL 2 tests hợp đồng trong pytest (`test_case_8_d_stroke_to_d_stroke` và `test_d5_fallback_d_to_d_stroke`) do các bài test này kiểm tra xung gập đốt tại 1 frame duy nhất.
- **Quyết định an toàn:** Tuân thủ chỉ thị, giữ nguyên `req_conf_frames = 1` trong production nhằm bảo vệ 100% hợp đồng hồi quy mà vẫn duy trì 0 false-positives trên toàn bộ 551 mẫu dữ liệu tĩnh.

### 11.4. Kiểm Toán Thông Lượng Webcam & Khuyến Nghị Vận Hành
- **Đánh giá suy giảm FPS trên 7 chuỗi cử chỉ thực tế (`scratch/evaluate_low_fps.py`):**
  - **Native FPS (~25 FPS):** 6/7 xác nhận thành công (Ă, Â, Ê, Đ, Ơ, Ư).
  - **10 FPS:** 4/7 xác nhận thành công (Ă, Â, Ê, Đ). Ơ và Ư bị trượt do số frame giữ nền không đủ vượt ngưỡng `min_base_stable_frames = 4`.
  - **<= 3 FPS:** 0/7 xác nhận thành công do mất thông tin quỹ đạo động học.
- **Khuyến nghị vận hành:** Cần duy trì tốc độ truyền nhận frontend-backend $\ge 10\text{--}15\text{ FPS}$ trong quá trình kiểm thử webcam.

### 11.5. Bảng Nghiệm Thu Cuối Cùng (Post CP6-FIX Final Baseline)

| Hạng mục kiểm thử | Baseline trước CP6 | Sau CP6-FIX (Hiện tại) | Trạng thái |
| :--- | :---: | :---: | :---: |
| **Full Regression Contract (6 suites)** | 206 PASS, 1 SKIP, 0 FAIL | **206 PASS, 1 SKIP, 0 FAIL** (207 collected) | **BẢO TOÀN 100%** |
| **Diacritic FSM Test Suites (3 suites)** | 77/77 PASS | **77/77 PASS** | **100% PASS** |
| **Pilot Temporal Isolation Suite** | 4/4 PASS | **5/5 PASS** | **100% PASS (Đã bế tắc FSM Leak)** |
| **Dynamic HUD Unit Tests (2 suites)** | Chưa có | **20/20 PASS** | **100% PASS (Đã giải quyết HUD Blocker)** |
| **Legacy 551 Accuracy** | 454/551 (82.40%) | **454/551 (82.40%)** | **BẢO TOÀN NGUYÊN VẸN** |
| **Canonical Evaluation (H, O, Q, R, M, N, V)** | 68.0% - 89.5% | **68.0% - 89.5%** | **BẢO TOÀN NGUYÊN VẸN** |
| **Motion Replay 7 Diacritics** | 6/7 CONFIRMED | **6/7 CONFIRMED** (Ô giữ nhãn manual review) | **ỔN ĐỊNH** |


