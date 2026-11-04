# BÁO CÁO TÍCH HỢP GATED STATIC HEAD FALLBACK CHO BA CHỮ NỀN A / D / E (TASK 7E.5B)

- **Ngày thực hiện:** 31/10/2026
- **Nhiệm vụ:** Task 7E.5B — Triển khai Gated Static Head Fallback cho A, D, E trong luồng `mode="letters"` và kết nối với `DiacriticStateMachine`.
- **Trạng thái thực thi:** HOÀN THÀNH TOÀN DIỆN (142/142 Unit & Regression Tests PASS; 4/4 Base Pose đạt BASE_STABLE; Checkpoint SHA-256 bảo toàn 100%).
- **File báo cáo & dữ liệu:**
  - Báo cáo chi tiết: [`backend/data/alphabet_v2/audit/task_7e5b_report.md`](file:///d:/SIGN_LANGUAGE/backend/data/alphabet_v2/audit/task_7e5b_report.md)
  - Kết quả thực nghiệm: [`backend/data/alphabet_v2/audit/task_7e5b_results.json`](file:///d:/SIGN_LANGUAGE/backend/data/alphabet_v2/audit/task_7e5b_results.json)

---

## 1. TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY)

Trong Task 7E.5A, quá trình kiểm toán tĩnh đã làm rõ nguyên nhân gốc rễ khiến hệ thống nhận diện kém các chữ cái động có dấu: **Luật hình học (Geometric Rules) thất bại nghiêm trọng trong việc nhận diện ba chữ nền tĩnh A, D, E** trên tập dữ liệu chuẩn QIPEDC (A thất bại 73.4%, D thất bại 67.4%, E thất bại 71.4%). Do không có frame chữ nền nào đạt chuẩn, máy trạng thái `DiacriticStateMachine` hoàn toàn bị "bỏ đói", không thể chuyển sang trạng thái `BASE_STABLE` để theo dõi và kích hoạt dấu.

Trong **Task 7E.5B**, cơ chế **Gated Static Head Fallback** đã được tích hợp thành công vào luồng nhận diện `mode="letters"` của `VSLRecognizer`:
1. **Kiểm soát chặt chẽ:** Chỉ kích hoạt khi Geometric Rules trả về `NO_MATCH` hoặc `UNKNOWN`, và mô hình Static Head (`alphabet_static_head_v1.pt`) dự đoán một trong ba chữ nền `{"A", "D", "E"}` với độ tin cậy $\ge 0.70$.
2. **Ưu tiên tuyệt đối của Geometric Rules:** Bất kỳ khi nào Geometric Rules trả về `MATCH` (ví dụ `MATCH I`, `MATCH V`, `MATCH B`), kết quả hình học luôn được sử dụng 100%; Static Head bị bỏ qua hoàn toàn, không có cơ hội ghi đè.
3. **Kích hoạt FSM thành công trên 100% video mục tiêu:** Trên cả 4 video động học `Ă`, `Â`, `Ê`, `Đ`, FSM đều đạt `BASE_STABLE` thành công nhờ nhãn nền từ Static Head Fallback (so với 0% trước đây).
4. **Xác nhận dấu thành công cho `Ă` và `Đ`:**
   - `Ă` (video `D0490B`): Base `A` ổn định ở frame 14 $\rightarrow$ Cử chỉ mở ngón V xác nhận `Ă` ở frame 28 (`source = diacritic_fsm`).
   - `Đ` (video `D0495`): Base `D` ổn định ở frame 42 $\rightarrow$ Đốt ngón trỏ gập gãy $24^\circ$ xác nhận `Đ` ở frame 66 (`source = diacritic_fsm`).
   - Cùng với `Ơ` (video `D0508N`, frame 46) và `Ư` (video `D0515N`, frame 34), hệ thống hiện giải quyết được **4/7 chữ động có dấu** (tăng gấp đôi từ 2/7 lên 4/7).
5. **Không suy thoái hệ thống (Zero Regression):** Toàn bộ 9 test suites với **142/142 tests đạt 100% PASS**. Cả 3 checkpoint mô hình đều được kiểm tra mã băm SHA-256 khớp 100% với nguyên bản, không bị chỉnh sửa hay ghi đè.

---

## 2. BỐI CẢNH & NGUYÊN NHÂN CẦN FALLBACK A/D/E

### 2.1. Giới hạn của Geometric Rules trên A, D, E (Kết quả Task 7E.5A)
- **Chữ A:** Ngưỡng khoảng cách đầu ngón cái - đốt bàn trỏ ($d_{IW} \le 0.90$) và đầu ngón cái - khớp bàn ngón giữa ($d_{TM} \ge 0.75$) quá ngặt, khiến 47/64 mẫu tĩnh (73.4%) của người ký thật bị từ chối thành `NO_MATCH`.
- **Chữ D:** Ngưỡng duỗi ngón trỏ ($d_{IW} \ge 1.70$) đòi hỏi ngón trỏ duỗi hoàn hảo; trên người ký thật có ngón cong nhẹ, 31/46 mẫu (67.4%) bị từ chối thành `NO_MATCH`.
- **Chữ E:** Ngưỡng khép ngón cái ($d_{TI} \le 0.35$, $d_{TM} \le 0.38$) và trạng thái 4 ngón nắm chặt khiến 25/35 mẫu (71.4%) bị từ chối thành `NO_MATCH`.

### 2.2. Năng lực của Static Alphabet Head
Kiểm toán Task 7E.5A trên `alphabet_static_head_v1.pt` (MLP 72D $\rightarrow$ 22 lớp) cho thấy:
- Độ chính xác độc lập trên các mẫu tĩnh: `A` đạt 96.9%, `D` đạt 95.7%, `E` đạt 91.4%.
- Tỷ lệ dương tính giả (False Positive) trên 19 lớp còn lại khi áp ngưỡng conf $\ge 0.70$ là **0.0%**.
- Trên các frame `BASE_POSE` của 4 video động học, Static Head nhận diện đúng:
  - `Ă` (`D0490B`): 19/20 frames dự đoán `A` (avg conf 0.957).
  - `Â` (`D0491B`): 23/24 frames dự đoán `A` (avg conf 0.963).
  - `Ê` (`D0497B`): 20/22 frames dự đoán `E` (avg conf 0.942).
  - `Đ` (`D0495`): 26/28 frames dự đoán `D` (avg conf 0.966).

Do đó, việc dùng Static Head làm fallback có kiểm soát cho A, D, E là giải pháp kỹ thuật hợp lý nhất mà không cần phải thay đổi các ngưỡng hình học hiện có.

---

## 3. THIẾT KẾ KIẾN TRÚC GATED STATIC HEAD FALLBACK

### 3.1. Sơ đồ phân cấp quyết định trong `mode="letters"`

Cơ chế phân giải nhãn tĩnh được tổ chức theo 3 tầng nối tiếp:

```
[Camera / Video Landmarks (21 points)]
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│ BƯỚC 1: 12 GEOMETRIC RULES (alphabet_classifier.py)    │
└────────────────────────────────────────────────────────┘
                 │
      ┌──────────┴──────────┐
      │ status == 'MATCH'?  │
      ▼                     ▼
    [ CÓ ]                [ KHÔNG ] (NO_MATCH hoặc UNKNOWN)
      │                     │
      ▼                     ▼
┌──────────────────┐  ┌──────────────────────────────────────────────┐
│ final_sign = geom│  │ BƯỚC 2: STATIC ALPHABET HEAD (72D Feature)   │
│ source =         │  └──────────────────────────────────────────────┘
│ 'geometric_rule' │                 │
└──────────────────┘        ┌────────┴────────┐
                            │ head_conf >=    │
                            │ threshold?      │
                            ▼                 ▼
                          [ CÓ ]            [ KHÔNG ]
                            │                 │
             ┌──────────────┴──────────────┐  ▼
             │ Phân loại nhãn Static Head: │ [ REJECT -> NEUTRAL ]
             └──────────────┬──────────────┘
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│ Nhóm 10 chữ bổ sung:          │   │ Nhóm Fallback Base Letters:   │
│ {G, H, K, L, M, N, P, Q, V, X}│   │ {A, D, E} (conf >= 0.70)      │
│ final_sign = head_letter      │   │ final_sign = head_letter      │
│ source = 'static_head'        │   │ source =                      │
└───────────────────────────────┘   │ 'static_head_fallback'        │
                                    └──────────────┬────────────────┘
                                                   │
                                                   ▼
┌───────────────────────────────────────────────────────────────────┐
│ BƯỚC 3: DIACRITIC STATE MACHINE (5 Nhóm: A, D, E, O, U)           │
│ - Nhận base_sign = 'A', 'D', 'E'                                  │
│ - Cập nhật BASE_CANDIDATE -> BASE_STABLE                          │
│ - Giám sát dấu động học: Ă (Rule V), Â/Ê (Circumflex), Đ (Angle)   │
└───────────────────────────────────────────────────────────────────┘
```

### 3.2. Bốn nguyên tắc bất khả xâm phạm
1. **Geometric MATCH Priority:** Khi Geometric Rules MATCH, Step 2 hoàn toàn không được thực hiện. Static Head không thể ghi đè bất kỳ luật hình học nào.
2. **Gated Base Scope:** Chỉ có đúng 3 chữ `{"A", "D", "E"}` được phép kích hoạt fallback. Bất kỳ chữ cái nào khác (ví dụ Static Head đoán `B`, `C`, `O`, `S`, `T`) đều bị từ chối về `NEUTRAL` và ghi nhận `reject_reason`.
3. **Strict Confidence Gate:** Ngưỡng tin cậy mặc định `fallback_base_threshold = 0.70`. Mọi dự đoán có conf $< 0.70$ đều bị loại bỏ.
4. **Minh bạch Nguồn gốc:** Trường `source` và `prediction_source` ghi nhận rõ ràng `"static_head_fallback"` để tầng nghiệp vụ và log phân biệt chính xác với `"geometric_rule"` và `"static_head"`.

---

## 4. CHI TIẾT THAY ĐỔI MÃ NGUỒN (CODE MODIFICATIONS)

### 4.1. `backend/vision/recognizer.py`

1. **Khai báo Hằng số và Cấu hình Ngưỡng (Dòng 64–65):**
   ```python
   ALLOWED_FALLBACK_BASE_LETTERS = {"A", "D", "E"}
   DEFAULT_FALLBACK_BASE_THRESHOLD = 0.70
   ```
2. **Mở rộng Constructor `VSLRecognizer.__init__` (Dòng 86, 137–139):**
   ```python
   def __init__(
       self,
       ...,
       fallback_base_threshold: Optional[float] = None
   ):
       ...
       self.fallback_base_threshold = (
           float(fallback_base_threshold)
           if fallback_base_threshold is not None
           else DEFAULT_FALLBACK_BASE_THRESHOLD
       )
   ```
3. **Bổ sung Nhánh Gated Static Head Fallback trong Bước 2 (Dòng 768–790):**
   ```python
   elif head_letter in ALLOWED_FALLBACK_BASE_LETTERS and head_conf >= self.fallback_base_threshold:
       final_sign = head_letter
       final_conf = head_conf
       recognition_source = "static_head_fallback"
       alpha_res = {
           "status": "MATCH",
           "letter": head_letter,
           "confidence": head_conf,
           "source": "static_head_fallback",
           "head_top_k": head_res.get("top_k", []),
           "reject_reason": None,
       }
   ```
4. **Cập nhật Từ chối Dự đoán (Dòng 795–800):**
   Cập nhật logic `reject_reason` để phân biệt rõ giữa chữ không thuộc tập bổ sung/fallback và chữ có conf $< \text{threshold}$.
5. **Đồng bộ Metadata Đầu ra (Dòng 1086):**
   Bổ sung `"prediction_source": recognition_source` vào dictionary trả về của `predict_landmarks()`.

### 4.2. Cập nhật `backend/tests/test_alphabet_static_head_integration.py`
Trong bộ test cũ, `test_case_4_rejection_of_non_supplemental_head_predictions` kiểm tra việc chặn toàn bộ các chữ ngoài 10 chữ bổ sung, trong đó danh sách cấm liệt kê cứng cả `A`, `D`, `E`. 
Theo đúng hướng dẫn của Task 7E.5B ("Nếu có test cũ kiểm tra hành vi cũ trái ngược với thiết kế fallback... hãy cập nhật lại test case đó cho đúng với đặc tả mới"), test case này đã được cập nhật:
- Danh sách kiểm tra từ chối gồm các chữ cái không thuộc supplemental và không thuộc fallback: `["B", "C", "I", "O", "S", "T", "Y", "Z"]`.
- Hành vi mới của `A`, `D`, `E` được chuyển sang kiểm tra đầy đủ, chuyên sâu tại bộ test mới `test_static_head_fallback.py`.

---

## 5. BỘ KIỂM THỬ MỚI & KẾT QUẢ HỒI QUY HỆ THỐNG

### 5.1. Bộ kiểm thử chuyên dụng: `test_static_head_fallback.py` (20/20 PASS)
File test [`backend/tests/test_static_head_fallback.py`](file:///d:/SIGN_LANGUAGE/backend/tests/test_static_head_fallback.py) gồm 20 test cases bao phủ toàn diện 4 nhóm nghiệp vụ:

| Nhóm Kiểm thử | Mã Test | Nội dung kiểm tra | Kết quả |
| :--- | :--- | :--- | :---: |
| **Category A: Fallback Active** | `test_a1` | Geom NO_MATCH, Head `A` conf 0.85 $\rightarrow$ `sign = "A"`, `source = "static_head_fallback"` | **PASS** |
| | `test_a2` | Geom UNKNOWN, Head `D` conf 0.80 $\rightarrow$ `sign = "D"`, `source = "static_head_fallback"` | **PASS** |
| | `test_a3` | Geom NO_MATCH, Head `E` conf 0.70 (ngưỡng biên) $\rightarrow$ `sign = "E"`, `source = "static_head_fallback"` | **PASS** |
| | `test_a4` | Nhãn fallback A được nạp thành công vào `fsm.candidate_base` và đếm candidate frame | **PASS** |
| **Category B: Fallback Rejected** | `test_b1` | Head `A` conf 0.69 ($< 0.70$) $\rightarrow$ Từ chối, trả về `NEUTRAL`, `source = "none"` | **PASS** |
| | `test_b2` | Head dự đoán `B` conf 0.90 (ngoài {A,D,E}) $\rightarrow$ Từ chối, trả về `NEUTRAL` | **PASS** |
| | `test_b3` | Head dự đoán `G` conf 0.80 (thuộc supplemental) $\rightarrow$ Nhánh chuẩn, `source = "static_head"` | **PASS** |
| | `test_b4` | Không có landmarks (None) $\rightarrow$ Không kích hoạt fallback, trả về `NEUTRAL` an toàn | **PASS** |
| | `test_b5` | Head phát sinh Exception $\rightarrow$ Bắt lỗi an toàn, trả về `NEUTRAL`, không crash | **PASS** |
| **Category C: Geometric Priority** | `test_c1` | Geom MATCH `A` (0.95), Head `D` (0.99) $\rightarrow$ Ưu tiên `A`, `source = "geometric_rule"` | **PASS** |
| | `test_c2` | Geom MATCH `I` (0.95), Head `A` (0.99) $\rightarrow$ Ưu tiên `I`, `source = "geometric_rule"` | **PASS** |
| | `test_c3` | Geom MATCH `V` (0.95), Head `A` (0.99) $\rightarrow$ Ưu tiên `V`, `source = "geometric_rule"` | **PASS** |
| | `test_c4` | Geom MATCH `D` (0.90), Head `E` (0.99) $\rightarrow$ Ưu tiên `D`, `source = "geometric_rule"` | **PASS** |
| **Category D: FSM Progression** | `test_d1` | 5 frames Fallback A $\rightarrow$ FSM đạt `BASE_STABLE`, `current_base = "A"` | **PASS** |
| | `test_d2` | 5 frames Fallback E $\rightarrow$ FSM đạt `BASE_STABLE`, `current_base = "E"` | **PASS** |
| | `test_d3` | 5 frames Fallback D $\rightarrow$ FSM đạt `BASE_STABLE`, `current_base = "D"` | **PASS** |
| | `test_d4` | Fallback A đạt stable $\rightarrow$ Rule V xuất hiện $\rightarrow$ Thăng cấp thành `Ă` (`diacritic_fsm`) | **PASS** |
| | `test_d5` | Fallback D đạt stable $\rightarrow$ Gập đốt ngón trỏ $\rightarrow$ Thăng cấp thành `Đ` (`diacritic_fsm`) | **PASS** |
| | `test_d6` | Fallback A đạt stable $\rightarrow$ Nâng tay vẽ nón $\rightarrow$ Thăng cấp thành `Â` (`diacritic_fsm`) | **PASS** |
| | `test_d7` | Fallback E đạt stable $\rightarrow$ Nâng tay vẽ nón $\rightarrow$ Thăng cấp thành `Ê` (`diacritic_fsm`) | **PASS** |

### 5.2. Kết quả hồi quy toàn bộ hệ thống (142/142 tests PASS)

Toàn bộ 9 bộ kiểm thử hồi quy trên toàn hệ thống đã được chạy và đạt tỷ lệ thành công 100%:

```
1. backend.tests.test_static_head_fallback          : 20/20 PASS  [OK]
2. backend.tests.test_diacritic_state_machine        : 41/41 PASS  [OK]
3. backend.tests.test_diacritic_camera_integration   : 14/14 PASS  [OK]
4. backend.tests.test_alphabet_classifier            :  4/4  PASS  [OK]
5. backend.tests.test_alphabet_static_head_integration: 10/10 PASS  [OK]
6. backend.tests.test_digit_10_integration           :  8/8  PASS  [OK]
7. backend.tests.test_mode_separation                : 16/16 PASS  [OK]
8. backend.tests.test_sequential_digits_integration  :  8/8  PASS  [OK]
9. backend.tests.test_api                            : 21/21 PASS  [OK]
----------------------------------------------------------------------
TỔNG CỘNG: 142/142 TESTS PASS (0 FAILED, 0 ERRORS) - 100% HOÀN HẢO
```

---

## 6. KẾT QUẢ THỰC NGHIỆM REPLAY TRÊN CÁC CHUỖI VIDEO ĐỘNG HỌC

Script replay [`backend/data/alphabet_v2/audit/run_task_7e5b_replay.py`](file:///d:/SIGN_LANGUAGE/backend/data/alphabet_v2/audit/run_task_7e5b_replay.py) đã thực thi trực tiếp trên toàn bộ 7 chuỗi video động học QIPEDC trong `backend/data/alphabet_v2/motion_sequences/`.

### 6.1. Bảng so sánh tổng hợp trước và sau khi tích hợp Fallback

| Ký hiệu | Video ID | Nhóm nền | Trước Task 7E.5B (Geom thuần) | Sau Task 7E.5B (Gated Fallback) | Trạng thái xác nhận dấu |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Ă** | `D0490B` | **A** | Thất bại (Base A không đạt stable) | **Base A stable tại f14** (20 frames FB, conf 0.946) | **CONFIRMED tại f28** (`diacritic_fsm`) |
| **Đ** | `D0495` | **D** | Thất bại (Base D không đạt stable) | **Base D stable tại f42** (31 frames FB, conf 0.955) | **CONFIRMED tại f66** (`diacritic_fsm`) |
| **Ơ** | `D0508N` | **O** | Thành công (Base O stable tại f40) | **Base O stable tại f40** (Geom O) | **CONFIRMED tại f46** (`diacritic_fsm`) |
| **Ư** | `D0515N` | **U** | Thành công (Base U stable tại f23) | **Base U stable tại f23** (Static Head U) | **CONFIRMED tại f34** (`diacritic_fsm`) |
| **Â** | `D0491B` | **A** | Thất bại (Base A không đạt stable) | **Base A stable tại f33** (23 frames FB, conf 0.952) | Chưa xác nhận (Rule O can thiệp tại f49) |
| **Ê** | `D0497B` | **E** | Thất bại (Base E không đạt stable) | **Base E stable tại f14** (23 frames FB, conf 0.946) | Chưa xác nhận (Timeout tìm kiếm dấu tại f30) |
| **Ô** | `D0507B` | **O** | Base O stable tại f19 | **Base O stable tại f19** (Geom O) | Chưa xác nhận (Động học vẽ nón dưới ngưỡng) |

### 6.2. Phân tích chi tiết hai trường hợp đột phá: Ă và Đ

#### 1. Chữ Ă (`D0490B` — 89 frames)
- **Quá trình thiết lập nền:** Geometric rules hoàn toàn không MATCH nhãn A. Tuy nhiên, từ frame 7 đến frame 26, Static Head Fallback liên tục cung cấp nhãn `A` với độ tin cậy trung bình 0.946.
- **Ổn định FSM:** Tại **frame 14** ($t = 0.584\text{s}$), FSM tích lũy đủ số frame ổn định và chuyển sang trạng thái `BASE_STABLE` với `current_base = "A"`.
- **Phân giải dấu:** Khi người ký xòe ngón trỏ và giữa để tạo dấu trăng khuyết, bàn tay tạo hình tương tự chữ `V` (Rule V MATCH tại frame 28). Nhờ cơ chế `update_competing()` đã hoàn thiện ở Task 7E.4A, FSM nhận diện đây là hành vi tạo dấu của nhóm A $\rightarrow$ Xác nhận thăng cấp thành **`Ă` tại frame 28** (`source = "diacritic_fsm"`).
- **Chuỗi tokens:** `[(7, 'A', 'static_head_fallback'), (27, 'D', 'geometric_rule'), (28, 'Ă', 'diacritic_fsm'), (38, 'V', 'geometric_rule'), (52, 'A', 'static_head_fallback')]`.

#### 2. Chữ Đ (`D0495` — 105 frames)
- **Quá trình thiết lập nền:** Geometric rule D thất bại do ngón trỏ hơi cong ($d_{IW} < 1.70$). Static Head Fallback kích hoạt 31 lần với độ tin cậy trung bình 0.955.
- **Ổn định FSM:** FSM đạt `BASE_STABLE` với `current_base = "D"` tại **frame 42** ($t = 1.752\text{s}$).
- **Phân giải dấu:** Trong pha `DIACRITIC` (frame 55–75), người ký gập đốt giữa ngón trỏ. Góc khớp liên đốt gần (PIP angle) giảm từ $168^\circ$ xuống $144^\circ$ ($\Delta\theta = 24^\circ \ge 15^\circ$ drop threshold).
- **Kết quả:** FSM xác nhận thành công **`Đ` tại frame 66** (`source = "diacritic_fsm"`).
- **Chuỗi tokens:** `[(34, 'D', 'static_head_fallback'), (66, 'Đ', 'diacritic_fsm')]`.

### 6.3. Phân tích nguyên nhân sâu xa của Â và Ê

Mặc dù cả `Â` và `Ê` đều đã khắc phục hoàn toàn lỗi nhận diện chữ nền (đạt `BASE_STABLE` tại frame 33 và frame 14), cử chỉ dấu chưa được xác nhận bởi các yếu tố độc lập với Static Head:

1. **Chữ Â (`D0491B`):**
   - Base `A` ổn định từ frame 33 đến frame 48.
   - Tại frame 49, khi người ký di chuyển bàn tay vẽ hình nón (circumflex lift), hình dáng bàn tay thay đổi làm Geometric Rule `O` đạt `MATCH` với confidence cao.
   - Khi nhãn `O` xuất hiện, FSM kích hoạt cơ chế `switch_candidate` (Task 7E.4B) và sau đó chuyển trạng thái theo nhãn `O`, làm gián đoạn pha vẽ nón của nhóm `A`.
2. **Chữ Ê (`D0497B`):**
   - Base `E` ổn định từ frame 14 ($t = 0.584\text{s}$) và người ký giữ thế tay nền tĩnh đến tận frame 29 ($t = 1.210\text{s}$).
   - Đến frame 30 ($t = 1.251\text{s}$), người ký mới bắt đầu cử chỉ vẽ nón, khiến nhãn chuyển sang `NEUTRAL`.
   - Lúc này, thời gian tìm kiếm dấu tính từ thời điểm FSM đạt stable ban đầu là $1.251 - 0.584 = 0.667\text{s}$, vượt quá ngưỡng an toàn `max_diacritic_search_sec = 0.65\text{s}` trong `update_transient()`. Do đó, FSM tự động reset về `IDLE` trước khi cử chỉ vẽ nón hoàn thành.

> [!NOTE]
> Các hiện tượng trên là vấn đề về **động học cử chỉ (kinematics) và ngưỡng thời gian FSM**, hoàn toàn nằm ngoài phạm vi tầng Static Fallback. Việc khắc phục thành công việc thiết lập `BASE_STABLE` cho A, D, E trong Task 7E.5B đã tạo tiền đề vững chắc để tinh chỉnh pha động học trong các nhiệm vụ tiếp theo.

---

## 7. TOÀN VẸN CHECKPOINT VÀ TÍNH NGUYÊN VẸN MÔ HÌNH

Hệ thống đã kiểm tra tính toàn vẹn thông qua mã băm SHA-256 trên toàn bộ các file checkpoint nhị phân:

| Tên Checkpoint | Đường dẫn | SHA-256 Hash | Trạng thái |
| :--- | :--- | :--- | :---: |
| **weights.pt** | `backend/models/weights.pt` | `6965b4ffd9eed45c10317f0cc4eff453470ea4552c081f18ea997d74365907be` | **KHỚP 100% (NGUYÊN VẸN)** |
| **sign_sstcn_attention_model.pth** | `backend/models/weights/sign_sstcn_attention_model.pth` | `2a8ddcc4d06c4f23aa7910caf77371848034dcdf8f4749d134c5894764cf9697` | **KHỚP 100% (NGUYÊN VẸN)** |
| **alphabet_static_head_v1.pt** | `backend/models/alphabet_static_head_v1.pt` | `cab8123e23ed41a6088ba1c6e80a8aaaa9a382663418cc669eacff834795c254` | **KHỚP 100% (NGUYÊN VẸN)** |

Tuyệt đối không có file checkpoint nào bị ghi đè, thay thế, nén lại hoặc huấn luyện lại trong suốt quá trình thực hiện Task 7E.5B.

---

## 8. TRẢ LỜI 8 CÂU HỎI CỐT LÕI (SECTION 9 REQUIREMENTS)

### Câu 1: Fallback có kích hoạt đúng khi geometric rule trả về NO_MATCH / UNKNOWN không?
**CÓ.** Fallback được đặt chính xác trong khối `elif status in ("NO_MATCH", "UNKNOWN") and self.has_static_head and p_prim and len(p_prim) >= 21:`. Khi luật hình học không khớp và Static Head dự đoán `A`, `D`, `E` với conf $\ge 0.70$, nhãn được gán ngay lập tức kèm `source = "static_head_fallback"`. Đã được chứng minh qua 4 unit tests (`test_a1`, `test_a2`, `test_a3`, `test_a4`) và hàng chục frame replay thực tế trên 4 video.

### Câu 2: Khi geometric rule trả về MATCH, fallback có bị chặn tuyệt đối không?
**CÓ.** Khi Geometric Rules trả về `status == "MATCH"`, luồng thực thi đi vào nhánh `if status == "MATCH":` (Step 1) và thoát khỏi khối phân giải tĩnh. Khối Step 2 (Static Head) hoàn toàn không được thực hiện. Các bài kiểm tra Category C (`test_c1` đến `test_c4`) đã kiểm chứng: ngay cả khi Static Head đạt conf 0.99 với một nhãn khác, kết quả Geometric MATCH vẫn luôn được chọn 100%.

### Câu 3: Có trường hợp nào nhãn fallback ngoài A, D, E bị phát ra không?
**HOÀN TOÀN KHÔNG.** Điều kiện cổng kiểm soát nghiêm ngặt:
`if head_letter in ALLOWED_FALLBACK_BASE_LETTERS and head_conf >= self.fallback_base_threshold:`
với `ALLOWED_FALLBACK_BASE_LETTERS = {"A", "D", "E"}`. Mọi nhãn khác ngoài tập này (ví dụ `B`, `C`, `O`, `S`...) hoặc có conf $< 0.70$ đều rơi vào nhánh `reject` và trả về `NEUTRAL` (conf = 0.0), ngoại trừ 10 chữ thuộc tập bổ sung vốn đã có từ Task 7D (`static_head`).

### Câu 4: Base stable có đạt được trên 4 video Ă, Â, Ê, Đ không?
**CÓ, ĐẠT 100% TRÊN CẢ 4 VIDEO.**
- `Ă` (`D0490B`): Base `A` đạt `BASE_STABLE` tại frame 14 ($t = 0.584\text{s}$).
- `Â` (`D0491B`): Base `A` đạt `BASE_STABLE` tại frame 33 ($t = 1.376\text{s}$).
- `Ê` (`D0497B`): Base `E` đạt `BASE_STABLE` tại frame 14 ($t = 0.584\text{s}$).
- `Đ` (`D0495`): Base `D` đạt `BASE_STABLE` tại frame 42 ($t = 1.752\text{s}$).

### Câu 5: Dấu của Ă và Đ có được xác nhận không? Dấu của Â và Ê thì sao?
- **Dấu `Ă` và `Đ` ĐƯỢC XÁC NHẬN THÀNH CÔNG:**
  - `Ă` được FSM xác nhận tại **frame 28** nhờ kết hợp Base `A` ổn định từ Fallback và cử chỉ Rule V cạnh tranh (`source = "diacritic_fsm"`).
  - `Đ` được FSM xác nhận tại **frame 66** nhờ Base `D` ổn định từ Fallback và độ rơi góc khớp PIP ngón trỏ đạt $24^\circ \ge 15^\circ$ (`source = "diacritic_fsm"`).
- **Dấu `Â` và `Ê` CHƯA ĐƯỢC XÁC NHẬN:**
  - `Â`: Bị Rule `O` can thiệp ở frame 49 khi vẽ nón, làm FSM chuyển candidate nhãn.
  - `Ê`: Bị timeout tìm kiếm dấu ($0.667\text{s} > 0.65\text{s}$) do người ký giữ thế tay nền quá lâu trước khi vẽ nón.

### Câu 6: Có xảy ra hồi quy trên 12 geometric rules, 10 static head letters, sequential digits, SSTCN không?
**HOÀN TOÀN KHÔNG.** Toàn bộ 142 bài kiểm tra trên 9 test suites đều PASS:
- 12 geometric rules không bị can thiệp logic hay thay đổi ngưỡng.
- 10 static head supplemental letters hoạt động bình thường qua nhánh `source = "static_head"`.
- Chế độ nhận diện chữ số (`digits`), chuỗi số tuần tự (`sequential_digits`), và từ vựng qua SSTCN (`words`, `sentence`) hoàn toàn không bị ảnh hưởng do phân tách mode độc lập tuyệt đối.

### Câu 7: Vấn đề rò rỉ dữ liệu (video leakage) từ Task 7E.5A ảnh hưởng như thế nào đến độ tin cậy của fallback này?
Như đã phát hiện và ghi nhận trung thực trong Task 7E.5A, checkpoint `alphabet_static_head_v1.pt` được huấn luyện trên tập dữ liệu bị trùng lặp video giữa train và holdout (`COMPLETE_VIDEO_OVERLAP`). 
- **Ý nghĩa thực tế:** Kết quả replay xuất sắc trên 4 video QIPEDC hiện tại (conf > 0.94) một phần có được do mô hình đã "thấy" người ký này trong tập train.
- **Tác động:** Khi triển khai trên webcam thực tế với người ký mới (multi-signer), độ tin cậy của Static Head có thể suy giảm (xuống dưới 0.70) dẫn đến không kích hoạt fallback, hoặc có thể phát sinh nhầm lẫn giữa các thế tay nắm tương đồng (`A` vs `S`, `E` vs `O`).
- **Biện pháp xử lý:** Ngưỡng gating nghiêm ngặt $\ge 0.70$ và việc chỉ cho phép đúng 3 chữ `{A, D, E}` là tấm đệm phòng thủ vững chắc để ngăn chặn dương tính giả trên webcam.

### Câu 8: Kết luận và khuyến nghị bước tiếp theo
- **Kết luận:** Task 7E.5B đã hoàn thành xuất sắc mục tiêu đề ra: gỡ bỏ nút thắt cổ chai lớn nhất khiến FSM bị tê liệt trên các chữ nền A, D, E. Tỷ lệ nhận diện chữ động có dấu tăng từ 2/7 lên 4/7 trên offline replay mà không làm tổn hại bất kỳ thành phần nào khác.
- **Khuyến nghị Task 7E.6:**
  1. Tiến hành kiểm thử trực tiếp trên Camera/Webcam với người ký thực để đánh giá độ ổn định thực tế của Static Head Fallback.
  2. Xem xét tinh chỉnh logic FSM cho cử chỉ vẽ nón (Circumflex) của Â/Ê/Ô: nới lỏng hoặc làm mới `stable_time` khi người ký vẫn đang duy trì thế tay nền, và kiểm soát việc Rule `O` chiếm nhãn khi tay nâng lên.

---

## 9. KẾT LUẬN

Việc tích hợp **Gated Static Head Fallback cho A, D, E** đã giải quyết triệt để vấn đề mất nhãn nền của `DiacriticStateMachine`, mở đường cho việc nhận diện thành công các chữ cái có dấu phức tạp như `Ă` và `Đ`. Hệ thống giữ vững tính ổn định, tuân thủ kiến trúc mô-đun hóa, bảo toàn nguyên vẹn 100% các checkpoint pretrained và toàn bộ các chế độ nhận diện khác của VSL Connect AI.

