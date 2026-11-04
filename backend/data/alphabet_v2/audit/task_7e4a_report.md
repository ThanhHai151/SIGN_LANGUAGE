# BÁO CÁO KIỂM TOÁN VÀ TRIỂN KHAI TASK 7E.4A: FSM-AWARE SUPPRESSION OF COMPETING RULE I/V

**Dự án:** VSL Connect AI  
**Mã nhiệm vụ:** TASK 7E.4A  
**Thời gian thực hiện:** 31/10/2026 15:58:00+07:00  
**Tình trạng:** HOÀN THÀNH TOÀN DIỆN (Đã xử lý hiện tượng Rule I cướp nhãn trên nhóm U/O, bảo toàn nguyên vẹn Checkpoint và Geometric Rules)

---

## 1. Xác nhận Nguyên nhân gốc rễ (Root Cause) bằng Code & Dữ liệu

Trong Task 7E.3, quá trình kiểm toán và replay đã phát hiện ra nghịch lý:
- Trên video **Ư (D0515N)**: FSM nhận diện thành công cử chỉ tạo dấu của `Ư` ở frame 34 và chuyển sang trạng thái `STATE_COOLDOWN`. Tuy nhiên, ngay frame 35, ngón út vẫn đang duỗi thẳng (thế tay tự nhiên của chữ Ư), dẫn đến bộ phân loại tĩnh [alphabet_classifier.py](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py) kích hoạt `Rule I` (`status="MATCH", letter="I", conf=0.95`).
- Vì `"I"` không thuộc `BASE_GROUPS = ("A", "E", "O", "U", "D")` và không phải `"NEUTRAL"`, luồng cũ trong [recognizer.py](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py) lập tức gọi `fsm.reset()`, xóa sạch trạng thái cooldown của `Ư` và ghi đè nhãn thành `I` (`[(17, 'U'), (34, 'Ư'), (35, 'I')]`).
- Trên video **Ơ (D0508N)**: Khi mở ngón út vươn tách xa khỏi khuyên tròn O, ở frame 57 `Rule I` chiếm nhãn, chặn đứng FSM.
- Trên video **Ă (D0490B)**: Động tác xòe ngón trỏ và ngón giữa khiến `Rule V` chiếm nhãn. Tuy nhiên, ở video này `valid_base_frames = 0` do luật hình học tĩnh A quá khắt khe (Root Cause 3), nên FSM chưa từng đạt `BASE_STABLE`.

**Truy vết luồng xử lý trước khi sửa (Stage A):**
1. **Thời điểm đánh giá:** `alphabet_classifier.classify_detailed()` được đánh giá trước FSM. Khi Rule I/V MATCH, nhãn `base_sign` mang giá trị `"I"` hoặc `"V"`.
2. **Hành vi reset FSM:** Vì `"I"` và `"V"` không nằm trong `BASE_GROUPS`, recognizer rơi vào nhánh `else: fsm.reset()`, vừa xóa sạch baseline vừa ghi đè nhãn FSM ngay tức thì (chỉ sau 1 frame duy nhất).
3. **Khả năng nhận diện cửa sổ dấu:** FSM có đầy đủ `current_base`, `baseline`, `state`, và `max_diacritic_search_sec` (0.65s) / `cooldown_sec` (0.40s) để xác định cửa sổ tìm kiếm dấu hợp lệ.
4. **Phân biệt cạnh tranh vs chuyển chữ:** Hệ thống cũ không có bộ đếm frame cạnh tranh (`competing_frames_count`), coi mọi frame I/V đều là chuyển chữ thật.

---

## 2. Danh mục tập tin sửa đổi

| Tập tin | Loại thay đổi | Mô tả thay đổi |
| :--- | :--- | :--- |
| [diacritic_state_machine.py](file:///d:/SIGN_LANGUAGE/backend/vision/diacritic_state_machine.py) | **Mã nguồn lõi** | Thêm `COMPETING_RULES_MAP = {"O": "I", "U": "I", "A": "V"}`, phương thức `is_competing_label()`, và phương thức điều phối chuyên trách `update_competing()`. Bổ sung đếm frame cạnh tranh liên tiếp `competing_frames_count` (ngưỡng chuyển đổi chủ động: `max_competing_frames = 3`). |
| [recognizer.py](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py) | **Pipeline tích hợp** | Bổ sung nhánh **Tình huống D (Nhãn cạnh tranh I/V)**: Khi `fsm.baseline is not None` và FSM đang ở `BASE_STABLE`/`DIACRITIC_CANDIDATE`/`COOLDOWN`, nếu gặp nhãn cạnh tranh tương ứng (`I` cho O/U, `V` cho A), gọi `fsm.update_competing()` thay vì gọi `fsm.reset()`. |
| [test_diacritic_state_machine.py](file:///d:/SIGN_LANGUAGE/backend/tests/test_diacritic_state_machine.py) | **Kiểm thử đơn vị** | Bổ sung 6 test cases mới (Test 22–27) kiểm tra toàn diện các kịch bản cạnh tranh Rule I/V, bảo vệ baseline, cooldown Ư, chuyển chữ thật sau 3 frames, và hết timeout. |
| [test_diacritic_camera_integration.py](file:///d:/SIGN_LANGUAGE/backend/tests/test_diacritic_camera_integration.py) | **Kiểm thử tích hợp camera** | Bổ sung 4 test cases mới (Test 17–20) kiểm tra pipeline camera, bảo vệ cooldown Ư trước Rule I, chuyển chữ thật sau 3 frames, cách ly phiên độc lập, và không chặn Rule V khi không có baseline A. |

---

## 3. Chi tiết cơ chế xử lý I/V cạnh tranh (Giai đoạn B)

### 3.1. Bản đồ cạnh tranh (Competing Rules Mapping)
Quy định cặp xung đột hình học trực tiếp:
- Nhóm **`U`** (`variants: {"U", "Ư"}`): Nhãn cạnh tranh là **`"I"`** (ngón út duỗi).
- Nhóm **`O`** (`variants: {"O", "Ô", "Ơ"}`): Nhãn cạnh tranh là **`"I"`** (ngón út vươn râu).
- Nhóm **`A`** (`variants: {"A", "Ă", "Â"}`): Nhãn cạnh tranh là **`"V"`** (ngón trỏ và ngón giữa xòe).

### 3.2. Điều kiện kích hoạt ưu tiên FSM
Cơ chế chỉ được kích hoạt khi thỏa mãn **toàn bộ 5 điều kiện**:
1. Mốc xương bàn tay hợp lệ (`has_valid_landmarks` = 21 điểm).
2. FSM **đã có baseline snapshot hợp lệ** (`fsm.baseline is not None`). *(Nếu A không có baseline, Rule V không bao giờ bị chặn).*
3. FSM đang ở một trong các trạng thái: `STATE_BASE_STABLE`, `STATE_DIACRITIC_CANDIDATE`, `STATE_COOLDOWN`.
4. Nhãn từ bộ phân loại tĩnh khớp với bản đồ cạnh tranh: `fsm.is_competing_label(base_sign) == True`.
5. Thời gian nằm trong cửa sổ hợp lệ:
   - Trong `COOLDOWN`: $(t - confirmed\_time) < cooldown\_sec$ (0.40s).
   - Trong `BASE_STABLE`/`DIACRITIC_CANDIDATE`: $(t - baseline.stable\_time) \le max\_diacritic\_search\_sec$ (0.65s).

### 3.3. Xử lý trong phương thức `update_competing()`
1. **Khi đang trong `STATE_COOLDOWN` (ví dụ vừa xác nhận `Ư`):**
   - Giữ nguyên `final_label = self.confirmed_diacritic` (`Ư`), `state = STATE_COOLDOWN`, `source = "diacritic_fsm"`.
   - `competing_frames_count = 0`.
   - Ngăn chặn tuyệt đối việc `Rule I` cướp nhãn trong thời gian cooldown!
2. **Khi đang trong `STATE_BASE_STABLE` hoặc `STATE_DIACRITIC_CANDIDATE`:**
   - Đưa frame vào `motion_buffer` và tính toán các đặc trưng động học ($v_y$, góc ngón trỏ, khoảng cách ngón, độ cong ngón út).
   - **Đánh giá dấu tức thời:** Nếu cử chỉ mở ngón út / mở ngón V thỏa mãn điều kiện tạo dấu của nhóm nền (`Ư`, `Ơ`, `Ă`), FSM **xác nhận dấu ngay lập tức** (`state = STATE_DIACRITIC_CONFIRMED`)!
   - Nếu chưa thỏa mãn dấu: Tăng `self.competing_frames_count += 1`.
     - Nếu `competing_frames_count < 3`: Tạm thời chặn nhãn cạnh tranh, bảo toàn baseline và motion buffer, trả về `final_label = "NEUTRAL"`, `suppressed = True`.
     - Nếu `competing_frames_count >= 3`: Người ký chủ động giữ thế tay I (hoặc V) $\ge 3$ frames liên tiếp $\rightarrow$ Kết thúc ưu tiên FSM, gọi `self.reset()`, trả về nhãn cạnh tranh gốc (`"I"` hoặc `"V"`, `confidence = 0.95`).
3. **Điều kiện kết thúc ưu tiên FSM & Trở về luồng bình thường:**
   - **Hết timeout tìm kiếm (0.65s):** FSM tự động reset về `IDLE`, chuyển nhượng nhận diện cho Rule I/V.
   - **Mất tay thực sự (None/rỗng):** FSM reset về `IDLE`, trả về `NEUTRAL`.
   - **Chuyển sang chữ tĩnh khác (B, C, D...):** FSM reset về `IDLE` ngay lập tức.
   - **Giữ I/V liên tiếp 3 frames:** Nhận diện I/V bình thường.

---

## 4. Kiểm toán tính toàn vẹn Checkpoint

Mã băm SHA-256 của toàn bộ 3 checkpoints được kiểm tra trước và sau khi triển khai, **hoàn toàn trùng khớp 100%**:

| Checkpoint | Đường dẫn | SHA-256 Trước | SHA-256 Sau | Trạng thái |
| :--- | :--- | :--- | :--- | :--- |
| **VSLGestureNet** | [weights.pt](file:///d:/SIGN_LANGUAGE/backend/models/weights.pt) | `6965b4ffd9eed45c10317f0cc4eff453470ea4552c081f18ea997d74365907be` | `6965b4ffd9eed45c10317f0cc4eff453470ea4552c081f18ea997d74365907be` | Trùng khớp 100% |
| **SSTCN Dynamic** | [sign_sstcn_attention_model.pth](file:///d:/SIGN_LANGUAGE/backend/models/weights/sign_sstcn_attention_model.pth) | `2a8ddcc4d06c4f23aa7910caf77371848034dcdf8f4749d134c5894764cf9697` | `2a8ddcc4d06c4f23aa7910caf77371848034dcdf8f4749d134c5894764cf9697` | Trùng khớp 100% |
| **Static Alphabet Head** | [alphabet_static_head_v1.pt](file:///d:/SIGN_LANGUAGE/backend/models/alphabet_static_head_v1.pt) | `cab8123e23ed41a6088ba1c6e80a8aaaa9a382663418cc669eacff834795c254` | `cab8123e23ed41a6088ba1c6e80a8aaaa9a382663418cc669eacff834795c254` | Trùng khớp 100% |

- **Zero Retraining**: Không huấn luyện lại bất kỳ mạng nơ-ron nào.
- **Không sửa `alphabet_classifier.py`**: Không sửa thứ tự hay ngưỡng hình học của 12 rules tĩnh.

---

## 5. Kết quả kiểm thử tự động toàn diện (100% PASS)

Toàn bộ 8 test suites kiểm thử tự động của hệ thống đều vượt qua xuất sắc (**126/126 tests PASS, 0 lỗi, 0 regression**):

| Test Suite | Tập tin kiểm thử | Số test | Kết quả | Ghi chú |
| :--- | :--- | :---: | :---: | :--- |
| **Diacritic State Machine** | [test_diacritic_state_machine.py](file:///d:/SIGN_LANGUAGE/backend/tests/test_diacritic_state_machine.py) | 27 | **PASS** | +6 test cases (Test 22–27) bao phủ toàn diện Rule I/V |
| **Diacritic Camera Integration** | [test_diacritic_camera_integration.py](file:///d:/SIGN_LANGUAGE/backend/tests/test_diacritic_camera_integration.py) | 20 | **PASS** | +4 test cases (Test 17–20) bao phủ camera pipeline |
| **Geometric Alphabet Rules** | [test_alphabet_classifier.py](file:///d:/SIGN_LANGUAGE/backend/tests/test_alphabet_classifier.py) | 12 | **PASS** | Bảo toàn 12 geometric rules |
| **Static Alphabet Head** | [test_alphabet_static_head_integration.py](file:///d:/SIGN_LANGUAGE/backend/tests/test_alphabet_static_head_integration.py) | 10 | **PASS** | 10 chữ tĩnh bổ sung hoạt động chuẩn xác |
| **Digit 10 Detector** | [test_digit_10_integration.py](file:///d:/SIGN_LANGUAGE/backend/tests/test_digit_10_integration.py) | 12 | **PASS** | Chữ số 10 độc lập |
| **Mode Separation** | [test_mode_separation.py](file:///d:/SIGN_LANGUAGE/backend/tests/test_mode_separation.py) | 7 | **PASS** | Tách biệt tuyệt đối letters/digits/words |
| **Sequential Digits** | [test_sequential_digits_integration.py](file:///d:/SIGN_LANGUAGE/backend/tests/test_sequential_digits_integration.py) | 6 | **PASS** | Nhận diện dãy số nguyên vẹn |
| **Full API Server & WebSocket** | [test_api.py](file:///d:/SIGN_LANGUAGE/backend/tests/test_api.py) | 32 | **PASS** | Hệ thống Web API ổn định 100% |
| **TỔNG CỘNG** | **8 test suites** | **126** | **PASS (100%)** | **0 lỗi, 0 regression** |

---

## 6. Kết quả Offline Replay trên các Video thực tế (Trước vs Sau Fix)

Dữ liệu replay chi tiết được ghi nhận tại [task_7e4a_replay_results.json](file:///d:/SIGN_LANGUAGE/backend/data/alphabet_v2/audit/task_7e4a_replay_results.json):

| Ký hiệu | Video ID | Tổng frames | Nhãn cạnh tranh | Số frame MATCH nhãn cạnh tranh | Trước Fix (Task 7E.3 Tokens) | Sau Fix (Task 7E.4A Tokens) | Số frame bị ghi đè sau fix | Đánh giá chuyển biến thực tế |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Ư** | D0515N | 118 | **I** | 4 frames (frames 35, 36, 37, 43) | `[(17, 'U'), (34, 'Ư'), (35, 'I')]` | **`[(17, 'U'), (34, 'Ư')]`** | **0 frame** | **THÀNH CÔNG VƯỢT TRỘI**: `Ư` được bảo lưu trọn vẹn trong cooldown, loại bỏ hoàn toàn frame rác `I` ở frame 35! |
| **Ơ** | D0508N | 89 | **I** | 1 frame (frame 57) | `[..., (45, 'O'), (57, 'I')]` | `[..., (45, 'O'), (57, 'I')]` | 1 frame | Ở video này, tại frame 43 xuất hiện nhầm nhãn `E` làm gián đoạn chuỗi O trước đó, dẫn đến frame 57 FSM chưa kịp tích lũy đủ 4 frames nền để đạt `BASE_STABLE` (thuộc Root Cause 3). |
| **Ă** | D0490B | 89 | **V** | 18 frames (frames 29–47) | `[(27, 'D'), (29, 'V')]` | `[(27, 'D'), (29, 'V')]` | 18 frames | Do quy tắc hình học A quá khắt khe, video này có 0 frame A hợp lệ (`valid_base_frames = 0`). Vì FSM không có baseline A, cơ chế bảo vệ FSM không kích hoạt để tránh chặn sai Rule V thật (theo đúng yêu cầu). |

---

## 7. Phân biệt rõ ràng giữa các môi trường kiểm định

1. **Unit Tests (Kiểm thử đơn vị FSM):** Xác nhận tính đúng đắn toán học của logic FSM, chứng minh thuật toán có khả năng phân biệt hoàn hảo giữa nhãn cạnh tranh tạm thời vs người ký cố tình giữ thế tay I/V $\ge 3$ frames.
2. **Offline Replay (Replay trên chuỗi video QIPEDC thực tế):**
   - Đã triệt tiêu hoàn toàn lỗi Rule I hijacking trên video **Ư** (D0515N), giữ vững token `Ư`.
   - Vạch trần chính xác sự phụ thuộc vào chữ nền: Trên video **Ơ** và **Ă**, FSM không bị lỗi thuật toán cạnh tranh mà bị nghẽn ở khâu nhận diện chữ nền ban đầu (chữ nền bị gián đoạn hoặc không nhận được).
3. **Live Webcam (Thực tế người dùng):** Người ký thực tế trước webcam có thể giữ chữ nền U hoặc O ổn định sẽ không còn bị hiện tượng Rule I bất ngờ cướp nhãn khi mở ngón út.

---

## 8. Những vấn đề còn tồn tại & Đề xuất bước tiếp theo (Task 7E.4B / 7E.5)

- **Vấn đề còn tồn tại duy nhất (Root Cause 3 - Strict Base Geometry):**
  - Các luật hình học tĩnh cho chữ **A, E, D** và độ ổn định của **O** trong [alphabet_classifier.py](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py) hiện đang dùng các ngưỡng hình học lý tưởng, dẫn đến việc người ký thực tế có bàn tay hơi nghiêng hoặc ngón hơi cong không thể kích hoạt được chữ nền.
- **Đề xuất nhiệm vụ tiếp theo (Task 7E.4B / Task 7E.5):**
  1. Cho phép `VSLStaticAlphabetHead` (hoặc mở rộng dung sai kiểm soát cho Rule A/E/D) hỗ trợ nhận diện chữ nền khi luật hình học tĩnh không khớp.
  2. Bổ sung cơ chế dung sai chuyển tiếp chữ nền ngắn (ví dụ frame 43 của video Ơ bị nhầm thành E trong 1 frame đơn lẻ không nên reset tức thì chuỗi 7 frames O trước đó).

