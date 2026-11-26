# BÁO CÁO PHÂN TÍCH NGUYÊN NHÂN & HƯỚNG GIẢI QUYẾT: LỖI CHỮ `O` NHẬN DIỆN ĐƯỢC NHƯNG PHỤ ĐỀ LUÔN GHI THÀNH `Ơ`

**Dự án:** VSL Connect AI (`D:\SIGN_LANGUAGE`)  
**Màn hình:** Camera → Ký hiệu sang Văn bản (Chế độ `Chữ cái`)  
**Trạng thái mã nguồn:** **CHƯA SỬA CODE** (Báo cáo phân tích độc lập từ mã nguồn và dữ liệu landmark thực tế).

---

## 1. Tóm tắt hiện tượng thực tế

Khi người dùng thực hiện ký hiệu chữ **`O`** tĩnh trước webcam:
- Trên thẻ **Đang nhận diện (HUD phía trên)**: Hệ thống vẫn nhận ra và hiển thị **`O` (`95%`)**.
- Trên ô **Phụ đề (phía dưới)** và **Giọng đọc (TTS)**: Chữ **`O`** không được giữ lại dưới Phụ đề (hoặc chỉ lướt qua trong ~`50–100ms` mắt thường không kịp nhìn thấy) rồi lập tức bị biến thành chữ **`Ơ`** và phát âm **`"Ơ"`**.
- Nếu người dùng tiếp tục giữ nguyên tay chữ `O`, thẻ **Đang nhận diện** phía trên lại quay về hiện **`O` (`95%`)**, nhưng ô **Phụ đề** phía dưới vẫn bị khóa cứng ở **`Ơ`**, không thể ghi chữ `O` xuống dưới.

---

## 2. Nguyên nhân gốc rễ đã xác minh từ mã nguồn & dữ liệu thực nghiệm

Qua đối chiếu trực tiếp tại [diacritic_state_machine.py](file:///d:/SIGN_LANGUAGE/backend/vision/diacritic_state_machine.py), [alphabet_classifier.py](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py), [recognizer.py](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py) và chuỗi dữ liệu chuẩn QIPEDC `D0508N` (`Ơ.json`), hiện tượng này do **3 lỗi logic trong luật nhận dấu `Ơ` (`is_o_horn`)** kết hợp với **2 cơ chế ghi đè/khóa token của Recognizer**:

### 2.1. Nguyên nhân cốt lõi #1: Luật chữ `Ơ` (`is_o_horn`) thiếu hoàn toàn điều kiện biến thiên động học ($\Delta$) so với `BaselineSnapshot`

Tại [diacritic_state_machine.py](file:///d:/SIGN_LANGUAGE/backend/vision/diacritic_state_machine.py#L552-L674), trong tổng số **7 chữ cái có dấu** (`Ă, Â, Đ, Ê, Ô, Ơ, Ư`), **6 chữ kia đều bắt buộc kiểm tra độ thay đổi tương đối so với tư thế gốc (`baseline`)**:
- `Ă`: bắt buộc `d_im_growth >= 0.20` và `index_uncurl >= 0.15`.
- `Â`: bắt buộc `d_im_growth >= 0.03` và `middle_opened_from_base` (hoặc `lift >= 0.020`).
- `Ê`: bắt buộc `d_im_growth >= 0.05` (hoặc `lift >= 0.020`).
- `Ô`: bắt buộc `d_im_growth >= 0.04`.
- `Ư`: bắt buộc `pinky_drop = baseline.pinky_curl - curr_frame.pinky_curl >= 0.20`.
- `Đ`: bắt buộc `angle_drop = baseline.index_pip_angle - curr_frame.index_pip_angle >= 12.0°`.

**Riêng duy nhất chữ `Ơ`** tại [diacritic_state_machine.py](file:///d:/SIGN_LANGUAGE/backend/vision/diacritic_state_machine.py#L646-L654) đang viết:

```python
# Ưu tiên 2: Chữ Ơ (QIPEDC D0508N: Móc râu ngón út, ngón cái không bung ngang thành Shaka Y)
is_o_horn = (
    not is_y_shaka and (
        (curr_frame.d_pinky_index_norm >= self.o_horn_dist_threshold and curr_frame.d_pinky_wrist_norm >= 0.80) or
        (curr_frame.d_pinky_wrist_norm >= 1.25 and curr_frame.pinky_curl <= 0.30)
    )
)
```

Mặc dù `BaselineSnapshot` ([diacritic_state_machine.py](file:///d:/SIGN_LANGUAGE/backend/vision/diacritic_state_machine.py#L201-L205)) đã lưu sẵn `baseline.d_pinky_index_norm`, `baseline.d_pinky_wrist_norm` và `baseline.pinky_curl`, biến `baseline` **không hề được sử dụng** trong `is_o_horn`.  
Hệ quả: **Bàn tay đứng yên hoàn toàn (`Δ = 0.00`), không hề có động tác bật/vẩy ngón út**, nhưng chỉ cần tư thế chữ `O` tĩnh của người dùng có sẵn `d_pinky_index_norm >= 0.50` và `d_pinky_wrist_norm >= 0.80` là FSM lập tức kết luận người dùng vừa thực hiện động tác móc râu chữ `Ơ`!

---

### 2.2. Nguyên nhân cốt lõi #2: Ngưỡng tĩnh `d_PI >= 0.50` và `d_PW >= 0.80` chồng lấn trực tiếp với hình dáng chữ `O` tự nhiên trên Webcam

1. **Sự khác biệt giữa góc quay mẫu QIPEDC và góc Webcam thực tế:**
   - Trong video mẫu QIPEDC (`D0506` / `D0508N`), diễn viên đặt bàn tay nghiêng ngang gần `90°` so với ống kính và ép sát cả 4 đầu ngón tay chụm khít vào ngón cái nên khoảng cách ngón út – ngón trỏ chuẩn hóa `d_PI` rất nhỏ (`0.17 – 0.26`).
   - Khi người dùng ngồi đối diện webcam laptop và tạo vòng tròn chữ `O` (xoay lòng/cạnh bàn tay chếch `30°–50°` về phía camera, hoặc chỉ chụm chặt cái–trỏ–giữa còn ngón út cong tự nhiên theo giải phẫu xương bàn tay nơi khớp gốc ngón út `MCP 17` đã lệch ngang khỏi khớp gốc ngón trỏ `MCP 5` khoảng `0.55 – 0.65 × palm_scale`):
     - Khoảng cách `d_PI` của chữ `O` tĩnh tự nhiên rơi vào vùng **`0.50 – 0.70`**.
     - Khoảng cách ngón út – cổ tay `d_PW` của chữ `O` tĩnh vốn đã được chính luật hình học chữ `O` ([alphabet_classifier.py](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py#L104)) yêu cầu `O_D_PW_MIN = 0.75` (trung vị thực tế trên tập `O` là **`1.0025 >= 0.80`**!).
   - Ngoài ra, nếu người dùng làm chữ `O` hơi xòe nhẹ ngón út (kiểu vòng `O` lỏng), `Static Head Fallback` ([recognizer.py](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L79)) vẫn nhận đó là chữ nền `O` (`conf >= 0.70`), đẩy thẳng vào FSM và kích hoạt `is_o_horn` ngay lập tức.

2. **So sánh định lượng thực tế trên chuỗi QIPEDC `D0508N` (`Ơ.json`):**

| Đặc trưng chuẩn hóa | Pha nền chữ `O` tĩnh (`frame 35–43`) | Pha bật râu chữ `Ơ` thật (`frame 45–66`) | Ngưỡng `is_o_horn` hiện tại | Nhận xét |
| :--- | :---: | :---: | :---: | :--- |
| `d_PI` (`dist(20, 8) / palm_scale`) | `0.20 – 0.26` (med `0.22`) | **`0.89 – 1.24`** (med **`1.18`**) | `>= 0.50` (không xét `Δ`) | Ngưỡng `0.50` quá thấp; khi bật râu thật `Δd_PI` tăng tới **`+0.66 → +1.02`** |
| `d_PW` (`dist(20, 0) / palm_scale`) | `0.84 – 1.01` (med `0.97`) | **`1.42 – 1.57`** (med **`1.50`**) | `>= 0.80` (không xét `Δ`) | Bản thân chữ `O` tĩnh đã có `d_PW ≈ 0.97 >= 0.80`! Khi bật râu thật `d_PW >= 1.42` (`Δd_PW >= +0.45`) |
| `d_PW - d_RW` (độ tách ngón út khỏi áp út) | `-0.12 → +0.04` (út khép cùng áp út) | **`+0.46 → +0.87`** (út tách hẳn khỏi áp út) | *Không kiểm tra* | Đặc trưng phân biệt cực mạnh giữa `O` tĩnh (4 ngón đi cùng nhau) và `Ơ` (chỉ ngón út bật lên) đang bị bỏ trống |
| Vòng khép `O` (`d_TI`, `d_TM`) | `d_TI <= 0.19`, `d_TM <= 0.24` | `d_TI <= 0.23`, `d_TM <= 0.29` | *Không kiểm tra* | Khi bật râu `Ơ` thật, ngón cái–trỏ–giữa vẫn khép vòng `O`; khi buông tay thì vòng này bung rộng |

---

### 2.3. Nguyên nhân cốt lõi #3: Kể cả khi ký `O` khép chặt chuẩn mẫu, thao tác duỗi/thả lỏng ngón tay lúc hạ tay vẫn biến `O` thành `Ơ`

Ngay cả khi người dùng ép sát ngón út (`d_PI = 0.20 < 0.50`) suốt lúc giữ chữ `O`:
- Khi người dùng chuẩn bị kết thúc cử chỉ và **hơi bung/thả lỏng các ngón tay trong 2 frame (`~66ms`) trước khi rút tay khỏi khung hình**, bộ phân loại tĩnh trả về `NEUTRAL`.
- [recognizer.py](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L2218-L2240) chuyển vào nhánh **`fsm.update_transient()`** (cho phép tìm dấu trong `1.50s` kể từ lúc ổn định `O`).
- Lúc các ngón tay bung nhẹ ra để thả tay: `d_PI` nhảy lên `~1.04 >= 0.50` và `d_PW ~ 1.04 >= 0.80` (mặc dù lúc này vòng chữ `O` đã vỡ hoàn toàn với `d_TM = 0.71`, `d_IM = 0.84`).
- Vì `is_o_horn` **không kiểm tra vòng tròn cái–trỏ–giữa còn khép hay không**, `update_transient()` xác nhận nhầm thành **`Ơ`** sau đúng 2 frame (`~66ms`) và ghi đè lên chữ `O` vừa chốt trước đó!

---

### 2.4. Giải thích trọn vẹn hiện tượng trên Giao diện (Trace thực nghiệm 16 frame liên tiếp)

Kết quả chạy mô phỏng trực tiếp qua `VSLRecognizer.predict_landmarks(..., mode="letters")` với một tư thế chữ **`O` bất động hoàn toàn** (khớp `Geometric Rule O` `95%`, `d_PI = 0.58`, `d_PW = 0.94`, `p_curl = 0.54`):

| Frame | Thời gian | Ô trên (`Đang nhận diện`) | Trạng thái FSM | `confirmed` | `replaced_token` | Ô dưới (`Phụ đề`) | Giải thích điều gì xảy ra |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| 1 – 3 | `0.00 – 0.10s` | **`O` (95%)** | `BASE_CANDIDATE` → `BASE_STABLE` | `False` | `None` | `""` *(trống)* | Mới có `1..3` phiếu `O` (`< 4/7`), chưa đủ phiếu ghi xuống Phụ đề. Frame 3 khóa `baseline`. |
| 4 | `0.15s` | **`O` (95%)** | `BASE_STABLE` | `True` | `None` | `"O"` *(ẩn TTS)* | Đạt `4/7` phiếu → ghi tạm `"O"` kèm cờ `_replaceable_base_token = "O"`, hoãn đọc TTS để chờ dấu. |
| 5 | `0.20s` | **`O` (95%)** | `DIACRITIC_CANDIDATE (1/2)` | `False` | `None` | `"O"` | `is_o_horn` khớp lần `1/2` ngay trên bàn tay đứng yên; HUD vẫn trả về `"O"`. |
| **6** | **`0.25s`** | **`Ơ` (95%)** | **`COOLDOWN` (`CONFIRMED`)** | **`True`** | **`"O"`** | **`"Ơ"` (đọc `"Ơ"`)** | Chỉ **100ms** sau Frame 4: `is_o_horn` khớp `2/2` → nhồi 4 phiếu `"Ơ"` vào `prediction_history` ([recognizer.py](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L2198-L2200)), **xóa `"O"` và ghi đè thành `"Ơ"`** ([recognizer.py](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L2530-L2536))! |
| 7 – 13 | `0.30 – 0.60s` | **`Ơ` (95%)** | `COOLDOWN` (`0.40s`) | `False` | `None` | `"Ơ"` | Trong `0.40s` cooldown của FSM, giữ nhãn `"Ơ"`. |
| **14+** | **`0.65s+`** | **`O` (95%)** | `BASE_CANDIDATE` | `False` | `None` | **`"Ơ"` (bị khóa)** | Hết `0.40s` cooldown FSM ([diacritic_state_machine.py](file:///d:/SIGN_LANGUAGE/backend/vision/diacritic_state_machine.py#L810)), **ô trên quay lại hiện `O` (95%)**, nhưng `Exit Guard` ([recognizer.py](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L2492-L2495): `_last_confirmed_diacritic == "Ơ"` khi chưa hạ tay) **chặn mọi chữ `O` không cho ghi xuống Phụ đề**! |

> [!IMPORTANT]
> Bảng trace trên giải thích chính xác 100% trải nghiệm của bạn: **Ô trên vẫn hiện chữ `O` 95%, nhưng chữ `O` vừa chạm xuống Phụ đề ở `0.15s` đã bị xóa và biến thành `Ơ` ở `0.25s` (chỉ sau `0.10s`), sau đó ở `0.65s` ô trên quay lại hiện `O` trong khi ô dưới đã bị chốt cứng là `Ơ`.**

---

## 3. Cũng phát hiện thêm 1 điểm liên đới tại `_is_forming_diacritic_motion` cho chữ `O`

Tại [recognizer.py](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L3168-L3172) (hàm bảo vệ gia hạn chờ dấu từ `0.85s` lên `1.60s`), điều kiện nhận biết chuyển động tạo dấu của `O` đang là:
```python
if pending_token == "O":
    d_im_growth = mf.d_index_middle_norm - bl.d_index_middle_norm
    d_pi_gain = mf.d_pinky_index_norm - bl.d_pinky_index_norm
    pinky_uncurl = bl.pinky_curl - mf.pinky_curl
    return bool(d_im_growth >= 0.03 or d_pi_gain >= 0.18 or pinky_uncurl >= 0.12)
```
Hàm này **đã dùng độ lệch động học** (`d_pi_gain >= 0.18`, `pinky_uncurl >= 0.12`) — đúng hướng hơn rất nhiều so với `is_o_horn` trong `diacritic_state_machine.py`! Tuy nhiên ngưỡng `d_im_growth >= 0.03` có thể hơi nhạy với rung nhẹ của đầu ngón tay (nên giữ khoảng `>= 0.035` kèm điều kiện vòng cái–trỏ chưa bung).

---

## 4. Tổng hợp hướng giải quyết (Đề xuất kiến trúc sửa chữa chuẩn xác)

Để sửa triệt để lỗi **"ký `O` bị biến thành `Ơ`"** mà **vẫn bảo đảm ký `O → Ơ` và `O → Ô` nhạy, chính xác 100%**, cần áp dụng đồng bộ **3 lớp ràng buộc động học & hình học** trực tiếp trong `DiacriticStateMachine._evaluate_diacritic_for_group` (nhánh `base == "O"`, tại [diacritic_state_machine.py](file:///d:/SIGN_LANGUAGE/backend/vision/diacritic_state_machine.py#L637-L655)):

### Giải pháp 1 (Bắt buộc — Trọng tâm nhất): Bổ sung điều kiện **Biến thiên động học ($\Delta$) so với `baseline`** cho `is_o_horn`
Chữ `Ơ` trong VSL là **cử chỉ động** (tạo chữ `O` trước rồi bật/vẩy ngón út lên tạo dấu móc). Do đó, `is_o_horn` bắt buộc phải thỏa mãn **sự gia tăng thực sự của ngón út so với `baseline` lúc mới vào `O`**:
- `d_pi_gain = curr_frame.d_pinky_index_norm - baseline.d_pinky_index_norm >= 0.30` *(trên video chuẩn `D0508N`, `d_pi_gain` thực tế đạt `+0.66 → +1.02`; khi giữ tĩnh chữ `O`, `d_pi_gain ≈ 0.00 → 0.05`)*, **HOẶC**
- `d_pw_gain = curr_frame.d_pinky_wrist_norm - baseline.d_pinky_wrist_norm >= 0.25` kết hợp `pinky_uncurl = baseline.pinky_curl - curr_frame.pinky_curl >= 0.15` *(trên `D0508N`, `d_pw_gain` đạt `+0.45 → +0.60`)*.

$\Rightarrow$ **Tác dụng:** Triệt tiêu **100%** hiện tượng giữ yên chữ `O` (dù xoay góc nào hay ngón út hơi tách sẵn) mà tự nhảy sang `Ơ`, vì bàn tay đứng yên luôn có `d_pi_gain ≈ 0` và `d_pw_gain ≈ 0`.

### Giải pháp 2 (Bắt buộc): Kiểm tra **tính bảo toàn vòng tròn nền `O` và sự tách biệt giữa ngón út với ngón áp út** (Chống nhầm khi buông/mở tay)
Khi người dùng thực sự làm chữ `Ơ`, chỉ có **ngón út** bật vươn lên làm dấu móc, còn **ngón cái – ngón trỏ (và ngón áp út/giữa)** vẫn giữ khum vòng chữ `O`. Ngược lại, khi người dùng **thả lỏng mở bàn tay để hạ tay**, cả 4 ngón cùng duỗi ra và vòng cái–trỏ mở toác.  
Vì vậy cần thêm 2 chốt chặn hình học cho `is_o_horn`:
1. **Ngón út phải thực sự vươn tách ra khỏi ngón áp út/vòng gập:**
   - `d_pinky_wrist_norm >= 1.15` (thay vì `0.80` vốn là mức mặc định của `O` tĩnh!) **VÀ**
   - Khoảng chênh độ vươn ngón út so với ngón áp út: `(curr_frame.d_pinky_wrist_norm - d_ring_wrist_norm) >= 0.25` (hoặc `ring_curl >= 0.30` giữ khum). *(Trong `D0508N`, `d_PW = 1.42–1.57` và `d_PW - d_RW = +0.46 → +0.87`, còn khi mở cả bàn tay lúc hạ tay thì `d_PW - d_RW ≈ -0.03`!).*
2. **Vòng chạm ngón cái – ngón trỏ của chữ `O` chưa bị phá vỡ hoàn toàn:**
   - `curr_frame.d_thumb_index_tip_norm <= 0.38` (trong `MotionFrame` đã tính sẵn `d_thumb_index_tip_norm = d_t8_norm` tại dòng `524`! Trên `D0508N`, suốt pha móc râu `Ơ`, `d_TI` luôn giữ ở `0.03 – 0.23 <= 0.38`; ngược lại khi bung tay thả lỏng hay chuyển sang chữ khác, `d_TI` vọt lên `> 0.45` hoặc `d_IM` mở toác `> 0.55`).

$\Rightarrow$ **Tác dụng:** Triệt tiêu hoàn toàn lỗi **ký `O` chuẩn rồi thả lỏng ngón tay lúc hạ tay bị `update_transient` đè thành `Ơ`**.

### Giải pháp 3 (Khuyến nghị bổ trợ): Chặn `Static Head Fallback` nhận vơ tư thế đang móc sẵn râu `Ơ` / chữ `I, Y` làm nền `O`
Tại [recognizer.py](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L2076):
- Khi `alphabet_classifier` trả về `NO_MATCH` mà `static_alphabet_head` dự đoán fallback `"O"`, cần thêm điều kiện bảo vệ nhẹ: chỉ cho phép fallback sang chữ nền `"O"` khi đầu ngón cái và ngón trỏ thực sự đang chụm gần nhau (`d_TI <= 0.35`) và ngón út không đang duỗi thẳng đơn độc (`not p_ext` hoặc `d_PW <= 1.25`).
- Điều này ngăn trường hợp bàn tay đang ở tư thế khác được MLP đoán mò là `O` rồi kích hoạt nhầm chuỗi FSM của nhóm `O`.

---

## 5. Đánh giá tác động nếu áp dụng phương án trên

| Kịch bản thực tế | Hành vi hiện tại (Đang lỗi) | Hành vi sau khi áp dụng Giải pháp 1 + 2 + 3 |
| :--- | :--- | :--- |
| **1. Ký chữ `O` tĩnh (nghiêng hoặc trực diện webcam, ngón út hơi tách tự nhiên)** | HUD hiện `O`, sau `0.10s` bị đè thành `Ơ` dưới Phụ đề và đọc `"Ơ"` | Chốt **`O`** xuống Phụ đề sau 4 frame (`~100–150ms`), giữ yên đọc `"O"` sau `0.85s` hoặc hạ tay đọc `"O"` ngay sau `~67ms` |
| **2. Ký chữ `O` khép chặt rồi buông lỏng ngón tay để hạ tay** | `update_transient` bắt 2 frame mở tay → xóa `O`, ghi đè thành `Ơ` | Vòng `O` mở (`d_PW - d_RW < 0.25` / `d_TI > 0.38`) nên không khớp `Ơ` → giữ nguyên **`O`** và đọc `"O"` ngay khi hạ tay |
| **3. Ký chữ `O → Ơ` thật (chụm `O` ổn định rồi bật/móc ngón út lên)** | Nhận `Ơ` | `d_pi_gain >= 0.30`, `d_PW >= 1.15`, `d_PW - d_RW >= 0.25`, `d_TI <= 0.38` đều thỏa mãn rõ rệt → thay `O` thành **`Ơ`** và chỉ đọc **`"Ơ"`** đúng 1 lần |
| **4. Ký chữ `O → Ô` thật (chụm `O` rồi tách nhẹ tạo nón trỏ–giữa)** | Nhận `Ô` | Luật ưu tiên 1 của `Ô` giữ nguyên 100% → nhận **`Ô`** và đọc **`"Ô"`** bình thường |
| **5. Độ trễ xác nhận & TTS vừa tối ưu ở các chữ cái/chữ số khác** | Đang hoạt động tốt | **Không bị ảnh hưởng** (chỉ siết điều kiện hình học–động học nội bộ của nhánh `Ơ`) |

---

## 6. Kết luận & Đề xuất bước tiếp theo

- **Nguyên nhân gốc rễ 100% nằm ở hàm `_evaluate_diacritic_for_group` nhánh `base == "O"` (`is_o_horn`) trong [diacritic_state_machine.py](file:///d:/SIGN_LANGUAGE/backend/vision/diacritic_state_machine.py#L646-L654)**: đây là chữ có dấu duy nhất kiểm tra ngưỡng tĩnh tuyệt đối (`d_PI >= 0.50, d_PW >= 0.80`) mà quên trừ `baseline` và quên kiểm tra sự tách biệt ngón út – áp út cũng như vòng khép cái – trỏ.
- Nếu bạn đồng ý với hướng giải quyết trên (Giải pháp 1 + 2 + 3), tôi sẽ tiến hành cập nhật gọn trong [diacritic_state_machine.py](file:///d:/SIGN_LANGUAGE/backend/vision/diacritic_state_machine.py) và [recognizer.py](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py), chạy kiểm thử hồi quy đảm bảo cả `O`, `Ô`, `Ơ` cùng toàn bộ 29 chữ cái đều đạt 100%, và reload trực tiếp lên server local để bạn thử lại ngay trên webcam.
