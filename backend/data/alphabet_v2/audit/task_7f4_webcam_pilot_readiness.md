# BÁO CÁO KIỂM TOÁN TASK 7F.4 — CHUẨN BỊ THỬ NGHIỆM WEBCAM CÓ KIỂM SOÁT (CONTROLLED WEBCAM PILOT READINESS AUDIT)

**Dự án:** VSL Connect AI — Hệ thống Nhận diện Ngôn ngữ Ký hiệu Việt Nam  
**Ngày thực hiện:** 2026-11-01  
**Phạm vi:** Kiểm toán mã nguồn production, đánh giá cơ chế toggle/rollback, năng lực ghi log (logging capabilities), thiết kế protocol thử nghiệm webcam và đối chiếu thống kê False Positive.  
**Trạng thái kiểm toán:** READ-ONLY AUDIT (Không sửa source code, không thay đổi threshold production, không train lại model, không can thiệp FSM hay pipeline chữ động).  
**Kết luận Readiness:** **`NEEDS_LOGGING_FIRST`**

---

## TỔNG QUAN VÀ MỤC TIÊU KIỂM TOÁN

Dựa trên kết quả thực nghiệm mô phỏng toàn diện từ **Task 7F.3** (đạt độ chính xác tổng thể 82.03% trên 551 mẫu dataset V2, tăng ròng +20 mẫu, tỷ lệ dương tính giả mới = 0), Task 7F.4 tiến hành kiểm toán mã nguồn production hiện tại để chuẩn bị cho giai đoạn thử nghiệm thực tế trên webcam.

Cấu hình pilot dự kiến kiểm tra:
1. **Q:** Giảm ngưỡng tin cậy từ `0.85` xuống `0.75` trong `supplemental_thresholds`.
2. **B, C, T, L, Y:** Kích hoạt Gated Static Head Fallback với ngưỡng tin cậy `0.70` khi 12 luật hình học trả về `NO_MATCH` hoặc `UNKNOWN`.
3. **A, D, E:** Giữ nguyên Gated Fallback hiện tại ở ngưỡng `0.70` (đã tích hợp từ Task 7E.5B).
4. **I:** Tuyệt đối **KHÔNG** đưa vào Fallback (giữ nguyên nhận diện thuần luật hình học để bảo vệ FSM chữ động `Ơ` và `Ư`).
5. **N:** Giữ nguyên ngưỡng `0.75` (không giảm xuống 0.70 để ngăn ngừa nhầm lẫn sang `M`).
6. **Các chữ cái còn lại:** Giữ nguyên toàn bộ ngưỡng và luật hình học hiện có.

---

## PHẦN 1 — KIỂM TRA MÃ NGUỒN PRODUCTION (PRODUCTION CODE AUDIT)

### 1.1. Vị trí quản lý cấu hình và ngưỡng trong `backend/vision/recognizer.py`

Qua kiểm tra trực tiếp mã nguồn `backend/vision/recognizer.py`, các biến số và cấu hình liên quan đến nhận diện chữ cái tĩnh được phân bổ tại các vị trí chính xác như sau:

| Tên biến / Hằng số | Kiểu dữ liệu | Vị trí (Dòng) | Giá trị hiện tại | Vai trò trong hệ thống |
| :--- | :--- | :--- | :--- | :--- |
| `ALLOWED_SUPPLEMENTAL_LETTERS` | `set` | Dòng 61 | `{"G", "H", "K", "M", "N", "P", "Q", "R", "U", "X"}` | Tập hợp 10 chữ cái tĩnh bổ sung không có luật hình học, được phục vụ trực tiếp bởi Static Head MLP (Task 6B/6C). |
| `ALLOWED_FALLBACK_BASE_LETTERS` | `set` | Dòng 64 | `{"A", "D", "E"}` | Tập hợp các chữ cái được phép kích hoạt Gated Fallback từ Static Head khi luật hình học thất bại (Task 7E.5B). |
| `DEFAULT_FALLBACK_BASE_THRESHOLD` | `float` | Dòng 65 | `0.70` | Ngưỡng tin cậy tối thiểu để chấp nhận dự đoán fallback cho các chữ cái trong fallback set. |
| `DEFAULT_SUPPLEMENTAL_THRESHOLDS` | `dict` | Dòng 69–80 | `{"G": 0.70, "H": 0.70, "K": 0.75, "M": 0.70, "N": 0.75, "P": 0.70, "Q": 0.85, "R": 0.75, "U": 0.70, "X": 0.70}` | Bảng ngưỡng tin cậy thực nghiệm cho từng chữ cái bổ sung. `Q` đang ở mức bảo thủ `0.85`. |

### 1.2. Khởi tạo đối tượng `VSLRecognizer` (`backend/vision/recognizer.py`, Dòng 98–148)

Trong hàm `__init__`:
- **Ngưỡng bổ sung (`supplemental_thresholds`):**
  ```python
  self.supplemental_thresholds = dict(DEFAULT_SUPPLEMENTAL_THRESHOLDS)
  if supplemental_thresholds and isinstance(supplemental_thresholds, dict):
      self.supplemental_thresholds.update(supplemental_thresholds)
  ```
  *(Dòng 123–125)*: Tham số `supplemental_thresholds` đã được tham số hóa linh hoạt qua constructor kwargs.
- **Ngưỡng Fallback (`fallback_base_threshold`):**
  ```python
  self.fallback_base_threshold = float(fallback_base_threshold) if fallback_base_threshold is not None else DEFAULT_FALLBACK_BASE_THRESHOLD
  ```
  *(Dòng 128)*: Tham số `fallback_base_threshold` đã được tham số hóa qua constructor kwargs.
- **Hạn chế cấu trúc nghiêm trọng phát hiện được:**
  Tập danh sách `ALLOWED_FALLBACK_BASE_LETTERS` hiện tại **chưa được tham số hóa** trong `__init__`. Biến này đang được tham chiếu trực tiếp dưới dạng biến toàn cục (module-level constant) tại Dòng 766 và Dòng 798.

### 1.3. Quy trình ra quyết định: Ưu tiên Luật hình học vs. Static Head (`recognizer.py`, Dòng 745–828)

Quy trình nhận diện trong `mode == "letters"` diễn ra tuần tự theo 2 bước nghiêm ngặt:

```
[21 Điểm mốc Landmark (p_prim)]
             │
             ▼
[Bước 1: AlphabetClassifier (12 Luật hình học)]
             │
      ┌──────┴──────────────────────────┐
   status == "MATCH"             status in ("NO_MATCH", "UNKNOWN")
      │                                 │
      ▼                                 ▼
[final_sign = Geometric Rule]    [Bước 2: VSLStaticAlphabetHead (MLP 72D)]
(Ưu tiên tuyệt đối 100%)                │
                                ┌───────┴────────────────────────┐
                                │                                │
                      head_letter in                   head_letter in
               ALLOWED_FALLBACK_BASE_LETTERS    ALLOWED_SUPPLEMENTAL_LETTERS
               và conf >= fallback_thresh       và conf >= req_threshold
                                │                                │
                                ▼                                ▼
                        [final_sign = Fallback]        [final_sign = Static Head]
                                │                                │
                                └───────┬────────────────────────┘
                                        │ (Nếu không thỏa mãn)
                                        ▼
                                [final_sign = NEUTRAL]
```

- **Bước 1 (Ưu tiên tuyệt đối):** Dòng 750–754:
  `if status == "MATCH" and alpha_res.get("letter") in ALLOWED_LETTERS:`  
  Nếu luật hình học tìm thấy kết quả khớp duy nhất, hệ thống lập tức chốt nhãn với nguồn `"geometric_rule"`, hoàn toàn không tham khảo Static Head.
- **Bước 2a (Gated Static Head Fallback):** Dòng 766–777:
  `if head_letter in ALLOWED_FALLBACK_BASE_LETTERS and head_conf >= self.fallback_base_threshold:`  
  Kích hoạt khi luật hình học trả về `NO_MATCH` hoặc `UNKNOWN`. Gán nhãn với nguồn `"static_head_fallback"`.
- **Bước 2b (Chữ cái bổ sung):** Dòng 779–790:
  `elif head_letter in ALLOWED_SUPPLEMENTAL_LETTERS and head_conf >= req_threshold:`  
  Gán nhãn với nguồn `"static_head"`.
- **Bước 2c (Từ chối an toàn):** Dòng 792–810:
  Nếu `head_letter` không thuộc 2 tập trên, hoặc `head_conf` thấp hơn ngưỡng, lập tức hạ về `final_sign = "NEUTRAL"`, `recognition_source = "none"`.

### 1.4. Đối chiếu tính nhất quán giữa các module

1. **`backend/vision/alphabet_classifier.py`:**
   - Quản lý độc lập 12 quy tắc hình học: `A, B, C, D, E, I, L, O, S, T, V, Y`.
   - Các chữ cái còn lại (`G, H, K, M, N, P, Q, R, U, X`) không có quy tắc hình học và luôn trả về `NO_MATCH` hoặc `UNKNOWN`.
   - Hoàn toàn nhất quán với cấu trúc bổ trợ 2 tầng trong `recognizer.py`.
2. **`backend/models/alphabet_static_head.py`:**
   - Định nghĩa `STATIC_CLASSES_22`: Đầy đủ 22 chữ cái tĩnh tiếng Việt (`A, B, C, D, E, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, X, Y`).
   - Checkpoint `alphabet_static_head_v1.pt` vốn đã tính toán xác suất Softmax trên toàn bộ 22 lớp này từ khi huấn luyện.
   - Các chữ cái `B, C, T, L, Y` thực chất đã được mô hình dự đoán ở đầu ra, chỉ bị chặn lại tại tầng logic gating của `recognizer.py`. Do đó, việc mở rộng fallback sang `B, C, T, L, Y` **không đòi hỏi huấn luyện lại hay can thiệp vào checkpoint mô hình**.

---

## PHẦN 2 — ĐÁNH GIÁ CƠ CHẾ TOGGLE VÀ ROLLBACK

### 2.1. Đánh giá tính khả thi cấu hình Runtime

Hiện tại, việc thay đổi cấu hình pilot đang gặp trở ngại do sự phân tán trong cách khai báo:
- `Q`: Có thể cập nhật qua tham số `supplemental_thresholds={"Q": 0.75}` của `VSLRecognizer.__init__`.
- `B, C, T, L, Y`: Chưa thể truyền qua constructor vì `ALLOWED_FALLBACK_BASE_LETTERS` đang là biến module cứng (`{"A", "D", "E"}`).

**Giải pháp kiến trúc không xâm lấn đề xuất cho Task 7F.5:**
Thêm tham số tùy chọn vào `VSLRecognizer.__init__`:
```python
def __init__(
    self,
    ...,
    allowed_fallback_letters=None,
    enable_pilot_extended_fallback=False,
):
    # Cơ chế toggle kép: qua parameter hoặc environment variable
    pilot_env = os.environ.get("VSL_PILOT_EXTENDED_FALLBACK", "0") == "1"
    if enable_pilot_extended_fallback or pilot_env:
        self.allowed_fallback_letters = {"A", "D", "E", "B", "C", "T", "L", "Y"}
        self.supplemental_thresholds["Q"] = 0.75
    else:
        self.allowed_fallback_letters = set(allowed_fallback_letters or ALLOWED_FALLBACK_BASE_LETTERS)
```

### 2.2. Khả năng bật/tắt Pilot và Quy trình Rollback tức thì

| Kịch bản | Cơ chế thực hiện | Thời gian đáp ứng | Nguy cơ ảnh hưởng hệ thống |
| :--- | :--- | :--- | :--- |
| **Kích hoạt Pilot** | Đặt biến môi trường `VSL_PILOT_EXTENDED_FALLBACK=1` hoặc truyền cờ trong factory khởi tạo. | Khởi động lại service (< 5s) | Không có. Không cần re-compile hay build lại image. |
| **Rollback tức thì (Circuit Breaker)** | Gỡ bỏ biến môi trường hoặc set `VSL_PILOT_EXTENDED_FALLBACK=0`. | Tức thì | Trở về 100% baseline cũ (`A, D, E` fallback, `Q=0.85`). |

### 2.3. Đảm bảo an toàn tuyệt đối cho 7 chữ động (Ă, Â, Ê, Ô, Ơ, Ư, Đ) và FSM

Kiểm tra sự giao thoa giữa cấu hình pilot chữ tĩnh và bộ máy trạng thái `DiacriticStateMachine`:
- Bộ máy FSM quản lý cử chỉ tạo dấu dựa trên 5 nhóm chữ cái nền:
  $$\text{BASE\_GROUPS} = \{ \text{"A"}, \text{"E"}, \text{"O"}, \text{"U"}, \text{"D"} \}$$
- Các chữ cái mới thêm vào fallback trong pilot: $\{ \text{"B"}, \text{"C"}, \text{"T"}, \text{"L"}, \text{"Y"} \}$.
- Đối chiếu tập hợp:
  $$\{ \text{"B"}, \text{"C"}, \text{"T"}, \text{"L"}, \text{"Y"} \} \cap \text{BASE\_GROUPS} = \emptyset$$
- **Kết luận:** Các chữ cái mở rộng fallback hoàn toàn độc lập với các nhóm chữ nền tạo dấu.
- **Về chữ `I`:** Việc kiên quyết loại trừ `I` khỏi Fallback giúp ngăn chặn hoàn toàn nguy cơ Rule I hoặc Static Head I chiếm nhãn trong quá trình mở ngón út tạo dấu của `Ơ` (video D0508N) và `Ư` (video D0515N), như đã kiểm chứng tại Task 7E.4A.

---

## PHẦN 3 — KIỂM TRA NĂNG LỰC GHI LOG (LOGGING CAPABILITIES AUDIT)

### 3.1. Phân tích hàm `_log_diagnostic` hiện có (`recognizer.py`, Dòng 1130–1235)

Hàm `_log_diagnostic` được thiết kế an toàn I/O với cơ chế non-blocking thread lock. Tuy nhiên, qua phân tích chi tiết, hệ thống logging hiện tại **chưa đáp ứng được yêu cầu của một đợt thử nghiệm webcam khoa học** vì 5 thiếu sót nghiêm trọng:

```
[Dòng 1140] now - self._last_diag_log_time < 0.45 ──> GIỚI HẠN ~2.2 FPS (BỎ RƠI >93% FRAME)
[Dòng 1064] has_mlp=False, nn_sign="NEUTRAL"     ──> HARDCODE BỎ QUA STATIC HEAD
[Dòng 1073] alpha_res serializer cho digit       ──> THIẾU letter, source, head_top_k
[Dòng 1058] Thiếu prediction_source ở top-level  ──> KHÔNG PHÂN BIỆT ĐƯỢC NGUỒN DỰ ĐOÁN
[Frontend]  x-expected-sign không có UI nhập liệu ──> MẤT GROUND TRUTH TRÊN WEBCAM
```

1. **Vấn đề Throttling thời gian (Bỏ rơi 93% frame):**
   `if now - self._last_diag_log_time < 0.45: return`  
   Tần suất log bị giới hạn cứng ở ~2 frame/giây. Trong luồng webcam chuẩn 30 FPS, việc bỏ qua 28 frame/giây khiến chúng ta không thể quan sát được hiện tượng rung lắc nhãn (label jitter), độ trễ ổn định cử chỉ (stabilization latency), hoặc các frame chuyển tiếp (transitional frames) ngắn hạn.
2. **Hardcode vô hiệu hóa thông tin mạng nơ-ron:**
   Tại dòng 1064–1066, khi `_log_diagnostic` được gọi từ luồng nhận diện chữ cái, các tham số được truyền cứng:
   `has_mlp=False, nn_sign="NEUTRAL", nn_conf=0.0`.
   Do đó, log file JSONL hoàn toàn không ghi lại nhãn dự đoán và độ tin cậy của `VSLStaticAlphabetHead`.
3. **Serializer bị lệch sang chế độ Chữ số (Digit-biased serializer):**
   Đoạn trích xuất `classifier_details` chỉ đọc các key của `digit_classifier`:
   `record["classifier_details"] = {"status": ..., "digit": ..., "matched_digits": ...}`.
   Các trường sống còn của bảng chữ cái như `head_letter`, `head_confidence`, `head_top_k`, `reject_reason` bị vứt bỏ hoàn toàn.
4. **Thiếu trường `prediction_source` ở mức record:**
   Log không phản ánh rõ nhãn cuối cùng được sinh ra từ `geometric_rule`, `static_head`, `static_head_fallback` hay bị ép về `none`.
5. **Thiếu cơ chế truyền Ground Truth trực quan:**
   Mặc dù backend API `/api/recognize_landmarks` hỗ trợ nhận header `x-expected-sign`, nhưng giao diện webcam hiện tại chưa có giao diện test có kiểm soát (pilot testing harness) để người ký hiệu chọn chữ cái mục tiêu và tự động gửi kèm nhãn chuẩn.

### 3.2. Yêu cầu định dạng Log tối thiểu cần bổ sung trước khi chạy Pilot

Trước khi kích hoạt pilot, `_log_diagnostic` cần được nâng cấp (trong chế độ pilot hoặc khi bật cờ riêng) để xuất định dạng JSONL tiêu chuẩn:

```json
{
  "timestamp": 1730460000.123,
  "session_id": "pilot_signer01_20261101",
  "frame_idx": 145,
  "expected_sign": "B",
  "prediction_source": "static_head_fallback",
  "final_sign": "B",
  "final_confidence": 0.88,
  "confirmed": true,
  "geometric": {
    "status": "NO_MATCH",
    "letter": "UNKNOWN",
    "confidence": 0.0,
    "reject_reason": "thumb_lat > B_THUMB_LAT_MAX"
  },
  "static_head": {
    "raw_predicted_letter": "B",
    "confidence": 0.882,
    "top_k": [
      {"letter": "B", "confidence": 0.882},
      {"letter": "O", "confidence": 0.054},
      {"letter": "D", "confidence": 0.021}
    ]
  },
  "gating_status": {
    "is_fallback_candidate": true,
    "threshold_applied": 0.70,
    "passed": true
  }
}
```

---

## PHẦN 4 — THIẾT KẾ PROTOCOL THỬ NGHIỆM WEBCAM CÓ KIỂM SOÁT

Để đảm bảo kết quả đánh giá phản ánh chính xác năng lực thực tế và loại trừ hoàn toàn hiện tượng thiên kiến dữ liệu đơn người ký (single-signer bias từ QIPEDC), protocol thử nghiệm phải tuân thủ nghiêm ngặt các tiêu chuẩn sau:

### 4.1. Cấu trúc thử nghiệm A/B Testing song song

- **Nhánh A (Baseline Production):**
  - Geometric Rules: 12 chữ cái.
  - Supplemental Static Head: 10 chữ cái (`Q=0.85`).
  - Fallback: `A, D, E` (`threshold=0.70`).
- **Nhánh B (Proposed Pilot):**
  - Geometric Rules: Giữ nguyên.
  - Supplemental Static Head: 10 chữ cái (`Q=0.75`).
  - Fallback: `A, D, E, B, C, T, L, Y` (`threshold=0.70`).
  - `I` giữ nguyên geometric-only.

### 4.2. Yêu cầu về Người thực hiện (Signers)

- **Số lượng tối thiểu:** **3 đến 5 người**.
- **Phân tách đối tượng bắt buộc:**
  - **Nhóm 1 (Tương đồng tập huấn luyện):** 1 người có cấu trúc bàn tay và thói quen ký hiệu tương đương signer trong video QIPEDC.
  - **Nhóm 2 (Unseen Signers — Bắt buộc $\ge 2$ người):** Người mới hoàn toàn, chưa từng xuất hiện trong bất kỳ dataset huấn luyện hay kiểm toán nào trước đây, đa dạng về kích thước bàn tay (tay nhỏ/lớn), tỷ lệ ngón và độ linh hoạt khớp ngón.

### 4.3. Điều kiện môi trường và thiết bị

1. **Khoảng cách và góc nhìn:**
   - Khoảng cách từ bàn tay tới camera: $0.6\text{m} - 0.9\text{m}$.
   - Vị trí bàn tay: Nằm trọn trong khung hình, diện tích bàn tay chiếm khoảng $15\% - 30\%$ diện tích khung hình.
   - 3 biến thể góc nghiêng:
     - Trực diện ($0^\circ$).
     - Nghiêng chếch trái/phải ($15^\circ - 20^\circ$).
     - Hướng lên/chúi xuống ($10^\circ - 15^\circ$).
2. **Điều kiện ánh sáng:**
   - Kịch bản 1: Ánh sáng phòng tiêu chuẩn (đủ sáng, đồng đều, không bóng gắt).
   - Kịch bản 2: Ánh sáng yếu hoặc ngược sáng nhẹ (kiểm tra độ bền của MediaPipe landmark extractor).
3. **Phông nền (Background):**
   - Phông nền đồng nhất (tường trơn).
   - Phông nền phức tạp (văn phòng có đồ đạc, người di chuyển phía sau).

### 4.4. Quy trình thực hiện cử chỉ và ghi nhận kết quả

- **Số lần lặp lại:** Mỗi chữ cái tĩnh (trong số 22 chữ) được thực hiện **10 lần độc lập** cho mỗi người ký.
- **Thời gian giữ tư thế (Hold Duration):** Mỗi lần ký hiệu giữ tĩnh tối thiểu **5.0 giây** (~150 frames ở 30 FPS).
- **Cử chỉ chuyển tiếp (Transitions):** Giữa mỗi lần lặp lại, người ký phải hạ tay về trạng thái nghỉ (Neutral / Rest pose) trong 2.0 giây để kiểm tra khả năng nhận diện frame rác trong quá trình đưa tay lên.
- **Chỉ số đo lường độc lập:**
  1. **Tỷ lệ nhận diện đúng từng frame (Per-frame Frame Accuracy):** Tỷ lệ % frame trong 5s hold khớp với ground truth.
  2. **Thời gian đạt ổn định (Time to Stable Confirmation):** Số frame/mili-giây từ khi đưa tay lên đến khi có nhãn `confirmed=True`.
  3. **Tỷ lệ dương tính giả khi chuyển tiếp (Transition False Positive Rate):** Số lần phát sinh nhãn chữ cái sai trong 2s chuyển thế.
  4. **Ma trận nhầm lẫn thực tế (Webcam Confusion Matrix):** Xây dựng độc lập cho từng người ký.

---

## PHẦN 5 — ĐỐI CHIẾU THỐNG KÊ FALSE POSITIVE TỪ TASK 7F.3

Trong các phiên thảo luận của Task 7F.3, xuất hiện sự khác biệt giữa hai cách diễn giải số liệu: **6 mẫu False Positive** (trong bảng Part 5) so với **4 mẫu** (ở một số nhận định tổng kết). Kiểm toán Task 7F.4 đã truy xuất trực tiếp ma trận nhầm lẫn và file mẫu `vsl_alphabet_static_v2.npz` để làm rõ nguyên nhân bản chất.

### 5.1. Bảng đối chiếu 6 mẫu False Positive trong tập dữ liệu

Kiểm tra toàn bộ 551 mẫu dữ liệu thực nghiệm cho thấy cả Baseline và Pilot đều ghi nhận **chính xác 6 mẫu lỗi phân loại sai (misclassified samples)**. Chi tiết từng mẫu:

| STT | Index trong Dataset | Nhãn chuẩn (True) | Video nguồn | Nhãn dự đoán (Pred) | Nguồn sinh lỗi | Dự đoán của Static Head | Bản chất kỹ thuật của lỗi |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | `Idx 39` | **B** | D0492 | **O** | `geometric_rule` | **B** (0.90) | **Luật O chiếm nhãn (Rule Hijacking):** Bàn tay ký hiệu B nhưng 4 ngón cong nhẹ, điều kiện `d_TI <= 0.22` và `d_TM <= 0.20` vô tình thỏa mãn Rule O. Static Head dự đoán đúng B nhưng bị Bước 1 đè nhãn. |
| **2** | `Idx 462` | **A** | D0491B | **O** | `geometric_rule` | **A** (0.91) | **Luật O chiếm nhãn:** Ngón cái áp sát ngón trỏ khi nắm tay tạo A, vô tình thỏa mãn ngưỡng khoảng cách của O. Static Head dự đoán đúng A nhưng bị Bước 1 đè nhãn. |
| **3** | `Idx 502` | **O** | D0508N | **E** | `geometric_rule` | **O** (0.95) | **Luật E chiếm nhãn:** Frame kết thúc của cử chỉ O khi các ngón tay gập sát mu bàn tay, thỏa mãn điều kiện `d_TI <= 0.35` và `d_TM <= 0.38` của Rule E. Static Head dự đoán đúng O (0.95). |
| **4** | `Idx 510` | **O** | D0508N | **E** | `geometric_rule` | **O** (0.95) | **Luật E chiếm nhãn:** Tương tự mẫu 502 trong cùng video D0508N. |
| **5** | `Idx 382` | **X** | D0517B | **T** | `geometric_rule` | **X** (0.79) | **Luật T chiếm nhãn:** Ngón trỏ co móc của chữ X có vị trí đầu ngón gần ngón cái, vô tình thỏa mãn `d_TI <= 0.16` của Rule T. Static Head dự đoán đúng X nhưng bị Bước 1 đè nhãn. |
| **6** | `Idx 234` | **N** | D0505 | **M** | `static_head` | **M** (0.89) | **Static Head nhầm lẫn đặc trưng:** Chữ N (ngón cái kẹp dưới 2 ngón) và M (kẹp dưới 3 ngón) có tọa độ không gian 72D cực kỳ gần nhau. Static Head dự đoán nhầm sang M với độ tin cậy 0.89. |

### 5.2. Giải thích căn nguyên sự khác biệt giữa con số "4" và "6"

Sự chênh lệch giữa hai cách diễn giải xuất phát từ góc nhìn thống kê:

1. **Cách đếm theo Mẫu dữ liệu (Sample-level Count) $\rightarrow$ 6 mẫu:**
   Tổng số frame trong 551 frame mà hệ thống đưa ra nhãn khác với ground truth và khác `NEUTRAL` là **6 frame** ($39, 234, 382, 462, 502, 510$).
2. **Cách đếm theo Lớp nhận nhầm (Target Class Count) $\rightarrow$ 4 lớp:**
   Xét ma trận nhầm lẫn theo cột (các lớp bị gán nhãn sai):
   - Cột `O`: Nhận 2 False Positives (1 từ `A`, 1 từ `B`).
   - Cột `E`: Nhận 2 False Positives (2 từ `O`).
   - Cột `M`: Nhận 1 False Positive (1 từ `N`).
   - Cột `T`: Nhận 1 False Positive (1 từ `X`).
   $\rightarrow$ Chỉ có đúng **4 chữ cái đích** bị phát sinh False Positive là $\{ \text{"E"}, \text{"O"}, \text{"M"}, \text{"T"} \}$.
3. **Cách đếm theo Cặp nhầm lẫn (Confusion Pairs) $\rightarrow$ 4 hoặc 5 cặp:**
   Nếu gộp cụm nhầm lẫn hình học sang O thành một nhóm "Hijacked into O", ta có 4 kịch bản nhầm lẫn:
   (1) $O \rightarrow E$, (2) $\{A, B\} \rightarrow O$, (3) $X \rightarrow T$, (4) $N \rightarrow M$.

### 5.3. Kết luận cốt lõi về độ an toàn của cấu hình Pilot

- **5/6 mẫu False Positive bắt nguồn từ Luật hình học:** Luật hình học có tính ưu tiên tuyệt đối nhưng ngưỡng khoảng cách Euclidean đơn giản dễ bị vi phạm tại các tư thế biên.
- **Static Head không gây thêm lỗi nào trong Pilot:** Cấu hình pilot (mở fallback cho `B, C, T, L, Y` ở 0.70 và hạ `Q` xuống 0.75) tạo ra **0 mẫu False Positive mới** trên toàn bộ 551 frame.
- Cả 6 mẫu False Positive này là thuộc tính cố hữu của hệ thống hiện tại, không phải do cấu hình pilot tạo ra.

---

## PHẦN 6 — PHÂN TÍCH RỦI RO VÀ BIỆN PHÁP GIẢM THIỂU

| Nguy cơ tiềm ẩn trên Webcam | Cơ chế phát sinh | Mức độ rủi ro | Biện pháp kiểm soát & Giảm thiểu |
| :--- | :--- | :---: | :--- |
| **1. Nhận diện nhầm trong frame chuyển động (Transition frames)** | Khi người dùng di chuyển tay để tạo hình, các tư thế trung gian có thể vô tình kích hoạt Static Head B/C/T/L/Y. | Trung bình | Bộ đệm làm mượt thời gian `prediction_history` (kích thước 10 frame) và cơ chế yêu cầu tối thiểu $50\%$ số frame đồng thuận trước khi confirm sẽ lọc sạch các frame chuyển tiếp tức thời (< 150ms). |
| **2. Hạ ngưỡng Q (0.85 $\rightarrow$ 0.75) gây dương tính giả** | Trên webcam, góc nhìn nghiêng hoặc bàn tay co quắp tự nhiên có thể tạo đặc trưng giống chữ Q nhìn từ trên xuống. | Thấp – Trung bình | Kiểm tra chặt chẽ trên unseen signers trong kịch bản nhiều góc nghiêng. Nếu phát hiện FP với chữ X, lập tức điều chỉnh ngưỡng Q về 0.80. |
| **3. Cạnh tranh giữa Fallback và Luật hình học** | Người dùng ký hiệu chữ khác (ví dụ O), nhưng góc máy làm mất điều kiện Rule O $\rightarrow$ Static Head nhận nhầm sang C hoặc B. | Thấp | Ngưỡng Fallback 0.70 là ngưỡng kiểm soát chặt chẽ. Kết quả offline Task 7F.3 đã chứng minh 0 trường hợp xung đột mới. |
| **4. Rủi ro chữ I phá hủy FSM** | Nếu đưa I vào Fallback, ngón út mở khi ký hiệu Ơ/Ư sẽ bị Rule I hoặc Static Head I chiếm nhãn. | **CỰC KỲ CAO** | **TÁI KHẲNG ĐỊNH: TUYỆT ĐỐI KHÔNG ĐƯỢC ĐƯA CHỮ I VÀO FALLBACK.** Giữ I hoàn toàn ở nhận diện hình học đơn lẻ. |
| **5. Rủi ro giảm ngưỡng chữ N** | Hạ ngưỡng N từ 0.75 xuống 0.70 sẽ làm tăng nhầm lẫn với chữ M. | Cao | **TÁI KHẲNG ĐỊNH: GIỮ NGUYÊN NGƯỠNG N TẠI 0.75.** Chấp nhận tỷ lệ nhận diện thấp hơn một chút để bảo toàn độ đặc hiệu. |

### Tiêu chí Dừng Khẩn cấp (Circuit Breaker Criteria)

Trong quá trình thực nghiệm webcam pilot, hệ thống phải tự động ngắt hoặc nhóm kỹ thuật phải lập tức rollback cấu hình pilot nếu xuất hiện bất kỳ điều kiện nào sau đây:
1. Tỷ lệ False Positive trung bình trên nhóm Unseen Signers vượt quá **$10\%$** tổng số lần thử nghiệm.
2. Phát sinh hiện tượng chữ cái fallback (`B, C, T, L, Y`) chiếm nhãn của các chữ cái nền tạo dấu (`A, E, O, U, D`) trong quá trình giữ tư thế chuẩn.
3. Xuất hiện trên **$3$ lần** FSM bị reset ngoài ý muốn khi đang thực hiện chuỗi cử chỉ tạo dấu của 7 chữ động.

---

## PHẦN 7 — KẾT LUẬN READINESS

Dựa trên toàn bộ kết quả kiểm toán mã nguồn, cơ chế toggle và hệ thống ghi log, Antigravity kết luận trạng thái chuẩn bị cho thử nghiệm webcam như sau:

$$\mathbf{K\hat{E}T\ LU\hat{A}N:\ NEEDS\_LOGGING\_FIRST}$$

### Lý do kỹ thuật:
1. **Mã nguồn nhận diện đã rõ ràng và logic pilot đã được chứng minh an toàn:** Cấu hình pilot (`Q: 0.75`, Fallback `B, C, T, L, Y: 0.70`, loại trừ `I`, giữ nguyên `N: 0.75`) đã được kiểm toán 100% trong Task 7F.3 với kết quả tăng ròng +20 mẫu và 0 False Positive mới.
2. **Tuy nhiên, hạ tầng logging hiện tại chưa đủ khả năng hỗ trợ pilot:**
   - Hàm `_log_diagnostic` hiện tại đang bóp nghẹt tần số ghi xuống ~2.2 FPS (bỏ rơi 93% dữ liệu webcam).
   - Log bị hardcode bỏ qua Static Head (`has_mlp=False`, `nn_sign="NEUTRAL"`).
   - Log serializer chưa ghi lại `prediction_source`, `head_top_k`, hay `reject_reason` cho bảng chữ cái.
   - Chưa có giao diện hoặc harness để truyền và ghi nhận `expected_sign` một cách có hệ thống trên webcam trực tiếp.
3. Nếu triển khai pilot ngay khi chưa nâng cấp logging, chúng ta sẽ không thể thu thập được dữ liệu định lượng chính xác để tính toán Confusion Matrix, F1-score và độ ổn định thời gian thực trên các người ký mới.

---

## KẾ HOẠCH HÀNH ĐỘNG TIẾP THEO

1. **Task 7F.5 — Triển khai Nâng cấp Logging & Cấu hình Runtime Toggle (Pilot Harness):**
   - Bổ sung tham số runtime toggle `enable_pilot_extended_fallback` và `allowed_fallback_letters` vào `VSLRecognizer`.
   - Nâng cấp `_log_diagnostic` để ghi đầy đủ: `prediction_source`, `raw_predicted_letter`, `head_conf`, `head_top_k`, `reject_reason` và hỗ trợ chế độ log unthrottled (30 FPS) khi chạy pilot.
   - Xây dựng script hoặc endpoint thu thập dữ liệu webcam theo protocol định sẵn kèm `expected_sign`.
2. **Task 7F.6 — Tiến hành Controlled Webcam Pilot:**
   - Thực hiện kiểm thử trên 3–5 signers (bao gồm ít nhất 2 unseen signers) theo đúng protocol tại Phần 4.
   - Thu thập log JSONL và xuất báo cáo đối soát thực tế giữa Webcam và Offline Replay.
3. **Task 7F.7 — Đánh giá Quyết định Merge Production:**
   - Dựa trên ma trận nhầm lẫn webcam từ Task 7F.6 để chính thức quyết định chuyển cấu hình pilot thành cấu hình mặc định của hệ thống.
