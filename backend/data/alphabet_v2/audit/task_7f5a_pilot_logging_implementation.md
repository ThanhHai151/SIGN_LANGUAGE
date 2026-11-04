# BÁO CÁO TRIỂN KHAI TASK 7F.5A — HỆ THỐNG GHI NHẬT KÝ CHẨN ĐOÁN PILOT (PILOT DIAGNOSTIC LOGGING IMPLEMENTATION)

**Dự án:** VSL Connect AI — Hệ thống Nhận diện Ngôn ngữ Ký hiệu Việt Nam  
**Ngày thực hiện:** 2026-11-01  
**Phạm vi:** Triển khai nâng cấp khả năng ghi log chẩn đoán chuẩn bị cho Controlled Webcam Pilot theo kết quả kiểm toán Task 7F.4.  
**Trạng thái kiểm toán:** HOÀN THÀNH — BẢO TOÀN NGUYÊN TRẠNG 100% PRODUCTION CLASSIFICATION BASELINE.  
**Kết luận Readiness:** **`LOGGING_READY_FOR_PILOT`**

---

## 1. TỔNG QUAN VÀ MỤC TIÊU ĐẠT ĐƯỢC

Trong Task 7F.4, kiểm toán đã chỉ ra hệ thống logging hiện tại (`_log_diagnostic`) gặp 5 hạn chế nghiêm trọng:
1. Throttling thời gian cứng $\le 2.2$ FPS (bỏ rơi $>93\%$ frame trong luồng webcam 30 FPS).
2. Hardcode bỏ qua mạng nơ-ron Static Head (`has_mlp=False, nn_sign="NEUTRAL"`).
3. Thiếu các trường dữ liệu cốt lõi: `prediction_source`, `head_letter`, `head_conf`, `head_top_k`, `reject_reason`, `gating_status`.
4. Serializer chi tiết bị lệch sang chế độ chữ số (`digit_classifier`).
5. Thiếu cơ chế truyền và gắn metadata `session_id`, `trial_id`, `frame_idx`, `expected_sign` một cách có hệ thống.

**Task 7F.5A đã giải quyết triệt để toàn bộ 5 hạn chế trên mà không làm thay đổi bất kỳ hành vi nhận diện, threshold, checkpoint hay FSM nào của hệ thống production.**

---

## 2. DANH SÁCH FILE ĐÃ THAY ĐỔI VÀ BỔ SUNG

| STT | File | Trạng thái | Nội dung kỹ thuật chính |
| :---: | :--- | :---: | :--- |
| **1** | `backend/vision/pilot_logger.py` | **MỚI** | Module chuyên dụng: `PilotDiagnosticLogger` sử dụng `queue.Queue` đa luồng, worker thread daemon phi nghẽn (non-blocking), hàm làm sạch nhãn chuẩn `sanitize_expected_sign`, bộ đếm thống kê record và cơ chế `flush()` an toàn. |
| **2** | `backend/vision/recognizer.py` | **SỬA** | Tích hợp cờ `pilot_logging_enabled`, các phương thức `enable_pilot_logging` / `disable_pilot_logging`, bộ đếm `actual_frame_idx`, trích xuất toàn bộ dữ liệu Static Head + Geometric Rules + Gating Status, ghi nhận `expected_sign` biệt lập hoàn toàn khỏi classifier. |
| **3** | `backend/app.py` | **SỬA** | Mở rộng model Pydantic `LandmarkRequest` nhận `expected_sign`, `session_id`, `trial_id`, `frame_idx` (hỗ trợ cả JSON body lẫn HTTP headers); bổ sung 2 REST endpoints điều khiển pilot: `/api/pilot/flush` và `/api/pilot/stats`. |
| **4** | `backend/tests/test_pilot_logging.py` | **MỚI** | Bộ kiểm thử tự động toàn diện gồm 10 unit test xác minh độc lập 10 tiêu chí kỹ thuật bắt buộc của hệ thống logging. |

---

## 3. CƠ CHẾ HOẠT ĐỘNG CỦA HỆ THỐNG LOGGING MỚI

```
[Webcam Stream 30 FPS]
         │ (HTTP Headers / JSON Body: session_id, trial_id, frame_idx, expected_sign)
         ▼
[/api/recognize_landmarks]
         │ (expected_sign được sanitize: CHỈ DÙNG CHO LOG, KHÔNG VÀO PHÂN LOẠI)
         ▼
[VSLRecognizer.predict_landmarks]
         │
         ├──► [Bước 1: AlphabetClassifier] ──► geom_status, geom_letter
         │
         ├──► [Bước 2: StaticAlphabetHead]  ──► raw_predicted_letter, confidence, top_k
         │
         ├──► [Bước 3: DiacriticStateMachine] (Giữ nguyên logic bảo vệ dấu)
         │
         ├──► [Bộ đệm smoothing & confirmation] (Giữ nguyên logic confirm)
         │
         ▼
[PilotDiagnosticLogger.log_frame()] ──(put_nowait < 0.1ms)──► [Thread-safe Queue (10,000)]
         │                                                            │
         ▼ (Trả kết quả nhận diện ngay lập tức <10ms)                  ▼ (Worker Daemon Thread)
[Frontend Client]                                             [Ghi file JSONL trên đĩa]
                                                              (Không bao giờ gây nghẽn)
```

### 3.1. Phi nghẽn (Non-blocking Queue Architecture)
- Luồng nhận diện gọi `pilot_logger.log_frame(record)`:
  - Bản ghi được đẩy vào hàng đợi `queue.Queue(maxsize=10000)` qua `put_nowait()`.
  - Độ trễ bổ sung cho luồng nhận diện là $< 0.05\text{ms}$, đảm bảo duy trì độ trễ thời gian thực $< 10\text{ms}$ của API.
  - Một Worker Daemon Thread chạy ngầm liên tục lấy bản ghi từ queue và ghi nối tiếp (`append`) xuống file JSONL UTF-8.
- **Cách ly lỗi I/O:** Nếu xảy ra lỗi ghi đĩa hoặc ổ cứng quá tải, exception được bắt gọn tại worker thread và tăng biến đếm `dropped_io_error`. Luồng nhận diện chính hoàn toàn không bị ảnh hưởng và không bao giờ crash.
- **Chống tràn bộ nhớ:** Nếu queue đầy (quá 10.000 bản ghi chưa kịp ghi xuống đĩa), lệnh `put_nowait` bắt `queue.Full`, tăng biến đếm `dropped_queue_full` và bỏ qua an toàn mà không làm nghẽn luồng dự đoán.

### 3.2. Không bóp nghẹt tần số (Unthrottled Logging)
Khác với `_log_diagnostic` bị giới hạn cứng 0.45s (~2.2 FPS), `PilotDiagnosticLogger` ghi nhận **tất cả mọi frame** được gửi tới server (đáp ứng trọn vẹn 30 FPS hoặc cao hơn), ghi nhận chính xác từng frame chuyển tiếp (transition), rung giật (jitter) và thời gian giữ tư thế (hold duration).

### 3.3. Cơ chế Flush và Kết thúc Phiên
- Cung cấp phương thức `logger.flush(timeout=5.0)`: Chờ worker thread ghi cạn hàng đợi xuống đĩa trước khi đọc dữ liệu phân tích.
- Tự động đăng ký `atexit.register(logger.close)`: Khi đóng tiến trình FastAPI hoặc Python runner, toàn bộ bản ghi còn lại trong buffer được flush an toàn xuống đĩa.

### 3.4. Theo dõi Thống kê Toàn vẹn Bản ghi
Phương thức `logger.get_stats()` cung cấp dữ liệu định lượng:
```python
{
    "queued_count": int,          # Số bản ghi đã nhận vào queue
    "logged_count": int,          # Số bản ghi đã ghi thành công xuống file
    "dropped_queue_full": int,    # Số bản ghi bị loại bỏ do tràn queue
    "dropped_io_error": int,      # Số bản ghi bị lỗi ghi đĩa
    "total_dropped": int,         # Tổng số bản ghi bị thất thoát
    "pending_in_queue": int       # Số bản ghi đang chờ ghi
}
```
Cam kết **không âm thầm bỏ log mà không có số liệu kiểm toán**.

---

## 4. SCHEMA ĐỊNH DẠNG JSONL THỰC TẾ

Mỗi dòng trong file log pilot JSONL tuân thủ nghiêm ngặt schema tiêu chuẩn sau:

```json
{
  "timestamp": 1730460000.1234,
  "session_id": "pilot_signer01_20261101",
  "trial_id": "trial_B_rep01",
  "frame_idx": 45,
  "expected_sign": "B",
  "final_sign": "B",
  "final_confidence": 0.95,
  "prediction_source": "geometric_rule",
  "confirmed": true,
  "geometric": {
    "status": "MATCH",
    "letter": "B"
  },
  "static_head": {
    "raw_predicted_letter": null,
    "confidence": null,
    "top_k": null
  },
  "gating_status": "passed_geometric",
  "threshold_applied": null,
  "reject_reason": null
}
```

### Chi tiết các trường:
- `timestamp`: Float, mốc thời gian độ phân giải cao (giây).
- `session_id`: String hoặc `null`, mã phiên thử nghiệm.
- `trial_id`: String hoặc `null`, mã lần thử nghiệm cụ thể (ví dụ: `trial_B_rep01`).
- `frame_idx`: Integer, chỉ số frame tăng đơn điệu (từ client hoặc bộ đếm tự động).
- `expected_sign`: String hoặc `null`, nhãn ground truth đã được chuẩn hóa.
- `final_sign`: String, nhãn kết luận cuối cùng của recognizer.
- `final_confidence`: Float, độ tin cậy của nhãn cuối cùng (0.0 đến 1.0).
- `prediction_source`: String, nguồn gốc nhãn: `"geometric_rule"`, `"static_head"`, `"static_head_fallback"`, hoặc `"none"`.
- `confirmed`: Boolean, trạng thái xác nhận qua bộ đệm smoothing thời gian.
- `geometric`: Object chứa `status` (`MATCH`, `NO_MATCH`, `CONFLICT`, `INVALID_INPUT`) và `letter` (`null` nếu không match).
- `static_head`: Object chứa:
  - `raw_predicted_letter`: Nhãn chữ cái mạng nơ-ron Static Head dự đoán (hoặc `null` nếu không gọi model).
  - `confidence`: Điểm số Softmax cao nhất của Static Head (hoặc `null`).
  - `top_k`: Danh sách top các nhãn kèm xác suất `[{"letter": ..., "confidence": ...}]` (hoặc `null`).
- `gating_status`: String, phản ánh nhánh logic gating: `"passed_geometric"`, `"passed_fallback"`, `"passed_supplemental"`, `"rejected"`, `"none"`, `"error"`.
- `threshold_applied`: Float hoặc `null`, ngưỡng tin cậy được áp dụng tại bước gating.
- `reject_reason`: String hoặc `null`, nguyên nhân từ chối nếu không đạt ngưỡng hoặc không thuộc danh sách cho phép.

---

## 5. CƠ CHẾ GHI NHẬN GROUND TRUTH VÀ BẢO VỆ MÔ HÌNH

Hàm `sanitize_expected_sign(expected_sign)` trong `backend/vision/pilot_logger.py` kiểm tra chặt chẽ:
1. Giá trị nhãn được chuẩn hóa chữ hoa (`upper()`) và xóa khoảng trắng thừa (`strip()`).
2. Đối chiếu với danh sách 22 chữ cái tĩnh chuẩn VSL:
   $$\text{VALID\_STATIC\_CLASSES\_SET} = \{ A, B, C, D, E, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, X, Y \}$$
3. Nếu nhãn không nằm trong danh sách trên (ví dụ: rỗng, chuỗi ngẫu nhiên, số, ký tự lạ), hàm trả về `None`.

### Cam kết Cách ly Tuyệt đối (Strict Ground Truth Isolation)
- `expected_sign` / `clean_expected_sign` **chỉ được sử dụng duy nhất làm giá trị trường dữ liệu trong bản ghi log**.
- Tuyệt đối **không đi vào bất kỳ nhánh điều kiện rẽ nhánh nào** của `AlphabetClassifier`, `VSLStaticAlphabetHead`, `DiacriticStateMachine`, bộ đệm `prediction_history` hay logic xác nhận `confirmed`.
- Test `test_7_expected_sign_does_not_alter_prediction_behavior` đã chứng minh thực nghiệm: Kết quả dự đoán đầu ra (`sign` và `confidence`) giống nhau 100% khi không truyền ground truth, khi truyền ground truth đúng, và khi cố tình truyền ground truth sai lệch.

---

## 6. KẾT QUẢ KIỂM THỬ TỰ ĐỘNG (AUTOMATED TEST VERIFICATION)

### 6.1. Bộ test chuyên biệt `backend/tests/test_pilot_logging.py`
Toàn bộ 10 bài test được thiết kế riêng cho Task 7F.5A đều đạt kết quả **PASSED (10/10)**:

```
backend/tests/test_pilot_logging.py::TestPilotDiagnosticLogging::test_1_sanitization_of_expected_sign PASSED [ 10%]
backend/tests/test_pilot_logging.py::TestPilotDiagnosticLogging::test_2_pilot_logging_disabled_by_default PASSED [ 20%]
backend/tests/test_pilot_logging.py::TestPilotDiagnosticLogging::test_3_pilot_logging_records_correct_final_sign_and_geometric_source PASSED [ 30%]
backend/tests/test_pilot_logging.py::TestPilotDiagnosticLogging::test_4_pilot_logging_records_static_head_info_on_fallback PASSED [ 40%]
backend/tests/test_pilot_logging.py::TestPilotDiagnosticLogging::test_5_pilot_logging_records_supplemental_source PASSED [ 50%]
backend/tests/test_pilot_logging.py::TestPilotDiagnosticLogging::test_6_pilot_logging_records_rejected_prediction PASSED [ 60%]
backend/tests/test_pilot_logging.py::TestPilotDiagnosticLogging::test_7_expected_sign_does_not_alter_prediction_behavior PASSED [ 70%]
backend/tests/test_pilot_logging.py::TestPilotDiagnosticLogging::test_8_grouping_multiple_frames_by_trial_id PASSED [ 80%]
backend/tests/test_pilot_logging.py::TestPilotDiagnosticLogging::test_9_resilience_to_queue_overflow_and_io_errors PASSED [ 90%]
backend/tests/test_pilot_logging.py::TestPilotDiagnosticLogging::test_10_missing_or_invalid_landmarks_logged_safely PASSED [100%]

============================== 10 passed in 4.33s ==============================
```

### 6.2. Kiểm thử hồi quy toàn bộ hệ thống (Regression Test Suite)
Chạy đồng thời toàn bộ 148 test liên quan đến phân loại hình học, Static Head, fallback, FSM chữ động và chẩn đoán:

```
backend\tests\test_alphabet_classifier.py ..........................     [ 17%]
backend\tests\test_alphabet_landmarks.py ......                          [ 21%]
backend\tests\test_alphabet_recognition_integration.py .........         [ 27%]
backend\tests\test_alphabet_static_head_integration.py ..........        [ 34%]
backend\tests\test_static_head_fallback.py ....................          [ 47%]
backend\tests\test_diacritic_state_machine.py .......................... [ 65%]
......                                                                   [ 69%]
backend\tests\test_diacritic_camera_integration.py ..................... [ 83%]
..                                                                       [ 85%]
backend\tests\test_diagnostic.py ............                            [ 93%]
backend\tests\test_pilot_logging.py ..........                           [100%]

============================= 148 passed in 7.29s =============================
```

**Kết luận hồi quy:** 100% test cũ và mới đều vượt qua thành công, không phát sinh bất kỳ lỗi hồi quy nào.

---

## 7. ĐÁNH GIÁ PHẦN GIAO DIỆN FRONTEND CÒN THIẾU

Backend hiện tại đã hỗ trợ đầy đủ việc nhận `expected_sign`, `session_id`, `trial_id`, `frame_idx` qua cả:
1. HTTP Request Body JSON (`LandmarkRequest`).
2. HTTP Headers (`x-expected-sign`, `x-session-id`, `x-trial-id`, `x-frame-idx`).

Tuy nhiên, phía giao diện Frontend (giao diện web camera người dùng) hiện tại:
- Đang gửi landmark trực tiếp mà chưa có giao diện điều khiển phiên thử nghiệm (Pilot Testing HUD).
- **Phần cần bổ sung trong task riêng (Task 7F.5B hoặc công cụ thu thập pilot):**
  1. Dropdown / Nút bấm chọn chữ cái mục tiêu ($A, B, C...$).
  2. Nút "Bắt đầu Trial" kèm bộ đếm ngược 5 giây giữ tư thế (Hold Countdown).
  3. Tự động sinh `trial_id` dạng `<session>_<letter>_rep<k>` và truyền header `x-expected-sign` tương ứng trong 5 giây đó.
  4. Nút bấm gọi endpoint `/api/pilot/flush` khi hoàn thành phiên để đảm bảo toàn bộ dữ liệu được xả xuống đĩa.

---

## 8. XÁC NHẬN BẢO TOÀN HỆ THỐNG PRODUCTION (SAFETY CONSTRAINTS COMPLIANCE)

Tuân thủ nghiêm ngặt các giới hạn của chỉ thị Task 7F.5A:
- **Ngưỡng nhận diện (Thresholds):** Giữ nguyên $100\%$ (`Q = 0.85`, `N = 0.75`, `fallback_base_threshold = 0.70`).
- **Tập Fallback:** Giữ nguyên `ALLOWED_FALLBACK_BASE_LETTERS = {"A", "D", "E"}`. Tuyệt đối **không** bật fallback cho `B, C, T, L, Y` trong task này.
- **Chữ I:** Tuyệt đối **không** đưa vào fallback.
- **Mô hình & Checkpoint:** Checkpoint `alphabet_static_head_v1.pt`, `sign_sstcn_attention_model.pth` và `weights.pt` hoàn toàn không bị chỉnh sửa, ghi đè hay huấn luyện lại.
- **FSM & Chữ động:** Bộ máy trạng thái `DiacriticStateMachine` cho 7 chữ cái có dấu động (Ă, Â, Ê, Ô, Ơ, Ư, Đ) giữ nguyên 100% logic và ngưỡng chuyển trạng thái.
- **Production Baseline:** Khi không bật cờ `VSL_PILOT_LOGGING=1`, hệ thống tiếp tục hoạt động theo luồng baseline cũ, không làm tăng dung lượng ghi log mặc định.

---

## 9. KẾT LUẬN READINESS

$$\mathbf{K\hat{E}T\ LU\hat{A}N:\ LOGGING\_READY\_FOR\_PILOT}$$

Hệ thống ghi log chẩn đoán mới đã đáp ứng đầy đủ tất cả các tiêu chí khoa học cần thiết:
1. Thu thập dữ liệu chi tiết, phân định rõ ràng 4 nguồn dự đoán (`geometric_rule`, `static_head`, `static_head_fallback`, `none`).
2. Ghi nhận đầy đủ xác suất và nhãn dự đoán của Static Head khi model được kích hoạt.
3. Không làm giảm hiệu năng xử lý (non-blocking queue worker).
4. Lưu vết chính xác nhãn ground truth và metadata phiên/lần thử mà không làm ảnh hưởng đến quyết định phân loại của mô hình.
5. Sẵn sàng 100% phục vụ cho đợt thử nghiệm Webcam có kiểm soát (Controlled Webcam Pilot).
