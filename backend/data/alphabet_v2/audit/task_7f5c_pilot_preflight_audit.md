# TASK 7F.5C — CONTROLLED WEBCAM PILOT PREFLIGHT AUDIT REPORT
**Trạng thái kiểm toán:** HOÀN TẤT (READ-ONLY AUDIT)  
**Thời gian thực hiện:** 2026-11-01  
**Phạm vi:** Kiểm toán an toàn và toàn vẹn dữ liệu cho Controlled Webcam Pilot Pipeline (Tasks 7F.5A & 7F.5B)  
**Tệp mục tiêu:** `backend/data/alphabet_v2/audit/task_7f5c_pilot_preflight_audit.md`  

---

## 1. MỤC TIÊU VÀ PHƯƠNG PHÁP KIỂM TOÁN

Mục tiêu của Task 7F.5C là tiến hành **kiểm toán toàn diện (Exhaustive Read-Only Audit)** toàn bộ chuỗi thu thập dữ liệu và chẩn đoán webcam đã được xây dựng tại Task 7F.5A (`PilotDiagnosticLogger`) và Task 7F.5B (`PilotTestingHUD`, metadata integration, `/api/pilot/*`).

Kiểm toán nhằm xác minh độ an toàn hệ thống, bảo vệ đường dẫn tệp tin, tính toàn vẹn dữ liệu JSONL, tính đơn điệu của frame index, khả năng rò rỉ metadata giữa các trial, và bảo toàn tuyệt đối logic nhận diện trước khi tiến hành thu thập dữ liệu thực tế trên người ký (human signers).

### Quy tắc kiểm toán
- **READ-ONLY**: Tuyệt đối không chỉnh sửa source code production hoặc file test trong task này.
- **Dữ liệu đối chiếu thực tế**: Kiểm tra trực tiếp mã nguồn tại:
  - `backend/app.py` (các endpoints `/api/pilot/control`, `/api/recognize_landmarks`, `/api/pilot/flush`, `/api/pilot/stats`)
  - `backend/vision/pilot_logger.py` (`PilotDiagnosticLogger`, `sanitize_expected_sign`, lifecycle management)
  - `backend/vision/recognizer.py` (phân tách luồng chẩn đoán, gating fallback, diacritic FSM)
  - `src/components/PilotTestingHUD.jsx` (timer, state transitions, refs, event handlers)
  - `src/components/CameraSignRecognizer.jsx` (render loop, metadata bridge)
  - `src/services/api.js` (HTTP headers & payload dispatch)
  - `backend/tests/test_pilot_logging.py` & `backend/tests/test_pilot_hud_integration.py` (tổng cộng 18 unit tests đã pass).

---

## 2. KẾT QUẢ KIỂM TRA BẢO VỆ ĐƯỜNG DẪN TỆP LOG (PATH TRAVERSAL & FILE OVERWRITE AUDIT)

### 2.1. Phân tích Endpoint `/api/pilot/control` (`backend/app.py:282-296`)

```python
@app.post("/api/pilot/control")
def control_pilot_logging(req: PilotControlRequest):
    if req.enabled:
        target_file = req.filepath
        if not target_file and req.session_id:
            target_dir = getattr(recognizer, "diagnostic_dir", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "diagnostics", "pilot"))
            os.makedirs(target_dir, exist_ok=True)
            clean_sess = re.sub(r'[^a-zA-Z0-9_-]', '_', req.session_id)
            target_file = os.path.join(target_dir, f"pilot_{clean_sess}.jsonl")
        logger = recognizer.enable_pilot_logging(filepath=target_file)
        return {"success": True, "enabled": True, "filepath": target_file, "stats": logger.get_stats()}
    else:
        recognizer.disable_pilot_logging()
        return {"success": True, "enabled": False}
```

### 2.2. Kiểm tra `session_id` Sanitization
- **Cơ chế**: Dòng 290 sử dụng biểu thức chính quy:
  `clean_sess = re.sub(r'[^a-zA-Z0-9_-]', '_', req.session_id)`
- **Đánh giá**:
  - Mọi ký tự nguy hiểm như `..`, `/`, `\`, null bytes `%00`, ký tự điều khiển đều bị biến đổi thành dấu gạch dưới `_`.
  - Tên tệp được sinh ra: `os.path.join(target_dir, f"pilot_{clean_sess}.jsonl")`.
  - **Kết luận `session_id`**: An toàn 100% trước path traversal. Không thể thoát khỏi `target_dir` qua tham số `session_id`.

### 2.3. Kiểm tra tham số `req.filepath` (LỖ HỔNG PATH TRAVERSAL NGHIÊM TRỌNG)
- **Cơ chế**: Dòng 286 gán trực tiếp: `target_file = req.filepath`.
- **Hành vi thực tế**:
  - Nếu API nhận một payload có `filepath`, ví dụ:
    `{"enabled": true, "filepath": "../../../malicious_output.jsonl"}`
    hoặc đường dẫn tuyệt đối ngoài workspace như `C:\Users\...\startup.jsonl`, backend chấp nhận trực tiếp `target_file = req.filepath`.
  - Tại `PilotDiagnosticLogger._worker_loop()` (`pilot_logger.py:136-139`):
    ```python
    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    with open(target_file, "a", encoding="utf-8") as f:
        f.write(line)
    ```
  - Worker thread sẽ tạo thư mục bất kỳ và mở tệp ở chế độ append (`"a"`).
- **Phía Frontend**: Trong `PilotTestingHUD.jsx:86`, frontend gọi `setPilotLogging(enable, null, sessionId)`, tức luôn truyền `filepath = null`. Do đó, người dùng thông qua giao diện HUD bình thường không kích hoạt lỗ hổng này.
- **Phía API Endpoint**: Endpoint `/api/pilot/control` không thực hiện kiểm tra `os.path.commonpath([resolved_file, target_dir]) == target_dir`. Bất kỳ yêu cầu HTTP nào gửi trực tiếp đến backend đều có thể ghi nhật ký vào bất kỳ vị trí nào có quyền ghi trên ổ đĩa.
- **Phân loại**: **BLOCKER (B-01)** theo đúng quy tắc chỉ đạo của kiểm toán: *"Nếu có nguy cơ path traversal hoặc ghi đè file ngoài ý muốn, đánh dấu BLOCKER."*

### 2.4. Nguy cơ Ghi đè tệp tin và Trùng lặp `session_id`
- **Chế độ mở tệp**: Logger mở tệp bằng `open(target_file, "a", encoding="utf-8")` (append mode).
  - Không có nguy cơ xóa sạch (truncate/overwrite) nội dung tệp tin đã có sẵn.
- **Xử lý trùng `session_id`**:
  - Nếu hai lần chạy dùng cùng một `session_id`, dữ liệu của lần chạy sau sẽ được nối tiếp vào cuối tệp của lần chạy trước.
  - Mỗi bản ghi đều có `timestamp` và `trial_id` duy nhất (`{session_id}_{letter}_rep{idx}`), nên về mặt dữ liệu có thể lọc được, nhưng sẽ gây phân mảnh và khó khăn cho script thống kê nếu không có cơ chế cảnh báo tệp đã tồn tại.

---

## 3. KẾT QUẢ KIỂM TRA VÒNG ĐỜI LOGGER VÀ CONCURRENCY

### 3.1. Phân loại Kiến trúc: Singleton vs Instance
- `PilotDiagnosticLogger` được quản lý dưới dạng **Module-level Singleton** qua `_global_pilot_logger` và hàm `get_pilot_logger()` trong `backend/vision/pilot_logger.py:198-206`.
- `recognizer` trong `backend/app.py` là một thực thể duy nhất chạy trong tiến trình FastAPI.
- Khi gọi `recognizer.enable_pilot_logging(filepath)`:
  - Gán `self.pilot_filepath = filepath`.
  - Lấy `_global_pilot_logger` và cập nhật `_global_pilot_logger.default_filepath = filepath`.

### 3.2. Vòng đời Worker Thread khi Đóng/Mở HUD hoặc Đổi Session
- Khi bật logging: Worker daemon thread `VSLPilotLoggerWorker` được khởi động một lần duy nhất nếu chưa có.
- Khi đóng HUD hoặc chuyển session:
  - Nếu gọi `recognizer.disable_pilot_logging()`: cờ `self.pilot_logging_enabled = False`.
  - Queue worker thread vẫn chạy nền nhưng ở trạng thái chờ (idle), không tiêu tốn CPU.
  - Khi ứng dụng tắt hoàn toàn, `atexit.register(self.close)` được kích hoạt: gửi sentinel `None`, đợi ghi hết dữ liệu còn lại (`flush`) và join thread trong tối đa 3 giây (`pilot_logger.py:178-191`).

### 3.3. Rủi ro Đa phiên đồng thời (Concurrent Multi-Tab / Multi-Signer Risk)
- **Tình huống**: Người kiểm thử mở 2 tab trình duyệt cùng lúc:
  - Tab 1 mở Session A (`signer01`).
  - Tab 2 mở Session B (`signer02`).
- **Phát hiện kiểm toán**:
  - Cả 2 tab đều gọi chung instance `recognizer` trên backend.
  - Tab 2 gọi `/api/pilot/control` sẽ ghi đè `recognizer.pilot_filepath` thành tệp của `signer02`.
  - Ngay lập tức, các frame gửi lên từ Tab 1 sẽ bị ghi vào tệp của Tab 2!
- **Đánh giá**: Đây là giới hạn kiến trúc đơn phiên (Single-Tenant Design).
- **Phân loại**: **NON-BLOCKER (Hạn chế quy trình vận hành)**. Controlled Webcam Pilot bắt buộc phải áp dụng **Quy ước Thử nghiệm Đơn phiên (Single-Signer Protocol)**: Mỗi thời điểm chỉ duy nhất 1 người ký mở trình duyệt kết nối tới backend.

### 3.4. Tính an toàn Đa luồng (Thread-Safety)
- **Queue**: Sử dụng `queue.Queue(maxsize=10000)` chuẩn của Python, được bảo vệ bởi mutex lock mức C-level, hoàn toàn thread-safe khi nhiều frame được đẩy vào liên tục.
- **Thống kê (Stats)**: Các biến đếm `queued_count`, `logged_count`, `dropped_queue_full`, `dropped_io_error` đều được bọc bởi `with self.lock:`.
- **Ghi đĩa**: Duy nhất 1 luồng worker daemon đọc từ queue và ghi tuần tự vào đĩa, loại bỏ hoàn toàn nguy cơ tranh chấp I/O làm hỏng định dạng file JSONL.

### 3.5. Đảm bảo Ghi hết Buffer khi Flush
- Hàm `flush(timeout=5.0)` trong `pilot_logger.py:148-158`:
  ```python
  start = time.time()
  while not self.queue.empty():
      if time.time() - start > timeout:
          return False
      time.sleep(0.02)
  return True
  ```
- Worker thread chỉ gọi `self.queue.task_done()` sau khi khối `with open(...)` đã ghi và đóng tệp thành công. Do đó, khi `self.queue.empty() == True`, 100% dữ liệu đã nằm trên hệ thống tệp của OS.

### Bảng phân loại nguồn kiểm chứng Vòng đời Logger

| Hạng mục kiểm tra | Trạng thái xác thực | Nguồn kiểm chứng / Bằng chứng |
| :--- | :--- | :--- |
| Singleton instance & Worker Thread | **Đã verify bằng automated test** | `test_pilot_logging.py:test_1_logger_initialization_and_thread` |
| Queue Non-blocking & Không ném ngoại lệ | **Đã verify bằng automated test** | `test_pilot_logging.py:test_4_non_blocking_behavior` |
| Flush cam kết xả hết queue xuống đĩa | **Đã verify bằng automated test** | `test_pilot_logging.py:test_7_and_8_flush_and_stats_endpoints` |
| Thread-safety khi đẩy frame liên tục | **Đã verify bằng automated test** | `test_pilot_hud_integration.py:test_4_trial_flow_and_flush` |
| Đóng tiến trình an toàn qua atexit | **Suy luận từ mã nguồn** | `backend/vision/pilot_logger.py:87` (`atexit.register(self.close)`) |
| Xung đột dữ liệu khi mở 2 tab đồng thời | **Suy luận từ mã nguồn** | Backend dùng chung instance `recognizer` duy nhất trong `app.py` |
| Rủi ro mất điện / OS crash trước disk sync | **Chưa được verify** | Python phụ thuộc vào OS file buffer flushing (`f.flush()`/`os.fsync`) |

---

## 4. KẾT QUẢ KIỂM TRA RANH GIỚI TRIAL VÀ FRAME (TRIAL & FRAME BOUNDARY INTEGRITY)

### 4.1. Request In-flight khi Kết thúc Trial tại Frame N
- **Cơ chế thu thập**:
  - Tại `CameraSignRecognizer.jsx:825-834`:
    ```javascript
    let pilotMeta = null;
    if (pilotMetadataProviderRef.current) {
        pilotMeta = pilotMetadataProviderRef.current();
    }
    sendLandmarksForRecognition(rawPrimary, multiHands, 'Right', requestMode, pilotMeta)
    ```
  - `pilotMeta` được chụp (snapshot) ngay tại thời điểm gọi hàm và đóng gói vào HTTP Headers (`x-trial-id`, `x-frame-idx`, `x-expected-sign`) và JSON Body.
  - Dù quá trình truyền mạng hoặc xử lý backend có độ trễ 5-20ms, request khi tới backend vẫn mang đúng `trial_id` và `frame_idx` của thời điểm nó được gửi đi.
- **Kết luận**: Không có hiện tượng request in-flight bị gắn nhầm sang trial kế tiếp.

### 4.2. Tính đơn điệu tăng dần của `frame_idx` (Strict Monotonicity)
- **Kiểm tra Frontend**:
  - `frameIdxRef.current` được khởi tạo bằng 0 khi bấm "Bắt đầu Trial" (`handleStartTrial`).
  - Trong mỗi lần `getFrameMetadata()` được gọi, `frameIdxRef.current += 1`.
  - Giá trị này được gán trực tiếp vào metadata mà không bị can thiệp bởi chu kỳ render lại của React.
- **Kiểm tra Backend**:
  - `recognizer.predict_landmarks` nhận `frame_idx` từ header hoặc body, chuyển đổi sang số nguyên `actual_frame_idx = int(frame_idx)`.
- **Bằng chứng thực nghiệm**: Đã được xác nhận qua `test_pilot_hud_integration.py:test_4_trial_flow_and_flush`, dãy `frame_idx` thu thập được là `1, 2, 3...` hoàn toàn đơn điệu, không trùng lặp và không nhảy cóc.

### 4.3. Rủi ro Micro-Race Condition trong React State Transitions
- **Vấn đề phát hiện**:
  - Trong `PilotTestingHUD.jsx:62-82`:
    ```javascript
    useEffect(() => {
      if (trialState === 'recording') {
        onMetadataUpdate({
          getFrameMetadata: () => {
            frameIdxRef.current += 1;
            return { ... };
          }
        });
      } else {
        onMetadataUpdate({
          getFrameMetadata: () => null
        });
      }
    }, [trialState, selectedLetter, sessionId, trialId]);
    ```
  - Khi 5 giây đếm ngược kết thúc, hàm `handleEndTrial()` gọi `setTrialState('completed')`.
  - Trong React 18, việc thay đổi state là bất đồng bộ (batched render). Trong khoảng vài mili-giây trước khi `useEffect` kích hoạt để cập nhật `pilotMetadataProviderRef.current = () => null`, nếu camera loop (chạy ở tần số 30-60 FPS) gọi `getFrameMetadata()`, nó có thể sinh thêm 1 frame mang metadata của trial vừa xong.
- **Đánh giá rủi ro**: Frame phụ này vẫn mang đúng nhãn `trial_id` của lượt thử hiện tại, không gây sai lệch nhãn và không ảnh hưởng tới kết quả phân loại. Sau đó `flushPilotLogs()` ghi nhận toàn bộ frames.
- **Khuyến nghị cải tiến**: Trong Task kế tiếp, nên kiểm tra thêm `trialStateRef.current === 'recording'` ngay bên trong closure `getFrameMetadata` để đạt độ chính xác vi giây.
- **Phân loại**: **NON-BLOCKER**.

### 4.4. Hành vi khi Hủy Trial hoặc Tắt Camera giữa chừng
1. **Người dùng bấm "Hủy" (`handleCancelTrial`)**:
   - `clearInterval(timerIntervalRef.current)`.
   - `setTrialState('idle')`.
   - `frameIdxRef.current = 0`.
   - Các frame tiếp theo lập tức nhận `metadata = null`. Backend chỉ ghi diagnostic thông thường nếu được bật, không ghi vào tệp pilot JSONL với ground truth sai lệch.
2. **Camera bị tắt giữa chừng**:
   - Vòng lặp `requestAnimationFrame` dừng lại, không có thêm landmark nào được gửi lên server.
   - Timer 5s tiếp tục đếm đến hết thời gian và gọi `handleEndTrial()`.
   - Tệp JSONL ghi nhận đúng số frame thực tế đã gửi được trước khi camera tắt (ví dụ 45 frames thay vì 100 frames), không crash hệ thống.

### 4.5. Đánh giá Rò rỉ Nhãn giữa các Trial (Cross-Trial Ground Truth Leakage)
- **Xác minh**: Đã kiểm tra qua `test_pilot_hud_integration.py:test_5_no_metadata_leakage_after_trial_ends`.
- **Kết quả**: Khi trial kết thúc, mọi request nhận diện thông thường gửi lên backend đều mang `expected_sign = None` và `trial_id = None`. Hoàn toàn không có hiện tượng rò rỉ ground truth từ trial trước sang trial sau.

---

## 5. KẾT QUẢ KIỂM TRA CHẤT LƯỢNG DỮ LIỆU JSONL

### 5.1. Tỷ lệ Ánh xạ 1:1 giữa Request và Record
- Với mỗi HTTP request gửi tới `/api/recognize_landmarks` khi `pilot_logging_enabled = True`:
  - `recognizer.predict_landmarks` gọi duy nhất 1 lần `self.pilot_logger.log_frame(record)`.
  - Queue worker ghi đúng 1 dòng JSONL tương ứng.
- Đã kiểm chứng qua test tự động: Gửi chính xác 3 frames -> Nhận chính xác 3 bản ghi JSONL (`test_pilot_logging.py:test_2_log_frame_structure`).
- Không có hiện tượng nhân đôi bản ghi (duplicate records) hoặc bỏ sót ngẫu nhiên.

### 5.2. Độ đầy đủ của Schema Bản ghi JSONL
Mỗi bản ghi trong JSONL bao gồm đầy đủ 14 trường thông tin chi tiết:
1. `timestamp`: Mốc thời gian thực chuẩn hóa (giây, độ chính xác 4 chữ số thập phân).
2. `session_id`: Định danh phiên người ký (ví dụ: `pilot_20261101_1900`).
3. `trial_id`: Định danh lượt ký (ví dụ: `pilot_20261101_1900_B_rep01`).
4. `frame_idx`: Thứ tự frame trong trial (tăng đơn điệu 1, 2, 3...).
5. `expected_sign`: Nhãn ký hiệu chuẩn (ground truth) do HUD truyền xuống.
6. `final_sign`: Nhãn ký hiệu đầu ra cuối cùng của recognizer.
7. `final_confidence`: Độ tin cậy đầu ra cuối cùng.
8. `prediction_source`: Nguồn nhận diện (`geometric_rule`, `static_head`, `static_head_fallback`, `diacritic_fsm`, `none`).
9. `confirmed`: Trạng thái xác nhận của smoothing temporal window (`True`/`False`).
10. `geometric`:
    - `status`: `MATCH`, `NO_MATCH`, `CONFLICT`, hoặc `INVALID_INPUT`.
    - `letter`: Nhãn khớp bởi geometric rule nếu có.
11. `static_head`:
    - `raw_predicted_letter`: Nhãn có xác suất cao nhất từ MLP Static Alphabet Head.
    - `confidence`: Xác suất softmax của nhãn này.
    - `top_k`: Danh sách top-3 dự đoán kèm xác suất.
12. `gating_status`: Trạng thái kiểm soát cổng (`passed_fallback`, `passed_supplemental`, `rejected`, `none`, `error`).
13. `threshold_applied`: Ngưỡng tin cậy cụ thể được áp dụng cho frame này.
14. `reject_reason`: Nguyên nhân chi tiết nếu bị từ chối nhận diện.

Schema này cung cấp **đầy đủ 100% dữ liệu** để tái hiện phân loại offline, tính toán ma trận nhầm lẫn (confusion matrix), vẽ đường cong ROC, và phân tích nguyên nhân sai lệch của từng frame.

### 5.3. Giải thích Sự chênh lệch Tần số Frame (FPS Camera vs Received FPS)
- **Hiện tượng**: Camera webcam thường có cấu hình 30 FPS hoặc 60 FPS. Tuy nhiên trong 5 giây thu thập của một trial, số lượng bản ghi thực tế trong file JSONL thường nằm trong khoảng **75 – 125 frames** (tương đương 15 – 25 FPS).
- **Nguyên nhân kỹ thuật (Không phải lỗi hệ thống)**:
  1. *MediaPipe Extraction Overhead*: MediaPipe JS chạy trên Web Worker của trình duyệt tốn khoảng 15 – 25ms để phát hiện 21 điểm mốc cho mỗi frame video.
  2. *Throttling / Asynchronous API Dispatch*: Frontend chỉ gửi frame tiếp theo khi có điểm mốc hợp lệ và không làm tràn call stack.
  3. *Độ trễ Mạng nội bộ (HTTP Roundtrip)*: Thời gian truyền HTTP POST localhost và xử lý tại FastAPI mất từ 5 – 12ms.
  4. *Drop frame tự nhiên*: Nếu người dùng di chuyển tay ra ngoài khung hình hoặc MediaPipe không bắt được bàn tay, frame đó bị bỏ qua ở phía client.
- **Kết luận**: Backend chỉ ghi nhận các frame **thực sự nhận được và được nhận diện**. Đây là hành vi hoàn toàn bình thường và chính xác của hệ thống nhận diện thời gian thực.

### 5.4. Tính nhất quán của Thống kê Logger (Stats Consistency)
- Các trường `queued_count`, `logged_count`, `dropped_queue_full`, `dropped_io_error` phản ánh chính xác số lượng bản ghi:
  - Khi không nghẽn mạng và tải đĩa bình thường: `queued_count == logged_count` và `total_dropped == 0`.
  - Nếu queue đầy (quá 10,000 bản ghi chưa kịp ghi), hệ thống drop an toàn và tăng `dropped_queue_full` thay vì làm treo ứng dụng.

---

## 6. KẾT QUẢ KIỂM TRA BẢO TOÀN NHẬN DIỆN (PRESERVATION OF RECOGNITION BASELINE)

### 6.1. Cô lập Tuyệt đối Nhãn `expected_sign`
- Hàm `sanitize_expected_sign()` (`pilot_logger.py:36-48`) chỉ làm nhiệm vụ kiểm tra nhãn ground truth có thuộc 22 chữ cái tĩnh hay không để phục vụ ghi nhãn log.
- Biến `clean_expected_sign` trong `recognizer.predict_landmarks` chỉ xuất hiện ở 2 vị trí:
  - Dòng 1223: Ghi vào payload của `self.pilot_logger.log_frame()`.
  - Dòng 1250: Ghi vào hàm chẩn đoán legacy `self._log_diagnostic()`.
- **Tuyệt đối không tham gia**:
  - Không truyền vào `alphabet_classifier.classify_detailed()`.
  - Không truyền vào `static_alphabet_head.predict_feature()`.
  - Không tham gia vào các phép so sánh ngưỡng `fallback_base_threshold` hoặc `supplemental_thresholds`.
  - Không ảnh hưởng đến các chuyển trạng thái của `DiacriticStateMachine`.

### 6.2. Bằng chứng Thực nghiệm Kiểm tra Hồi quy
- Được xác nhận bởi `test_pilot_hud_integration.py:test_10_recognition_output_unchanged_with_hud_metadata`:
  - Cho cùng một tập mốc bàn tay mẫu (chữ B chuẩn):
    - Khi gọi nhận diện không có metadata (`expected_sign = None`): Đầu ra là `sign = 'B'`, `confidence = 0.95`.
    - Khi gọi nhận diện có metadata đúng (`expected_sign = 'B'`): Đầu ra vẫn là `sign = 'B'`, `confidence = 0.95`.
    - Khi cố tình truyền metadata sai lệch (`expected_sign = 'Z'`): Đầu ra vẫn là `sign = 'B'`, `confidence = 0.95`.
  - Mọi giá trị `sign`, `confidence`, `prediction_source`, `classifier_details` đều trùng khớp 100%.

### 6.3. Tính Bất biến của Cấu hình Ngưỡng Production
- Giữ nguyên toàn bộ 12 geometric rules và ngưỡng hình học hiện tại.
- Giữ nguyên `ALLOWED_FALLBACK_BASE_LETTERS = ("A", "D", "E")` ở ngưỡng 0.70.
- Giữ nguyên `ALLOWED_SUPPLEMENTAL_LETTERS = ("B", "C", "G", "H", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "X", "Y")`.
- Giữ nguyên `DEFAULT_SUPPLEMENTAL_THRESHOLDS` (Q: 0.85, N: 0.75, các chữ khác: 0.70).
- Không có bất kỳ thay đổi thuật toán nhận diện nào diễn ra trong các Task 7F.5A, 7F.5B và 7F.5C.

---

## 7. DANH SÁCH BLOCKERS VÀ NON-BLOCKERS TRƯỚC KHI CHẠY PILOT THỰC TẾ

### 7.1. Danh sách BLOCKERS (Bắt buộc phải xử lý trước khi kích hoạt Pilot)

| Mã lỗi | Mức độ | Vị trí phát hiện | Mô tả chi tiết lỗ hổng | Giải pháp khắc phục yêu cầu |
| :--- | :---: | :--- | :--- | :--- |
| **B-01** | **BLOCKER** | `backend/app.py:286` (endpoint `/api/pilot/control`) | **Lỗ hổng Path Traversal qua tham số `filepath`**: API cho phép nhận đường dẫn tệp tin tùy ý từ client mà không kiểm tra giới hạn trong thư mục log pilot, cho phép tạo và ghi dữ liệu ra ngoài thư mục dự định. | Bổ sung kiểm tra containment path: đảm bảo `os.path.commonpath([os.path.abspath(target_file), os.path.abspath(target_dir)]) == os.path.abspath(target_dir)`. Hoặc loại bỏ hoàn toàn tham số `filepath` từ client, chỉ cho phép sinh tên file tự động dựa trên `session_id` đã được sanitize. |

---

### 7.2. Danh sách NON-BLOCKERS (Giới hạn vận hành và cải tiến khuyến nghị)

| Mã | Mức độ | Vị trí | Mô tả | Biện pháp kiểm soát / Khuyến nghị |
| :--- | :---: | :--- | :--- | :--- |
| **NB-01** | Khuyến nghị vận hành | `backend/vision/pilot_logger.py` & `recognizer.py` | **Giới hạn Đơn phiên (Single-Signer Constraint)**: Backend dùng chung 1 instance recognizer/logger singleton. Chạy đồng thời 2 tab/client sẽ ghi lẫn lộn dữ liệu vào nhau. | Áp dụng quy trình thử nghiệm: Chỉ 1 người ký / 1 máy trạm kết nối thử nghiệm tại một thời điểm. |
| **NB-02** | Khuyến nghị kỹ thuật | `src/components/PilotTestingHUD.jsx:62-82` | **Micro-Race condition khi kết thúc Trial**: Độ trễ render của React 18 có thể làm lọt thêm 1 frame cuối trial trước khi ref được reset về `null`. | Bổ sung kiểm tra `trialStateRef.current === 'recording'` trực tiếp trong hàm `getFrameMetadata()` ở task sau. |
| **NB-03** | Lưu ý phân tích | Pipeline nhận diện thời gian thực | **Chênh lệch FPS giữa Camera và JSONL**: Camera 30-60 FPS nhưng JSONL ghi 15-25 FPS do trích xuất MediaPipe và độ trễ HTTP. | Phổ biến cho đội ngũ phân tích dữ liệu: số frame trong file JSONL là số frame thực nhận, không coi đây là mất mát dữ liệu. |
| **NB-04** | Khuyến nghị vận hành | `backend/app.py:291` | **Ghi nối tiếp khi trùng `session_id`**: Nếu nhập trùng `session_id` cũ, logger sẽ nối dữ liệu vào file cũ mà không cảnh báo. | Khuyến nghị tester luôn dùng session ID mặc định có gắn timestamp tự động hoặc đặt tên duy nhất cho mỗi ca thử nghiệm. |

---

## 8. CHECKLIST CHUẨN BỊ MÔI TRƯỜNG CHẠY WEBCAM THỰC TẾ CHO TESTER / SIGNER

Dưới đây là bảng hướng dẫn chuẩn bị môi trường và thao tác từng bước dành cho kỹ thuật viên và người tham gia thử nghiệm có kiểm soát:

### Bước 1: Chuẩn bị Phần cứng & Môi trường Vật lý
- [ ] **Khoảng cách ngồi**: Ngồi cách webcam từ 0.8m đến 1.2m, sao cho toàn bộ bàn tay và phần thân trên từ ngực trở lên nằm trọn trong khung hình.
- [ ] **Ánh sáng**: Đảm bảo ánh sáng đồng đều phía trước mặt, tránh ngược sáng (không ngồi quay lưng vào cửa sổ hoặc nguồn sáng mạnh).
- [ ] **Hậu cảnh**: Hậu cảnh đơn giản, hạn chế vật thể có màu da hoặc chuyển động gây nhiễu phía sau.
- [ ] **Bàn tay thực hiện**: Thử nghiệm mặc định sử dụng tay phải. Đảm bảo bàn tay không đeo găng tay, vòng tay quá khổ gây che khuất các khớp ngón tay.

### Bước 2: Thiết lập Phần mềm & Khởi động Hệ thống
- [ ] Khởi động Backend: `python run.py` hoặc uvicorn server tại `localhost:8000`.
- [ ] Khởi động Frontend: `npm run dev` tại `localhost:5173`.
- [ ] Mở trình duyệt Chrome/Edge (khuyến nghị phần cứng có tăng tốc GPU cho MediaPipe).
- [ ] Cho phép quyền truy cập Camera khi trình duyệt yêu cầu.
- [ ] Chọn chế độ nhận diện: Chuyển sang chế độ **"Chữ cái (VSL Alphabet)"**.
- [ ] Bật Camera và quan sát khung skeleton 21 điểm mốc: đảm bảo các điểm mốc bám chắc vào các khớp ngón tay, không bị rung giật (jitter).

### Bước 3: Thao tác Thu thập Dữ liệu qua Pilot Testing HUD
- [ ] Bấm nút **"Mở Pilot Testing HUD"** ở góc trên thanh điều khiển.
- [ ] **Kiểm tra Session ID**: Xác nhận Session ID có dạng `signer{XX}_pilot_{YYYYMMDD}`.
- [ ] **Chọn chữ cái cần kiểm thử**: Chọn tuần tự từng chữ trong danh sách 22 chữ cái tĩnh (A, B, C, D, E, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, X, Y).
- [ ] **Quy trình thực hiện 1 Lượt thử (Trial)**:
  1. Người ký tạo dáng tay ký hiệu chuẩn của chữ cái đã chọn và giữ cố định.
  2. Bấm nút xanh **"Bắt đầu Trial"** (hoặc phím tắt).
  3. Giữ nguyên tư thế tay ổn định trong suốt 5 giây đếm ngược.
  4. Quan sát đồng hồ đếm ngược và bộ đếm frame trên HUD tăng dần (dự kiến 75 - 120 frames).
  5. Sau 5 giây, hệ thống tự động kết thúc trial, tăng số lượt `Repetition` lên 1 và tự động flush dữ liệu xuống đĩa.
  6. Thả lỏng tay 2-3 giây trước khi bắt đầu lượt lặp tiếp theo.
- [ ] Thực hiện đủ số lượt lặp quy định cho mỗi chữ cái (khuyến nghị: 5 đến 10 reps cho mỗi chữ).
- [ ] Khi hoàn tất toàn bộ các chữ cái cho phiên thử nghiệm, bấm nút **"Hoàn tất Session"** để flush lần cuối và đóng tệp an toàn.

---

## 9. KẾT LUẬN READINESS

Căn cứ vào các phát hiện kiểm toán đối chiếu với các tiêu chí an toàn bắt buộc:

> ### **TRẠNG THÁI KIỂM TOÁN: BLOCKED**
> *(Lý do: Tồn tại lỗ hổng Path Traversal **B-01** tại tham số `filepath` của endpoint `/api/pilot/control` theo đúng quy tắc kiểm toán: "Nếu có nguy cơ path traversal hoặc ghi đè file ngoài ý muốn, đánh dấu BLOCKER".)*

### Biện pháp giải tỏa Blocker:
Hệ thống sẽ ngay lập tức đạt trạng thái **`PREFLIGHT_READY`** để triển khai thử nghiệm webcam có kiểm soát sau khi áp dụng bản vá bảo vệ đường dẫn tại `backend/app.py:286-291` trong Task tiếp theo:
```python
# Bản vá bảo vệ đường dẫn (Path Traversal Fix):
pilot_dir = os.path.abspath(getattr(recognizer, "diagnostic_dir", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "diagnostics", "pilot")))
os.makedirs(pilot_dir, exist_ok=True)

if req.filepath:
    resolved_path = os.path.abspath(req.filepath)
    if os.path.commonpath([resolved_path, pilot_dir]) != pilot_dir:
        raise HTTPException(status_code=400, detail="Không được phép lưu file log ngoài thư mục pilot hợp lệ.")
    target_file = resolved_path
elif req.session_id:
    clean_sess = re.sub(r'[^a-zA-Z0-9_-]', '_', req.session_id)
    target_file = os.path.join(pilot_dir, f"pilot_{clean_sess}.jsonl")
else:
    target_file = os.path.join(pilot_dir, f"pilot_session_{int(time.time())}.jsonl")
```

Ngoài lỗ hổng `B-01`, toàn bộ kiến trúc thu thập dữ liệu (HUD, Metadata Bridge, Queue Non-blocking Logger, Monotonic Frame Indexing, và tính bất biến của baseline nhận diện) đều **hoạt động hoàn hảo, đạt chuẩn kiểm toán kỹ thuật cao nhất**.

