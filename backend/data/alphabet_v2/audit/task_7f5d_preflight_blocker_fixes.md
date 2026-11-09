# TASK 7F.5D — FIX PILOT PREFLIGHT BLOCKERS: PATH SAFETY & RELIABLE FLUSH
**Trạng thái kiểm toán:** HOÀN THÀNH (BLOCKERS_FIXED)  
**Thời gian thực hiện:** 2026-11-01  
**Mục tiêu:** Xử lý triệt để hai Blocker B-01 (Path Safety) và B-02 (Reliable Flush Barrier), ngăn ngừa xung đột session và micro-race ranh giới trial.  
**Tệp mục tiêu:** `backend/data/alphabet_v2/audit/task_7f5d_preflight_blocker_fixes.md`  

---

## 1. NGUYÊN NHÂN GỐC RỄ (ROOT CAUSE) CỦA B-01 VÀ B-02

### 1.1. Root Cause của B-01 (Lỗ hổng Path Traversal & Ghi đè/nối tệp cũ)
1. **Chấp nhận `filepath` trực tiếp từ client**:
   - Trong `backend/app.py:286`, endpoint `/api/pilot/control` gán thẳng:
     `target_file = req.filepath`
   - Không có kiểm tra path containment (`os.path.commonpath`). Bất kỳ client HTTP nào cũng có thể gửi `filepath: "../../../evil.jsonl"` hoặc đường dẫn tuyệt đối ra ngoài thư mục pilot dự định.
2. **Nguy cơ nối nhầm dữ liệu session cũ**:
   - Khi `session_id` được cung cấp, backend tự sinh `pilot_{clean_sess}.jsonl`.
   - Logger mở tệp bằng chế độ append (`"a"`). Nếu một session mới vô tình trùng tên `session_id` với phiên trước, dữ liệu mới sẽ tự động bị ghi nối vào cuối tệp cũ, gây xáo trộn bản ghi giữa các lượt thử nghiệm độc lập.

### 1.2. Root Cause của B-02 (Cơ chế Flush thiếu tin cậy / Race condition I/O)
1. **Kiểm tra `queue.empty()` không đủ chứng minh I/O đã hoàn tất**:
   - Triển khai cũ của `PilotDiagnosticLogger.flush()` trong `pilot_logger.py`:
     ```python
     while not self.queue.empty():
         time.sleep(0.02)
     return True
     ```
   - Khi worker lấy bản ghi cuối cùng ra khỏi queue (`queue.get()`), queue lập tức ở trạng thái **rỗng** (`self.queue.empty() == True`).
   - Tuy nhiên, tại thời điểm này, worker **chưa thực hiện xong việc ghi đĩa** (đang ở `json.dumps` hoặc `f.write(line)`).
   - Hàm `flush()` thấy queue rỗng liền trả về `True` ngay lập tức, trong khi bản ghi chưa thực sự được ghi hoặc có thể sắp văng ngoại lệ `IOError`!
2. **Không phân biệt được lỗi I/O**:
   - Nếu worker ném ngoại lệ I/O khi ghi tệp, `flush()` trước đây vẫn báo thành công giả (`True`), gây mất dấu dữ liệu thực tế.

---

## 2. DANH SÁCH FILE VÀ HÀM ĐÃ SỬA ĐỔI

1. **`backend/vision/pilot_logger.py`**:
   - Thêm class `FlushBarrier`: Đối tượng đồng bộ hóa hàng đợi mang `threading.Event`, theo dõi `initial_error_count` và phát hiện lỗi I/O phát sinh trước barrier.
   - Bổ sung hàm `create_safe_pilot_file(target_dir, clean_session_id)`: Tạo tệp log nguyên tử ở cấp hệ điều hành bằng cờ `os.O_CREAT | os.O_EXCL | os.O_WRONLY`, tự động đánh số index khi trùng `session_id` và kiểm tra containment chống symlink escape.
   - Sửa `PilotDiagnosticLogger.__init__`: Thêm hook `_test_worker_pre_write_hook` hỗ trợ kiểm soát thời gian trong unit test.
   - Sửa `PilotDiagnosticLogger._worker_loop`: Xử lý `FlushBarrier` tuần tự FIFO, gọi `f.flush()` đẩy dữ liệu xuống OS buffer, quản lý ngoại lệ chính xác.
   - Sửa `PilotDiagnosticLogger.flush(timeout)`: Đặt `FlushBarrier` vào queue, đợi `barrier.completed.wait(timeout)` và trả về `barrier.success`.
   - Sửa `PilotDiagnosticLogger.close(timeout)`: Gọi flush trước khi gửi sentinel và dừng luồng.

2. **`backend/vision/recognizer.py`**:
   - Thêm thuộc tính `active_pilot_session_id = None` và `_pilot_lock = threading.Lock()`.
   - Cập nhật `enable_pilot_logging(session_id, filepath)`: Áp dụng cơ chế độc quyền phiên (Session Exclusivity), ngăn session B ghi đè session A khi A đang chạy.
   - Cập nhật `disable_pilot_logging()`: Giải phóng `active_pilot_session_id`, flush toàn bộ log và đóng session an toàn.

3. **`backend/app.py`**:
   - Cập nhật model `PilotControlRequest`: Giữ `filepath` để phát hiện và từ chối mọi yêu cầu từ client.
   - Sửa `control_pilot_logging(req: PilotControlRequest)`:
     - Từ chối ngay lập tức nếu `req.filepath is not None` (HTTP 400).
     - Kiểm tra và sanitize `session_id` nghiêm ngặt; từ chối chuỗi rỗng hoặc chỉ chứa ký tự traversal (HTTP 400).
     - Kiểm tra xung đột session đang hoạt động; trả về HTTP 409 Conflict nếu phiên khác đang kích hoạt.
     - Gọi `create_safe_pilot_file` để sinh tệp mới an toàn, không ghi đè file cũ.
   - Bổ sung điều kiện chặn prefetch video nền QIPEDC trong môi trường kiểm thử tự động.

4. **`src/components/PilotTestingHUD.jsx`**:
   - Sửa `getFrameMetadata()`: Bổ sung kiểm tra đồng bộ vi giây `if (trialStateRef.current !== 'recording') return null;`.
   - Cập nhật `handleEndTrial()` và `handleCancelTrial()`: Cập nhật `trialStateRef.current` ngay dòng đầu tiên trước khi gọi `setState`, loại bỏ triệt để micro-race condition giữa các frame.

5. **`src/services/api.js`**:
   - Cập nhật `setPilotLogging(enabled, session_id)`: Loại bỏ việc gửi `filepath` qua HTTP body, chỉ truyền `{ enabled, session_id }`.

6. **`backend/tests/test_pilot_hud_integration.py`**:
   - Cập nhật toàn bộ các bài test phù hợp với API contract an toàn mới (dùng `session_id` thay vì truyền `filepath`).

7. **`backend/tests/test_pilot_blocker_fixes.py` (File test mới)**:
   - Xây dựng bộ kiểm thử chuyên biệt 15 test cases xác minh toàn diện Path Safety, Flush Barrier, và Session Exclusivity.

---

## 3. THIẾT KẾ MỚI CỦA PATH SAFETY (FIX B-01)

### 3.1. Loại bỏ hoàn toàn quyền kiểm soát filepath của Client
- API endpoint `/api/pilot/control` kiểm tra:
  ```python
  if req.filepath is not None:
      raise HTTPException(
          status_code=400,
          detail="Chỉ định filepath qua API bị từ chối vì lý do an toàn bảo mật. Vui lòng chỉ truyền session_id."
      )
  ```
- Client HTTP bên ngoài chỉ có thể truyền `session_id`. Toàn bộ quá trình tạo đường dẫn và định vị tệp do backend quản lý 100%.

### 3.2. Sanitize và Kiểm tra Session ID
- Loại bỏ mọi ký tự không an toàn:
  `clean_sess = re.sub(r'[^a-zA-Z0-9_-]', '', raw_sess).strip('_')`
- Nếu sau khi lọc `clean_sess` rỗng (ví dụ: client gửi `../../` hoặc khoảng trắng): Trả về HTTP 400 Bad Request.

### 3.3. Cơ chế Tạo tệp Nguyên tử và Chống Ghi Nối/Đè (`create_safe_pilot_file`)
```python
def create_safe_pilot_file(target_dir: str, clean_session_id: str) -> str:
    target_dir_real = os.path.realpath(target_dir)
    os.makedirs(target_dir_real, exist_ok=True)

    base_candidate = os.path.join(target_dir_real, f"pilot_{clean_session_id}.jsonl")
    candidate_real = os.path.realpath(base_candidate)

    # 1. Kiểm tra path containment và symlink escape
    if os.path.commonpath([candidate_real, target_dir_real]) != target_dir_real:
        raise ValueError(f"Đường dẫn file không hợp lệ hoặc thoát khỏi thư mục pilot: {base_candidate}")

    # 2. Tạo file nguyên tử cấp OS kernel bằng O_CREAT | O_EXCL
    try:
        fd = os.open(base_candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return base_candidate
    except FileExistsError:
        pass

    # 3. Trùng session_id: Tự động đánh số index và timestamp, không ghi nối/đè file cũ
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    counter = 1
    while True:
        unique_candidate = os.path.join(target_dir_real, f"pilot_{clean_session_id}_{timestamp_str}_{counter}.jsonl")
        unique_real = os.path.realpath(unique_candidate)
        if os.path.commonpath([unique_real, target_dir_real]) != target_dir_real:
            raise ValueError(f"Đường dẫn file không hợp lệ hoặc thoát khỏi thư mục pilot: {unique_candidate}")
        try:
            fd = os.open(unique_candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return unique_candidate
        except FileExistsError:
            counter += 1
```

---

## 4. THIẾT KẾ MỚI CỦA FLUSH VÀ WORKER SYNCHRONIZATION (FIX B-02)

### 4.1. Cơ chế FlushBarrier theo thứ tự FIFO
Thay vì chỉ thăm dò `self.queue.empty()`, logger đặt một đối tượng barrier vào hàng đợi đa luồng:

```
[Record 1] -> [Record 2] -> [Record N (đang ghi)] -> [FlushBarrier(Event)]
                                                            ^
                                                   Worker xử lý tuần tự
```

- Worker tiêu thụ hàng đợi tuần tự:
  1. Record N được ghi xong hoàn toàn xuống tệp (`with open(...)`, `f.flush()`).
  2. Worker gọi `task_done()`.
  3. Worker lấy phần tử tiếp theo: `item = FlushBarrier`.
  4. Worker kiểm tra chênh lệch lỗi I/O:
     `errors_delta = (current_io_errors + current_dropped) - barrier.initial_error_count`
  5. Nếu `errors_delta == 0`: `barrier.success = True`. Nếu có lỗi: `barrier.success = False`.
  6. Worker kích hoạt `barrier.completed.set()`.
  7. Worker gọi `task_done()` cho barrier.

### 4.2. Bảo đảm của Hàm `flush(timeout)`
- Thread gọi `flush(timeout)` ngủ chờ trên `barrier.completed.wait(timeout)`.
- Khi thức dậy:
  - Nếu hết thời gian mà barrier chưa hoàn thành: trả về `False` (Timeout).
  - Nếu barrier hoàn thành nhưng có bản ghi bị lỗi I/O: trả về `False` (Không báo thành công giả).
  - Chỉ trả về `True` khi **toàn bộ bản ghi trước barrier đã ghi đĩa thành công**.

---

## 5. CƠ CHẾ SESSION EXCLUSIVITY VÀ QUẢN LÝ CONCURRENCY

Nhằm đảm bảo an toàn cho quy trình Controlled Webcam Pilot:
- Trong `VSLRecognizer`:
  ```python
  with self._pilot_lock:
      if self.pilot_logging_enabled and self.active_pilot_session_id is not None:
          if session_id is not None and session_id != self.active_pilot_session_id:
              raise RuntimeError(f"Phiên '{self.active_pilot_session_id}' đang hoạt động...")
          return self.pilot_logger
  ```
- Trong `backend/app.py`:
  - Nếu Session A đang bật, mà client gửi request bật Session B: Backend từ chối ngay với **HTTP 409 Conflict**.
  - File đích và trạng thái của Session A được bảo toàn tuyệt đối, không bị thay đổi âm thầm.
  - Phải gọi disable Session A (`enabled=False`), hệ thống mới cho phép kích hoạt Session B.

---

## 6. KẾT QUẢ TỪNG TEST MỚI (`test_pilot_blocker_fixes.py`)

Bộ kiểm thử mới gồm 15 bài test tự động chuyên sâu, kết quả **15/15 PASSED (100%)**:

| STT | Tên bài kiểm thử | Nội dung kiểm tra | Kết quả |
| :---: | :--- | :--- | :---: |
| 1 | `test_path_1_reject_client_filepath` | Từ chối client truyền `filepath` tùy ý (HTTP 400) | **PASSED** |
| 2 | `test_path_2_reject_path_traversal_in_session_id` | Từ chối `session_id` chứa `../` hoặc chuỗi rỗng (HTTP 400) | **PASSED** |
| 3 | `test_path_3_create_safe_pilot_file_blocks_containment_breach` | Chặn symlink escape và thoát khỏi thư mục pilot | **PASSED** |
| 4 | `test_path_4_duplicate_session_id_avoids_overwriting_or_appending` | Trùng `session_id` tự sinh file index mới, không ghi đè/nối | **PASSED** |
| 5 | `test_path_5_valid_session_id_generates_file_successfully` | `session_id` hợp lệ tạo file JSONL thành công trong thư mục pilot | **PASSED** |
| 6 | `test_path_6_hud_normal_invocation_works` | HUD bật/tắt chuẩn qua API hoạt động chính xác | **PASSED** |
| 7 | `test_flush_1_timing_controlled_barrier` | **Test kiểm soát thời gian**: Worker bị block ghi dở dang, flush từ thread khác KHÔNG trả về sớm trước khi worker hoàn tất | **PASSED** |
| 8 | `test_flush_2_multiple_records_processed` | Queue 50 records: flush đảm bảo xử lý hết 100% | **PASSED** |
| 9 | `test_flush_3_worker_io_error_reflected_in_flush` | Lỗi I/O xảy ra trước barrier làm `flush()` trả về `False` | **PASSED** |
| 10 | `test_flush_4_timeout_returns_false` | Flush timeout trả về `False` khi worker bị nghẽn | **PASSED** |
| 11 | `test_flush_5_empty_queue_returns_true_immediately` | Flush khi queue rỗng trả về `True` ngay lập tức (<0.01s) | **PASSED** |
| 12 | `test_flush_6_repeated_consecutive_flushes` | Flush liên tiếp nhiều lần hoạt động ổn định | **PASSED** |
| 13 | `test_flush_7_shutdown_flushes_remaining_records` | `close()` xả hết queue trước khi kết thúc worker thread | **PASSED** |
| 14 | `test_session_1_prevent_concurrent_session_override` | Chặn session B ghi đè session A đang chạy (HTTP 409 Conflict) | **PASSED** |
| 15 | `test_session_2_disable_A_allows_session_B` | Tắt session A cho phép kích hoạt session B bình thường | **PASSED** |

---

## 7. KẾT QUẢ REGRESSION TEST SUITE

Đã chạy toàn bộ các bộ kiểm thử liên quan đến hệ thống nhận diện và chẩn đoán:

1. **Bộ kiểm thử Pilot Suite (33 tests)**:
   - `test_pilot_logging.py`: 10/10 passed (0.29s).
   - `test_pilot_hud_integration.py`: 8/8 passed (0.15s).
   - `test_pilot_blocker_fixes.py`: 15/15 passed (0.39s).
   - **Tổng Pilot Tests**: **33/33 PASSED (100%)**.

2. **Bộ kiểm thử Phân loại & Hình học Cốt lõi (137 tests)**:
   - `test_alphabet_classifier.py`: 26/26 passed.
   - `test_alphabet_static_head_integration.py`: 10/10 passed.
   - `test_static_head_fallback.py`: 20/20 passed.
   - `test_diacritic_state_machine.py`: 32/32 passed.
   - `test_geometry.py`: 45/45 passed.
   - `test_pilot_letter_a.py`: 4/4 passed.
   - **Tổng Core Recognition Tests**: **137/137 PASSED (100%)**.

3. **Frontend Build Check**:
   - `npm run build`: Hoàn tất thành công không lỗi trong 864ms (`dist/assets/index-OpiwPItH.js`).

---

## 8. XÁC NHẬN BẢO TOÀN RECOGNITION BASELINE

Khẳng định chắc chắn:
- **12 Geometric Rules**: Không thay đổi bất kỳ dòng code hoặc điều kiện hình học nào.
- **Static Alphabet Head**: Giữ nguyên toàn bộ checkpoint `alphabet_static_head_v1.pt`, feature extractor 72-dim, không train lại model.
- **Gating Fallback**: Giữ nguyên tập hợp chữ nền fallback `ALLOWED_FALLBACK_BASE_LETTERS = ("A", "D", "E")` ở ngưỡng 0.70.
- **Supplemental Thresholds**: Giữ nguyên `Q = 0.85`, `N = 0.75`, các chữ khác `0.70`.
- **Diacritic FSM**: Giữ nguyên toàn bộ logic FSM và Rule I/V của Task 7E.
- **Smoothing & Cooldown**: Giữ nguyên temporal window và logic xác nhận chữ cái.
- **Kết quả nhận diện**: Hoàn toàn đồng nhất 100% giữa hai trạng thái bật và tắt logging (đã kiểm chứng bởi `test_10`).

---

## 9. CÁC GIỚI HẠN VẪN CÒN (KNOWN RESIDUAL LIMITATIONS)

1. **Cơ chế Đơn phiên (Single-Signer Constraint)**:
   - Backend sử dụng kiến trúc Singleton cho `VSLRecognizer` và `PilotDiagnosticLogger`.
   - Hệ thống được thiết kế tối ưu cho quy trình Controlled Pilot thử nghiệm tuần tự từng người ký. Hệ thống cố tình từ chối HTTP 409 nếu có phiên thứ 2 cố gắng chen ngang.
2. **Cam kết Tồn tại Dữ liệu sau Mất điện / OS Crash**:
   - Tệp JSONL được flush ở cấp Python stdio (`f.flush()`), dữ liệu được chuyển giao cho OS file buffer cache.
   - Hệ thống **không** gọi `os.fsync(f.fileno())` ở từng frame để tránh gây độ trễ disk seek (10-50ms/frame) làm tụt FPS của webcam. Do đó, dữ liệu được bảo vệ an toàn khi ứng dụng crash hoặc đóng bình thường, nhưng không cam kết 100% trong trường hợp mất điện đột ngột hoặc kernel panic trước khi OS kịp sync đĩa.

---

## 10. KẾT LUẬN READINESS

> ### **TRẠNG THÁI CUỐI CÙNG: BLOCKERS_FIXED**  
>
> Cả hai vấn đề kỹ thuật nghiêm trọng **B-01 (Path Safety)** và **B-02 (Reliable Flush Barrier)** cùng rủi ro xung đột session và micro-race đã được khắc phục hoàn toàn. Toàn bộ 15 bài test chuyên biệt mới và 170+ bài test hồi quy hệ thống đều **vượt qua 100%**.  
> Hạ tầng phần mềm, API và HUD logging đã chính thức sẵn sàng cho việc tiến hành các ca thử nghiệm Controlled Webcam Pilot trên người ký thực tế.

