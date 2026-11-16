# TASK 7F.6B — SMOKE TEST VERIFICATION SCRIPT & OPERATIONAL CONSISTENCY AUDIT REPORT

**Trạng thái phê duyệt:** `READY_FOR_SMOKE_TEST`  
**Ngày kiểm định:** 2026-11-01  
**Môi trường:** Python 3.12, PyTorch 2.2.2, FastAPI, React 18, Vite 8.3.0, Windows 10/11  
**Thành phần thực hiện:** `backend/scripts/verify_pilot_smoke_test.py`, `backend/tests/test_verify_pilot_smoke_test.py`

---

## 1. TỔNG QUAN & MỤC TIÊU

Task 7F.6B thực hiện rà soát sâu, đối chiếu mã nguồn thực tế và hoàn thiện công cụ nghiệm thu dữ liệu tự động (`verify_pilot_smoke_test.py`) cho Controlled Webcam Pilot Smoke Test sau Task 7F.6A.

Mục tiêu cốt lõi:
1. Xác minh thực nghiệm các hành vi vận hành chưa rõ trong Task 7F.6A (định dạng Trial ID, semantics tăng/reset của `frame_idx`, bản ghi inter-trial ngoài trial, độ phân lập của ground-truth).
2. Xây dựng script kiểm tra tự động toàn diện, đánh giá hai tầng độc lập (**Frame-level** và **Trial-level**), loại bỏ các giả định sai lầm (như coi các frame liên tiếp là mẫu độc lập thống kê, hoặc bắt buộc mọi frame trong file phải có `trial_id`).
3. Đảm bảo script không tạo "PASS giả" (False Positive) do thiếu assertion, đồng thời không tạo "FAIL oan" (False Negative) do hiểu sai semantics của HUD.
4. Kiểm thử script xác minh bằng bộ test tự động chuyên biệt với các synthetic fixtures.

---

## 2. KẾT QUẢ ĐỐI CHIẾU HÀNH VI MÃ NGUỒN (SOURCE CODE AUDIT)

Qua rà soát chi tiết từng dòng code trong `PilotTestingHUD.jsx`, `CameraSignRecognizer.jsx`, `api.js`, `app.py`, `recognizer.py`, và `pilot_logger.py`:

### A. Session và Trial ID
1. **Session ID:**
   - Được khởi tạo trên HUD (`src/components/PilotTestingHUD.jsx:L25-L29`) mặc định theo mẫu `pilot_YYYYMMDD_HHMM` (ví dụ `pilot_20261101_2015`) và cho phép người vận hành chỉnh sửa với regex `^[a-zA-Z0-9_-]{3,64}$`.
   - Được gửi lên Backend qua endpoint `/api/pilot/control` để tạo file và xác thực qua hàm `validate_session_id()`.
   - Được lưu trong biến `recognizer.active_pilot_session_id`.
   - Trong quá trình webcam stream, HUD truyền `session_id` qua header `X-Session-Id` và JSON payload vào `/api/recognize_landmarks`.
2. **Trial ID:**
   - Được tính toán tự động trong HUD (`src/components/PilotTestingHUD.jsx:L59`):
     ```javascript
     const trialId = `${sessionId}_${selectedLetter}_rep${String(currentRep).padStart(2, '0')}`;
     ```
   - **Định dạng chuẩn:** `{sessionId}_{letter}_rep{currentRep:02d}` (ví dụ: `smoke_signer01_20261101_A_rep01`).
   - Khi đổi chữ cái (`selectedLetter`): `currentRep` tự động reset về `1`, `trialId` được tính toán lại ngay lập tức.
   - Khi hoàn thành 1 trial 5.0s: `currentRep` tự động tăng lên 1 (từ 1 lên 2), `trialId` cập nhật cho lần thử tiếp theo.
3. **Semantics của `frame_idx` (Tăng theo Session hay Reset theo Trial?):**
   - **Xác minh từ mã nguồn:** Trong `PilotTestingHUD.jsx:L123`:
     ```javascript
     const handleStartTrial = async () => {
       ...
       frameIdxRef.current = 0;
       setTrialFrameCount(0);
       ...
     ```
     Và trong `getFrameMetadata()` (`PilotTestingHUD.jsx:L70`):
     ```javascript
     frameIdxRef.current += 1;
     return {
       ...
       frame_idx: frameIdxRef.current
     };
     ```
   - **KẾT LUẬN QUAN TRỌNG:** `frame_idx` **ĐƯỢC RESET VỀ 1 Ở ĐẦU MỖI TRIAL**. Trong một trial, `frame_idx` tăng đơn điệu nghiêm ngặt không trùng lặp ($1, 2, 3, \dots, N$). Khi sang trial mới, `frame_idx` bắt đầu lại từ $1$. Do đó, kiểm tra tính duy nhất (uniqueness) của `frame_idx` chỉ được áp dụng **trong phạm vi từng trial**, tuyệt đối không kiểm tra uniqueness trên toàn bộ file JSONL (tránh FAIL oan).
4. **Bản ghi ngoài Trial (Inter-trial / Idle Records):**
   - Khi `trialState !== 'recording'` (trạng thái chờ giữa các trial hoặc trước khi bấm trial), HUD trả về `getFrameMetadata() => null`.
   - API client không đính kèm `session_id` hay `trial_id`.
   - Tuy nhiên, vì `pilot_logging_enabled = True` trên Backend trong suốt phiên, các frame webcam gửi về giữa các lần thử vẫn được worker ghi đĩa với `session_id: null`, `trial_id: null`, `expected_sign: null`.
   - **KẾT LUẬN QUAN TRỌNG:** File JSONL chứa cả hai loại bản ghi: **Trial records** (có `trial_id`) và **Inter-trial records** (`trial_id: null`). Script nghiệm thu phải phân tách hai loại bản ghi này, không được bắt buộc toàn bộ dòng trong file đều phải có `trial_id`.

### B. File JSONL & Nhận biết tệp
1. **Đường dẫn thực tế:** `backend/data/diagnostics/pilot/pilot_{session_id}.jsonl`.
2. **Cơ chế khi trùng Session ID:**
   - Sử dụng kernel flag `os.O_CREAT | os.O_EXCL` trong `create_safe_pilot_file()`.
   - Nếu tệp đã tồn tại, tự động sinh tên duy nhất: `pilot_{session_id}_{YYYYMMDD_HHMMSS}_{counter}.jsonl`.
   - Backend trả về đường dẫn chính xác này trong response của `/api/pilot/control`.
   - Script xác minh hỗ trợ tự động tìm kiếm tệp mới nhất khớp tiền tố `pilot_{session_id}*.jsonl` trong thư mục pilot nếu người dùng chỉ truyền `session_id`.
3. **Kiểm tra Telemetry Stats sau khi đóng logger:**
   - Khi phiên kết thúc qua `/api/pilot/control` (`enabled: false`), `recognizer.disable_pilot_logging()` gọi `pilot_logger.flush()` và hủy kích hoạt logging.
   - Thống kê bộ nhớ trong logger vẫn tồn tại cho đến khi tiến trình dừng hoặc server restart, nhưng không bền vững (ephemeral).
   - **Nguồn dữ liệu chân lý duy nhất (Single Source of Truth):** Toàn bộ việc xác minh phải được thực hiện trực tiếp từ các bản ghi lưu trên **file JSONL đĩa cứng**, việc đối chiếu với telemetry stats qua API/JSON là tùy chọn bổ trợ.

### C. Tính độc lập của Ground-Truth (Ground-Truth Isolation)
- Đã rà soát toàn bộ các vị trí xuất hiện `expected_sign` trong `backend/vision/recognizer.py`.
- Trong chế độ Chữ cái (`mode == "letters"`):
  - Dòng 587: `clean_expected_sign = sanitize_expected_sign(expected_sign)` chỉ chuẩn hóa chuỗi nhãn.
  - Bộ trích xuất đặc trưng `normalize_hand_landmarks()` không nhận `expected_sign`.
  - Bộ phân loại hình học `alphabet_classifier.classify_detailed(p_prim)` không nhận `expected_sign`.
  - Mô hình học sâu `static_alphabet_head.predict_feature()` không nhận `expected_sign`.
  - Cơ chế Gated Static Head Fallback và ngưỡng `fallback_base_threshold` không nhận `expected_sign`.
  - Máy trạng thái dấu phụ `diacritic_fsms` không nhận `expected_sign`.
  - Bộ đệm làm mượt thời gian `prediction_history` không nhận `expected_sign`.
  - Dòng 1240: `clean_expected_sign` chỉ được đóng gói vào từ điển gửi cho `pilot_logger.log_frame()`.
  *(Lưu ý: Dòng 1329 có điều kiện so sánh `expected_sign` nhưng chỉ nằm trong nhánh `is_legacy_mode` (`letters_numbers`) cũ, hoàn toàn không được kích hoạt khi chạy chế độ chữ cái `letters`).*
- **KẾT LUẬN:** **Ground-Truth Isolation đạt 100% VERIFIED**. `expected_sign` không thể can thiệp làm sai lệch kết quả dự đoán của mô hình.

---

## 3. CÁC ĐIỂM CÒN `NOT VERIFIED`

Sau khi kiểm tra trực tiếp toàn bộ chuỗi từ frontend React đến backend Python:
- **TẤT CẢ các hành vi quy trình đã được xác minh trực tiếp từ mã nguồn thực tế.**
- Không còn điểm nào ở trạng thái `NOT VERIFIED`.

---

## 4. HOÀN THIỆN SCRIPT XÁC MINH SMOKE TEST

Đã triển khai công cụ chính thức tại:
[`backend/scripts/verify_pilot_smoke_test.py`](file:///d:/SIGN_LANGUAGE/backend/scripts/verify_pilot_smoke_test.py)

### 4.1. Các tính năng cốt lõi
1. **Giao diện dòng lệnh linh hoạt:**
   - Chấp nhận tham số dạng positional hoặc options:
     ```powershell
     python backend/scripts/verify_pilot_smoke_test.py --file backend/data/diagnostics/pilot/pilot_smoke_signer01_20261101.jsonl --session-id smoke_signer01_20261101
     # Hoặc ngắn gọn:
     python backend/scripts/verify_pilot_smoke_test.py smoke_signer01_20261101
     ```
2. **Kiểm tra cú pháp & tính hợp lệ 100%:** Kiểm tra từng dòng `json.loads()`, báo lỗi chi tiết số dòng vi phạm nếu file bị hỏng.
3. **Phân tách hai loại bản ghi:** Tách biệt Trial records và Inter-trial records, không báo lỗi oan cho các frame quay giữa các trial.
4. **Kiểm tra Session ID & Trial ID:**
   - Đối chiếu Session ID của các trial frame với `expected_session_id`.
   - Kiểm tra Trial ID theo regex chuẩn: `^{sessionId}_([A-Z])_rep(\d{2})$`.
   - Đảm bảo đủ 6 trials ($3 \text{ chữ } A, B, C \times 2 \text{ reps}$).
5. **Kiểm tra Frame Index:**
   - Bắt đầu từ 1 trong mỗi trial.
   - Tăng đơn điệu, không trùng lặp trong trial.
6. **Kiểm tra Timestamp & Tính FPS:**
   - Timestamp dương, tăng đơn điệu nghiêm ngặt trong trial.
   - Kiểm tra tính tuần tự: Trial sau không được bắt đầu trước khi trial trước kết thúc.
   - Tính FPS thực tế chính xác: $\text{FPS} = \frac{N - 1}{t_{\text{end}} - t_{\text{start}}}$, yêu cầu $\text{FPS} \ge 10.0$.
7. **Báo cáo hai tầng rõ rệt (Two-Tier Evaluation):**
   - **Tầng Frame-level:** Thống kê tổng số frame, tỷ lệ frame đúng (`final_sign == expected_sign`), tỷ lệ frame đạt `confirmed`, phân bố confidence và prediction sources.
   - **Tầng Trial-level:** Xác định **Nhãn đại diện (Representative Label)** theo đa số phiếu non-neutral. Xử lý triệt để trường hợp `NEUTRAL` (nếu không có frame nhận diện nào, nhãn là `NEUTRAL`).

---

## 5. TIÊU CHÍ NGHIỆM THU CUỐI CÙNG (PASS/FAIL CRITERIA)

Một phiên Smoke Test được công nhận là **PASSED** khi thỏa mãn đồng thời:

1. **Tính toàn vẹn kỹ thuật (Integrity):**
   - Không có lỗi cú pháp JSON.
   - Không có trường dữ liệu bắt buộc bị thiếu hoặc `null` trong trial frames.
   - `session_id` khớp 100% với phiên đăng ký.
   - `expected_sign` khớp với chữ cái trong `trial_id`.
   - `frame_idx` tăng đơn điệu từ 1, không trùng lặp trong trial.
   - Timestamp tăng đơn điệu, thứ tự trial tuần tự.
   - Không có frame bị drop do nghẽn queue (`dropped_queue_full == 0`) hoặc lỗi ghi đĩa (`dropped_io_error == 0`).
2. **Tiêu chuẩn dữ liệu Trial (Data Sufficiency):**
   - Hoàn thành đủ 6/6 trials dự kiến.
   - Mỗi trial có ít nhất 50 frames (đảm bảo giữ cử chỉ đủ 5 giây).
   - FPS thực tế của mỗi trial $\ge 10.0$.
3. **Tiêu chuẩn nhận diện (Recognition Quality):**
   - Nhãn đại diện của mỗi trial phải khớp với chữ cái mục tiêu (`representative_label == expected_sign`).
   - Tỷ lệ frame dự đoán đúng trong mỗi trial $\ge 50.0\%$.
   - Mỗi trial phải có ít nhất 1 frame đạt trạng thái xác nhận ổn định (`confirmed == True`).

---

## 6. KẾT QUẢ KIỂM THỬ SCRIPT XÁC MINH (TEST SUITE)

Đã xây dựng test suite chuyên biệt [`backend/tests/test_verify_pilot_smoke_test.py`](file:///d:/SIGN_LANGUAGE/backend/tests/test_verify_pilot_smoke_test.py) với 11 test case synthetic fixtures:

| STT | Kịch bản kiểm thử | Kết quả mong đợi | Thực tế |
|:---:|:---|:---:|:---:|
| 1 | `test_parse_trial_id`: Phân tích đúng/sai định dạng trial_id | Khớp định dạng | **PASSED** |
| 2 | `test_valid_smoke_test_file_passes`: File chuẩn 6 trials | `passed == True` | **PASSED** |
| 3 | `test_valid_smoke_test_with_inter_trial_frames_passes`: File có frame idle | `passed == True` | **PASSED** |
| 4 | `test_invalid_jsonl_syntax_fails`: Dòng JSON bị hỏng | `passed == False` | **PASSED** |
| 5 | `test_wrong_session_id_fails`: Sai session ID | `passed == False` | **PASSED** |
| 6 | `test_missing_trial_fails`: Chỉ có 5/6 trials | `passed == False` | **PASSED** |
| 7 | `test_expected_sign_mismatch_fails`: expected_sign sai lệch trial | `passed == False` | **PASSED** |
| 8 | `test_duplicate_frame_idx_fails`: Trùng lặp frame_idx trong trial | `passed == False` | **PASSED** |
| 9 | `test_non_monotonic_timestamp_fails`: Timestamp bị lùi thời gian | `passed == False` | **PASSED** |
| 10 | `test_low_fps_fails`: FPS < 10.0 | `passed == False` | **PASSED** |
| 11 | `test_telemetry_stats_cross_check_detects_dropped`: Thất thoát dropped > 0 | `passed == False` | **PASSED** |

**Kết quả chạy test:** `11/11 tests PASSED` (0.177s).  
**Toàn bộ hệ thống test Pilot:** `62/62 tests PASSED` (3.352s).

---

## 7. HƯỚNG DẪN NGƯỜI VẬN HÀNH SỬ DỤNG SCRIPT

Sau khi thực hiện xong 6 trials trên giao diện HUD và bấm **"Flush & Xong"**, người vận hành chạy lệnh sau từ thư mục dự án:

```powershell
python -X utf8 backend/scripts/verify_pilot_smoke_test.py --session-id smoke_signer01_20261101
```

Script sẽ tự động tìm kiếm tệp JSONL tương ứng trong `backend/data/diagnostics/pilot/`, phân tích cú pháp, tính toán FPS và in bảng đánh giá 2 tầng cùng kết luận nghiệm thu rõ ràng.

---

## 8. KẾT LUẬN CUỐI CÙNG

```
================================================================================
                    TASK 7F.6B AUDIT & VERIFICATION SCRIPT
                    STATUS: READY_FOR_SMOKE_TEST
================================================================================
  - Source Code Audit: 100% verified (Session/Trial ID, frame_idx reset, inter-trial)
  - Ground-Truth Isolation: 100% verified in letters mode
  - Verification Script: Built & Tested with Two-Tier Evaluation (Frame + Trial)
  - Synthetic Test Suite: 11/11 Tests Passed
  - Regression Suite: 62/62 Pilot Tests Passed
================================================================================
```
Hệ thống kiểm định đã hoàn tất, không còn mâu thuẫn hay điểm nghẽn logic. Dự án đã sẵn sàng bước vào giai đoạn thu thập dữ liệu webcam thực tế.

