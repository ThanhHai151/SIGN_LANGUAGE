# BÁO CÁO TRIỂN KHAI TASK 7F.5B — GIAO DIỆN THỬ NGHIỆM PILOT HUD & TÍCH HỢP METADATA ĐẦU-CUỐI (PILOT TESTING HUD & END-TO-END METADATA INTEGRATION)

**Dự án:** VSL Connect AI — Hệ thống Nhận diện Ngôn ngữ Ký hiệu Việt Nam  
**Ngày thực hiện:** 2026-11-01  
**Phạm vi:** Xây dựng Pilot Testing HUD phía Frontend React, kết nối truyền metadata đầu-cuối (End-to-End) tới Backend FastAPI và ghi nhận vào file JSONL chẩn đoán.  
**Trạng thái kiểm toán:** HOÀN THÀNH — BẢO TOÀN NGUYÊN TRẠNG 100% THUẬT TOÁN VÀ NGƯỠNG PRODUCTION.  
**Kết luận Readiness:** **`PILOT_HUD_READY_FOR_CONTROLLED_TEST`**

---

## 1. TỔNG QUAN VÀ MỤC TIÊU ĐẠT ĐƯỢC

Trong Task 7F.5A, hạ tầng ghi log phi nghẽn `PilotDiagnosticLogger` đã được thiết lập thành công trên Backend. Tuy nhiên, phía Frontend Webcam chưa có giao diện điều khiển phiên thử nghiệm và chưa thể gửi kèm các trường metadata (`expected_sign`, `session_id`, `trial_id`, `frame_idx`) một cách có hệ thống.

**Task 7F.5B đã hoàn thành việc tích hợp toàn diện đầu-cuối:**
1. **Pilot Testing HUD:** Xây dựng component trực quan [`src/components/PilotTestingHUD.jsx`](file:///d:/SIGN_LANGUAGE/src/components/PilotTestingHUD.jsx) cho phép cấu hình phiên thử nghiệm, chọn chữ cái mục tiêu trong 22 chữ cái tĩnh, đếm ngược 5 giây giữ tư thế (Hold Countdown), theo dõi lần lặp (Repetitions) và hiển thị số liệu telemetry thời gian thực từ Backend.
2. **Kênh truyền Metadata Đầu-Cuối:** Nâng cấp [`src/services/api.js`](file:///d:/SIGN_LANGUAGE/src/services/api.js) và [`src/components/CameraSignRecognizer.jsx`](file:///d:/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx) để tự động gắn metadata vào cả HTTP Headers và JSON Request Body trong suốt 5 giây của mỗi trial.
3. **Chống rò rỉ Metadata (Zero-Leakage Guarantee):** Đảm bảo ngoài thời gian trial (khi người dùng nghỉ hoặc chuẩn bị), metadata gửi lên là `null` tuyệt đối; không có bất kỳ thông tin nào của trial trước bị rò rỉ sang trial tiếp theo.
4. **Phân lập hoàn toàn khỏi thuật toán phân loại:** `expected_sign` chỉ đóng vai trò nhãn ground truth lưu trữ trong JSONL, tuyệt đối không can thiệp vào bất kỳ nhánh phân loại, ngưỡng hay FSM nào.

---

## 2. DANH SÁCH CÁC FILE ĐÃ BỔ SUNG VÀ CẬP NHẬT

| STT | File | Trạng thái | Nội dung kỹ thuật chính |
| :---: | :--- | :---: | :--- |
| **1** | `src/components/PilotTestingHUD.jsx` | **MỚI** | Component React giao diện HUD: quản lý `session_id`, dropdown 22 chữ cái tĩnh, chọn lần lặp ($1 \dots 10$), bộ đếm ngược 5.0s, thanh tiến trình visual, nút Flush & Kết thúc phiên, hiển thị telemetry từ backend. |
| **2** | `src/services/api.js` | **CẬP NHẬT** | Mở rộng `sendLandmarksForRecognition` nhận tham số `metadata`, tự động đính kèm `X-Expected-Sign`, `X-Session-Id`, `X-Trial-Id`, `X-Frame-Idx` vào HTTP Headers và JSON Body; bổ sung các hàm gọi API: `setPilotLogging()`, `flushPilotLogs()`, `fetchPilotStats()`. |
| **3** | `src/components/CameraSignRecognizer.jsx` | **CẬP NHẬT** | Tích hợp `<PilotTestingHUD>`, quản lý `pilotMetadataProviderRef` thời gian thực (zero React re-render overhead trong luồng 60 FPS), truyền metadata an toàn vào vòng lặp suy luận. |
| **4** | `backend/app.py` | **CẬP NHẬT** | Bổ sung Pydantic model `PilotControlRequest` và endpoint `/api/pilot/control` cho phép Frontend bật/tắt logger và chỉ định session file; tích hợp các endpoint `/api/pilot/flush` và `/api/pilot/stats`. |
| **5** | `backend/tests/test_pilot_hud_integration.py` | **MỚI** | Bộ kiểm thử tích hợp tự động gồm 8 method (10 assertions) xác minh toàn diện vòng đời truyền nhận metadata từ Frontend đến JSONL. |

---

## 3. LUỒNG DỮ LIỆU ĐẦU-CUỐI: FRONTEND $\rightarrow$ BACKEND $\rightarrow$ JSONL

```
[Người thử nghiệm tương tác trên Pilot HUD]
         │
         ├──► Chọn chữ cái (ví dụ: "B"), lần lặp (rep: 1/10)
         ├──► Bấm "Bắt đầu Trial (5s Hold)"
         │
         ▼
[React State Machine: trialState = 'recording']
         │
         ├──► Đếm ngược 5.0s -> 0.0s kèm progress bar
         ├──► pilotMetadataProviderRef sinh metadata cho từng frame:
         │      { expected_sign: "B", session_id: "...", trial_id: "..._B_rep01", frame_idx: k }
         │
         ▼
[api.sendLandmarksForRecognition(..., metadata)]
         │
         ├──► HTTP Headers: X-Expected-Sign, X-Session-Id, X-Trial-Id, X-Frame-Idx
         └──► JSON Body: { landmarks, mode: 'letters', expected_sign, session_id, trial_id, frame_idx }
         │
         ▼
[FastAPI: POST /api/recognize_landmarks]
         │
         ├──► Đọc metadata từ Request Body / Headers
         ├──► Chuyển tiếp vào recognizer.predict_landmarks()
         │
         ▼
[VSLRecognizer.predict_landmarks]
         │
         ├──► sanitize_expected_sign: kiểm tra thuộc 22 chữ tĩnh (B -> hợp lệ)
         ├──► KHÔNG ĐƯA expected_sign vào phân loại hình học hay FSM
         ├──► Thực hiện nhận diện: Geometric Rule / Static Head / Fallback
         │
         ▼
[PilotDiagnosticLogger.log_frame(record)]
         │
         ├──► Đẩy vào queue.Queue(maxsize=10000) (non-blocking < 0.05ms)
         │
         ▼ (Background Worker Daemon Thread)
[File JSONL trên đĩa: backend/data/diagnostics/pilot/pilot_<session>.jsonl]
         │
         ▼ (Sau khi hết 5 giây)
[Pilot HUD tự động gọi POST /api/pilot/flush]
         └──► Worker xả sạch hàng đợi xuống file -> HUD hiển thị số frames đã lưu.
```

---

## 4. QUẢN LÝ SESSION, TRIAL VÀ FRAME KHÔNG RÒ RỈ DỮ LIỆU

### 4.1. Quy ước định danh duy nhất (Unique Identifiers)
- **`session_id`:** Định danh phiên thử nghiệm, mặc định tự động sinh theo ngày giờ: `pilot_YYYYMMDD_HHMM` (ví dụ: `pilot_20261101_1830`), hoặc do người điều hành đặt tên (ví dụ: `signer01_baseline`).
- **`trial_id`:** Định danh duy nhất cho từng lần thử của một chữ cái cụ thể:
  $$\text{trial\_id} = \{\text{session\_id}\} \_ \{\text{selectedLetter}\} \_ \text{rep}\{\text{currentRep:02d}\}$$
  *Ví dụ:* `pilot_20261101_1830_B_rep01`, `pilot_20261101_1830_B_rep02`.
- **`frame_idx`:** Số nguyên tăng đơn điệu ($1, 2, 3 \dots$) trong suốt 5 giây của trial, phản ánh chính xác số frame thực tế mà camera và MediaPipe đã gửi tới server.

### 4.2. Cơ chế Chống Rò Rỉ Metadata (Zero-Leakage Architecture)
- Khi `trialState === 'idle'` hoặc `'completed'`:
  `pilotMetadataProviderRef.current()` trả về `null`.
- Hàm `sendLandmarksForRecognition` kiểm tra: nếu `metadata === null`, request gửi đi **hoàn toàn không có các header `X-Expected-Sign`, `X-Session-Id`, `X-Trial-Id`, `X-Frame-Idx`**, và body JSON không chứa các trường này.
- Khi người dùng hạ tay nghỉ ngơi giữa 2 lần thử, các frame trung gian được backend xử lý bình thường và không bị gán nhãn nhầm vào trial trước hay trial sau.

---

## 5. HỆ THỐNG REST API ĐIỀU KHIỂN PILOT

| Endpoint | Method | Payload | Mô tả chức năng |
| :--- | :---: | :--- | :--- |
| `/api/pilot/control` | `POST` | `{"enabled": true/false, "session_id": "...", "filepath": "..."}` | Bật hoặc tắt chế độ ghi nhật ký chẩn đoán unthrottled từ UI. Tự động khởi tạo file log theo `session_id`. |
| `/api/pilot/flush` | `POST` | Không có | Chờ worker thread ghi cạn toàn bộ hàng đợi bản ghi xuống đĩa, trả về thống kê số record đã ghi. |
| `/api/pilot/stats` | `GET` | Không có | Lấy số liệu giám sát trực tiếp: `queued_count`, `logged_count`, `dropped_queue_full`, `dropped_io_error`, `total_dropped`, `pending_in_queue`. |

---

## 6. KẾT QUẢ KIỂM THỬ TỰ ĐỘNG VÀ HỒI QUY

### 6.1. Kiểm thử tích hợp `backend/tests/test_pilot_hud_integration.py`
Toàn bộ 8 method kiểm thử tích hợp (đáp ứng trọn vẹn 10 yêu cầu kỹ thuật) đều đạt kết quả **PASSED (8/8)**:

```
backend/tests/test_pilot_hud_integration.py::TestPilotHUDIntegration::test_10_recognition_output_unchanged_with_hud_metadata PASSED [ 12%]
backend/tests/test_pilot_hud_integration.py::TestPilotHUDIntegration::test_1_and_2_frontend_metadata_received_and_logged PASSED [ 25%]
backend/tests/test_pilot_hud_integration.py::TestPilotHUDIntegration::test_3_trial_id_changes_between_trials PASSED [ 37%]
backend/tests/test_pilot_hud_integration.py::TestPilotHUDIntegration::test_4_frame_idx_increments_monotonically PASSED [ 50%]
backend/tests/test_pilot_hud_integration.py::TestPilotHUDIntegration::test_5_no_metadata_leakage_after_trial_ends PASSED [ 62%]
backend/tests/test_pilot_hud_integration.py::TestPilotHUDIntegration::test_6_isolated_sessions_do_not_mix PASSED [ 75%]
backend/tests/test_pilot_hud_integration.py::TestPilotHUDIntegration::test_7_and_8_flush_and_stats_endpoints PASSED [ 87%]
backend/tests/test_pilot_hud_integration.py::TestPilotHUDIntegration::test_9_default_logging_is_disabled PASSED [100%]

======================== 8 passed in 9.84s ========================
```

### 6.2. Kiểm thử hồi quy toàn bộ hệ thống (Full Regression Suite)
Chạy đồng thời toàn bộ 156 test của hệ thống (bao gồm phân loại hình học, Static Head, fallback, FSM chữ động, logger và HUD integration):

```
backend\tests\test_pilot_logging.py ..........                           [  6%]
backend\tests\test_pilot_hud_integration.py ........                     [ 11%]
backend\tests\test_alphabet_classifier.py ..........................     [ 28%]
backend\tests\test_alphabet_landmarks.py ......                          [ 32%]
backend\tests\test_alphabet_recognition_integration.py .........         [ 37%]
backend\tests\test_alphabet_static_head_integration.py ..........        [ 44%]
backend\tests\test_static_head_fallback.py ....................          [ 57%]
backend\tests\test_diacritic_state_machine.py .......................... [ 73%]
......                                                                   [ 77%]
backend\tests\test_diacritic_camera_integration.py ..................... [ 91%]
..                                                                       [ 92%]
backend\tests\test_diagnostic.py ............                            [100%]

====================== 156 passed in 15.43s =======================
```

### 6.3. Kiểm thử đóng gói Frontend (Vite Production Build)
Chạy lệnh `npm run build`:
```
vite v8.3.0 building client environment for production...
✓ 1899 modules transformed.
dist/index.html                   2.01 kB │ gzip:   0.95 kB
dist/assets/index-C8m3tDRW.css    4.28 kB │ gzip:   1.58 kB
dist/assets/index-B_DJQt4U.js   912.06 kB │ gzip: 241.76 kB
✓ built in 17.43s
```
Bundle React Frontend biên dịch hoàn toàn thành công, không có lỗi cú pháp, JSX hay tham chiếu thiếu.

---

## 7. BẢO TOÀN NGUYÊN TRẠNG PRODUCTION (SAFETY CONSTRAINTS COMPLIANCE)

Tuân thủ nghiêm ngặt các điều kiện an toàn:
1. **Geometric Rules:** Giữ nguyên 100% ngưỡng và quy tắc hình học trong `alphabet_classifier.py`.
2. **Static Head & Checkpoint:** Checkpoint `alphabet_static_head_v1.pt`, `weights.pt` và `sign_sstcn_attention_model.pth` không bị can thiệp.
3. **Thresholds:** Giữ nguyên $Q = 0.85$, $N = 0.75$, `fallback_base_threshold = 0.70`.
4. **Fallback Allowlist:** Giữ nguyên `ALLOWED_FALLBACK_BASE_LETTERS = {"A", "D", "E"}`. Tuyệt đối **không** bật fallback cho `B, C, T, L, Y` và không đưa `I` vào fallback trong task này.
5. **Chữ động & FSM:** Bộ máy trạng thái `DiacriticStateMachine` cho 7 chữ cái có dấu động (Ă, Â, Ê, Ô, Ơ, Ư, Đ) giữ nguyên 100%.
6. **Mặc định:** Khi không sử dụng HUD, hệ thống chạy baseline như trước, không gửi metadata và không sinh log pilot.

---

## 8. HƯỚNG DẪN THỰC HIỆN THỬ NGHIỆM THỦ CÔNG QUA HUD (MANUAL TESTING WALKTHROUGH)

Khi nhóm kỹ thuật hoặc người thử nghiệm bắt đầu thu thập dữ liệu webcam:

1. **Khởi động hệ thống:**
   - Khởi động Backend: `uvicorn backend.app:app --host 127.0.0.1 --port 8000`
   - Khởi động Frontend: `npm run dev` (truy cập `http://localhost:5173`)
2. **Mở giao diện Pilot:**
   - Trong thẻ Camera Nhận diện, bấm nút **"Mở Pilot Testing HUD"** (icon bình thí nghiệm xanh).
   - Bảng điều khiển Controlled Webcam Pilot HUD sẽ hiển thị phía trên khung hình camera.
3. **Cấu hình phiên thử nghiệm:**
   - Nhập `Session ID` (ví dụ: `signer01_20261101` hoặc để mặc định).
   - Chọn chữ cái mục tiêu từ dropdown: ví dụ `Chữ cái B`.
   - Chọn lần lặp: ví dụ `Lần 1 / 10`.
4. **Thực hiện một Trial (5 giây):**
   - Bấm **"Bật Camera"** (nếu camera chưa bật).
   - Bấm nút **"Bắt đầu Trial (5s Hold)"**.
   - Người ký hiệu đưa tay vào khung hình, tạo dáng chữ mục tiêu và giữ yên tư thế.
   - Thanh tiến trình sẽ hiển thị thời gian đếm ngược từ 5.0s về 0.0s kèm số lượng frame gửi đi.
   - Khi hết 5 giây, hệ thống tự động chốt trial, gọi flush dữ liệu xuống file JSONL và tự động tăng lần lặp lên `Lần 2 / 10`.
5. **Kết thúc Session:**
   - Sau khi hoàn thành đủ các lần lặp hoặc các chữ cái, bấm nút **"Flush & Xong"**.
   - Toàn bộ dữ liệu được đảm bảo đã ghi trọn vẹn vào thư mục `backend/data/diagnostics/pilot/`.

---

## 9. KẾT LUẬN READINESS

$$\mathbf{K\hat{E}T\ LU\hat{A}N:\ PILOT\_HUD\_READY\_FOR\_CONTROLLED\_TEST}$$

Hệ thống thu thập dữ liệu Controlled Webcam Pilot đã hoàn tất 100% tích hợp đầu-cuối:
- Metadata được truyền chính xác từ tương tác người dùng trên giao diện web qua HTTP headers & body tới Backend.
- Mỗi frame trong trial được gắn đúng `expected_sign`, `trial_id` duy nhất và `frame_idx` tăng đơn điệu.
- Ngoài trial, hệ thống tự động ngắt metadata, ngăn chặn tuyệt đối hiện tượng rò rỉ nhãn chuẩn.
- Giao diện và API đã sẵn sàng phục vụ cho đợt thử nghiệm webcam khoa học có kiểm soát.
