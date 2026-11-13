**PROMPT ID:** A01
**PROMPT NAME:** ANTIGRAVITY | CẢI THIỆN NHẬN DIỆN CỬ CHỈ & ĐÁNH GIÁ CHECKPOINT
**GIAI ĐOẠN:** A01.1 — Khắc phục Hands fallback và kiểm tra landmark

# BÁO CÁO TIẾN ĐỘ GIAI ĐOẠN A01.1: WORDS HOLISTIC RUNTIME INTEGRATION

- **Thời gian thực hiện:** 2026-11-11
- **Phạm vi:** Tích hợp và khóa cứng MediaPipe Holistic cho chế độ `words` (63 từ vựng, lớp 39..101); chặn triệt để `hands_fallback_pose` và pose giả đưa vào SSTCN_Attention; giữ nguyên interface tensor `(1, 3, 30, 75)`; bảo toàn hoàn toàn các chế độ `letters`, `digits`, `sentence`.

---

## 1. Trạng Thái Pipeline Trước Khi Chỉnh Sửa

Qua kiểm tra mã nguồn thực tế và đối chiếu 14 session replay từ audit VSL-P010:
1. **Kiến trúc Dual-Engine gây lọt Hands Fallback:**
   - Trong [`CameraSignRecognizer.jsx`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx), mặc dù hàm `handleModeChange('words')` có lệnh chuyển `targetScope = 'holistic'`, người dùng vẫn có thể bấm nút "Bàn tay" (Scope A) trên giao diện trong khi tab "Từ (102 Từ)" đang hoạt động.
   - Khi camera bật trước khi chuyển mode hoặc khi chuyển đổi tracker gặp lỗi mạng/Wasm, hệ thống âm thầm giữ engine `Hands` và gọi hàm `extract75KeypointsFromHands(results)`.
2. **Pose giả tĩnh (`DEFAULT_POSE`):**
   - Hàm `extract75KeypointsFromHands` gán cứng 29/33 khớp pose (bao gồm hai vai `[0.65, 0.45]` và `[0.35, 0.45]`) là hằng số tĩnh ($\sigma = 0$), khuỷu tay là trung điểm tuyến tính giả với độ sâu $z = 0$.
   - Khi người dùng thực hiện các từ cần chuyển động cánh tay và thân trên (ví dụ: *"Buổi tối"*, *"Chào"*, *"Bạn"*, *"Tên là gì"*), mô hình SSTCN_Attention nhận đồ thị 75 khớp bị đóng băng thân trên, làm xác suất Top-1 sụt giảm nghiêm trọng từ ~15% (`holistic_raw`) xuống ~3% (`hands_fallback_pose`), thậm chí 0% đối với *"Tên là gì"*.
3. **Thiếu cơ chế xử lý khi Holistic không khả dụng:**
   - Trước đây nếu Holistic CDN không tải được hoặc gặp lỗi Wasm, hệ thống có nguy cơ crash hoặc tự động rơi về `Hands`, tạo cảm giác hệ thống vẫn nhận diện nhưng thực chất đang nạp dữ liệu rác vào SSTCN.

---

## 2. Danh Sách File Đã Kiểm Tra & Chỉnh Sửa

### 2.1. File đã kiểm tra (Read & Verify):
- [`backend/models/sstcn_model.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/models/sstcn_model.py): Xác minh kiến trúc `SSTCN_Attention`, đồ thị `build_vsl_adjacency()`, đầu vào `(Batch, Channels=3, Time=30, Joints=75)`.
- [`backend/models/vsl_102_dictionary.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/models/vsl_102_dictionary.py): Xác minh 102 lớp (`0..28`: chữ cái, `29..38`: chữ số, `39..101`: 63 từ vựng, `102`: UNKNOWN/NEUTRAL).
- [`backend/training/prepare_sstcn.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/training/prepare_sstcn.py) & [`backend/training/sstcn_data.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/training/sstcn_data.py): Xác minh hợp đồng dữ liệu chuẩn hóa `mediapipe_image_normalized_xyz`, thứ tự `33 Pose + 21 Left Hand + 21 Right Hand`, điền `[0, 0, 0]` khi thiếu khớp.
- [`index.html`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/index.html): Xác minh nạp song song `@mediapipe/holistic` và `@mediapipe/hands`.

### 2.2. File đã chỉnh sửa (Minimal & Targeted Changes):
1. **[`src/components/CameraSignRecognizer.jsx`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx):**
   - **Khóa cứng Holistic cho mode `words`:** Trong `handleScopeChange`, nếu `conversionMode === 'words'` và người dùng chọn `hands`, hành động bị từ chối và ghi log `ENGINE_FALLBACK_BLOCKED`.
   - **Vô hiệu hóa trực quan nút "Bàn tay" ở mode `words`:** Thêm `disabled={conversionMode === 'words'}` với opacity `0.45` và tooltip giải thích rõ *"Chế độ Từ vựng yêu cầu Toàn thân (Holistic 75 khớp thật)"*.
   - **Chốt chặn luồng frame tại `onHandsResultsFast`:** Bổ sung kiểm tra `if (conversionModeRef.current === 'words') return;` để ngăn chặn bất kỳ callback Hands cũ nào gửi frame vào buffer khi đang ở chế độ từ vựng.
   - **Xử lý ngắt mềm khi Holistic không khả dụng:** Thêm `applyHolisticUnavailableState` đặt nhãn `Holistic không khả dụng — Ngừng dự đoán`, xóa buffer, dừng gửi frame suy luận, không gây crash và ghi log `HOLISTIC_UNAVAILABLE`.
   - **Chuẩn hóa trích xuất 75 khớp Holistic:** Hàm `extract75KeypointsFromHolistic` đọc an toàn `[safeCoord(lm?.x), safeCoord(lm?.y), safeCoord(lm?.z)]`, điền đúng `[0, 0, 0]` cho bộ phận thiếu, tuyệt đối không dùng `DEFAULT_POSE`.
   - **Truyền `debugMeta` chi tiết:** Luôn đính kèm `tracker_engine` và `active_instance_engine` khi gửi chuỗi 30 frame của mode `words` về backend.
2. **[`backend/vision/recognizer.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/vision/recognizer.py):**
   - **Chốt chặn an toàn tại `predict_sequence`:** Trước khi đưa tensor vào `self.sstcn_model`, kiểm tra tính hợp lệ bằng `analyze_sequence_landmarks(arr, debug_meta)`.
   - **Chặn đứng Pose giả / chuỗi mất Pose:** Nếu phát hiện `hands_fallback_pose` (vai cố định hằng số `DEFAULT_POSE` hoặc $\ge 25$ khớp pose bất động khác 0) hoặc chuỗi mất toàn bộ pose/hai bàn tay, trả về ngay lập tức:
     - `fallback_blocked: True`
     - `sign: "NEUTRAL"`
     - `confidence: 0.0`
     - `confirmed: False`
     - Thông báo nhãn rõ ràng: *"Đã chặn Pose giả (Yêu cầu MediaPipe Holistic)"* hoặc *"Tạm mất khung xương thân trên (Pose) — Ngừng dự đoán"*.
     - Ghi nhận sự kiện `FALLBACK_BLOCKED` vào hệ thống Word Debug Logging nếu đang có phiên test.
   - **Giữ nguyên 100% logic cho các chế độ khác:** `predict_landmarks` cho `letters`/`digits` và `predict_sequence` cho `sentence` hoàn toàn không bị ảnh hưởng.
3. **[`backend/vision/word_debugger.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/vision/word_debugger.py):**
   - Bổ sung hàm `analyze_sequence_landmarks`: phân tích độ biến thiên của vai, khuỷu tay, cổ tay; phân biệt chính xác giữa khớp bị thiếu (tọa độ bằng 0) và khớp giả mạo (tọa độ khác 0 nhưng $\sigma < 10^{-4}$).
   - Bổ sung hàm `record_blocked_sequence` và mở rộng whitelist sự kiện `FALLBACK_BLOCKED`, `HOLISTIC_UNAVAILABLE`, `ENGINE_FALLBACK_BLOCKED`.
4. **[`backend/app.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/app.py) & [`src/services/wordDebugApi.js`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/services/wordDebugApi.js):**
   - Mở rộng model `WordDebugEventRequest` và hàm `logWordDebugEvent` để truyền tải đầy đủ trạng thái engine và mốc xương bị mất.

---

## 3. Kết Quả Kiểm Thử

### 3.1. Kiểm thử tự động (Automated Regression Tests)
Kịch bản kiểm thử độc lập [`scratch/test_a01_1_holistic_integration.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/scratch/test_a01_1_holistic_integration.py) đã thực thi trên môi trường Python dự án với **36/36 bài test đạt PASS (100%)**:
1. **TEST 1 — Cấu trúc từ điển & Mapping 102 lớp:**
   - `[PASS]` Đủ 102 lớp mô hình (`0..101`) trong `SIGN_DICT_102`.
   - `[PASS]` Đúng 63 từ vựng trong khoảng `[39, 101]` (`Thời gian` → `Bộ đội`).
2. **TEST 2 — Phân tích Landmark & Phát hiện Pose giả:**
   - `[PASS]` Nhận diện chính xác `holistic_raw` khi vai và cánh tay chuyển động thật.
   - `[PASS]` Phát hiện và gắn cờ `is_synthetic_pose = True`, `landmarks_source = 'hands_fallback_pose'` khi có dấu hiệu `DEFAULT_POSE`.
   - `[PASS]` Nhận diện chính xác `holistic_pose_missing` khi toàn bộ pose bằng 0.
   - `[PASS]` Nhận diện chính xác `both_hands_missing` khi cả hai bàn tay đều bằng 0.
3. **TEST 3 — Chốt chặn Runtime tại `recognizer.predict_sequence`:**
   - `[PASS]` Khi đưa `hands_fallback_pose` vào mode `words`: trả về `fallback_blocked=True`, `sign='NEUTRAL'`, `confidence=0.0`, không crash mô hình.
   - `[PASS]` Khi đưa `holistic_pose_missing` vào mode `words`: trả về `fallback_blocked=True`, `blocked_reason='holistic_pose_missing'`.
   - `[PASS]` Khi đưa `holistic_raw` thật vào mode `words`: suy luận thành công, sinh ra Top-5 dự đoán với Top-1 thuộc đúng khoảng `[39, 101]`.
4. **TEST 4 — Đảm bảo tính độc lập của các chế độ khác:**
   - `[PASS]` Mode `letters` (nhận diện 21 điểm mốc ngón tay qua MLP & Rules) hoạt động bình thường, định dạng trả về không đổi.
   - `[PASS]` Mode `digits` (nhận diện chữ số 1–9) hoạt động bình thường.
   - `[PASS]` Mode `sentence` hoạt động bình thường.
5. **TEST 5 — Word Debug Logging:**
   - `[PASS]` Ghi nhận chính xác sự kiện `FALLBACK_BLOCKED` khi phát hiện chuỗi không đạt chuẩn.
   - `[PASS]` Ghi nhận sự kiện `PREDICTION` với đầy đủ thống kê độ biến thiên vai/khuỷu tay khi chuỗi đạt chuẩn.
6. **Kiểm tra Frontend Build:**
   - `[PASS]` Lệnh `npm run build` biên dịch thành công production bundle trong `832ms` không phát sinh bất kỳ cảnh báo lỗi cú pháp nào.

### 3.2. Kiểm thử Webcam thực tế (Manual Webcam Testing)
*Lưu ý minh bạch theo quy định A01.1: Hiện tại các bài test đã chạy là automated test trên môi trường backend/frontend code; chưa thực hiện phiên mở webcam người thật trước máy.*

Dưới đây là **Quy trình Kiểm thử Thủ công (Manual Webcam Test Guide)** dành cho người dùng kiểm chứng trên 3 từ trọng điểm:

#### Chuẩn bị kiểm thử:
1. Chạy Backend: `.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000`
2. Mở trình duyệt với cờ debug: `http://localhost:5173/?wordDebug=1`
3. Nhấp chọn tab **"Từ (102 Từ)"** (Kiểm tra nút "Bàn tay" đã bị vô hiệu hóa mờ, chỉ còn "Toàn thân" hoạt động).
4. Bấm **"Bật Camera"**: kiểm tra camera bật, HUD hiển thị `FPS` và vòng tròn frame `0/30`.

#### Test Case 1: Từ "Buổi tối" (Lớp 49 — Chuyển động cánh tay & vai)
- **Mục tiêu:** Xác minh chuyển động vai và cánh tay không bị đóng băng.
- **Thực hiện:**
  - Trong Word Debug Panel, chọn Target: `Buổi tối (class 49)`, Trial: `1`, bấm **"Start Test"**.
  - Người thực hiện đứng cách camera 1.2–1.5m, hai tay giơ trước ngực rồi hạ chéo xuống mô phỏng trời tối.
  - Sau khi kết thúc cử chỉ, bấm **"Stop Test"**.
- **Tiêu chí PASS:**
  - Trong bảng kết quả Trial: `landmarks_source` hiển thị `holistic_raw`.
  - Khớp vai hiển thị `(Vai động ✓)`, không có cảnh báo `(Vai tĩnh ⚠️)`.
  - File `events.jsonl` ghi nhận `fixed_pose_landmarks_count == 0` và `shoulder_left_std > 0.005`.

#### Test Case 2: Từ "Chào" (Lớp 52 — Hai nhịp chuyển động vẫy tay)
- **Mục tiêu:** Xác minh tracking bàn tay và khớp cổ tay/khuỷu tay qua cử chỉ động.
- **Thực hiện:**
  - Chọn Target: `Chào (class 52)`, Trial: `1`, bấm **"Start Test"**.
  - Đưa tay phải lên ngang thái dương và vẫy nhẹ 2 nhịp.
  - Bấm **"Stop Test"**.
- **Tiêu chí PASS:**
  - `landmarks_source` là `holistic_raw`.
  - `right_hand_zero_frames < 10` trong suốt cử chỉ.
  - Không xuất hiện event `FALLBACK_BLOCKED`.

#### Test Case 3: Từ "Bạn" (Lớp 60 — Động tác chỉ tay về phía trước)
- **Mục tiêu:** Xác minh khớp khuỷu tay và chiều sâu $z$ khi tay duỗi thẳng về phía camera.
- **Thực hiện:**
  - Chọn Target: `Bạn (class 60)`, Trial: `1`, bấm **"Start Test"**.
  - Đưa ngón trỏ chỉ thẳng về phía camera.
  - Bấm **"Stop Test"**.
- **Tiêu chí PASS:**
  - `landmarks_source` là `holistic_raw`.
  - Không xuất hiện nhầm lẫn sang `Tháng` như trong các session `hands_fallback_pose` cũ.

---

## 4. Các Vấn Đề Còn Tồn Tại & Cần Xử Lý ở Giai Đoạn Sau

Mặc dù việc tích hợp Holistic và chặn Pose giả đã hoàn tất 100%, các hạn chế sau thuộc phạm vi của các giai đoạn tiếp theo (chưa can thiệp ở A01.1 theo đúng ràng buộc):
1. **Tốc độ khung hình của Holistic WebAssembly (~13–15 FPS):**
   - Khi chạy MediaPipe Holistic trên trình duyệt, tốc độ trung bình đạt ~13.6 FPS khiến cửa sổ 30 frame kéo dài ~2.2–2.5 giây (dài hơn thời lượng thực tế của cử chỉ ~1.2 giây). Vấn đề này sẽ được giải quyết ở giai đoạn **A01.2 (Temporal Resampling & Motion-based Segmentation)**.
2. **Hiện tượng cắt cử chỉ khi mất tay 4 frame (`SEQUENCE_RESET`):**
   - Bộ đệm FIFO vẫn reset khi mất tay 4 frame liên tiếp (~150ms). Khi người dùng thực hiện cử chỉ có nhịp xoay úp bàn tay, cử chỉ vẫn có nguy cơ bị đứt đoạn. Vấn đề này thuộc phạm vi cải thiện phân đoạn của **A01.2**.
3. **Độ hội tụ của Checkpoint:**
   - Checkpoint `sign_sstcn_reconstructed_model.pth` hiện tại ở local dừng ở Epoch 10. Checkpoint mới nhất được train trên Kaggle (Epoch 16, Test Acc 92.25%) sẽ được đánh giá đối chứng độc lập trong **A01.3**.

---

## 5. Xác Nhận Phạm Vi Code Không Bị Ảnh Hưởng

- **Chế độ Chữ cái (`letters` / `letters_numbers`):** Giữ nguyên MediaPipe Hands 60 FPS, mạng MLP và hệ thống Geometric Rules; không thay đổi bất kỳ logic nào.
- **Chế độ Chữ số (`digits`):** Giữ nguyên quy tắc phân loại hình học và bộ phát hiện chữ số 10.
- **Chế độ Ghép câu (`sentence`):** Giữ nguyên logic debouncing và ghép chuỗi từ.
- **Trọng số & Kiến trúc Model:** Không thay đổi checkpoint, không cập nhật trọng số, giữ nguyên kiến trúc 102 lớp.
- **Training Code & Dataset:** Không can thiệp mã nguồn huấn luyện hay tập dữ liệu gốc.

---

## 6. Đề Xuất Bước Tiếp Theo

1. Chuyển sang **Giai đoạn A01.2**: Triển khai phân đoạn cử chỉ theo năng lượng chuyển động (Motion-Energy / Wrist Velocity Segmentation) kết hợp lấy mẫu tuyến tính `np.linspace(0, T-1, 30)` để loại bỏ hoàn toàn độ trễ 2.5 giây của cửa sổ FIFO, đồng thời khắc phục triệt để hiện tượng cắt một cử chỉ phức thành 2 lần dự đoán.
2. Chuyển sang **Giai đoạn A01.3**: Tải checkpoint Reconstructed Best mới từ Kaggle Version 3 (Epoch 16, Test Acc 92.25%) về thư mục weights và chạy đối chứng A/B/C với Historical (Epoch 32) và Reconstructed cũ (Epoch 10).

