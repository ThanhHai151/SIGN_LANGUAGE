# BÁO CÁO GIAI ĐOẠN 2C.11 — KHẮC PHỤC LỖI CHUYỂN CHẾ ĐỘ TRỰC TIẾP TRÊN CAMERA (LIVE MODE SWITCHING)

**Dự án:** VSL Connect AI  
**Thời điểm thực hiện:** 29/10/2026  
**Mục tiêu:** Khắc phục triệt để hiện tượng khi chuyển tab từ `letters` sang `digits` trên giao diện người dùng lúc camera đang chạy nhưng hệ thống vẫn tiếp tục trả về kết quả chữ cái thay vì chuyển sang nhận diện chữ số.  
**Cam kết phạm vi:** Tuyệt đối không thay đổi thuật toán nhận diện chữ số (0–9), không chỉnh sửa trọng số model hay checkpoint, không sửa fixture dữ liệu chuẩn, giữ nguyên hợp đồng 4 chế độ độc lập.

---

## 1. Nguyên nhân gốc rễ được xác minh (Root Cause Verification)

Sau khi truy vết toàn diện luồng dữ liệu từ giao diện React, vòng lặp camera MediaPipe tới API backend, ba nguyên nhân kỹ thuật cốt lõi đã được xác định:

### 1.1. Hiện tượng Stale Closure trong Vòng lặp MediaPipe Callback (Nguyên nhân chính)
- Vòng lặp camera MediaPipe (`Hands` hoặc `Holistic`) đăng ký hàm callback kết quả một lần duy nhất khi khởi động camera (`tracker.onResults(onHandsResultsFast)` trong `startCamera`).
- Hàm `onHandsResultsFast` và `handleMotionAndSequence` đóng (closure) lấy giá trị trạng thái React tại thời điểm khởi tạo (`conversionMode = 'letters'`).
- Khi người dùng click nút đổi sang `digits`, React kích hoạt re-render nhưng vòng lặp WebAssembly / requestAnimationFrame của MediaPipe vẫn gọi callback cũ được closure với biến `conversionMode` và `recognitionMode` ban đầu.
- Hậu quả: Dù giao diện đã hiển thị tab "Chữ số", MediaPipe loop vẫn liên tục gửi payload với `mode: 'letters'` về backend.

### 1.2. Hiện tượng Out-of-Order của In-Flight Requests (Race Condition)
- Khi người dùng bấm chuyển tab, một hoặc nhiều request nhận diện chữ cái gửi đi từ frame trước đó vẫn đang trong quá trình truyền qua mạng (in-flight).
- Khi Promise của request cũ hoàn tất (`.then((resp) => ...)`), kết quả phản hồi của chế độ `letters` đến trễ đã ghi đè vào `currentResult` của giao diện người dùng, tạo cảm giác hệ thống "không chịu chuyển mode".

### 1.3. Trạng thái Debounce và Smoothing trên Backend
- Khi chuyển đổi mode trên cùng một phiên camera trực tiếp (`session_id`), Recognizer trên backend đã có cơ chế dọn dẹp `prediction_history` nhưng chưa xóa `last_confirmed_sign`, dẫn đến việc debounce ký hiệu của chế độ trước có khả năng ảnh hưởng đến ký hiệu đầu tiên của chế độ mới.

---

## 2. Luồng Xử lý Trước và Sau Khi Khắc phục

### 2.1. Trước khi sửa (Lỗi Stale Closure & Out-of-Order Override)
```
[User Clicks "Chữ số"]
       │
       ▼
React State: conversionMode = 'digits'
       │
       ├──> MediaPipe Loop (Stale Closure) ──> Đọc conversionMode cũ ('letters') ──> Gửi POST /api/recognize_landmarks (mode: 'letters')
       │                                                                                 │
       │                                                                                 ▼
       │                                                                           Backend xử lý Mode Letters (Trả về chữ cái D)
       │                                                                                 │
       └──> In-Flight Request cũ hoàn tất ───────────────────────────────────────────────┴──> UI ghi đè: currentResult = 'D' (CHỮ CÁI)
```

### 2.2. Sau khi sửa (Đồng bộ tức thì qua Refs, Generation Counter & UI Guard)
```
[User Clicks "Chữ số"]
       │
       ├──> conversionModeRef.current = 'digits' (Đồng bộ tức thì)
       ├──> recognitionModeRef.current = 'instant'
       ├──> requestGenRef.current += 1 (Tăng thế hệ, vô hiệu hóa toàn bộ request cũ)
       ├──> isInferringRef.current = false
       │
       ▼
MediaPipe Loop (Frame tiếp theo)
       │
       ├──> Đọc trực tiếp conversionModeRef.current ('digits')
       ├──> Gửi POST /api/recognize_landmarks (mode: 'digits', requestGen: N)
       │
       ├──> Backend nhận diện: Reset debounce (last_confirmed_sign = None) -> Trả về '1' (Số)
       │
       └──> Promise .then():
              ├── Kiểm tra: currentGen === requestGenRef.current && conversionModeRef.current === requestMode? (HỢP LỆ)
              └── Chốt chặn bảo vệ UI (UI Guard): Bỏ qua nếu nhãn không khớp regex /^[0-9]$/
                     │
                     ▼
              Cập nhật an toàn: currentResult = '1' (CHỮ SỐ VSL)
```

---

## 3. Các File và Hàm đã Chỉnh sửa

### 3.1. `src/components/CameraSignRecognizer.jsx`
- **Thêm Refs đồng bộ trạng thái thời gian thực:**
  ```javascript
  const conversionModeRef = useRef(conversionMode);
  const recognitionModeRef = useRef(recognitionMode);
  const requestGenRef = useRef(0);

  useEffect(() => { conversionModeRef.current = conversionMode; }, [conversionMode]);
  useEffect(() => { recognitionModeRef.current = recognitionMode; }, [recognitionMode]);
  ```
- **Cập nhật `handleModeChange(newMode)`:**
  - Đồng bộ ngay lập tức giá trị mới vào `conversionModeRef.current` và `recognitionModeRef.current`.
  - Tăng `requestGenRef.current += 1` để đánh dấu thế hệ mới, tự động hủy bỏ kết quả của tất cả request đang bay của mode cũ.
  - Reset cờ `isInferringRef.current = false` để frame tiếp theo được gửi đi ngay lập tức không bị nghẽn.
  - Xóa buffer chuỗi động `sequenceBufferRef.current = []`.
- **Cập nhật `handleMotionAndSequence`:**
  - Đọc `currentConversionMode` và `currentRecognitionMode` trực tiếp từ các refs thay vì đọc closure state.
  - Trong callback `.then()`, kiểm tra nghiêm ngặt `currentGen !== requestGenRef.current || conversionModeRef.current !== requestMode` để loại bỏ phản hồi stale.
  - Bổ sung **Chốt chặn bảo vệ UI (UI Guard)**:
    - Ở chế độ `digits`: Loại bỏ mọi nhãn không phải chữ số `0–9` hoặc `NEUTRAL`.
    - Ở chế độ `letters`: Loại bỏ mọi nhãn là chữ số `0–9`.

### 3.2. `backend/vision/recognizer.py`
- Trong hàm `predict_landmarks`: Khi phát hiện phiên (`session_id`) thay đổi `mode`, bổ sung thêm `self.last_confirmed_sign = None` cùng với `self.prediction_history.clear()`, đảm bảo trạng thái debounce của chế độ trước được dọn sạch hoàn toàn.

### 3.3. `backend/tests/test_mode_separation.py`
- Bổ sung 4 kiểm thử hồi quy nghiêm ngặt:
  1. `test_13_live_camera_mode_switching_letters_to_digits`: Mô phỏng chuỗi 5 frame `letters` liên tục, người dùng chuyển sang `digits` trực tiếp trên cùng một camera session; các frame sau đó lập tức trả về chữ số mà không bị kẹt nhãn chữ cái.
  2. `test_14_mode_switch_clears_prediction_history_and_last_confirmed_sign`: Kiểm tra `prediction_history` và `last_confirmed_sign` bị xóa sạch khi đổi mode, triệt tiêu nguy cơ debounce rò rỉ.
  3. `test_15_strict_no_cross_mode_labels_during_rapid_switch`: Kiểm tra chuyển đổi qua lại nhanh liên tục giữa `letters` và `digits` 5 chu kỳ liên tiếp; mode `digits` không bao giờ sinh nhãn chữ và mode `letters` không bao giờ sinh nhãn số.
  4. `test_16_zero_digit_vs_letter_mode_switching`: Kiểm tra chuyển đổi giữa số 0 và chữ cái trên cùng mẫu mốc hình học số 0; mode `digits` trả về '0', mode `letters` tuyệt đối không trả về '0'.

---

## 4. Kết quả Kiểm thử Thực tế

### 4.1. Bộ Kiểm thử Hồi quy Backend (Backend Test Suite)
- **Công cụ:** Pytest 9.1.1 kết hợp Python 3.12 (môi trường dự án)
- **Kết quả:**
  ```
  collected 206 items
  backend/tests/test_api.py .................... PASSED
  backend/tests/test_diagnostic.py ............ PASSED
  backend/tests/test_digit_0_landmarks.py ...... PASSED
  backend/tests/test_digit_classifier.py ....... PASSED
  backend/tests/test_digit_classifier_e2e.py ... PASSED
  backend/tests/test_digit_classifier_integration.py PASSED
  backend/tests/test_digit_letter_design.py .... PASSED
  backend/tests/test_digits.py ................. PASSED
  backend/tests/test_geometry.py ............... PASSED
  backend/tests/test_landmark_extractor.py ..... PASSED
  backend/tests/test_mode_filtering.py ......... PASSED
  backend/tests/test_mode_separation.py (16 tests) PASSED
  ====================== 206 passed, 2 warnings in 11.35s =======================
  ```
- **Tỷ lệ đạt:** **206 / 206 tests (100% PASSED)**, 0 failed, 0 errors.

### 4.2. Kiểm tra Frontend Build & Lint
- **Vite Build (`npm run build`):**
  - Biên dịch thành công 1898 modules trong 435ms, mã nguồn bundle hợp lệ 100%, không phát sinh lỗi cú pháp hay liên kết.
- **Oxlint (`npm run lint`):**
  - Quét 15 files với 104 rules, 0 errors.

---

## 5. Xác nhận Cam kết và Ranh giới Kỹ thuật

1. **Không sửa đổi thuật toán nhận diện số:** Các quy tắc hình học, phân loại và ngưỡng nhận diện số 0–9 trong `backend/vision/digit_classifier.py` được giữ nguyên vẹn 100%.
2. **Không sửa checkpoint và trọng số mô hình:** Mô hình MLP và SSTCN được giữ nguyên.
3. **Bảo tồn tính toàn vẹn của hợp đồng 4 chế độ:** Bốn chế độ `letters`, `digits`, `words`, `sentence` cùng chế độ lai tương thích ngược `letters_numbers` hoạt động hoàn toàn độc lập và không can thiệp chéo lẫn nhau.

