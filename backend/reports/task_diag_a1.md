# BÁO CÁO ĐIỀU TRA KỸ THUẬT: TRUY VẾT LỖI KÝ HIỆU CHỮ A BỊ NHẬN THÀNH CHỮ KHÁC (TASK DIAG-A1)

**Dự án:** VSL Connect AI — Hệ thống giao tiếp hai chiều Ngôn ngữ Ký hiệu Việt Nam  
**Giai đoạn:** Task DIAG-A1 — Chẩn đoán lỗi nhận diện ký hiệu chữ A  
**Ngày thực hiện:** 30/10/2026  
**Kỹ sư thực hiện:** Senior Computer Vision / AI Engineer  
**Trạng thái:** Đã tái hiện thành công 100% nguyên nhân gốc rễ (Root Cause Confirmed)  

---

## 1. Tóm tắt Hiện tượng và Phát hiện Cốt lõi

- **Hiện tượng:** Khi người dùng thực hiện ký hiệu chữ cái **A** trước camera trong chế độ `letters`, hệ thống không hiển thị chữ A mà hiển thị chữ cái khác (cụ thể là **chữ Z** với độ tin cậy tuyệt đối $\approx 1.0$).
- **Phát hiện cốt lõi (Root Cause):**
  1. **Luồng camera production KHÔNG HỀ GỌI `VSLAlphabetClassifier`:** Module độc lập `backend/vision/alphabet_classifier.py` (vừa xây dựng ở Task 3B.1) chưa từng được tích hợp vào luồng nhận diện camera của backend (`recognizer.py`).
  2. **Hàm heuristic cũ bị mạng nơ-ron MLP ghi đè (Override):** Trong luồng camera hiện tại, chữ cái được nhận diện bằng hàm cũ `classify_geometry` trong `recognizer.py`. Trong hàm này, độ tin cậy của quy tắc chữ A chỉ được đặt là `0.84`. Vì $0.84 < 0.85$ (ngưỡng ưu tiên hình học của ensemble), nhánh MLP (`VSLGestureNet`) được kích hoạt.
  3. **Mạng MLP phân loại nhầm cử chỉ A thành Z:** Mạng nơ-ron tĩnh MLP dự đoán đặc trưng của chữ A thành nhãn **"Z"** với xác suất $99.9\% - 100\%$, dẫn đến nhãn cuối cùng bị ép thành "Z" và hiển thị lên giao diện người dùng.

---

## 2. Truy vết Chi tiết Luồng Xử lý Thực tế từ Camera đến Giao diện

Bảng truy vết từng chặng xử lý từ lúc webcam thu hình ảnh đến khi nhãn xuất hiện trên UI:

```
[Webcam] -> [Canvas 320x240] -> [MediaPipe Hands (Wasm)]
                |
                v
[CameraSignRecognizer.jsx: onHandsResultsFast]
                |
                v (21 landmarks [x,y,z])
[CameraSignRecognizer.jsx: sendLandmarksForRecognition]
                |
                v (HTTP POST /api/recognize_landmarks, mode="letters")
[backend/app.py: recognize_landmarks()]
                |
                v
[backend/vision/recognizer.py: predict_landmarks(mode="letters")]
   ├── Gọi: self.classify_geometry(p_prim) -> Trả về "A", conf = 0.84
   │        (LƯU Ý: VSLAlphabetClassifier KHÔNG ĐƯỢC GỌI!)
   ├── Kiểm tra Ensemble: conf = 0.84 < 0.85 (Trượt ngưỡng hình học!)
   ├── Kích hoạt MLP: self.model(tensor_in) -> Trả về "Z", conf = 1.00
   └── Quyết định cuối: final_sign = "Z" (MLP ghi đè "A" thành "Z")
                |
                v (Response JSON: {"sign": "Z", "label": "Z", ...})
[CameraSignRecognizer.jsx: setCurrentResult] -> HIỂN THỊ "Z" TRÊN MÀN HÌNH
```

### Vị trí các File và Hàm cụ thể:
1. **Lấy hình ảnh từ camera:**
   - File: [`src/components/CameraSignRecognizer.jsx`](file:///d:/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L225-L315)
   - Hàm: `startCamera()` (dòng 225) và vòng lặp `processCameraFrame()` (dòng 290) dùng Canvas 320x240 tối ưu GPU để gửi frame vào MediaPipe.
2. **Phát hiện bàn tay bằng MediaPipe:**
   - File: [`src/components/CameraSignRecognizer.jsx`](file:///d:/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L233-L243)
   - Tracker: `window.Hands` (MediaPipe Hands Lite modelComplexity: 0, 60 FPS).
   - Callback nhận mốc: `onHandsResultsFast(results)` (dòng 589).
3. **Chuyển landmark thành dữ liệu đầu vào:**
   - File: [`src/components/CameraSignRecognizer.jsx`](file:///d:/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L761-L767)
   - Trích xuất: `rawPrimary = primary.map(lm => [lm.x, lm.y, lm.z])`.
   - Gửi API: `sendLandmarksForRecognition(rawPrimary, multiHands, 'Right', 'letters')` trong [`src/services/api.js`](file:///d:/SIGN_LANGUAGE/src/services/api.js#L139-L158).
4. **Tiếp nhận API Backend:**
   - File: [`backend/app.py`](file:///d:/SIGN_LANGUAGE/backend/app.py#L230-L252)
   - Hàm: `recognize_landmarks(req: LandmarkRequest, request: Request)` gọi:
     `recognizer.predict_landmarks(landmarks_list=req.landmarks, mode="letters")`.
5. **Bộ nhận diện trong backend:**
   - File: [`backend/vision/recognizer.py`](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L408-L640)
   - Hàm: `predict_landmarks(..., mode="letters")`.
   - **Xác nhận:** `VSLAlphabetClassifier` (`backend/vision/alphabet_classifier.py`) **HOÀN TOÀN KHÔNG ĐƯỢC GỌI**. Thay vào đó, hệ thống gọi hàm nội bộ cũ `self.classify_geometry(p_prim, finger_states)` (dòng 571) và mạng nơ-ron MLP `self.model` (dòng 605).
6. **Chuyển kết quả thành nhãn trên giao diện:**
   - File: [`src/components/CameraSignRecognizer.jsx`](file:///d:/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L772-L780)
   - State: `setCurrentResult({ sign: data.sign, label: data.label, ... })`.

---

## 3. Tái hiện Lỗi và Bằng chứng Thực nghiệm

Chúng tôi đã chạy kiểm thử trực tiếp trên toàn bộ các mẫu landmark thực tế của chữ A trích xuất từ video QIPEDC chuẩn (`backend/tests/fixtures/real_alphabet_landmarks.json` — nguồn gốc video `D0489.mp4` của Bộ GD&ĐT):

### Bảng kết quả so sánh giữa các tầng xử lý:

| Frame ID | Pha cử chỉ | VSLAlphabetClassifier (Task 3B.1) | `recognizer.classify_geometry` cũ | Mạng MLP `VSLGestureNet` | Kết quả cuối cùng tại `predict_landmarks` (Hiển thị UI) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Frame 23** | transition | **A** (conf = 0.95, MATCH) | A (conf = 0.84) | **Z** (conf = 1.00) | ❌ **sign = "Z", label = "Z"** |
| **Frame 37** | stable_pose | **A** (conf = 0.95, MATCH) | A (conf = 0.84) | **Z** (conf = 0.99) | ❌ **sign = "Z", label = "Z"** |
| **Frame 46** | stable_pose | **A** (conf = 0.95, MATCH) | A (conf = 0.84) | **Z** (conf = 0.99) | ❌ **sign = "Z", label = "Z"** |
| **Frame 55** | stable_pose | **A** (conf = 0.95, MATCH) | A (conf = 0.84) | **Z** (conf = 0.99) | ❌ **sign = "Z", label = "Z"** |

### Chi tiết các đặc trưng hình học ghi nhận được:
- Số mốc: Đủ 21 điểm `[x, y, z]` hợp lệ, không có NaN/Inf.
- Trạng thái ngón: `thumb_extended = True`, các ngón trỏ, giữa, áp út, út đều gập chặt (`curl > 0.50`).
- Độ vươn ngón cái: $d_{\text{TW}} = 1.40 - 1.49 \ge 1.32$, $d_{\text{TM}} = 0.80 - 0.90 \ge 0.75$.
- Module `VSLAlphabetClassifier` đánh giá: **MATCH chữ A hoàn hảo với confidence 0.95**.

---

## 4. Phân loại và Kết luận Nguyên nhân

Theo các nhóm nguyên nhân định nghĩa trong yêu cầu:

> **Kết luận phân loại: Nhóm A kết hợp Nhóm D**
> - **Nguyên nhân chính (Nhóm A): Luồng camera không gọi classifier dự kiến.**  
>   Module độc lập `VSLAlphabetClassifier` (`backend/vision/alphabet_classifier.py`) đã hoàn thành và đạt 100% unit tests, nhưng chưa được tích hợp vào hàm `predict_landmarks` của `backend/vision/recognizer.py` cho chế độ `mode="letters"`.
> - **Cơ chế gây lỗi trực tiếp (Nhóm D): Bước hậu xử lý / Ensemble bị mạng MLP ghi đè nhãn.**  
>   Trong `backend/vision/recognizer.py`:
>   ```python
>   # Dòng 337-343: Hàm hình học cũ classify_geometry hardcode conf = 0.84
>   if thumb_curl <= 0.40 and d_thumb_index >= 0.35:
>       return "A", 0.84
>
>   # Dòng 621-626: Ensemble logic
>   elif selected_geom_conf >= 0.85:
>       final_sign = selected_geom_sign
>   elif has_mlp and nn_conf > 0.6:
>       final_sign = nn_sign  # <-- VÌ 0.84 < 0.85, MLP ĐƯỢC CHỌN VÀ ÉP NHÃN THÀNH "Z"!
>   ```
>   Vì độ tin cậy của quy tắc chữ A cũ là **0.84**, thấp hơn ngưỡng chặn **0.85**, nên mạng nơ-ron MLP được phép ghi đè, và mô hình MLP phân loại sai chữ A thành chữ **Z**!

---

## 5. Hướng dẫn Lấy Log Chẩn đoán Thời gian Thực khi Người dùng Ký hiệu A

Dự án đã có sẵn hệ thống chẩn đoán `VSL_DIAGNOSTIC` được thiết kế trong `recognizer.py`. Để theo dõi log trực tiếp khi người dùng đứng trước camera:

1. **Bật chế độ chẩn đoán khi khởi động backend:**
   ```powershell
   $env:VSL_DIAGNOSTIC="1"
   python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
   ```
2. **Kiểm tra file log chẩn đoán:**
   Các bản ghi sẽ được tự động lưu theo định dạng JSONL tại:
   `backend/data/diagnostics/diag_session_<timestamp>.jsonl`
3. **Trường dữ liệu quan sát được:**
   Mỗi frame sẽ ghi nhận:
   - `geom_sign`: Kết quả từ lớp hình học (ví dụ: `"A"`, conf: `0.84`).
   - `nn_sign`: Kết quả từ mạng MLP (ví dụ: `"Z"`, conf: `0.99`).
   - `final_sign`: Nhãn quyết định cuối cùng (ví dụ: `"Z"`).
   - `mode`: `"letters"`.

---

## 6. Đề xuất Hướng Xử lý Kỹ thuật (Cho bước tiếp theo)

Theo đúng quy tắc của Task DIAG-A1: *"Trước tiên hãy điều tra và báo cáo. Chỉ sửa sau khi tìm được nguyên nhân có bằng chứng"*.

Khi được phép chỉnh sửa, giải pháp triệt để và an toàn nhất gồm 2 bước:
1. **Tích hợp `VSLAlphabetClassifier` vào `recognizer.predict_landmarks` cho chế độ `mode == "letters"`:**
   Tương tự như cách đã triển khai thành công cho chữ số (`is_digits_mode` gọi `digit_classifier` độc lập), khi `is_letters_mode`, hệ thống sẽ gọi trực tiếp `alphabet_classifier.classify_detailed(p_prim)`.
2. **Thiết lập quyền ưu tiên tuyệt đối cho `VSLAlphabetClassifier` khi `status == "MATCH"`:**
   Khi bộ phân loại hình học chuyên biệt đã xác nhận chữ cái với độ tin cậy cao (ví dụ: `confidence = 0.95`), không để mạng nơ-ron MLP ghi đè nhãn.

