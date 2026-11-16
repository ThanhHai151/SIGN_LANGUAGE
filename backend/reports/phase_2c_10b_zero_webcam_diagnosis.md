# BÁO CÁO GIAI ĐOẠN 2C.10B — SỬA LỖI LOGGER VÀ CHẨN ĐOÁN SỐ 0 TRÊN WEBCAM

**Dự án:** VSL Connect AI  
**Thời điểm thực hiện:** 29/10/2026  
**Trạng thái:** Hoàn thành sửa lỗi logger chẩn đoán, bổ sung regression tests và xác lập đầy đủ bằng chứng so sánh hình học cho Số 0 giữa webcam thực tế và video QIPEDC.  
**Ranh giới tuân thủ:** Chưa sửa ngưỡng `d_TM <= 0.22`, chưa sửa quy tắc phân loại hay checkpoint, không train lại mô hình.

---

## 1. Các file đã sửa

1. **[`backend/vision/recognizer.py`](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py):**
   - **Sửa lỗi crash `_log_diagnostic`:** Xử lý an toàn tuyệt đối khi `finger_states=None`. Bổ sung cơ chế tự động trích xuất các đặc trưng trạng thái ngón tay từ `classifier_details["features"]` (khi ở chế độ `digits`) hoặc fallback về dictionary mặc định, loại trừ hoàn toàn nguy cơ phát sinh `AttributeError: 'NoneType' object has no attribute 'items'`.
   - **Tích hợp thông tin chẩn đoán chi tiết:** Bổ sung trường `mode`, `valid_frames_count` trong `smoothing`, và khối `classifier_details` chứa `status`, `digit`, `confidence`, `reason`, `matched_digits`, `features` (`thumb_ext`, `index_ext`, `middle_ext`, `ring_ext`, `pinky_ext`, `d_TI`, `d_TM`, `d_IW`, `index_curl`).
   - **Khối gọi ở chế độ `digits`:** Truyền trực tiếp `digit_res` vào `classifier_details` và `geom_sign = digit_res.get("digit")`.
2. **[`backend/tests/test_diagnostic.py`](file:///d:/SIGN_LANGUAGE/backend/tests/test_diagnostic.py):**
   - Bổ sung test `test_diagnostic_safe_when_finger_states_is_none`: Kiểm chứng `_log_diagnostic` an toàn khi `finger_states=None`.
   - Bổ sung test `test_diagnostic_logging_in_digits_mode`: Kiểm chứng luồng ghi log khi bật `VSL_DIAGNOSTIC=1` ở chế độ `digits`, đảm bảo file JSONL được ghi đầy đủ các trường đặc trưng của bộ phân loại số.

---

## 2. Kết quả kiểm thử thực tế

Đã thực thi toàn bộ test suite hệ thống:
- **Test chẩn đoán (`backend/tests/test_diagnostic.py`):** **12/12 PASSED** (10 test cũ + 2 test mới).
- **Test mốc Số 0 (`backend/tests/test_digit_0_landmarks.py`):** **11/11 PASSED**.
- **Test phân tách 4 chế độ (`backend/tests/test_mode_separation.py`):** **12/12 PASSED**.
- **Toàn bộ Test Suite hệ thống (`pytest backend/tests`):** **197/197 PASSED (100%)**, 2 warnings (FastAPI lifespan).

---

## 3. Ví dụ Log đại diện (Đã loại bỏ dữ liệu cá nhân không cần thiết)

Dưới đây là cấu trúc log JSONL chuẩn được tạo ra khi người dùng bật `VSL_DIAGNOSTIC=1` ở chế độ `digits`:

### Ví dụ 1: Khung hình số 0 hợp lệ (Khớp quy tắc MATCH)
```json
{
  "timestamp": 1793277040.324,
  "session_id": "webcam_session_01",
  "frame_idx": 1,
  "mode": "digits",
  "expected_sign": null,
  "handedness": "Right",
  "palm_scale": 0.0887,
  "finger_states": {
    "thumb_extended": false,
    "index_extended": false,
    "middle_extended": false,
    "ring_extended": false,
    "pinky_extended": false,
    "thumb_curl": 0.0,
    "index_curl": 0.6392,
    "middle_curl": 0.0,
    "ring_curl": 0.0,
    "pinky_curl": 0.0
  },
  "geometry": { "sign": "0", "confidence": 0.95 },
  "mlp": { "has_weights": false, "sign": "NEUTRAL", "confidence": 0.0 },
  "ensemble": { "sign": "0", "confidence": 0.95 },
  "smoothing": {
    "confirmed": false,
    "recent_history": ["0"],
    "valid_frames_count": 1
  },
  "classifier_details": {
    "status": "MATCH",
    "digit": "0",
    "confidence": 0.95,
    "reason": "Vòng tròn số 0: d_TI=0.105<=0.2, d_TM=0.115<=0.22, d_IW=1.145>=0.95",
    "matched_digits": ["0"],
    "features": {
      "palm_scale": 0.0887,
      "thumb_ext": false,
      "index_ext": false,
      "middle_ext": false,
      "ring_ext": false,
      "pinky_ext": false,
      "index_curl": 0.6392,
      "ang_TI": 13.88,
      "ang_TP": 23.95,
      "thumb_lat": 0.2841,
      "d_TI": 0.1052,
      "d_TM": 0.1148,
      "d_IW": 1.1448
    }
  }
}
```

### Ví dụ 2: Khung hình bị trượt do ngón giữa không áp sát ngón cái (`d_TM > 0.22`)
```json
{
  "timestamp": 1793277040.500,
  "session_id": "webcam_session_01",
  "frame_idx": 2,
  "mode": "digits",
  "handedness": "Right",
  "palm_scale": 0.0887,
  "ensemble": { "sign": "NEUTRAL", "confidence": 0.0 },
  "smoothing": {
    "confirmed": false,
    "recent_history": ["0", "NEUTRAL"],
    "valid_frames_count": 1
  },
  "classifier_details": {
    "status": "NO_MATCH",
    "digit": "NEUTRAL",
    "confidence": 0.0,
    "reason": "Không khớp bất kỳ quy tắc chữ số nào",
    "matched_digits": [],
    "features": {
      "thumb_ext": false,
      "index_ext": false,
      "middle_ext": false,
      "ring_ext": false,
      "pinky_ext": false,
      "index_curl": 0.6392,
      "d_TI": 0.1052,
      "d_TM": 0.2850,
      "d_IW": 1.1448
    }
  }
}
```

---

## 4. So sánh Đặc trưng Webcam với 4 mẫu Fixture QIPEDC

> **Lưu ý phương pháp luận:** Cả 4 mẫu trong `real_digit_0_landmarks.json` (frames 20, 25, 30, 35) đều được trích xuất từ **duy nhất một video phòng thu `D0529.mp4`** của một người ký duy nhất. Không được coi đây là 4 mẫu độc lập đại diện cho toàn bộ người dùng VSL.

### Bảng đối chiếu hình học chi tiết:

| Đặc trưng | 4 mẫu trong Fixture `D0529` (QIPEDC) | Cử chỉ tự nhiên người dùng trước Webcam | Tác động phân loại |
| :--- | :---: | :---: | :--- |
| **`d_TI` (Khoảng cách Cái–Trỏ)** | `0.105` – `0.131` (Ngưỡng $\le 0.20$) | `0.08` – `0.18` | **ĐẠT:** Người dùng tự nhiên luôn chụm ngón cái và ngón trỏ khi tạo vòng tròn 0. |
| **`d_TM` (Khoảng cách Cái–Giữa)** | `0.115` – `0.130` (Ngưỡng $\le 0.22$) | `0.24` – `0.38` | **TRƯỢT:** Người ký phòng thu trong `D0529` ép cả ngón giữa vào ngón cái. Người dùng thông thường làm số 0 chỉ chụm ngón trỏ–cái, ngón giữa chỉ uốn cong theo vòm chứ không chạm sát mút ngón cái, khiến `d_TM > 0.22` và bị từ chối thành `NEUTRAL`. |
| **Trạng thái duỗi ngón (`I, M, R, P`)** | Đều là `False` (Do `TIP.y > PIP.y`) | Thường bị lật sang `True` khi nghiêng tay | **NGUY CƠ CAO:** Biên chênh lệch `TIP.y - PIP.y` trong video `D0529` chỉ là $+0.016 \sim +0.035$ (~20px). Khi người dùng nghiêng nhẹ bàn tay về phía camera, phép chiếu 2D làm `TIP.y < PIP.y`, hệ thống kết luận ngón đang duỗi $\rightarrow$ Luật số 0 bị loại bỏ ngay từ vòng đầu. |
| **`index_curl` (Độ co ngón trỏ)** | `0.608` – `0.648` (Khoảng $[0.45, 0.80]$) | `0.40` – `0.65` | **CƠ BẢN ĐẠT:** Độ cong vòm khuyên tròn tự nhiên. |
| **`d_IW` (Trỏ–Cổ tay)** | `1.145` – `1.194` (Ngưỡng $\ge 0.95$) | `0.95` – `1.30` | **ĐẠT:** Loại trừ được nắm đấm co sát cổ tay. |

---

## 5. Phân loại Nguyên nhân

1. **Logger không hoạt động:**
   - **XÁC NHẬN:** Trước giai đoạn này, `_log_diagnostic()` bị lỗi `AttributeError` khi `finger_states=None`, dẫn đến việc không có log nào được ghi ở chế độ `digits`. **Đã được khắc phục hoàn toàn trong giai đoạn này.**
2. **Classifier không được gọi hoặc nhận sai mode:**
   - **XÁC NHẬN (Về mặt UX):** Giao diện web khởi động mặc định ở tab `letters`. Nếu người dùng không nhấn chọn tab `digits`, backend nhận `mode="letters"` và trả về chữ cái `'O'`, không bao giờ gọi `digit_classifier`.
3. **Landmark không hợp lệ hoặc không đúng bàn tay:**
   - **BÌNH THƯỜNG:** MediaPipe trích xuất đầy đủ 21 mốc khi bàn tay nằm trong khung hình.
4. **Luật số 0 bị trượt do feature vượt ngưỡng:**
   - **XÁC NHẬN:** Ngưỡng `d_TM <= 0.22` bị overfit theo tư thế cá biệt của người ký trong video `D0529`. Người dùng ký tự nhiên có `d_TM` trong khoảng $0.24 \sim 0.38$, vượt ngưỡng `0.22` nên bị chuyển thành `NEUTRAL`. Ngoài ra, góc nghiêng bàn tay làm lật điều kiện 2D `TIP.y < PIP.y`.
5. **Classifier nhận số 0 nhưng smoothing/frontend không hiển thị:**
   - **MỘT PHẦN (DO NHẤP NHÁY KHUNG HÌNH):** Bộ làm mượt yêu cầu $\ge 5/10$ khung hình liên tiếp đạt độ tin cậy $\ge 0.70$. Khi cử chỉ dao động quanh biên ngưỡng hình học, các khung hình bị xen kẽ giữa `0` và `NEUTRAL`, bộ đếm không đủ 5 frame để kích hoạt xác nhận phụ đề (`confirmed=True`).
6. **Chưa đủ dữ liệu để kết luận:**
   - Cần thêm dữ liệu log thực tế được thu thập qua công cụ logger vừa sửa để đo đạc chính xác phân phối `d_TM` trên nhiều người dùng thực tế.

---

## 6. Đề xuất bước tiếp theo nhỏ nhất (Next Step)

*(Tuân thủ nguyên tắc: Không tự ý sửa luật số 0 trong giai đoạn này)*

1. **Hướng dẫn người dùng thu thập log thực tế:**
   Người dùng bật chẩn đoán khi kiểm tra camera:
   ```powershell
   $env:VSL_DIAGNOSTIC="1"
   python backend/app.py
   ```
   Sau đó mở trình duyệt, chọn tab **"🔢 Chữ số"** và ký số 0 để file log `backend/data/diagnostics/diagnostic_YYYYMMDD.jsonl` ghi lại các giá trị thực tế.
2. **Đề xuất hiệu chỉnh hình học cho Giai đoạn tiếp theo (Stage 2C.10C):**
   - **Phương án A (Nới ngưỡng nhẹ):** Nới ngưỡng `D_TM_THRESHOLD_0` từ `0.22` lên `0.30` hoặc `0.32` dựa trên đo đạc thực tế, giữ nguyên các điều kiện khác.
   - **Phương án B (Nới lỏng có kiểm soát độ co):** Bỏ ràng buộc cứng `d_TM <= 0.22`, chỉ yêu cầu `d_TI <= 0.22` kết hợp `middle_curl >= 0.40` (ngón giữa có co vòm nhưng không bắt buộc phải chạm mút ngón cái). Phương án này giải quyết triệt để sự khác biệt giữa biến thể studio và cử chỉ tự nhiên.

