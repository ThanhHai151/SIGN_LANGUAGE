# TÀI LIỆU KỸ THUẬT: SCHEMA JSONL & ĐỒNG BỘ VIDEO THU THẬP DỮ LIỆU
## TASK 3.10.2 — STANDARDIZATION & TECHNICAL SPECIFICATION
**Phiên bản Schema**: `2.0-kinematic`  
**Hệ thống**: VSL Controlled Pilot Logging & Kinematic Data Collection  
**Trạng thái**: SPECIFICATION & IMPLEMENTATION ALIGNED (READ-ONLY PRESERVATION)

---

### 1. PROTOCOL TRIAL 6.0 GIÂY CHUẨN HOÁ (5 PHA ĐỘNG HỌC)

Theo thiết kế đã được User phê duyệt tại TASK 3.9 & TASK 3.9.1, thời lượng mỗi trial thu thập cử chỉ là **6.0 giây** và được chia thành 5 pha cố định không co giãn theo tỷ lệ:

| Thứ Tự | Tên Pha | Ranh Giới Thời Gian | Thời Lượng | Mô Tả Hành Động & Ý Nghĩa Kỹ Thuật |
|:---:|:---|:---:|:---:|:---|
| **Pha 1** | `PRE_IDLE` | $[0.0, 0.5)\text{s}$ | 0.5 giây | Người ký buông lỏng hai tay ở tư thế nghỉ, camera thu thập baseline không có cử chỉ. |
| **Pha 2** | `ENTRY` | $[0.5, 1.5)\text{s}$ | 1.0 giây | Người ký đưa tay lên vùng quan sát, bắt đầu chuyển động tạo hình dạng chữ cái. |
| **Pha 3** | `STABLE_HOLD` | $[1.5, 4.5)\text{s}$ | **3.0 giây** | Người ký giữ ổn định hình dạng chữ cái mục tiêu. **Pha duy nhất dùng đánh giá tĩnh (Static Classifier Recall).** |
| **Pha 4** | `EXIT` | $[4.5, 5.5)\text{s}$ | 1.0 giây | Người ký buông lỏng cử chỉ, hạ tay xuống. Đánh giá khả năng về NEUTRAL, chống False Activation. |
| **Pha 5** | `POST_IDLE` | $[5.5, 6.0]\text{s}$ | 0.5 giây | Người ký đã hạ tay hoàn toàn, kết thúc trial và chuẩn bị cho lần lặp tiếp theo. |

- **Tính đồng nhất của nguồn thời gian**:
  - `countdownSeconds`, `currentKinematicPhase`, và `handleEndTrial` đều được tính toán từ một nguồn duy nhất: `Date.now() - startTime`.
  - Cấu hình mặc định trong `PilotTestingHUD.jsx`: `DEFAULT_TRIAL_DURATION_SEC = 6.0`.
  - Danh sách presets: `DURATION_PRESETS = [3.0, 5.0, 6.0, 8.0, 10.0]` (với nhãn `6s (Chuẩn)`).

---

### 2. CẤU TRÚC SCHEMA JSONL THỰC TẾ (`2.0-kinematic`)

Tệp log được ghi thành các dòng JSONL tại `backend/data/diagnostics/pilot_<session_id>.jsonl`. Mỗi bản ghi đại diện cho 1 khung hình xử lý:

#### 2.1. Bản ghi khi Tracking Hợp Lệ (`tracking_status = "VALID"`)
- `image_landmarks_21` là top-level key chứa danh sách 21 điểm mốc $[x, y, z]$ trong không gian khung nhìn $[0, 1] \times [0, 1]$ kèm $z$ tương đối.
- `normalized_landmarks_21` là top-level key chứa danh sách 21 điểm mốc sau khi dời gốc cổ tay về $[0.0, 0.0, 0.0]$ và chuẩn hóa theo kích thước lòng bàn tay $d_{\text{palm}} = \|\mathbf{p}_0 - \mathbf{p}_9\|_2$.
- **Không có key cha bọc `raw_landmarks_21`**.

```json
{
  "schema_version": "2.0-kinematic",
  "timestamp": 1731112456.7891,
  "session_id": "pilot_20261109_023000",
  "trial_id": "pilot_20261109_023000_O_rep01",
  "signer_id": "signer_01",
  "frame_idx": 45,
  "expected_sign": "O",
  "tracking_status": "VALID",
  "image_width": 640,
  "image_height": 480,
  "handedness": "Right",
  "handedness_score": 0.998,
  "tracking_confidence": 0.95,
  "feature_extractor_version": "2.0-kinematic",
  "classifier_rule_version": "Candidate-03",
  "image_landmarks_21": [
    [0.512345, 0.781250, -0.000120],
    [0.451200, 0.521300, -0.031200],
    "..."
  ],
  "normalized_landmarks_21": [
    [0.0000, 0.0000, 0.0000],
    [-0.3214, -1.2451, -0.1542],
    "..."
  ],
  "final_sign": "O",
  "final_confidence": 0.985,
  "prediction_source": "geometric_rule",
  "confirmed": true,
  "has_static_head": false,
  "geometric": {
    "status": "MATCH",
    "letter": "O",
    "reason": "Rule O match",
    "features": {
      "d_TI": 0.1652,
      "d_TM": 0.2841,
      "idx_straight": 0.3812,
      "d_IW": 1.1520,
      "d_MW": 1.0831,
      "d_RW": 0.9542,
      "d_PW": 0.8810,
      "d_TW": 0.9421
    },
    "failed_conditions": []
  },
  "static_head": {
    "status": "NOT_CALLED",
    "head_sign": null,
    "head_confidence": null,
    "top3": []
  },
  "gating_status": "none",
  "threshold_applied": null,
  "reject_reason": null,
  "sync": {
    "frame_idx": 45,
    "client_capture_ts": 15.2410,
    "landmark_extract_ts": 15.2620,
    "request_send_ts": 15.2630,
    "backend_receive_ts": 1731112456.7784,
    "video_pts_sec": 1.5023,
    "video_frame_idx": 45
  },
  "temporal": {
    "smoothing_window": 10,
    "vote_threshold": 5,
    "vote_count": 8,
    "most_common_sign": "O",
    "recent_predictions_10": [
      {"sign": "O", "confidence": 0.98},
      {"sign": "O", "confidence": 0.99}
    ],
    "last_confirmed_sign": "O",
    "last_confirmed_time": 1731112456.5123,
    "cooldown_sec": 1.0,
    "cooldown_remaining_sec": 0.723,
    "token_confirmation_event": true,
    "new_token_emitted": "O",
    "model_has_weights": false,
    "has_static_head": false
  }
}
```

---

#### 2.2. Bản ghi khi Mất Dấu Bàn Tay (`tracking_status = "NO_HAND"`)
- Xuất hiện khi `p_prim is None` hoặc danh sách landmarks rỗng `[]`.
- `image_landmarks_21` và `normalized_landmarks_21` đều là `null` (None trong Python).
- `final_sign` là `"NEUTRAL"`, `final_confidence` là `0.0`. Không sinh tọa độ giả.

```json
{
  "schema_version": "2.0-kinematic",
  "timestamp": 1731112457.1234,
  "session_id": "pilot_20261109_023000",
  "trial_id": "pilot_20261109_023000_O_rep01",
  "signer_id": "signer_01",
  "frame_idx": 55,
  "expected_sign": "O",
  "tracking_status": "NO_HAND",
  "image_width": 640,
  "image_height": 480,
  "handedness": "Right",
  "handedness_score": null,
  "tracking_confidence": null,
  "feature_extractor_version": "2.0-kinematic",
  "classifier_rule_version": "Candidate-03",
  "image_landmarks_21": null,
  "normalized_landmarks_21": null,
  "final_sign": "NEUTRAL",
  "final_confidence": 0.0,
  "prediction_source": "none",
  "confirmed": false,
  "has_static_head": false,
  "geometric": {
    "status": "INVALID_INPUT",
    "letter": null,
    "reason": "No valid hand landmarks detected",
    "features": {},
    "failed_conditions": []
  },
  "static_head": {
    "status": "NOT_CALLED",
    "head_sign": null,
    "head_confidence": null,
    "top3": []
  },
  "gating_status": "none",
  "threshold_applied": null,
  "reject_reason": "No valid hand landmarks detected",
  "sync": {
    "frame_idx": 55,
    "client_capture_ts": 15.5740,
    "landmark_extract_ts": null,
    "request_send_ts": 15.5750,
    "backend_receive_ts": 1731112457.1120,
    "video_pts_sec": 1.8340,
    "video_frame_idx": 55
  },
  "temporal": { ... }
}
```

---

#### 2.3. Bản ghi khi Dữ liệu Landmarks Không Hợp Lệ (`tracking_status = "INVALID_SHAPE"`)
- Xuất hiện khi:
  1. Số lượng điểm landmark khác 21 ($0 < \text{len} < 21$ hoặc $> 21$).
  2. Mảng tọa độ không đúng kích thước $(21, 3)$ hoặc thiếu tọa độ con.
  3. Tọa độ chứa giá trị không hợp lệ: `NaN`, `+Inf`, `-Inf`.
- `image_landmarks_21` và `normalized_landmarks_21` đều là `null`.
- `final_sign` là `"NEUTRAL"`.

```json
{
  "schema_version": "2.0-kinematic",
  "timestamp": 1731112457.3456,
  "session_id": "pilot_20261109_023000",
  "trial_id": "pilot_20261109_023000_O_rep01",
  "signer_id": "signer_01",
  "frame_idx": 62,
  "expected_sign": "O",
  "tracking_status": "INVALID_SHAPE",
  "image_width": 640,
  "image_height": 480,
  "handedness": "Right",
  "handedness_score": null,
  "tracking_confidence": null,
  "feature_extractor_version": "2.0-kinematic",
  "classifier_rule_version": "Candidate-03",
  "image_landmarks_21": null,
  "normalized_landmarks_21": null,
  "final_sign": "NEUTRAL",
  "final_confidence": 0.0,
  "prediction_source": "none",
  "confirmed": false,
  "has_static_head": false,
  "geometric": {
    "status": "INVALID_INPUT",
    "letter": null,
    "reason": "Non-finite or invalid shape landmarks",
    "features": {},
    "failed_conditions": []
  },
  "gating_status": "none",
  "threshold_applied": null,
  "reject_reason": "Non-finite or invalid shape landmarks",
  "sync": { ... },
  "temporal": { ... }
}
```

---

#### 2.4. Bản ghi khi Lỗi Xử Lý Ngoại Lệ Nội Bộ (`tracking_status = "ERROR"`)
- Xuất hiện khi xảy ra ngoại lệ không dự tính trước trong khối `try...except` xử lý landmark (ví dụ lỗi bộ nhớ, lỗi thư viện).
- `image_landmarks_21` và `normalized_landmarks_21` là `null`.
- `final_sign` là `"NEUTRAL"`, `reject_reason` ghi rõ ngoại lệ.

```json
{
  "schema_version": "2.0-kinematic",
  "tracking_status": "ERROR",
  "image_landmarks_21": null,
  "normalized_landmarks_21": null,
  "final_sign": "NEUTRAL",
  "final_confidence": 0.0,
  "reject_reason": "Landmark validation exception: <exception message>",
  "..." : "..."
}
```

---

### 3. GHI NHẬN TRUNG THỰC VỀ ĐỒNG BỘ VIDEO VÀ SIDECAR (LIMITATIONS)

Hệ thống ghi nhận và khẳng định trung thực các đặc tính kỹ thuật của tầng đồng bộ:

1. **`videoElement.currentTime` là Preview Clock**:
   - Giá trị này được đọc trực tiếp từ thuộc tính `currentTime` của thẻ `<video>` hiển thị luồng webcam tại client.
   - Đây **không phải** là Presentation Timestamp (PTS) mã hóa cứng bên trong các gói tin container WebM của MediaRecorder.
2. **Hệ quy chiếu thời gian Monotonic vs Wall Clock**:
   - Client sử dụng `performance.now() / 1000` (Monotonic clock tính từ lúc load trang, thường có giá trị nhỏ dạng $[0, 3600]\text{s}$).
   - Backend sử dụng `time.time()` (UNIX Epoch wall clock, dạng $\approx 1.73 \times 10^9\text{s}$).
   - Do đó, hai mốc thời gian này có gốc tọa độ khác nhau; không thể thực hiện phép trừ đại số trực tiếp nếu chưa căn chỉnh clock offset.
3. **`video_frame_idx` không phải chỉ số khung hình WebM phần cứng**:
   - `video_frame_idx` được gán bằng `frame_idx` (số thứ tự khung hình xử lý bởi MediaPipe).
   - Nếu MediaPipe chạy ở 20 FPS (do quá tải CPU) trong khi MediaRecorder ghi ở 30 FPS, số lượng frame trong sidecar sẽ ít hơn số frame trong video WebM.
4. **Phân loại cấp độ đồng bộ**:
   - Cơ chế này được xác định là **Đồng bộ gần đúng (Approximate Best-Effort Synchronization)**, hỗ trợ con người đối chiếu trực quan khi xem video, **không phải là đồng bộ chính xác cấp độ khung hình (Frame-accurate hardware synchronization)**.
   - Nghiêm cấm bịa đặt PTS, không giả định frame rate cố định, và không tuyên bố sai lệch về khả năng đồng bộ tuyệt đối.

