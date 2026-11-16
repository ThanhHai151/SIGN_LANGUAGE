# THIẾT KẾ VÀ ĐÁNH GIÁ TÍNH KHẢ THI THU THẬP DỮ LIỆU BỔ SUNG (VSL CONNECT AI)
## TASK 3.9 & 3.9.1 — DATA COLLECTION PROTOCOL, SCHEMA & QUALITY CONTROL AUDIT

---

## 1. TỔNG QUAN ĐIỀU HÀNH (EXECUTIVE SUMMARY)

- **Mục tiêu**: Thiết kế quy trình thu thập dữ liệu bổ sung có kiểm chứng, độc lập và chống rò rỉ (leakage-free) nhằm phục vụ điều tra và giải quyết dứt điểm lỗi nhận diện chữ O trong VSL Connect AI; kiểm toán năng lực thu thập và lưu trữ hiện tại của kho mã nguồn (repository).
- **Phạm vi quyền hạn**: **READ-ONLY 100%**.
  - Không sửa đổi mã nguồn production (`alphabet_classifier.py`, `recognizer.py`, `feature_extractor.py`).
  - Không thay đổi bất kỳ ngưỡng hình học nào (`O_D_TI_MAX = 0.22`, `O_IDX_STRAIGHT_MAX = 0.48`...).
  - Không thay đổi model weights, dataset NPZ hay fixtures hiện hành.
  - Tuyệt đối không tự ý thu thập video hay dữ liệu người tham gia thật trong nhiệm vụ này.
- **Bối cảnh kỹ thuật đã được bảo toàn**:
  - Baseline hiện tại là O-Candidate-03 (triển khai tại TASK 3.5 và xác minh tại TASK 3.7).
  - Kết quả kiểm định webcam holdout độc lập (TASK 3.7): Chữ O chỉ đạt **53/349 active frames (15.19%)** và **0/10 trials** đạt tiêu chí Recall $\ge 70\%$.
  - Trạng thái nghiệm thu bắt buộc tiếp tục duy trì:
    $$\mathbf{O:\ WEBCAM\ VERIFIED\ —\ FAIL\ (TASK\ 3.7:\ 0/10\ TRIALS\ \ge\ 70\%,\ NOT\ FIXED)}$$
  - Phát hiện mấu chốt từ TASK 3.8 đến 3.8.2: Thiếu dữ liệu video RGB đồng bộ khiến mọi nhận định về cơ sinh học (chạm cạnh ngón trỏ, góc chúc camera, chuyển động ngón cái ở B frame 54) chỉ dừng lại ở tầng **Giả thuyết (Tier C)**; sự phụ thuộc vào 1-2 người ký khiến mô hình thiếu khả năng tổng quát hóa (generalization).

---

## 2. KIỂM TOÁN NĂNG LỰC THU THẬP DỮ LIỆU HIỆN TẠI CỦA REPOSITORY

Đối chiếu trực tiếp mã nguồn tại:
- [backend/vision/pilot_logger.py](file:///d:/SIGN_LANGUAGE/backend/vision/pilot_logger.py)
- [backend/vision/recognizer.py](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py)
- [src/components/PilotTestingHUD.jsx](file:///d:/SIGN_LANGUAGE/src/components/PilotTestingHUD.jsx)
- [src/components/CameraSignRecognizer.jsx](file:///d:/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx)
- [backend/app.py](file:///d:/SIGN_LANGUAGE/backend/app.py)

### 2.1. Các trường dữ liệu hiện đang được ghi nhận
| Thành phần dữ liệu | Trạng thái hiện tại | Vị trí mã nguồn đối chiếu | Đánh giá & Giới hạn |
|:---|:---:|:---|:---|
| **21 Hand Landmarks** | **KHÔNG CÓ (trong Pilot JSONL)** | [backend/vision/recognizer.py#L1615-L1634](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L1615-L1634) | Payload của `pilot_logger.log_frame` chỉ nhận derived features, **hoàn toàn không lưu 21 mốc tọa độ thô** `[x, y, z]`. Module legacy `_log_diagnostic` ([recognizer.py#L1736](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L1736)) có lưu `raw_landmarks` nhưng bị giới hạn rate-limit ~2.2 FPS và không được kích hoạt khi chạy Pilot HUD. |
| **Tọa độ x, y, z & visibility** | **MỘT PHẦN (Chỉ trong tệp tĩnh)** | [backend/training/extract_video_landmarks.py#L55-L65](file:///d:/SIGN_LANGUAGE/backend/training/extract_video_landmarks.py#L55-L65) | Tệp tĩnh `O.json` có tọa độ chuẩn hóa $[x, y, z]$. MediaPipe Hands chuẩn **không xuất trường visibility/presence** (chỉ có ở MediaPipe Pose/Holistic). |
| **Feature Vector & Diagnostics** | **ĐẦY ĐỦ** | [backend/vision/recognizer.py#L394-L407](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L394-L407), [alphabet_classifier.py#L300-L380](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py#L300-L380) | Lưu đầy đủ 35 đặc trưng hình học (`idx_straight`, `d_TI`, `d_TM`, `d_IW`, `d_MW`, `d_RW`, `d_PW`, `d_TW`...) và toàn bộ diagnostic rules (`o_rule_diagnostics`, `h_rule_diagnostics`...). |
| **Prediction, Conf, Source** | **ĐẦY ĐỦ** | [backend/vision/recognizer.py#L1621-L1623](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L1621-L1623) | Lưu `final_sign`, `final_confidence`, `prediction_source` (`"geometric_rule"` hoặc `"none"`). |
| **Timestamp, Frame, Trial** | **ĐẦY ĐỦ** | [backend/vision/recognizer.py#L1616-L1619](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L1616-L1619) | Lưu `timestamp` (Unix float giây), `frame_idx` (tăng dần trong trial), `session_id`, `trial_id` (`<session>_<letter>_rep<XX>`). |
| **Temporal Event & Token** | **MỘT PHẦN** | [backend/vision/recognizer.py#L1624](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L1624) | Lưu cờ `confirmed: bool` khi phát sinh token. **Không lưu snapshot 10 frame của `prediction_history`** trong Pilot JSONL (chỉ lưu trong legacy `_log_diagnostic` [recognizer.py#L1691](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L1691)). |
| **Trạng thái MLP & Fallback** | **ĐẦY ĐỦ** | [backend/vision/recognizer.py#L410-L420](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L410-L420), [L1625-L1630](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L1625-L1630) | Lưu `has_static_head`, `load_status`, `pytorch_version`, `raw_predicted_letter`, `confidence`. |

### 2.2. Khả năng ghi Video RGB đồng bộ với Landmark JSONL
1. **Hiện trạng sẵn có**: **CHƯA CÓ (NOT IMPLEMENTED)**.
   - Kho mã nguồn hiện tại không có thành phần nào ghi lại video webcam. Không có `MediaRecorder` trong frontend ([src/components/CameraSignRecognizer.jsx](file:///d:/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx)) và không có `cv2.VideoWriter` trong backend ([backend/app.py](file:///d:/SIGN_LANGUAGE/backend/app.py)).
   - Luồng hoạt động hiện tại: Trình duyệt mở webcam $\rightarrow$ MediaPipe Hands JS chạy trực tiếp trên client trích xuất 21 điểm $\rightarrow$ gửi JSON mốc tọa độ qua HTTP POST `/api/recognize_landmarks` $\rightarrow$ Backend xử lý và ghi log JSONL. Ảnh video RGB bị hủy ngay trên canvas của client sau khi MediaPipe chạy xong.
2. **Khả năng đồng bộ hóa**:
   - Để đồng bộ chính xác, client phải capture cặp khóa nguyên tử: `(frame_idx, client_capture_ts)` và nhúng vào cả stream video lẫn payload HTTP gửi về backend.
3. **Phát hiện Frame Drop**:
   - `frame_idx` tăng tuần tự từng frame do frontend quản lý ([PilotTestingHUD.jsx#L175](file:///d:/SIGN_LANGUAGE/src/components/PilotTestingHUD.jsx#L175)). Nếu request bị drop, `frame_idx` trong JSONL sẽ bị nhảy cóc. Hiệu số timestamp giữa 2 frame liên tiếp $\Delta t > 70\text{ms}$ cho phép phát hiện frame drop.
4. **Giới hạn hiệu năng & dung lượng**:
   - Dung lượng video 720p 30 FPS $\approx 20\text{ MB/phút}$, hoàn toàn khả thi trên ổ đĩa cục bộ.
   - Rủi ro hiệu năng: Vừa render, vừa chạy MediaPipe Hands, vừa encode video trên máy không có GPU rời có thể làm tụt frame rate xuống $<20\text{ FPS}$.

---

## 3. XÁC ĐỊNH STABLE HOLD CÓ CĂN CỨ & QUY TRÌNH GÁN NHÃN GROUND TRUTH

### 3.1. Phân biệt ba thành phần ổn định tư thế
Không được dùng riêng vận tốc dịch chuyển bàn tay để kết luận tư thế ổn định. Phải phân tách 3 thành phần kỹ thuật độc lập:
1. **Hand Translation Stability (Độ ổn định vị trí bàn tay)**:
   - Đo bằng dịch chuyển không gian của Wrist (mốc 0) và Middle MCP (mốc 9) giữa 2 frame liên tiếp:
     $$v_{trans} = \frac{\|\mathbf{p}_0(t) - \mathbf{p}_0(t-1)\|_2}{\Delta t}$$
   - Phản ánh việc bàn tay có bị rung lắc, dịch chuyển tịnh tiến trong khung hình hay không.
2. **Finger Configuration Stability (Độ ổn định cấu hình các ngón tay)**:
   - Đo bằng biến thiên khoảng cách nội bộ giữa các mốc đầu ngón tay so với gốc cổ tay:
     $$v_{config} = \frac{1}{5} \sum_{i \in \{4, 8, 12, 16, 20\}} \frac{\|(\mathbf{p}_i(t) - \mathbf{p}_0(t)) - (\mathbf{p}_i(t-1) - \mathbf{p}_0(t-1))\|_2}{\Delta t}$$
     và độ biến thiên của hai đặc trưng hình học cốt lõi: $|\Delta idx\_straight| < \epsilon_1$ và $|\Delta d_{TI}| < \epsilon_2$.
   - Phản ánh việc ngón tay có đang co, duỗi, ngập ngừng, đổi dáng hay không (kể cả khi cổ tay đứng yên).
3. **Overall Pose Stability (Độ ổn định tổng thể của tư thế)**:
   - Là giao điểm khi cả $v_{trans}$ và $v_{config}$ cùng duy trì dưới ngưỡng giới hạn ổn định trong một khoảng thời gian liên tục $\ge 1.0\text{ giây}$.

### 3.2. Quy trình gán nhãn Ground Truth 5 pha động học
**NGUYÊN TẮC BẮT BUỘC**: Tuyệt đối không để thuật toán phân loại (như Rule 6 hay Candidate-03) tự xác định pha rồi dùng chính kết quả đó làm Ground Truth (tránh ngụy biện vòng tròn - circular logic).

Quy trình gán nhãn Ground Truth độc lập gồm 5 pha:
1. **Pha 1: Pre-trial Idle (0.0s – 0.5s)**: Tay ở vị trí nghỉ (dưới ngực hoặc trên bàn).
2. **Pha 2: Entry Phase (0.5s – 1.5s)**: Tay bắt đầu đưa lên vùng quan sát, các ngón tay bắt đầu uốn khum tạo vòng chữ O.
3. **Pha 3: Stable Hold Phase (1.5s – 4.5s, kéo dài 3.0s)**:
   - **Xác định Ground Truth**: Được gán nhãn độc lập bởi **Người kiểm toán thị giác (Human Visual Inspection)** đối chiếu trên video RGB:
     - Đánh dấu chính xác `hold_start_frame` (thời điểm bàn tay hoàn thành tạo vòng O và dừng chuyển động).
     - Đánh dấu chính xác `hold_end_frame` (thời điểm bàn tay bắt đầu buông lỏng hoặc hạ xuống).
   - Đây là **pha duy nhất** được dùng làm Ground Truth để đánh giá bộ phân loại tĩnh (Static Geometric Classifier).
4. **Pha 4: Exit Phase (4.5s – 5.5s)**: Tay buông lỏng vòng O và hạ xuống vị trí nghỉ.
5. **Pha 5: Post-trial Idle (5.5s – 6.0s)**: Tay đã trở về vị trí nghỉ hoàn toàn.

- **Phân định rõ ranh giới**:
  - **Human Ground Truth Annotation**: Nhãn pha do con người quan sát video đối chứng gán (dùng làm Ground Truth).
  - **Automated QC Metrics**: Các chỉ số tính toán tự động ($v_{trans}, v_{config}$) chỉ dùng làm cờ kiểm tra chất lượng (QC flags), **tuyệt đối không dùng làm Ground Truth**.

---

## 4. THIẾT KẾ ĐỒNG BỘ VIDEO VÀ LANDMARKS (TIMESTAMP & FRAME CORRESPONDENCE)

### 4.1. Sáu mốc thời gian bắt buộc phân biệt
Để phát hiện chính xác frame drop, network lag và video drift, hệ thống phải phân biệt rạch ròi 6 mốc thời gian:
1. $t_{cap}$ (`client_frame_capture_ts`): Thời điểm webcam client chụp khung hình từ camera (`performance.now()`).
2. $t_{extract}$ (`landmark_extraction_ts`): Thời điểm MediaPipe JS hoàn tất trích xuất 21 mốc xương trên client.
3. $t_{send}$ (`request_send_ts`): Thời điểm client phát request HTTP POST sang backend.
4. $t_{recv}$ (`backend_receive_ts`): Thời điểm backend nhận được request và giải mã JSON.
5. $t_{pts}$ (`video_pts_sec`): Presentation Timestamp của khung hình trong tệp video ghi hình (`MediaRecorder`).
6. `frame_idx`: Số thứ tự khung hình tăng tuần tự do client quản lý.

### 4.2. Cơ chế đồng bộ không dựa trên Fixed Offset
- Không được giả định một fixed offset (ví dụ 15ms) vì độ trễ mạng và CPU jitter thay đổi liên tục.
- **Giải pháp đồng bộ**:
  - Tại mỗi khung hình, Client gắn cặp khóa nguyên tử `(frame_idx, client_frame_capture_ts)` vào metadata video chunk và vào payload HTTP gửi sang backend.
  - Backend ghi nhận nguyên vẹn `client_frame_capture_ts` vào dòng log JSONL.
  - Khi kiểm toán/replay: Khớp chính xác frame thứ $i$ trong video với dòng log thứ $i$ qua `frame_idx` và kiểm tra $t_{pts} \approx client\_frame\_capture\_ts - session\_start\_ts$.

### 4.3. Phát hiện mất frame và lệch thời gian
- **Phát hiện Drop Frame**: Nếu `frame_idx[k] - frame_idx[k-1] > 1`, ghi nhận drop frame tại khoảng cách đó.
- **Phát hiện Time Jitter / Drift**: Nếu khoảng cách giữa 2 frame liên tiếp $\Delta t = t_{cap}[k] - t_{cap}[k-1] > 2 \times \text{frame\_period}$ (ví dụ $>66.6\text{ms}$ ở 30 FPS), đánh dấu cờ `timing_jitter_detected`.

---

## 5. HOÀN THIỆN SCHEMA RAW LANDMARKS & LOGGING NGUYÊN TỬ

### 5.1. Quy định hệ tọa độ và dữ liệu mốc
- **Chính xác 21 Mốc**: Lưu mảng đúng 21 điểm tương ứng chuẩn MediaPipe Hands (0: Wrist, 1-4: Thumb, 5-8: Index, 9-12: Middle, 13-16: Ring, 17-20: Pinky).
- **Hệ tọa độ Raw Coordinates (`raw_landmarks_21`)**:
  - $x \in [0.0, 1.0]$: Chiều ngang viewport (trái sang phải).
  - $y \in [0.0, 1.0]$: Chiều dọc viewport (trên xuống dưới).
  - $z$: Chiều sâu tương đối so với cổ tay do MediaPipe ước tính.
- **Hệ tọa độ Normalized Shifted/Scaled (`normalized_landmarks_21`)**:
  - Gốc tọa độ dịch về Wrist $\mathbf{p}_0 = (0, 0, 0)$.
  - Tọa độ chia cho `palm_scale` ($\|\mathbf{p}_9 - \mathbf{p}_0\|$).
- **Trường hợp thiếu landmark hoặc tracking loss**:
  - Không có bàn tay: `raw_landmarks_21: null`, `quality_status: "no_hand_detected"`.
  - Mốc suy biến ($palm\_scale < 1e-5$): `raw_landmarks_21: [[...]]`, `quality_status: "degenerate_palm_scale"`.
- **Handedness & Confidence**:
  - `handedness`: `"Right"` hoặc `"Left"`.
  - `handedness_score`: float từ `multiHandedness[0].score`.
  - **Tuyệt đối không tự ý thêm visibility/presence** vì MediaPipe Hands JS không cung cấp các trường này.

### 5.2. Bản ghi JSONL đầy đủ minh họa
```json
{
  "protocol_version": "vsl_pilot_v3",
  "session_id": "pilot_20261115_143000",
  "signer_id": "signer_03",
  "trial_id": "pilot_20261115_143000_O_rep04",
  "frame_idx": 45,
  "timing": {
    "client_capture_ts": 1794760205.1234,
    "client_extract_ts": 1794760205.1298,
    "client_send_ts": 1794760205.1310,
    "backend_recv_ts": 1794760205.1385
  },
  "video_sync": {
    "video_file": "videos/pilot_20261115_143000_signer03.mp4",
    "video_frame_idx": 45,
    "video_pts_sec": 1.500,
    "sync_status": "matched"
  },
  "expected_sign": "O",
  "kinematic_phase_ground_truth": "stable_hold",
  "kinematic_phase_annotator": "human_audit_v1",
  "qc_flags": {
    "hand_trans_velocity": 0.0125,
    "finger_config_velocity": 0.0084,
    "pose_stable_auto": true
  },
  "raw_landmarks_21": [
    [0.512345, 0.789123, -0.000123],
    [0.485123, 0.723456, -0.012345],
    "..."
  ],
  "handedness": "Right",
  "handedness_score": 0.9852,
  "quality_status": "ok",
  "versioning": {
    "feature_extractor_version": "vsl_feature_extractor_v2.1",
    "classifier_rule_version": "alphabet_rule_v2_candidate_03"
  },
  "features": {
    "palm_scale": 0.1852,
    "idx_straight": 0.4215,
    "d_TI": 0.1824,
    "d_TM": 0.1420,
    "d_TW": 1.1205,
    "d_IW": 1.0850,
    "d_MW": 1.0620,
    "d_RW": 1.0540,
    "d_PW": 1.0420,
    "thumb_ext": false,
    "index_ext": false,
    "middle_ext": false,
    "ring_ext": false,
    "pinky_ext": false
  },
  "geometry_evaluation": {
    "status": "MATCH",
    "matched_rules": ["O"],
    "rule_details": {
      "rule_6_matched": true,
      "failed_conditions": []
    }
  },
  "temporal_pipeline_state": {
    "final_sign": "O",
    "final_confidence": 0.95,
    "confirmed": true,
    "recent_predictions_10": [
      ["NEUTRAL", 0.0], ["O", 0.95], ["O", 0.95], ["O", 0.95], ["NEUTRAL", 0.0],
      ["O", 0.95], ["O", 0.95], ["O", 0.95], ["O", 0.95], ["O", 0.95]
    ],
    "vote_count_target": 8,
    "last_confirmed_sign": "O",
    "cooldown_remaining_sec": 0.0
  }
}
```

---

## 6. PHÂN TÁCH HAI LOẠI ĐÁNH GIÁ (STATIC CLASSIFIER VS TEMPORAL PIPELINE)

Thiết kế bắt buộc phải phân tách hai tầng đánh giá độc lập:

### 6.1. Static Classifier Evaluation (Đánh giá hình học tĩnh)
- **Mục tiêu**: Đánh giá độ chính xác của các quy tắc hình học Rule 6 trên tư thế chuẩn.
- **Tập dữ liệu**: **CHỈ ĐÁNH GIÁ TRÊN PHA STABLE HOLD** (theo nhãn Ground Truth của con người).
- **Chỉ số đo lường**:
  - *Stable Hold Frame Recall*:
    $$\text{Recall}_{hold} = \frac{\text{Số frame khớp Rule 6 trong pha Stable Hold}}{\text{Tổng số frame trong pha Stable Hold}}$$
  - *Phân phối đặc trưng vi phạm*: Đo giá trị thực tế của $idx\_straight$ và $d_{TI}$ trong pha giữ tĩnh để xác định ranh giới cơ sinh học thật.
  - *False Positive tĩnh*: Tỷ lệ nhận diện nhầm O trong pha Stable Hold của các chữ đối chứng (B, C, E, T, S, A, P, X). Tiêu chuẩn: **Bắt buộc = 0.00%**.

### 6.2. Temporal Pipeline Evaluation (Đánh giá luồng thời gian thực)
- **Mục tiêu**: Đánh giá cơ chế làm mượt thời gian (Temporal Smoothing Buffer) và khả năng phát sinh token khi chuyển động.
- **Tập dữ liệu**: **ĐÁNH GIÁ TRÊN TOÀN BỘ CHUỖI TRIAL** (Pre-idle $\rightarrow$ Entry $\rightarrow$ Stable Hold $\rightarrow$ Exit $\rightarrow$ Post-idle).
- **Chỉ số đo lường**:
  - *Trial Success Rate*: Tỷ lệ trial phát sinh chính xác token mong muốn ($\ge 7/10$ trials).
  - *Time to Confirmation*: Thời gian từ khi bắt đầu Stable Hold đến khi token được phát ra.
  - *Exit False Positive / Phantom Token*: Kiểm tra xem pha buông tay hạ xuống (Exit) có vô tình làm nổ thêm token O rác hay không.
  - *B-to-O Exit Collision*: Kiểm tra xem khi người ký hạ tay ở chữ B có kích hoạt token O hay không.

---

## 7. CHIẾN LƯỢC CHIA TẬP DỮ LIỆU & BẢO VỆ CHỐNG RÒ RỈ (ANTI-LEAKAGE)

### 7.1. Ba phân vùng dữ liệu bắt buộc (Data Partitions)
```
                                   TỔNG TẬP DỮ LIỆU THU THẬP MỚI
                                                 │
                   ┌─────────────────────────────┴─────────────────────────────┐
                   ▼                                                           ▼
        TẬP PHÁT TRIỂN & CHUẨN ĐOÁN                                TẬP KIỂM ĐỊNH ĐỘC LẬP
         (Development & Validation)                              (Independent Signer Holdout)
           [Signers 01, 02, 03, 04]                                  [Signers 05, 06, 07, 08]
                   │                                                           │
         ┌─────────┴─────────┐                                                 │
         ▼                   ▼                                                 ▼
      DEV SET             VAL SET                                      ĐÁNH GIÁ NGHIỆM THU
   (Tối ưu hóa &     (Kiểm tra hội tụ &                              (Nghiệm thu độc lập,
    phân tích lỗi)     chống overfit)                                 tuyệt đối KHÔNG tune)
```

1. **Phân chia theo Người ký (Signer-Level Separation)**:
   - Dữ liệu của một Signer ID chỉ được phép nằm trong tập Dev/Val **HOẶC** tập Independent Holdout.
2. **Cấm chia ngẫu nhiên cấp Frame (No Frame-Level Splitting)**:
   - Tất cả các frame của cùng 1 trial/video bắt buộc phải thuộc về cùng một partition.
3. **Giới hạn quy mô thiết kế 5–8 Signers**:
   - Khẳng định rõ: Quy mô 5–8 signers là **pilot nghiên cứu kỹ thuật ban đầu** trong phòng lab, không phải bằng chứng về tính đại diện thống kê cho toàn thể cộng đồng VSL tại Việt Nam.
4. **Bảo vệ Independent Holdout khỏi việc Tuning**:
   - Tập Independent Holdout chỉ được mở ra đánh giá nghiệm thu cuối cùng (Giai đoạn 7).
   - Tuyệt đối không sử dụng kết quả Holdout để quay ngược lại tinh chỉnh ngưỡng hay thiết kế Candidate-04.

---

## 8. KIỂM TOÁN CÁC TIÊU CHUẨN QC (PROVISIONAL THRESHOLD AUDIT)

Rà soát toàn diện 4 ngưỡng kiểm soát chất lượng kỹ thuật:

| Tiêu chuẩn QC | Mục đích | Cách tính | Căn cứ lựa chọn | Hậu quả nếu vi phạm | Phân loại trạng thái |
|:---|:---|:---|:---|:---|:---:|
| **FPS $\ge 25$** | Bảo đảm temporal buffer 10 frame bao phủ khoảng thời gian sinh học hợp lý ($\approx 300–400\text{ms}$) | $\frac{\text{Frames}}{\text{Giây}}$ qua `performance.now()` | Chuẩn webcam thông thường (30 FPS); dưới 25 FPS cử chỉ bị giật cục | Gắn cờ `low_fps_warning`; nếu $<15\text{ FPS}$ loại trial khỏi benchmark | **PROVISIONAL (Đề xuất kỹ thuật)** |
| **Landmark range $[-0.5, 1.5]$** | Loại bỏ frame mốc bị bắn ra ngoài vô cực do lỗi giải thuật MediaPipe | $\min(x, y) \ge -0.5$ và $\max(x, y) \le 1.5$ | [extract_video_landmarks.py#L64](file:///d:/SIGN_LANGUAGE/backend/training/extract_video_landmarks.py#L64) đang dùng ngưỡng này để lọc mốc rác | Đánh dấu frame `quality_status: "out_of_bounds"`, loại frame khỏi tính toán | **VERIFIED IN REPO (Đã có trong code)** |
| **Tracking gap $\le 100\text{ms}$** | Tránh mất dấu bàn tay quá lâu trong pha giữ tĩnh | Thời gian mất dấu liên tục giữa 2 frame có landmark | Mất dấu $>100\text{ms}$ (~3 frame) làm đứt chuỗi tích lũy của bộ đệm trượt 10 frame | Gắn nhãn trial `incomplete: true`, loại khỏi tập đánh giá Stable Hold | **PROVISIONAL (Đề xuất kỹ thuật)** |
| **Hand velocity $\Delta d < 0.03\text{ palm/frame}$** | Cờ kiểm tra tự động xem người ký có bị rung tay quá mức không | Dịch chuyển mốc cổ tay $\mathbf{p}_0$ giữa 2 frame chia cho `palm_scale` | Tham khảo ngưỡng ổn định tư thế trong các nghiên cứu kinematic | Chỉ dùng làm **cờ kiểm tra chất lượng (QC flag)**, **TUYỆT ĐỐI KHÔNG DÙNG LÀM GROUND TRUTH** | **PROVISIONAL (Đề xuất kiểm tra)** |

---

## 9. RỦI RO, GIỚI HẠN VÀ CÁC QUYẾT ĐỊNH CẦN USER PHÊ DUYỆT

### 9.1. Các rủi ro kỹ thuật cần lưu ý
1. **Rủi ro suy giảm hiệu năng client (Client Throttling)**: Vừa render, vừa chạy MediaPipe Hands JS 60 FPS, vừa encode video MediaRecorder trên máy cấu hình yếu có thể làm tụt FPS.
2. **Rủi ro đồng bộ thời gian (Clock Drift)**: Lệch timestamp giữa client và server cần được xử lý triệt để bằng cặp khóa `(frame_idx, client_capture_ts)`.

### 9.2. Các quyết định cần User phê duyệt trước khi chuyển sang TASK 3.10
1. **Phê duyệt phương án Video RGB**:
   - Đồng ý triển khai ghi video RGB kèm làm mờ khuôn mặt để có visual ground truth kiểm chứng, hay chỉ thu thập landmark JSONL thuần túy.
2. **Phê duyệt số lượng Signers mục tiêu**:
   - Chấp thuận mục tiêu thiết kế ban đầu 5–8 signers độc lập cho đợt thu thập dữ liệu chuyên sâu chữ O.
3. **Phê duyệt nâng cấp Schema Logging**:
   - Đồng ý cho phép nâng cấp `pilot_logger` và `recognizer.py` trong TASK 3.10 để lưu thêm `raw_landmarks_21`, `timing`, và `video_sync`.

---

## 10. CHECKLIST TRIỂN KHAI CHO TASK 3.10 (IMPLEMENTATION CHECKLIST)
- [ ] Bổ sung trường `raw_landmarks` vào payload `log_frame` trong [backend/vision/recognizer.py](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py).
- [ ] Bổ sung cấu trúc `timing` (client capture, extract, send, backend recv) vào request và log.
- [ ] Tích hợp `MediaRecorder` ở frontend [src/components/PilotTestingHUD.jsx](file:///d:/SIGN_LANGUAGE/src/components/PilotTestingHUD.jsx) để ghi video clip theo từng trial.
- [ ] Cập nhật giao diện HUD hướng dẫn rõ ràng 5 pha động học cho người tham gia.
- [ ] Chuẩn bị kịch bản kiểm tra cho các nhóm đối chứng âm tính (Negative Controls: B, C, E, T, S, A, P, X).

---
*Tài liệu được lập và rà soát hoàn toàn ở chế độ READ-ONLY. Tuyệt đối không thay đổi mã nguồn production, các ngưỡng quy tắc hoặc bộ dữ liệu hiện hành.*
