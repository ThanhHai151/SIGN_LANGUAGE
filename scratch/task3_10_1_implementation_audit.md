# BÁO CÁO AUDIT ĐỘC LẬP TASK 3.10.1
## KIỂM TOÁN TÍNH TOÀN VẸN TRIỂN KHAI VÀ TUÂN THỦ THIẾT KẾ THU THẬP DỮ LIỆU
**Chế độ**: READ-ONLY AUDIT — KHÔNG SỬA ĐỔI PRODUCTION CODE / TEST / THRESHOLD
**Thời gian thực hiện**: 2026-11-09T02:30:00+07:00
**Tài liệu tham chiếu SSOT**: `docs/VSL_8_LETTERS_FIX_PLAN.md`, `scratch/task3_9_data_collection_design.md`, Báo cáo TASK 3.9.1, Báo cáo TASK 3.10.

---

### 1. DANH SÁCH CÁC TỆP VÀ HÀM ĐÃ KIỂM TRA (AUDITED FILES & FUNCTIONS)

| Tệp Kiểm Tra | Vị Trí / Hàm Kiểm Tra | Mục Đích Đối Chiếu |
|:---|:---|:---|
| `src/components/PilotTestingHUD.jsx` | `DEFAULT_TRIAL_DURATION_SEC` (L36), `DURATION_PRESETS` (L37), `handleStartTrial` (L402-442), `handleEndTrial` (L445-470), `startMediaRecorder` (L209-242), `stopMediaRecorder` (L244-272), Signer Dropdown (L780-803), Badge 5 pha (L1110-1120) | Đối chiếu thời lượng 5 pha (Audit A), phân chia Signer (Audit B), ghi video và đồng bộ (Audit D), quyền riêng tư (Audit F) |
| `src/components/CameraSignRecognizer.jsx` | `getUserMedia` (L304-310), `onResults` & `pilotMeta` (L830-870) | Luồng camera (`audio: false`), thời điểm lấy `currentTime` và monotonic clock |
| `backend/vision/recognizer.py` | `_build_pilot_frame_record` (L432-576), call sites (L1029, L1080, L1126, L1623, L1887), `predict_landmarks` smoothing (L1570-1615) | Schema 21 raw landmarks (Audit C), fix lỗi `most_common/count` và tính nhân quả temporal (Audit E) |
| `backend/vision/pilot_logger.py` | `compute_file_sha256` (L462-473), `save_sidecar_sync` (L475-498), `create_session_manifest` (L500-532), `save_session_manifest` (L534-558) | Toàn vẹn dữ liệu, SHA256 và lưu manifest nguyên tử (Audit F) |
| `backend/app.py` | `LandmarkRequest` (L418-440), `/api/recognize_landmarks` (L480-510), `/api/pilot/manifest` (L460-475), `/api/pilot/sidecar_sync` (L442-458) | Timing backend, schema API và nhận diện wall clock (Audit D, F) |
| `backend/tests/test_pilot_task3_10.py` | 12 test cases thuộc 6 nhóm kiểm thử | Chất lượng kiểm thử, mock vs real files, invariant protection (Audit G) |
| `.gitignore` | Dòng cuối: `*.webm`, `*.mp4`, `*.mkv` | Bảo vệ video không bị đưa vào Git (Audit F) |

---

### 2. KẾT QUẢ KIỂM TOÁN TỔNG QUAN THEO TỪNG NHÓM (A – G)

| Nhóm Kiểm Toán | Nội Dung Kiểm Toán | Phân Loại Kết Quả | Trạng Thái Đánh Giá |
|:---|:---|:---|:---:|
| **AUDIT A** | Tuân thủ Protocol 5 pha & Thời lượng Trial | **FAIL** | Phát hiện sai lệch nghiêm trọng giữa thiết kế đã duyệt (6.0s) và code thực tế (5.0s default, Stable Hold chỉ 2.0s), đồng thời báo cáo TASK 3.10 ghi sai 4.0s (ảo giác báo cáo). |
| **AUDIT B** | Phân chia Người ký (Signer Split) | **VERIFIED** | Đủ 8 signer, phân định 3 tập riêng biệt `(Dev)`, `(Val)`, `(Holdout)` trong dropdown và manifest. |
| **AUDIT C** | Schema 21 Raw Landmarks & Metadata | **PARTIAL PASS / DISCREPANCY** | Bảo toàn đủ 21 điểm mốc thô và chuẩn hóa trên đĩa. Tuy nhiên cấu trúc thực tế là phẳng (`image_landmarks_21`), không có wrapper `raw_landmarks_21` như báo cáo mô tả; dùng `ERROR` thay vì `INVALID_SHAPE`. |
| **AUDIT D** | Đồng bộ Video RGB và Tệp Sidecar | **APPROXIMATE ONLY** | Đồng bộ dựa trên preview `currentTime` và monotonic clock client, không phải hardware PTS container. Không thể coi là đồng bộ chính xác cấp độ khung hình. |
| **AUDIT E** | Hồi quy Temporal Pipeline & Sửa lỗi biến | **VERIFIED** | Sửa `most_common/count` là bugfix thuần túy giải quyết `UnboundLocalError` cho logging, zero hồi quy thuật toán smoothing. Snapshot temporal đảm bảo tính nhân quả. |
| **AUDIT F** | Quyền riêng tư, Hash SHA256 & Manifest | **VERIFIED** | Ghi video cục bộ, `audio: false`, loại trừ khỏi Git, SHA256 thật trên JSONL/sidecar, video hash ghi `None` trung thực, ghi đĩa nguyên tử. |
| **AUDIT G** | Chất lượng Kiểm thử (Test Suite Quality) | **VERIFIED** | 10/12 test kiểm tra tệp thực tế trên đĩa (không lạm dụng mock), 163/163 test PASS, frontend build thành công. |

---

### 3. CHI TIẾT AUDIT A — PROTOCOL COMPLIANCE (SAI LỆCH THỜI LƯỢNG PHA)

#### 3.1. Đối chiếu Thiết kế đã duyệt vs Code thực tế vs Báo cáo TASK 3.10

| Pha Động Học | Thiết Kế Duyệt (TASK 3.9/3.9.1) | Code Thực Tế (`PilotTestingHUD.jsx` @ 5.0s Default) | Code Thực Tế (Nếu chỉnh lên 6.0s) | Báo Cáo TASK 3.10 Đã Ghi |
|:---|:---:|:---:|:---:|:---:|
| **Pha 1: Pre-idle** | 0.5 giây | **0.5 giây** (`elapsedSec < 0.5`) | 0.5 giây | 1.0 giây *(Sai lệch)* |
| **Pha 2: Entry** | 1.0 giây | **1.0 giây** (`0.5 <= elapsedSec < 1.5`) | 1.0 giây | 0.5 giây *(Sai lệch)* |
| **Pha 3: Stable Hold** | **3.0 giây** | **2.0 giây** (`1.5 <= elapsedSec < 3.5`) | **3.0 giây** (`1.5 <= elapsedSec < 4.5`) | 1.5 giây *(Sai lệch)* |
| **Pha 4: Exit** | 1.0 giây | **1.0 giây** (`3.5 <= elapsedSec < 4.5`) | 1.0 giây (`4.5 <= elapsedSec < 5.5`) | 0.5 giây *(Sai lệch)* |
| **Pha 5: Post-idle** | 0.5 giây | **0.5 giây** (`4.5 <= elapsedSec <= 5.0`) | 0.5 giây (`5.5 <= elapsedSec <= 6.0`) | 0.5 giây *(Đúng)* |
| **Tổng Thời Lượng** | **6.0 giây / trial** | **5.0 giây / trial** *(Mặc định khi mở)* | **6.0 giây / trial** | **4.0 giây / trial** *(Ảo giác)* |

#### 3.2. Bằng chứng trực tiếp từ Code (`src/components/PilotTestingHUD.jsx`)
1. Dòng 36: `export const DEFAULT_TRIAL_DURATION_SEC = 5.0;`
2. Dòng 37: `export const DURATION_PRESETS = [3.0, 5.0, 8.0, 10.0];` $\rightarrow$ Preset 6.0s hoàn toàn không có trong danh sách nút chọn nhanh!
3. Dòng 424–434: Logic phân định pha:
   ```javascript
   if (elapsedSec < 0.5) {
     setCurrentKinematicPhase('pre_idle');
   } else if (elapsedSec < 1.5) {
     setCurrentKinematicPhase('entry');
   } else if (elapsedSec < durationSec - 1.5) {
     setCurrentKinematicPhase('stable_hold');
   } else if (elapsedSec < durationSec - 0.5) {
     setCurrentKinematicPhase('exit');
   } else {
     setCurrentKinematicPhase('post_idle');
   }
   ```
4. Dòng 1117: UI hiển thị chữ: `'Pha 3: Stable Hold (Giữ ổn định 3s - Đánh giá tĩnh)'`.
   - **Mâu thuẫn logic nội bộ**: Khi mở ứng dụng với cấu hình mặc định `5.0s`, Stable Hold kết thúc tại $5.0 - 1.5 = 3.5\text{s}$, tức chỉ kéo dài $3.5 - 1.5 = \mathbf{2.0\text{ giây}}$. Trong khi đó, dòng chữ hướng dẫn trên màn hình lại ghi là "Giữ ổn định 3s"!
5. Nguồn thời gian: Cả đồng hồ đếm ngược `countdownSeconds` và pha `currentKinematicPhase` đều được tính từ cùng một nguồn: `Date.now() - startTime` trong `setInterval` 50ms (`VERIFIED`).
6. **Nguồn gốc thay đổi**:
   - Con số 5.0s là giá trị legacy từ HUD cũ (Task 7F). Khi triển khai TASK 3.10, công thức phân pha `0.5 + 1.0 + (durationSec - 3.0) + 1.0 + 0.5` được thiết kế đúng cho tổng 6.0s, nhưng hằng số `DEFAULT_TRIAL_DURATION_SEC` bị bỏ quên ở mức 5.0s.
   - Báo cáo TASK 3.10 ghi "tổng 4s (1.0s, 0.5s, 1.5s, 0.5s, 0.5s)" là sự suy diễn/ảo giác của tác giả báo cáo, không hề có căn cứ trong code và chưa từng được User phê duyệt.
- **Kết luận AUDIT A**: **`FAIL`**. Protocol mặc định hiện tại không tuân thủ thiết kế 6.0s đã duyệt. Cần đưa `DEFAULT_TRIAL_DURATION_SEC` về `6.0` và cập nhật `DURATION_PRESETS` bổ sung `6.0`.

---

### 4. CHI TIẾT AUDIT B — SIGNER SPLIT

#### 4.1. Bằng chứng trực tiếp từ Code (`src/components/PilotTestingHUD.jsx`)
- Dòng 797–801:
  ```javascript
  {['signer_01', 'signer_02', 'signer_03', 'signer_04', 'signer_05', 'signer_06', 'signer_07', 'signer_08'].map((s) => (
    <option key={s} value={s}>
      {s} {s === 'signer_01' || s === 'signer_02' ? '(Dev)' : s === 'signer_03' || s === 'signer_04' ? '(Val)' : '(Holdout)'}
    </option>
  ))}
  ```
- Dropdown hiển thị đầy đủ 8 signer với nhãn rõ ràng:
  - `signer_01`, `signer_02`: Gắn nhãn `(Dev)` (Development Set).
  - `signer_03`, `signer_04`: Gắn nhãn `(Val)` (Validation Set).
  - `signer_05`, `signer_06`, `signer_07`, `signer_08`: Gắn nhãn `(Holdout)` (Independent Holdout Set).
- `signer_id` được truyền thông suốt từ UI $\rightarrow$ `CameraSignRecognizer.jsx` $\rightarrow$ `recognizer.py` $\rightarrow$ JSONL record $\rightarrow$ `pilot_logger.py` $\rightarrow$ `manifest.json`.
- Tập Holdout (`signer_05–08`) được phân tách độc lập, không bị gộp chung vào tập huấn luyện hay tuning.
- **Kết luận AUDIT B**: **`VERIFIED — PASS`**.

---

### 5. CHI TIẾT AUDIT C — RAW LANDMARK SCHEMA

#### 5.1. Bằng chứng thực nghiệm từ Code và Output JSONL thực tế
Chạy kiểm tra độc lập trên `_build_pilot_frame_record`:

1. **Số lượng điểm mốc và quy ước tọa độ**:
   - `image_landmarks_21`: Đúng 21 điểm mốc theo thứ tự chuẩn của MediaPipe Hands (Index 0: Wrist, Index 4: Thumb Tip, Index 8: Index Tip, Index 12: Middle Tip, Index 16: Ring Tip, Index 20: Pinky Tip).
   - Tọa độ viewport $[0, 1] \times [0, 1]$ với $z$ tương đối được làm tròn 6 chữ số thập phân (`VERIFIED`).
   - `normalized_landmarks_21`: Lưu độc lập 21 điểm sau khi tịnh tiến gốc cổ tay về $[0.0, 0.0, 0.0]$ và chia cho khoảng cách cổ tay – gốc ngón giữa $d_{\text{palm}} = \|\mathbf{p}_0 - \mathbf{p}_9\|_2$. Cả hai trường đều được lưu riêng biệt, không ghi đè dữ liệu gốc (`VERIFIED`).
2. **Xử lý khi mất tracking bàn tay**:
   - Khi `p_prim is None`: `tracking_status = "NO_HAND"`, `image_landmarks_21 = None`, `normalized_landmarks_21 = None`. Không tạo tọa độ giả (`VERIFIED`).
   - Khi có tay nhưng thiếu điểm: `tracking_status = "PARTIAL_LANDMARKS"`.
   - Khi tọa độ chứa NaN/Inf hoặc sai kích thước $(21, 3)$: `tracking_status = "ERROR"`. (Lưu ý: Code dùng `"ERROR"`, chưa tách riêng enum `"INVALID_SHAPE"`).
3. **Các trường metadata đi kèm**:
   - `image_width`, `image_height`: Nhận từ kích thước video element (mặc định 640x480).
   - `handedness`, `handedness_score`, `tracking_confidence`: Lưu đầy đủ.
   - `feature_extractor_version: "2.0-kinematic"`, `classifier_rule_version: "Candidate-03"`, `schema_version: "2.0-kinematic"`.
4. **Sai lệch giữa Báo cáo TASK 3.10 và Code thực tế**:
   - Báo cáo TASK 3.10 trình bày mẫu JSONL có chứa wrapper cha `"raw_landmarks_21": { "image_landmarks_21": [...], "normalized_landmarks_21": [...] }`.
   - **Thực tế trong code**: Cả hai trường `"image_landmarks_21"` và `"normalized_landmarks_21"` nằm trực tiếp ở cấp cao nhất (top-level keys), KHÔNG có key bọc `"raw_landmarks_21"`.
   - Ví dụ 3 điểm trong báo cáo là do tác giả viết rút gọn (đại diện điểm 0, 4, 8), trên thực tế code ghi trọn vẹn đủ 21 điểm mốc.
- **Kết luận AUDIT C**: **`PARTIAL PASS`**. Dữ liệu 21 landmarks được bảo toàn nguyên vẹn và chuẩn xác trên đĩa. Điểm cần ghi nhận là sai khác tài liệu hóa (không có key bọc `raw_landmarks_21`) và dùng nhãn `ERROR` thay cho `INVALID_SHAPE`.

---

### 6. CHI TIẾT AUDIT D — VIDEO AND SIDECAR SYNCHRONIZATION

#### 6.1. Phân tích 7 tiêu chí đồng bộ

1. **Thời điểm lấy `videoElement.currentTime`**:
   - Được lấy trong callback `onResults` của MediaPipe Hands tại `CameraSignRecognizer.jsx` (L846), tức là thời điểm landmark đã được trích xuất xong từ video frame, ngay trước khi gửi API sang backend.
2. **Bản chất của giá trị `currentTime`**:
   - Đối với thẻ `<video>` phát luồng trực tiếp `MediaStream` từ webcam (`video.srcObject = stream`), `currentTime` phản ánh đồng hồ phát của thẻ preview trên trình duyệt.
   - Đây **KHÔNG PHẢI** là Presentation Timestamp (PTS) nhúng trong container video WebM do `MediaRecorder` sinh ra ở luồng mã hóa ngầm.
3. **Mapping từ `frame_idx` sang video frame**:
   - `video_frame_idx` trong code hiện tại chỉ đơn thuần gán bằng `currFidx` (chỉ số khung hình của MediaPipe).
   - Không có bộ đếm khung hình phần cứng từ encoder video.
4. **Khả năng lệch khung hình giữa MediaPipe và MediaRecorder**:
   - `MediaRecorder` ghi video theo tốc độ ổn định (ví dụ 30 FPS từ webcam).
   - Trong khi đó, `MediaPipe Hands` chạy xử lý AI có thể dao động (ví dụ 20–25 FPS tùy tải CPU máy người dùng).
   - Do đó, số lượng phần tử trong `syncMappingRef` không nhất thiết bằng tổng số khung hình trong tệp `.webm`. Khung hình thứ $N$ trong sidecar không đồng nhất tuyệt đối với frame thứ $N$ trong video.
5. **Cơ chế phát hiện bất thường**:
   - Code có theo dõi `jumps` (nếu $\Delta frame\_idx > 1$) và `stalls` (nếu khoảng cách gửi request $> 70\text{ms}$) lưu vào state `syncAnomalies` (`VERIFIED`).
6. **Xung đột hệ quy chiếu thời gian (Time Bases)**:
   - Client sử dụng `performance.now() / 1000` (Monotonic clock, tính từ lúc nạp trang, giá trị nhỏ dạng $\approx 15.24\text{s}$).
   - Backend sử dụng `time.time()` (UNIX Epoch wall clock, giá trị dạng $\approx 1731112456.78\text{s}$).
   - Hai hệ quy chiếu này không thể trừ trực tiếp cho nhau để tính độ trễ mạng nếu không có cơ chế bắt chéo đồng hồ (handshake offset).
7. **Xử lý UNKNOWN**:
   - Khi `currentTime` không hợp lệ (không hữu hạn), code ghi nhận trung thực `"UNKNOWN"`.
- **Kết luận AUDIT D**: **`APPROXIMATE ONLY`**. Cơ chế đồng bộ hiện tại là giải pháp xấp xỉ tốt nhất có thể thực hiện trên tầng Web API client, nhưng **chưa đủ bằng chứng để tuyên bố là đồng bộ chính xác cấp độ khung hình (Frame-accurate)**. Cần xác định rõ đây là "Đồng bộ gần đúng có giới hạn kỹ thuật".

---

### 7. CHI TIẾT AUDIT E — TEMPORAL PIPELINE REGRESSION

#### 7.1. Phân tích nguyên nhân và bản chất Bugfix `most_common/count`
1. **Trước TASK 3.10**:
   - Trong `recognizer.py`, hai biến `most_common` và `count` chỉ được khởi tạo và gán giá trị bên trong khối điều kiện:
     `if len(recent_signs) >= self.smoothing_window // 2:`
   - Khi cửa sổ chưa tích lũy đủ 5 dự đoán hợp lệ (ví dụ các khung hình 1, 2, 3, 4), hai biến này hoàn toàn không tồn tại trong phạm vi cục bộ (`local scope`).
2. **Nguyên nhân lỗi xuất hiện ở TASK 3.10**:
   - Khi thêm đoạn code lưu vết `temporal_snapshot`, tác giả truy cập `count` và `most_common` tại dòng log chẩn đoán:
     `vote_count_val = count if (len(recent_signs) >= self.smoothing_window // 2 and most_common in ALLOWED_LETTERS) else 0`
   - Khi `len(recent_signs) < 5`, việc đánh giá biến chưa khai báo đã kích hoạt `UnboundLocalError`.
3. **Cách khắc phục**:
   - Khởi tạo giá trị mặc định an toàn trước khối `if`:
     `most_common = None`
     `count = 0`
4. **Kiểm tra hồi quy thuật toán**:
   - Điều kiện xác nhận token: `count >= (self.smoothing_window // 2) and most_common in ALLOWED_LETTERS` $\rightarrow$ **100% Không đổi**.
   - Thời điểm phát token (Cooldown logic): Giữ nguyên phép so sánh `current_time_clock - self.last_confirmed_time > self.cooldown_sec` $\rightarrow$ **100% Không đổi**.
   - Độ tin cậy (Confidence), fallback MLP, quy tắc hình học: **100% Không đổi**.
   - Thời điểm tạo snapshot: Được tạo ngay sau khi cập nhật `prediction_history` của khung hình hiện tại $N$. Danh sách trượt chỉ chứa các frame $\le N$, không thể chứa frame $N+1$ (được chứng minh qua test case `test_3_2_no_future_frames_in_temporal_state`).
- **Kết luận AUDIT E**: **`VERIFIED — PASS`**. Đây là bugfix cú pháp bắt buộc cho tính năng logging, hoàn toàn không can thiệp hay làm biến đổi thuật toán làm mượt thời gian.

---

### 8. CHI TIẾT AUDIT F — PRIVACY, MANIFEST AND HASH

#### 8.1. Kiểm tra Quyền riêng tư
- MediaRecorder chỉ bắt đầu khi người dùng chủ động nhấn nút Bắt đầu Trial (`handleStartTrial`).
- Thẻ luồng camera được cấu hình rõ ràng: `{ video: true, audio: false }` $\rightarrow$ Tuyệt đối không thu âm micro.
- Video được giữ hoàn toàn trong RAM client dưới dạng `Blob`, chỉ tải về khi người dùng bấm nút tải xuống. Không có bất kỳ dòng code nào gửi video lên backend hoặc dịch vụ đám mây.
- Khi đóng hoặc hủy HUD giữa chừng, `useEffect` unmount gọi `mediaRecorder.stop()` để giải phóng luồng ghi hình.
- File `.gitignore` đã loại trừ toàn bộ đuôi mở rộng video (`*.webm`, `*.mp4`, `*.mkv`).

#### 8.2. Kiểm tra Manifest và Hàm băm SHA256
- `compute_file_sha256(filepath)`: Đọc nhị phân theo khối 64KB, tính mã băm SHA256 chính xác của tệp JSONL và tệp sidecar đã ghi xong trên đĩa.
- Manifest được lưu nguyên tử bằng cách ghi vào tệp `.tmp_<timestamp>` rồi gọi `os.replace()`, ngăn chặn hiện tượng tệp bị đọc dở dang (race conditions).
- Trường `video.sha256` trong manifest được gán giá trị trung thực là `null` với ghi chú: `"Client local storage via MediaRecorder (not tracked in Git)"`. Hệ thống không bịa đặt mã băm video khi video chưa được đưa lên máy chủ.
- **Kết luận AUDIT F**: **`VERIFIED — PASS`**.

---

### 9. CHI TIẾT AUDIT G — TEST QUALITY & RE-EXECUTION

#### 9.1. Đánh giá chất lượng 12 bài test trong `test_pilot_task3_10.py`
- 10 / 12 test case thực hiện ghi và đọc tệp thật trên đĩa trong thư mục tạm `tempfile.mkdtemp()`, không sử dụng mock cho luồng ghi log JSONL và manifest.
- Test `test_1_1` và `test_1_4` kiểm tra đủ 21 điểm mốc, xác thực chiều không gian và tính chuẩn hóa của gốc cổ tay.
- Test `test_3_2` kiểm tra trực tiếp tính nhân quả bằng cách duyệt qua 5 khung hình liên tiếp và xác minh rằng snapshot tại frame $k$ có độ dài đúng bằng $k$ và không chứa bất kỳ frame nào $> k$.
- Test `test_6_1` bảo vệ trực tiếp các hằng số ngưỡng của Candidate-03 (`O_D_TI_MAX = 0.22`, `O_IDX_STRAIGHT_MAX = 0.48`), Rule 15 (P) và Rule 18 (X).
- Giới hạn: Không thể kiểm thử MediaRecorder trong môi trường Python unittest do thiếu DOM browser; phần này được kiểm chứng qua lệnh build thành công của Vite.

#### 9.2. Kết quả chạy lại toàn bộ Test Suite và Build

| Bộ Kiểm Thử | Lệnh Thực Thi | Số Lượng Test | Kết Quả | Thời Gian |
|:---|:---|:---:|:---:|:---:|
| **Task 3.10 Suite** | `python -m pytest backend/tests/test_pilot_task3_10.py -v` | 12 items | **12 / 12 PASS** | 4.21s |
| **Alphabet Classifier** | `python -m pytest backend/tests/test_alphabet_classifier.py` | 132 items | **132 / 132 PASS** | 1.67s |
| **Pilot Logging** | `python -m pytest backend/tests/test_pilot_logging.py` | 13 items | **13 / 13 PASS** | 3.36s |
| **Pilot 29 Letters HUD** | `python -m pytest backend/tests/test_pilot_29_letters_hud.py` | 6 items | **6 / 6 PASS** | 4.10s |
| **Tổng Test Suites Liên Quan** | *Chạy độc lập từng bộ* | **163 items** | **163 / 163 PASS (100%)** | ~13.34s |
| **Frontend Production Build** | `npm run build` | 1899 modules | **PASS (0 errors)** | 994ms |

---

### 10. TỔNG HỢP DANH SÁCH VẤN ĐỀ CẦN SỬA TRONG TASK TIẾP THEO

Dựa trên kết quả kiểm toán, các vấn đề kỹ thuật sau đây cần được xử lý trong một task điều chỉnh riêng biệt trước khi tiến hành thu thập dữ liệu với người ký thật:

1. **Vấn đề 1: Đưa thời lượng trial mặc định về đúng 6.0 giây theo thiết kế đã duyệt** *(Mức độ: Cao)*:
   - Trong `src/components/PilotTestingHUD.jsx`:
     - Sửa `DEFAULT_TRIAL_DURATION_SEC = 6.0` (thay vì 5.0).
     - Cập nhật `DURATION_PRESETS = [3.0, 5.0, 6.0, 8.0, 10.0]`.
     - Điều này sẽ tự động khôi phục thời lượng Stable Hold đạt đúng **3.0 giây** chuẩn ($6.0 - 1.5 - 1.5 = 3.0\text{s}$) khớp hoàn toàn với nhãn giao diện "Giữ ổn định 3s".
2. **Vấn đề 2: Đồng bộ hóa tài liệu schema JSONL với code thực tế** *(Mức độ: Trung bình)*:
   - Cập nhật SSOT và tài liệu hướng dẫn ghi rõ hai trường là `image_landmarks_21` và `normalized_landmarks_21` ở cấp cao nhất (không có wrapper `raw_landmarks_21`).
3. **Vấn đề 3: Tách rõ trạng thái tracking `INVALID_SHAPE`** *(Mức độ: Thấp)*:
   - Trong `recognizer.py` (L1087): Cân nhắc phân biệt rõ `tracking_status = "INVALID_SHAPE"` khi mảng landmark sai kích thước hoặc chứa NaN, thay vì gộp chung vào nhãn `"ERROR"`.
4. **Vấn đề 4: Ghi chú rõ tính chất đồng bộ video là gần đúng** *(Mức độ: Trung bình)*:
   - Ghi nhận trong tài liệu phân tích dữ liệu rằng `video_pts_sec` là thời gian preview của thẻ video và client monotonic clock, không dùng để đối chiếu micro-second với backend wall clock.

---

### 11. STOP CONDITIONS & KHẲNG ĐỊNH BẢO TOÀN TRẠNG THÁI

1. Không sửa bất kỳ dòng mã production nào trong suốt quá trình thực hiện Audit TASK 3.10.1.
2. Không thay đổi ngưỡng hoặc thuật toán của Candidate-03, Rule 15 (P), Rule 18 (X).
3. Không triển khai Candidate-04.
4. Không bắt đầu thu thập dữ liệu thực tế.
5. Trạng thái của chữ O trên SSOT giữ nguyên tuyệt đối:
   $$\mathbf{O: \text{WEBCAM VERIFIED — FAIL (TASK 3.7: 0/10 TRIALS } \ge \text{70\%, NOT FIXED)}}$$

*Báo cáo kết thúc. Dừng lại và chờ quyết định từ User.*

