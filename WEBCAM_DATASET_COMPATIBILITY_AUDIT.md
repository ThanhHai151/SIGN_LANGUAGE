**PROMPT ID:** PROMPT 02
**PROMPT NAME:** KIỂM TRA TÍNH TƯƠNG THÍCH WEBCAM VÀ DATASET SSTCN

# BÁO CÁO KIỂM TOÁN TÍNH TƯƠNG THÍCH PIPELINE WEBCAM VÀ DATASET SSTCN (PROMPT 02)

- **Ngày thực hiện:** 2026-11-08
- **Phạm vi kiểm toán:** Pipeline nhận diện webcam chế độ `words` (63 từ vựng thuộc 102 lớp của mô hình [`SSTCN_Attention`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/models/sstcn_model.py#L48-L120)) đối chiếu với hợp đồng dữ liệu (data contract) của dataset huấn luyện chính thức do **Hải** cung cấp (`41.645` mẫu sau chuẩn bị, tensor shape `(41645, 3, 30, 75)`, manifest SHA-256 `e3340c0d55b27dce703a1575d721c339c73609dc19ad730d616f027099fb3920`).
- **Ràng buộc:** Báo cáo kiểm toán chỉ đọc (Read-only Audit). Không chỉnh sửa mã nguồn runtime, không can thiệp các chế độ `letters`, `digits`, `sentence`, không thay đổi checkpoint hay dữ liệu.

---

## 1. Bảng Tổng Hợp Kết Quả 10 Hạng Mục Kiểm Toán

| # | Hạng mục kiểm tra | Trạng thái trên `holistic` (`Toàn thân`) | Trạng thái trên `hands` (`Bàn tay` fallback) | Trạng thái tổng thể (`words` mode) | Mức độ ảnh hưởng tới Accuracy |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **1** | Engine MediaPipe đang dùng ở webcam (`Pose`, `Hands`, `Holistic`) | `PASS` (nhưng giảm độ phân giải & `modelComplexity: 0`) | `FAIL` (cho phép chạy `Hands` ở mode `words`) | **`FAIL`** | **Nghiêm trọng (Critical)** |
| **2** | Trích xuất đủ `33 pose + 21 left_hand + 21 right_hand` | `PASS` (75 khớp thật) | `FAIL` (33 khớp pose bị làm giả) | **`FAIL` (Khi chạy `hands`) / `PASS` (Khi chạy `holistic`)** | **Nghiêm trọng (Critical)** |
| **3** | Landmark vai, khuỷu tay, cổ tay đi vào tensor hay chỉ hiển thị | `PASS` (đi vào tensor thật) | `FAIL` (vai cố định 100%, khuỷu tay nội suy tuyến tính giả) | **`FAIL` (Khi chạy `hands`) / `PASS` (Khi chạy `holistic`)** | **Nghiêm trọng (Critical)** |
| **4** | Thứ tự 75 landmark: `pose (0..32) → left_hand (33..53) → right_hand (54..74)` | `PASS` | `PASS` (về chỉ số mảng) | **`PASS`** | Không gây lỗi |
| **5** | Hệ tọa độ khớp `mediapipe_image_normalized_xyz` | `PASS` | `FAIL` (`z=0` cố định cho pose giả) | **`PASS` (trên `holistic`)** | Trung bình – Cao (nếu rơi vào `hands`) |
| **6** | Xử lý landmark thiếu bằng `0` & không chuẩn hóa lệch với lúc train | `PASS` | `FAIL` (pose thiếu không gán `0`; tay mất nhưng cổ tay/khuỷu tay pose vẫn giữ hằng số) | **`FAIL` (Khi chạy `hands`) / `PASS` (Khi chạy `holistic`)** | **Cao (High)** |
| **7** | Lấy 30 frame liên tục trên webcam vs `linear_resample_to_30_after_frame_stride` | `FAIL` (`FIFO 30 frames` ~2.52s @ 13.6 FPS vs co giãn toàn bộ cử chỉ về 30 frames) | `FAIL` (`FIFO 30 frames` ~1.20s @ 26.1 FPS) | **`FAIL`** | **Nghiêm trọng nhất (Blocker)** |
| **8** | Mirror camera gây đảo trái/phải hoặc sai thứ tự landmark | `PASS` (chỉ lật CSS `scaleX(-1)`, ảnh gốc đưa vào Holistic không lật) | `FAIL` (`@mediapipe/hands` mặc định `selfieMode: false` gây đảo nhãn `Left`/`Right` trên ảnh không lật) | **`FAIL` (trên `hands`) / `CHƯA ĐỦ BẰNG CHỨNG` (về video MP4 gốc của dataset)** | **Cao (High)** |
| **9** | Mapping 63 lớp `words` khớp [`vsl_102_dictionary.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/models/vsl_102_dictionary.py) và 102 lớp của checkpoint | `PASS` (masking `0..38`, giữ `39..101`) | `PASS` | **`PASS`** | Không gây lỗi |
| **10** | Nguy cơ cắt một cử chỉ nhiều động tác thành 2 lần dự đoán riêng biệt | `FAIL` (cửa sổ trượt `FRAME_STEP=6`, xóa buffer khi mất tay 4 frame, cooldown chỉ chặn trùng nhãn) | `FAIL` | **`FAIL`** | **Nghiêm trọng (Critical)** |

---

## 2. Phân Tích Chi Tiết Từng Hạng Mục (1 → 10)

### Hạng mục 1: Webcam đang sử dụng MediaPipe Pose, Hands, Holistic hay kết hợp các thành phần nào?

- **Trạng thái:** **`FAIL`** *(Kiến trúc Dual-Engine cho phép chạy `Hands` không tương thích trong chế độ `words`, đồng thời cấu hình `Holistic` trên trình duyệt bị hạ cấp so với script tạo dataset)*.
- **Bằng chứng mã nguồn & thực nghiệm:**
  1. **Nạp thư viện CDN:** [`index.html`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/index.html#L18-L19) nạp đồng thời cả 2 thư viện WebAssembly từ jsDelivr:
     - `@mediapipe/holistic/holistic.js` (`window.Holistic`) tại dòng `18`.
     - `@mediapipe/hands/hands.js` (`window.Hands`) tại dòng `19`.
  2. **Khởi tạo mặc định là `hands`:** Trong [`CameraSignRecognizer.jsx`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L91-L99), state khởi tạo ban đầu khi mount component là `recognitionScope = 'hands'` (dòng `91`), `trackerEngine = 'hands'` (dòng `98`) và `activeScopeRef.current = 'hands'` (dòng `99`).
  3. **Chuyển đổi tự động nhưng cho phép người dùng ghi đè về `hands`:**
     - Khi người dùng bấm chọn tab `Từ vựng (Words)`, hàm [`handleModeChange`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L472-L491) (dòng `490`) đặt `const targetScope = (newMode === 'words') ? 'holistic' : 'hands'`.
     - **Tuy nhiên**, giao diện ở [`CameraSignRecognizer.jsx`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L1123-L1139) vẫn hiển thị nút chuyển đổi phạm vi **Tiêu chí A (`Bàn tay` / `Toàn thân`)** thông qua [`handleScopeChange`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L466-L469), cho phép người dùng (hoặc trạng thái khởi tạo) chạy chế độ `words` với engine `hands`. Thậm chí, thông báo UI tại [`getCompatibilityInfo`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L563-L569) còn hiển thị thông điệp gây hiểu nhầm: *"⚡ Hỗ trợ tốt: Nhận diện 102 từ động qua 2 bàn tay (Khớp Pose thân trên được ước lượng tự động)"*.
     - Bằng chứng thực nghiệm từ 14 phiên ghi log trong [`docs/vsl/word_ab_fixed_input/pose_temporal_audit.md`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/docs/vsl/word_ab_fixed_input/pose_temporal_audit.md#L11-L15): trong 501 cửa sổ dự đoán chế độ `words`, có tới **268/501 cửa sổ (53.5%)** được trích xuất bởi `hands_fallback_pose` và chỉ **233/501 cửa sổ (46.5%)** được trích xuất bởi `holistic_raw`.
  4. **Sự khác biệt cấu hình `Holistic` giữa Training và Webcam:**
     - **Lúc tạo dataset huấn luyện** ([`backend/training/prepare_sstcn.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/training/prepare_sstcn.py#L68-L69)): dùng Python `mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)` trên khung hình độ phân giải gốc của video.
     - **Lúc chạy webcam** ([`CameraSignRecognizer.jsx`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L285-L295) và [`L343-L371`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L343-L371)): dùng JS `window.Holistic` với `modelComplexity: 0, smoothLandmarks: false, minDetectionConfidence: 0.35, minTrackingConfidence: 0.30` và ép kích thước ảnh đầu vào xuống canvas phụ `480×360` (`holisticCanvasRef`, dòng `343-345`, `366-368`).
- **Ảnh hưởng đến nhận diện:**
  - Khi rơi vào engine `hands`, mô hình `SSTCN_Attention` nhận toàn bộ hình học thân trên giả (xem Hạng mục 2 & 3), làm accuracy Top-1 trên webcam giảm từ **15.0%** (`holistic_raw`) xuống **3.0%** (`hands_fallback_pose`).
  - Khi chạy `holistic`, việc đặt `modelComplexity: 0` và `smoothLandmarks: false` trên ảnh `480×360` với ngưỡng `0.35/0.30` làm tăng nhiễu rung (jitter) ở các khớp ngón tay và cổ tay so với `model_complexity=1` (có `smooth_landmarks=True` mặc định trong MediaPipe Python) của bộ dữ liệu huấn luyện.
- **Cách khắc phục:**
  1. Ở chế độ `words`, **khóa cứng (lock)** `recognitionScope = 'holistic'` và vô hiệu hóa nút chuyển về `'hands'` trên UI; sửa thông báo tại [`getCompatibilityInfo`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L563-L569).
  2. Trong [`createTracker('holistic')`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L287-L293), bật `smoothLandmarks: true` và nâng `minDetectionConfidence: 0.5, minTrackingConfidence: 0.5` để khớp với [`prepare_sstcn.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/training/prepare_sstcn.py#L68-L69).

---

### Hạng mục 2: Có trích xuất đủ 33 pose + 21 left hand + 21 right hand không?

- **Trạng thái:** **`PASS` (trên engine `holistic`) / `FAIL` (trên engine `hands` fallback)**.
- **Bằng chứng mã nguồn:**
  - **Hợp đồng dữ liệu huấn luyện:** [`backend/training/prepare_sstcn.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/training/prepare_sstcn.py#L79-L84) và [`backend/training/sstcn_data.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/training/sstcn_data.py#L14-L19) quy định mỗi frame gồm đúng 75 khớp: `33 pose` + `21 left_hand` + `21 right_hand`.
  - **Nhánh `holistic` (`PASS`):** Hàm [`extract75KeypointsFromHolistic(results)`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L185-L214) đọc trực tiếp:
    - `results.poseLandmarks` (33 điểm, dòng `187–194`).
    - `results.leftHandLandmarks` (21 điểm, dòng `196–203`).
    - `results.rightHandLandmarks` (21 điểm, dòng `205–212`).
  - **Nhánh `hands` (`FAIL`):** Hàm [`extract75KeypointsFromHands(results)`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L217-L269) chỉ có đầu vào từ `@mediapipe/hands` (`multiHandLandmarks` và `multiHandedness`), hoàn toàn **không có bộ dò Pose**. Để đủ 75 điểm, hàm này khởi tạo 33 điểm đầu tiên bằng mảng hằng số tĩnh `DEFAULT_POSE` (dòng `219–233`), chỉ ghi đè 4 điểm `[13, 14, 15, 16]` bằng phép nội suy hình học đơn giản từ cổ tay (dòng `246–254`), sau đó nối 21 điểm `leftHand` và 21 điểm `rightHand` (dòng `256–267`).
- **Ảnh hưởng đến nhận diện:**
  - Về mặt kích thước mảng (shape), cả 2 nhánh đều trả về mảng chiều dài `75 × 3` (do đó không gây crash kích thước tensor ở backend), nhưng về mặt ngữ nghĩa sinh học, nhánh `hands` cung cấp **33 khớp pose giả** hoàn toàn xa lạ với phân phối dữ liệu huấn luyện.
- **Cách khắc phục:**
  - Loại bỏ việc sử dụng [`extract75KeypointsFromHands`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L217-L269) cho mô hình `SSTCN_Attention` (mode `words`). Chỉ cho phép gọi [`extract75KeypointsFromHolistic`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L185-L214) khi `mode === 'words'`.

---

### Hạng mục 3: Các landmark vai, khuỷu tay, cổ tay có thực sự được đưa vào tensor hay chỉ hiển thị landmark bàn tay?

- **Trạng thái:** **`PASS` (trên engine `holistic`) / `FAIL` (trên engine `hands` fallback)**.
- **Bằng chứng mã nguồn & thực nghiệm:**
  1. **Tách biệt giữa luồng vẽ Canvas (hiển thị) và luồng tạo Tensor nhận diện:**
     - Trong [`onResults`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L693-L760), việc vẽ skeleton lên màn hình được thực hiện bởi [`renderSkeletonsOverlay`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L665-L691) (gọi tại dòng `719`), trong đó có dùng hàm dự đoán ngoại suy [`extrapolateLandmarks`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L587-L617) để làm mượt nét vẽ.
     - Ngược lại, tensor đưa vào `sequenceBufferRef` để gửi về backend được trích xuất từ `results` gốc tại dòng [`726`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L726) (`extract75KeypointsFromHands(results)`) hoặc dòng [`745`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L745) (`extract75KeypointsFromHolistic(results)`).
  2. **Khi chạy `holistic` (`PASS`):**
     - Các khớp vai trái/phải (`pose[11], pose[12]`), khuỷu tay trái/phải (`pose[13], pose[14]`) và cổ tay pose (`pose[15], pose[16]`) từ `results.poseLandmarks` **thực sự được đưa vào tensor** tại các chỉ số `11..16` ([`CameraSignRecognizer.jsx`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L187-L191)).
     - Kiểm chứng trên **233/233 file `.npy`** thuộc nhóm `holistic_raw` ([`pose_temporal_audit.md`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/docs/vsl/word_ab_fixed_input/pose_temporal_audit.md#L11-L15)): số khớp pose bất động là **`0 / 33`**; độ lệch chuẩn chuyển động vai trái/phải đạt $\sigma \approx 0.0086 - 0.0091$ và khuỷu tay đạt $\sigma \approx 0.0257 - 0.0363$ (khớp với phân phối thật của dataset huấn luyện trong [`data_provenance.md`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/docs/vsl/word_training_audit/data_provenance.md#L73-L92)).
  3. **Khi chạy `hands` fallback (`FAIL` nghiêm trọng):**
     - Tại [`CameraSignRecognizer.jsx`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L219-L254):
       - **Vai trái (`keypoints[11]`) và vai phải (`keypoints[12]`) bị gán cứng 100% bất động** tại `[0.65, 0.45, 0.0]` và `[0.35, 0.45, 0.0]` (dòng `224`), cùng với 27 khớp thân/mặt khác (`0..10`, `17..32`). Do đó **29/33 khớp pose hoàn toàn đứng yên ($\sigma = 0$)** xuyên suốt 30 frame.
       - **Cổ tay pose (`keypoints[15], keypoints[16]`)** chỉ là bản sao của `leftHand[0]` và `rightHand[0]` (dòng `247, 252`).
       - **Khuỷu tay (`keypoints[13], keypoints[14]`)** được bịa ra bằng trung điểm tuyến tính giữa vai cố định và cổ tay cộng hằng số lệch `±0.04` theo trục X và `z = 0`:
         - `keypoints[13] = [(0.65 + wx) / 2 + 0.04, (0.45 + wy) / 2, 0]` (dòng `248`)
         - `keypoints[14] = [(0.35 + wx) / 2 - 0.04, (0.45 + wy) / 2, 0]` (dòng `253`)
- **Ảnh hưởng đến nhận diện:**
  - Trong kiến trúc [`SSTCN_Attention`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/models/sstcn_model.py#L14-L45), ma trận kề đồ thị [`build_vsl_adjacency()`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/models/sstcn_model.py#L12-L45) nối trực tiếp chuỗi xương chính `(11, 13), (13, 15), (15, 33)` (vai trái → khuỷu trái → cổ tay pose → gốc bàn tay trái) và `(12, 14), (14, 16), (16, 54)` (bên phải). Khi 29/33 nút đồ thị bị đóng băng và khuỷu tay có `z = 0`, phép tích chập đồ thị không gian (`SpatialGraphConv`, dòng `67–72`) tạo ra vector đặc trưng hoàn toàn lệch khỏi không gian học được.
  - Bằng chứng thực nghiệm ([`conclusion.md`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/docs/vsl/word_ab_fixed_input/conclusion.md#L33-L45)): với từ **"Tên là gì"** (lớp 61), tỷ lệ lọt vào Top-5 trên `holistic_raw` là **28.9% (11/38 cửa sổ)** nhưng trên `hands_fallback_pose` là **0.0% (0/95 cửa sổ)** cho cả 2 checkpoint `Historical` và `Reconstructed Best`.
- **Cách khắc phục:**
  - Bắt buộc dùng `Holistic` cho chế độ `words`; tuyệt đối không đưa `DEFAULT_POSE` vào tensor của `SSTCN_Attention`.

---

### Hạng mục 4: Thứ tự 75 landmark có khớp dataset: `pose → left_hand → right_hand` không?

- **Trạng thái:** **`PASS`**.
- **Bằng chứng mã nguồn:**
  1. **Trong Dataset & Model:**
     - [`backend/training/prepare_sstcn.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/training/prepare_sstcn.py#L79-L84): ghép `np.concatenate([pose, left_hand, right_hand], axis=0)` với `pose` (33), `left_hand` (21), `right_hand` (21).
     - [`backend/training/sstcn_data.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/training/sstcn_data.py#L19): `"landmark_order": ["pose_33", "left_hand_21", "right_hand_21"]`.
     - [`backend/models/sstcn_model.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/models/sstcn_model.py#L14-L16): `0-32: Pose (33)`, `33-53: Left Hand (21)`, `54-74: Right Hand (21)`.
  2. **Trên Frontend Webcam:**
     - [`extract75KeypointsFromHolistic`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L185-L214) đẩy tuần tự 33 điểm `poseLandmarks` (`0..32`), 21 điểm `leftHandLandmarks` (`33..53`), và 21 điểm `rightHandLandmarks` (`54..74`).
  3. **Chuyển vị (Transpose) tại Backend Runtime:**
     - [`backend/vision/recognizer.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/vision/recognizer.py#L118-L120) nhận mảng `(30, 75, 3)` từ JSON payload và thực hiện `arr.transpose(2, 0, 1)[np.newaxis, ...]` để tạo tensor `(1, 3, 30, 75)` (`Batch, Channels=xyz, Frames=30, Nodes=75`), khớp 100% với hàm [`canonical_sequence()`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/training/sstcn_data.py#L67-L95) dùng khi huấn luyện.
- **Ảnh hưởng đến nhận diện:** Không có sai lệch về thứ tự chỉ số nút đồ thị hay thứ tự chiều tensor.
- **Cách khắc phục:** Không cần sửa.

---

### Hạng mục 5: Hệ tọa độ có khớp `mediapipe_image_normalized_xyz` không?

- **Trạng thái:** **`PASS` (trên engine `holistic`) / `FAIL` một phần (trên `hands` fallback do gán cứng `z = 0`)**.
- **Bằng chứng mã nguồn:**
  - **Trong Dataset:** [`backend/training/sstcn_data.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/training/sstcn_data.py#L20) khai báo `"coordinates": "mediapipe_image_normalized_xyz"` và [`prepare_sstcn.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/training/prepare_sstcn.py#L61-L65) lấy trực tiếp `[lm.x, lm.y, lm.z]` từ kết quả của MediaPipe Holistic (`pose_landmarks`, `left_hand_landmarks`, `right_hand_landmarks`).
  - **Trên Webcam (`holistic`):** [`CameraSignRecognizer.jsx`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L188-L210) lấy trực tiếp `[lm.x, lm.y, lm.z]` từ `results.poseLandmarks`, `results.leftHandLandmarks`, `results.rightHandLandmarks` (tọa độ `x, y` chuẩn hóa theo chiều rộng/cao khung hình trong khoảng `[0, 1]`, `z` là độ sâu tương đối).
  - **Lưu ý phụ về tỷ lệ khung hình (Aspect Ratio):** `prepare_sstcn.py` chạy trên khung hình gốc của video (thường là `16:9` hoặc `4:3`), còn webcam vẽ lên canvas `480×360` (`4:3`) từ luồng camera `640×480` (`4:3`, dòng `406–407`), nên hệ tọa độ chuẩn hóa `[0, 1]` hoàn toàn tương thích.
- **Ảnh hưởng đến nhận diện:** Tương thích hoàn toàn khi chạy `holistic`.
- **Cách khắc phục:** Chỉ cần đảm bảo luôn chạy `holistic` ở chế độ `words`.

---

### Hạng mục 6: Có xử lý đúng landmark thiếu bằng zero và có bước normalization nào khác với training không?

- **Trạng thái:** **`PASS` (trên engine `holistic`) / `FAIL` (trên engine `hands` fallback)**.
- **Bằng chứng mã nguồn:**
  1. **Kiểm tra bước Normalization phụ:**
     - Trong [`backend/training/sstcn_data.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/training/sstcn_data.py#L3-L4) và [`L20`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/training/sstcn_data.py#L20): `"normalization": "none"` (không dịch gốc về cổ tay/vai, không chia khoảng cách).
     - Trong [`backend/vision/recognizer.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/vision/recognizer.py#L100-L125) (`predict_sstcn`): không có bất kỳ bước chuẩn hóa tọa độ phụ nào trước khi đưa vào `self.sstcn_model(tensor)`. *(Lưu ý: hàm [`normalize_landmarks`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/vision/recognizer.py#L75-L98) chỉ được gọi trong `predict_mlp` cho `letters`/`digits`, hoàn toàn không ảnh hưởng tới `words`).*
  2. **Kiểm tra xử lý Landmark thiếu (Missing Landmarks):**
     - **Lúc huấn luyện:** [`prepare_sstcn.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/training/prepare_sstcn.py#L61-L65) trả về `np.zeros((count, 3), dtype=np.float32)` nếu một bộ phận (`pose`, `left_hand`, hoặc `right_hand`) không phát hiện được. Trong 150 file mẫu của dataset huấn luyện ([`preprocessing_comparison.md`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/docs/vsl/word_training_audit/preprocessing_comparison.md#L34-L38)), có **32.7%** số mẫu có các frame mà bàn tay trái là `[0, 0, 0]` và **14.0%** số mẫu có các frame mà bàn tay phải là `[0, 0, 0]`, trong khi `pose` thật vẫn hiện diện.
     - **Trên Webcam `holistic` (`PASS`):** [`extract75KeypointsFromHolistic`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L193-L212) điền `[0, 0, 0]` cho đúng bộ phận bị thiếu (`poseLandmarks`, `leftHandLandmarks`, `rightHandLandmarks`).
     - **Trên Webcam `hands` fallback (`FAIL`):** Trong [`extract75KeypointsFromHands`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L219-L267):
       - Khi không có Pose, thay vì điền `[0, 0, 0]`, hàm luôn điền `DEFAULT_POSE`.
       - Nghiêm trọng hơn: khi một bàn tay không xuất hiện trong khung hình (ví dụ `leftHand === null`), 21 điểm bàn tay trái (`33..53`) được điền `[0, 0, 0]` (dòng `260`), **nhưng** cổ tay trái pose (`keypoints[15]`) và khuỷu tay trái (`keypoints[13]`) vẫn giữ nguyên tọa độ hằng số của `DEFAULT_POSE` là `[0.68, 0.80, 0]` và `[0.72, 0.65, 0]` (dòng `225`)! Điều này tạo ra trạng thái mâu thuẫn giải phẫu chưa từng tồn tại trong tập huấn luyện.
- **Ảnh hưởng đến nhận diện:**
  - Trên `holistic`: khớp 100% với tập huấn luyện.
  - Trên `hands` fallback: tạo ra cạnh đồ thị nối từ nút `15` (`[0.68, 0.80, 0]`) tới nút `33` (`[0, 0, 0]`) với độ dài vector giả cực lớn ngay cả khi tay đó không hề giơ lên.
- **Cách khắc phục:**
  - Chặn hoàn toàn nhánh `extract75KeypointsFromHands` khi chạy `words`.
  - *(Tinh chỉnh nhỏ ở [`CameraSignRecognizer.jsx`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L190-L208)):* đổi `lm.x || 0` thành `Number.isFinite(lm.x) ? lm.x : 0` để rõ ràng về mặt kiểu dữ liệu.

---

### Hạng mục 7: Việc lấy 30 frame liên tục trên webcam có tương thích với phương pháp `linear_resample_to_30_after_frame_stride` của dataset không?

- **Trạng thái:** **`FAIL`** *(Đây là nguyên nhân kỹ thuật lớn nhất gây sụt giảm độ chính xác từ ~88% offline xuống ~6–15% trên webcam ngay cả khi đã bật `holistic`)*.
- **Bằng chứng mã nguồn & thực nghiệm:**
  1. **Cách tạo chuỗi 30 frame trong Dataset huấn luyện:**
     - Theo [`backend/training/prepare_sstcn.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/training/prepare_sstcn.py#L72-L90) và [`backend/training/sstcn_data.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/training/sstcn_data.py#L22,L89-L94) (`temporal_Rule: "linear_resample_to_30_after_frame_stride"`):
       - Mỗi mẫu huấn luyện là một **video clip độc lập đã được cắt gọn đầu–cuối (trimmed clip)** chứa trọn vẹn đúng 1 từ ký hiệu từ lúc bắt đầu đến lúc kết thúc.
       - Script đọc mỗi `frame_stride = 2` frame từ toàn bộ chiều dài video $N$ frame (thu được $M = \lceil N/2 \rceil$ frame), sau đó dùng **nội suy chỉ số tuyến tính trên toàn bộ chiều dài cử chỉ**:
         ```python
         positions = np.linspace(0, len(a) - 1, sequence_length) # sequence_length = 30
         indices = np.round(positions).astype(int)
         a = a[indices]
         ```
       - **Hệ quả bất biến thời gian (Temporal Invariance):** Trong mọi mẫu của tập train/val/test, **`frame[0]` luôn là thời điểm bắt đầu cử chỉ (0%), `frame[14..15]` luôn là đỉnh cử chỉ (50%), và `frame[29]` luôn là kết thúc cử chỉ (100%)**, bất kể video gốc dài 1 giây (30 frame) hay 3 giây (90 frame) và bất kể tốc độ FPS của camera quay video gốc.
  2. **Cách lấy 30 frame trên Webcam hiện tại:**
     - Tại [`CameraSignRecognizer.jsx`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L46-L47) và [`L829-L852`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L829-L852):
       - Hệ thống dùng bộ đệm FIFO thô `sequenceBufferRef.current.push(keypoints75)` lấy **30 frame liên tiếp theo thời gian thực** (`SEQ_LEN = 30`, `frame_stride = 1`), và trượt mỗi `FRAME_STEP = 6` frame (`sequenceBufferRef.current.splice(0, 6)`).
       - **Hoàn toàn không có bước phát hiện điểm bắt đầu/kết thúc cử chỉ (Gesture Boundary Detection / VAD cho chuyển động tay) và không có bước `np.linspace` co giãn chuỗi cử chỉ về 30 frame.**
  3. **Bằng chứng thực nghiệm về tốc độ khung hình (FPS) và độ dài cửa sổ thời gian:**
     - Theo thống kê trên 501 cửa sổ webcam tại [`pose_temporal_audit.md`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/docs/vsl/word_ab_fixed_input/pose_temporal_audit.md#L11-L15):
       - Khi chạy `holistic_raw` trên trình duyệt, do chi phí tính toán nặng của WebAssembly Holistic, tốc độ thực tế chỉ đạt trung bình **`13.6 FPS`** (trung vị `12.0 FPS`), khiến 30 frame liên tiếp kéo dài tới **`2.518 giây ± 0.296 giây`**!
       - Khi chạy `hands_fallback_pose`, tốc độ đạt **`26.1 FPS`**, nên 30 frame kéo dài **`1.199 giây`**.
     - Trong khi đó, một từ ký hiệu VSL thực tế do người dùng thực hiện trước webcam chỉ kéo dài khoảng **`1.0 – 1.5 giây`**.
- **Ảnh hưởng đến nhận diện:**
  - Khi cửa sổ `holistic_raw` dài **2.52 giây** nhưng cử chỉ thực tế chỉ dài **1.2 giây**: cử chỉ chỉ chiếm ~14–16 frame trong tổng số 30 frame, còn lại ~14–16 frame là giai đoạn nghỉ/chuẩn bị hạ tay hoặc giơ tay lên.
  - Đối với các lớp từ động của `SSTCN_Attention` (vốn sử dụng các nhánh tích chập thời gian `TemporalConv` với `dilation=1, 2, 4` tại [`sstcn_model.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/models/sstcn_model.py#L75-L94) và `TemporalAttention` trên 30 bước thời gian), việc cử chỉ bị lệch pha, co cụm ở một nửa cửa sổ hoặc bị cắt ngang giữa chừng làm biến dạng hoàn toàn mẫu vận tốc thời gian so với dữ liệu huấn luyện đã được chuẩn hóa bằng `linspace(0, T-1, 30)`.
- **Cách khắc phục:**
  1. **Giải pháp chuẩn xác nhất (Khuyến nghị):** Bổ sung cơ chế **Phân đoạn cử chỉ theo năng lượng chuyển động (Motion-Energy / Velocity Segmentation)** kết hợp **Linear Resampling về 30 frame**:
     - Theo dõi vận tốc dịch chuyển của 2 cổ tay và 2 bàn tay qua từng frame:
       - Khi vận tốc vượt ngưỡng `v_start` (bắt đầu ký hiệu): mở cờ `recording_gesture = true` và thu thập toàn bộ các frame của cử chỉ vào `gestureFrames` (cho phép thu từ `12` đến `90` frame, tương ứng cử chỉ dài `0.8s – 3.5s` ở mọi mức FPS từ 12–30 FPS).
       - Khi vận tốc giảm xuống dưới ngưỡng `v_rest` (dừng tay ở cuối ký hiệu trong ~4–6 frame liên tiếp) hoặc đạt giới hạn thời gian: đóng cử chỉ, dùng nội suy tuyến tính (`linspace(0, gestureFrames.length - 1, 30)`) để lấy mẫu đúng **30 frame trải đều từ đầu đến cuối cử chỉ**, rồi mới gửi 1 request duy nhất về backend.
  2. **Giải pháp hỗ trợ (Nếu vẫn giữ chế độ trượt liên tục):** Thu thập cửa sổ theo **thời gian thực cố định (~1.4 – 1.6 giây)** dựa trên timestamp `performance.now()` thay vì đếm cứng 30 frame ở 13.6 FPS, sau đó resample bằng `linspace` về đúng 30 frame trước khi gửi.

---

### Hạng mục 8: Camera có đang bị mirror gây đảo trái/phải hoặc sai thứ tự landmark không?

- **Trạng thái:**
  - **`PASS`** đối với luồng ảnh đưa vào `Holistic` trên webcam (chỉ mirror hiển thị CSS, không mirror dữ liệu tensor).
  - **`FAIL`** đối với nhánh `Hands` fallback (do `@mediapipe/hands` mặc định `selfieMode: false` đảo ngược nhãn `Left`/`Right` khi nhận ảnh không lật).
  - **`CHƯA ĐỦ BẰNG CHỨNG`** về việc các video MP4 gốc của Hải trước khi chạy `prepare_sstcn.py` có bị quay bằng camera trước (selfie-mirrored) hay không.
- **Bằng chứng mã nguồn:**
  1. **Hiển thị CSS vs Dữ liệu Canvas đưa vào MediaPipe:**
     - Trên giao diện [`CameraSignRecognizer.jsx`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L1259,L1273), thẻ `<video>` và `<canvas>` có thuộc tính CSS `transform: 'scaleX(-1)'` chỉ để người dùng nhìn thấy hình ảnh phản chiếu tự nhiên như soi gương.
     - Tuy nhiên, khi trích xuất frame đưa vào MediaPipe tại [`CameraSignRecognizer.jsx`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L366-L371):
       ```javascript
       hCtx.drawImage(videoRef.current, 0, 0, 480, 360);
       await currentTracker.send({ image: holisticCanvasRef.current });
       ```
       Hàm `drawImage` lấy trực tiếp pixel gốc từ `videoRef.current` (bỏ qua mọi thuộc tính CSS `transform`). Do đó, ảnh đưa vào MediaPipe là **ảnh gốc không bị lật (un-mirrored)**.
  2. **Hành vi của `Holistic` (`PASS`):**
     - Trong cả MediaPipe Python ([`prepare_sstcn.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/training/prepare_sstcn.py#L77-L84)) và `@mediapipe/holistic` JS ([`extract75KeypointsFromHolistic`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L185-L214)), `leftHandLandmarks` (`33..53`) và `rightHandLandmarks` (`54..74`) được xác định dựa trên giải phẫu cơ thể người trong ảnh (gắn với cổ tay trái `pose[15]` và cổ tay phải `pose[16]`). Vì cả `prepare_sstcn.py` (`cv2.VideoCapture` đọc frame không `cv2.flip`) và webcam `holisticCanvasRef` đều dùng ảnh không lật, thứ tự trái/phải và chiều trục $X$ của `Holistic` là nhất quán.
  3. **Hành vi của `Hands` fallback (`FAIL`):**
     - Trong [`createTracker('hands')`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L296-L303), `@mediapipe/hands` không được truyền tùy chọn `selfieMode: true` (mặc định của MediaPipe Hands JS là `selfieMode: false`, tức giả định ảnh đầu vào đã bị lật gương nên bộ phân loại handedness sẽ đảo ngược nhãn `'Left'` và `'Right'` trên ảnh camera gốc không lật!).
     - Tại [`extract75KeypointsFromHands`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L238-L243):
       ```javascript
       const label = results.multiHandedness?.[idx]?.label; // 'Left' or 'Right'
       if (label === 'Left') leftHand = landmarks;
       else rightHand = landmarks;
       ```
       Khi đưa ảnh không lật vào `@mediapipe/hands`, bàn tay phải thật của người dùng thường bị gán nhãn `'Left'` và bị nhét vào vị trí `33..53` (`leftHand`), trong khi bàn tay trái thật bị nhét vào `54..74` (`rightHand`), gây đảo ngược hoàn toàn 2 bàn tay so với `Holistic`!
  4. **Điểm `CHƯA ĐỦ BẰNG CHỨNG` về nguồn video gốc của Dataset:**
     - Kho dữ liệu `sstcn-dataset-v1` chỉ chứa các file `.npy` đã trích xuất, không đính kèm file video `.mp4` gốc. Do đó chưa thể khẳng định 100% liệu tất cả video gốc khi quay có dùng camera sau (không lật) hay có lẫn các clip quay bằng webcam/camera trước đã lật sẵn trong file MP4 hay không.
- **Ảnh hưởng đến nhận diện:**
  - Nếu chạy `hands` fallback: đảo lộn `left_hand` (`33..53`) và `right_hand` (`54..74`), làm hỏng hoàn toàn các ký hiệu 1 tay hoặc ký hiệu bất đối xứng 2 tay.
- **Cách khắc phục:**
  - Không dùng `hands` fallback cho chế độ `words`.
  - *(Nếu muốn kiểm tra thực nghiệm chiều trái/phải của dataset)*: có thể viết 1 script đọc thử vài file `.npy` của các từ ký hiệu 1 tay trong tập train để kiểm tra xem năng lượng chuyển động nằm ở `54..74` (`right_hand`) hay `33..53` (`left_hand`).

---

### Hạng mục 9: Mapping 63 lớp `words` có khớp với `vsl_102_dictionary.py` và mapping 102 lớp của checkpoint không?

- **Trạng thái:** **`PASS`**.
- **Bằng chứng mã nguồn & dữ liệu:**
  1. **Từ điển [`backend/models/vsl_102_dictionary.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/models/vsl_102_dictionary.py#L7-L112):**
     - Định nghĩa `SIGN_DICT_102` gồm đúng 102 chỉ số (`0..101`):
       - `0..28` (29 lớp): Chữ cái (`A` → `Y`).
       - `29..38` (10 lớp): Chữ số (`Số 1` → `Số 10`).
       - `39..101` (63 lớp): Từ vựng (`Thời gian` ở index `39` → `Bộ đội` ở index `101`).
  2. **Đối chiếu với `classes.json` và Checkpoint:**
     - Đã kiểm tra trong [`preprocessing_comparison.md`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/docs/vsl/word_training_audit/preprocessing_comparison.md#L56-L65): toàn bộ 102 nhãn trong `backend/data/sstcn/classes.json`, `ckpt["class_names"]` của `sign_sstcn_reconstructed_model.pth` / `sign_sstcn_reconstructed_last.pth` và `SIGN_DICT_102` khớp nhau **102/102 vị trí**.
  3. **Cơ chế Logit Masking tại [`backend/vision/recognizer.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/vision/recognizer.py#L126-L137):**
     - Khi `mode == "words"` (dòng `130–131`):
       ```python
       elif mode == "words":
           masked_logits[:, :39] = -float('inf')  # Keep 39-101 (Words)
       ```
     - Softmax sau đó chỉ tính xác suất trên đúng 63 lớp từ vựng (`39..101`), loại bỏ hoàn toàn khả năng dự đoán nhầm sang chữ cái hoặc chữ số.
- **Ảnh hưởng đến nhận diện:** Hoàn toàn chính xác, không có lỗi lệch chỉ số lớp (off-by-one hay class shift).
- **Cách khắc phục:** Không cần sửa.

---

### Hạng mục 10: Có nguy cơ cắt một cử chỉ nhiều động tác thành hai lần dự đoán riêng biệt không?

- **Trạng thái:** **`FAIL`** *(Nguy cơ rất cao, đã được chứng minh cả trong mã nguồn lẫn log thực nghiệm)*.
- **Bằng chứng mã nguồn:**
  Có **3 cơ chế nối tiếp nhau** trong pipeline hiện tại trực tiếp gây ra hiện tượng cắt một cử chỉ phức (nhiều nhịp/nhiều động tác như *"Tên là gì"*, *"Đi vệ sinh"*, *"Xin phép"*, *"Gia đình"*) thành 2 hoặc nhiều lần dự đoán rời rạc:
  1. **Cửa sổ trượt liên tục mỗi 6 frame (`FRAME_STEP = 6`) không chờ kết thúc cử chỉ:**
     - Tại [`CameraSignRecognizer.jsx`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L46-L47) và [`L829-L852`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L829-L852):
       ```javascript
       const SEQ_LEN = 30;
       const FRAME_STEP = 6;
       ...
       if (sequenceBufferRef.current.length >= SEQ_LEN) {
           const sequenceToSend = sequenceBufferRef.current.slice(0, SEQ_LEN);
           sequenceBufferRef.current.splice(0, FRAME_STEP);
           if (!isInferringRef.current) {
               sendSequenceToBackend(sequenceToSend, activeMode);
           }
       }
       ```
     - Ngay khi người dùng mới thực hiện được **nửa đầu** của một cử chỉ 2 nhịp (hợp với các frame chuẩn bị trước đó thành đủ 30 frame), hệ thống đã lập tức gửi cửa sổ thứ nhất đi dự đoán. Chỉ **6 frame sau (~230 ms – 440 ms)**, hệ thống lại gửi tiếp cửa sổ thứ hai chứa nửa sau của cử chỉ!
     - Trong thực nghiệm [`conclusion.md`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/docs/vsl/word_ab_fixed_input/conclusion.md#L33-L45), khi người dùng làm cử chỉ **"Tên là gì"** (cử chỉ 2 nhịp) hoặc **"Thế nào"**, các cửa sổ trượt ở giai đoạn đầu/giữa cử chỉ liên tục bị mô hình chốt nhầm thành từ đơn giản 1 nhịp là **"Bạn"** với độ tự tin rất cao ($\ge 0.65$).
  2. **Xóa sạch bộ đệm (`SEQUENCE_RESET`) chỉ sau 4 frame mất dấu bàn tay (~130 ms – 250 ms):**
     - Tại [`CameraSignRecognizer.jsx`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L804-L824):
       ```javascript
       missingHandsCountRef.current += 1;
       if (missingHandsCountRef.current >= 4 && sequenceBufferRef.current.length > 0) {
           sequenceBufferRef.current = [];
       ```
     - Trong các cử chỉ nhiều động tác có nhịp xoay cổ tay, vung tay nhanh (gây nhòe chuyển động - motion blur) hoặc hai bàn tay đan/che khuất nhau, MediaPipe rất dễ mất dấu bàn tay trong 4 frame liên tiếp. Việc xóa trắng `sequenceBufferRef.current = []` ngay lập tức **chặt đứt cử chỉ đang thực hiện làm đôi**: nửa đầu bị vứt bỏ, nửa sau phải tích lũy lại từ đầu.
     - Trong khi đó, bản thân dữ liệu huấn luyện (Hạng mục 6) vốn chứa sẵn các frame `[0, 0, 0]` khi tay bị khuất tạm thời và mô hình `SSTCN_Attention` đã học cách xử lý các frame trống ngắn hạn này.
  3. **Logic Cooldown ở Backend chỉ chặn 2 lần dự đoán trùng tên từ, KHÔNG chặn 2 từ khác nhau liên tiếp:**
     - Tại [`backend/vision/recognizer.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/vision/recognizer.py#L47,L161-L169):
       ```python
       self.cooldown_sec = 1.2
       ...
       if max_prob >= threshold:
           if pred_term != self.last_confirmed_sign or (current_time - self.last_confirmed_time > self.cooldown_sec):
               self.sentence_tokens.append(pred_term)
               self.last_confirmed_sign = pred_term
               self.last_confirmed_time = current_time
       ```
     - Vì điều kiện là `pred_term != self.last_confirmed_sign OR (...)`, nên nếu ở nhịp 1 của cử chỉ (tại $t = 0.0\text{s}$), cửa sổ trượt dự đoán ra **Từ A** (ví dụ `"Bạn"`, xác suất $\ge 0.65$), và ở nhịp 2 của chính cử chỉ đó (tại $t = 0.35\text{s}$), cửa sổ trượt dự đoán ra **Từ B** (ví dụ `"Tên là gì"`, xác suất $\ge 0.65$), biểu thức `pred_term != self.last_confirmed_sign` trả về **`True`**!
     - Hệ quả: **Cả 2 từ `"Bạn"` và `"Tên là gì"` đều được xác nhận ngay lập tức cách nhau chỉ 0.35 giây** và nối thẳng vào `sentence_tokens` từ một cử chỉ duy nhất của người dùng!
- **Ảnh hưởng đến nhận diện:**
  - Gây hiện tượng "nhảy từ kép" (1 cử chỉ sinh ra 2–3 từ liên tiếp trong câu) và làm các từ có cấu trúc phức tạp bị từ đơn giản cướp quyền ưu tiên (preempt) ngay từ nửa đầu động tác.
- **Cách khắc phục:**
  1. **Tại Backend ([`recognizer.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/vision/recognizer.py#L161-L169)):**
     - Áp dụng **Global Post-Confirmation Cooldown** (ví dụ `0.8s – 1.0s` sau khi vừa chốt bất kỳ một từ nào, hệ thống tạm khóa không chốt thêm từ mới để người dùng kết thúc cử chỉ và chuyển sang cử chỉ tiếp theo) **VÀ** yêu cầu **Temporal Voting / Stability** (một từ phải xuất hiện ở Top-1 trong ít nhất 2 cửa sổ liên tiếp hoặc sau khi cử chỉ đã đạt điểm dừng chuyển động).
  2. **Tại Frontend ([`CameraSignRecognizer.jsx`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L804-L824)):**
     - Tăng ngưỡng `missingHandsCountRef.current` từ `4` lên `10–12` frame (~0.5–0.8 giây nghỉ tay thực sự mới xóa buffer). Trong khoảng `< 10` frame mất dấu tay tạm thời giữa cử chỉ, vẫn giữ luồng frame (với bàn tay bằng `[0, 0, 0]` và pose thật từ Holistic) giống như trong tập huấn luyện.
     - Sau khi một từ được backend xác nhận (`confirmed: true`), chủ động xóa `sequenceBufferRef.current = []` để cửa sổ tiếp theo không dùng lại phần đuôi của cử chỉ cũ.

---

## 3. Phân Loại Nguyên Nhân Gốc Rễ (Root-Cause Taxonomy)

Để tránh sửa sai hướng hoặc train lại mô hình không cần thiết, bảng dưới đây tách bạch rõ 4 nhóm nguyên nhân gây ra chênh lệch giữa **Accuracy Offline (~86.6% – 88.7%)** và **Accuracy Webcam (~6% – 15%)**:

| Nhóm nguyên nhân | Tỷ trọng tác động ước tính | Các lỗi cụ thể đã xác minh | Bằng chứng thực nghiệm |
| :--- | :---: | :--- | :--- |
| **1. Lỗi Phân đoạn cử chỉ & Thời gian (Gesture Segmentation & Temporal Domain Shift)** | **~45%** | - Webcam lấy 30 frame FIFO liên tiếp (`2.52s` @ `13.6 FPS` trên Holistic) thay vì cắt trọn vẹn 1 cử chỉ rồi co giãn tuyến tính `np.linspace(0, T-1, 30)` như tập train.<br>- Cửa sổ trượt `FRAME_STEP=6` gửi dự đoán ngay khi cử chỉ mới đi được một nửa.<br>- Xóa trắng buffer khi mất tay chỉ 4 frame (`~150ms`) giữa cử chỉ.<br>- Cooldown tại [`recognizer.py:163`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/vision/recognizer.py#L163) dùng toán tử `OR` (`pred_term != last_confirmed_sign`), cho phép 1 cử chỉ nhiều nhịp sinh ra 2 từ khác nhau cách nhau `300ms`. | Ngay cả khi dùng checkpoint tốt nhất (`Historical`, epoch 32, val acc `91.02%`) và chạy đúng `holistic_raw`, accuracy trên webcam vẫn chỉ đạt **25.7%** (44/171 cửa sổ) do các cửa sổ trượt rơi vào vùng chuyển tiếp/giữa cử chỉ ([`conclusion.md`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/docs/vsl/word_ab_fixed_input/conclusion.md#L12-L20)). |
| **2. Lỗi Landmark / Tracker Engine (MediaPipe Config & Fallback)** | **~30%** | - Cho phép chế độ `words` chạy trên engine `hands` (`extract75KeypointsFromHands`), đóng băng cứng 29/33 khớp pose (`DEFAULT_POSE`), bịa khuỷu tay với `z=0`.<br>- `@mediapipe/hands` mặc định `selfieMode: false` trên ảnh canvas không lật làm đảo ngược nhãn bàn tay trái/phải.<br>- `Holistic` trên web đặt `modelComplexity: 0, smoothLandmarks: false` trên canvas `480×360` với ngưỡng `0.35/0.30` (khác với `model_complexity=1, confidence=0.5` lúc tạo dataset). | Trên cùng tập 501 cửa sổ replay ([`conclusion.md`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/docs/vsl/word_ab_fixed_input/conclusion.md#L22-L31)), khi chuyển từ `holistic_raw` sang `hands_fallback_pose`, Top-1 của `Historical` sụt từ **15.0%** xuống **3.0%** (giảm 5 lần), và từ *"Tên là gì"* sụt xuống **0/95 (0%)** ở cả Top-1 lẫn Top-5. |
| **3. Lỗi Tiền xử lý (Preprocessing Mismatch)** | **~5%** | - Trên `holistic`: **Không có lỗi tiền xử lý** (thứ tự `33+21+21`, hệ tọa độ `xyz`, điền `0` khi thiếu khớp và transpose `(1, 3, 30, 75)` đều khớp 100%).<br>- Trên `hands` fallback: Khi mất 1 bàn tay, `33..53` được điền `[0,0,0]` nhưng cổ tay/khuỷu tay pose (`13, 15`) vẫn giữ hằng số `DEFAULT_POSE`, tạo vector cạnh đồ thị dị thường. | Đã kiểm chứng đối chiếu từng dòng giữa [`prepare_sstcn.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/training/prepare_sstcn.py), [`sstcn_data.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/training/sstcn_data.py) và [`CameraSignRecognizer.jsx`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx). |
| **4. Lỗi do Model chưa train đủ (Model Under-training / Early Stop)** | **~20%** | - Checkpoint `reconstructed_best` hiện tại mới dừng ở **Epoch 10/11** (`val_acc = 85.79%`, `learning_rate = 0.0005` chưa giảm hết lịch Cosine Annealing), trong khi `historical` được train tới **Epoch 32** (`val_acc = 91.02%`).<br>- Tuy nhiên, trên tập Test offline (`4.185` mẫu), `reconstructed_best` đã đạt **88.66% Top-1 (97.30% Top-5)** ([`resume_readiness.md`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/docs/vsl/word_training_audit/resume_readiness.md#L56-L65)). | Khi đối chứng cố định đầu vào trên đúng 233 cửa sổ `holistic_raw`, `Historical` (epoch 32) đạt **15.02% Top-1 / 38.63% Top-5**, cao hơn `Reconstructed Best` (epoch 10) đạt **8.15% Top-1 / 25.32% Top-5**. Như vậy resume training từ epoch 11 lên 30–40 trên Kaggle (gói VSL-P014) sẽ giúp cải thiện độ bền vững của model, **nhưng KHÔNG THỂ thay thế việc sửa lỗi Số 1 và Số 2 ở pipeline webcam**. |

---

## 4. Lộ Trình Khắc Phục Đề Xuất (Thứ Tự Ưu Tiên Kỹ Thuật)

Để khắc phục triệt để vấn đề nhận diện webcam chế độ `words` mà không ảnh hưởng tới `letters`, `digits`, `sentence`:

1. **Bước 1 — Khóa chặt `Holistic` cho chế độ `words` & Đồng bộ cấu hình Tracker (Sửa lỗi Landmark):**
   - Trong [`CameraSignRecognizer.jsx`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx): khi `mode === 'words'`, bắt buộc `recognitionScope = 'holistic'` (vô hiệu hóa lựa chọn `'hands'` trên UI cho tab `words`, và chặn gửi frame nếu `trackerEngineRef.current !== 'holistic'`).
   - Cập nhật [`createTracker('holistic')`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L287-L293): bật `smoothLandmarks: true`, nâng `minDetectionConfidence: 0.5`, `minTrackingConfidence: 0.5`.
2. **Bước 2 — Chuẩn hóa Phân đoạn Cử chỉ & Lấy mẫu `linspace(0, T-1, 30)` (Sửa lỗi Temporal / Segmentation):**
   - Tăng ngưỡng xóa buffer khi mất tay tại [`CameraSignRecognizer.jsx:806`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L806) từ `4` frame lên `10–12` frame để không cắt đứt cử chỉ khi tay xoay/khuất ngắn hạn.
   - Thay vì gửi trực tiếp 30 frame FIFO ở 13.6 FPS (kéo dài 2.52s), thu thập chuỗi frame theo biên độ chuyển động của một cử chỉ (hoặc cửa sổ thời gian thực ~1.4s–1.6s khi phát hiện chuyển động rồi dừng tay) và nội suy tuyến tính (`linspace`) về đúng `30` frame trước khi gửi về `/api/recognize`.
3. **Bước 3 — Sửa Cooldown & Xác nhận Cử chỉ tại [`backend/vision/recognizer.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/vision/recognizer.py#L161-L169) (Chống cắt 1 cử chỉ thành 2 từ):**
   - Áp dụng khoảng nghỉ toàn cục (global cooldown ~`0.8s – 1.0s`) sau mỗi lần `confirmed = True` cho bất kỳ từ nào, kết hợp xóa `sequenceBufferRef` ở frontend ngay khi nhận được `confirmed = true`.
4. **Bước 4 — Resume Training `SSTCN_Attention` trên Kaggle (Gói `vsl_kaggle_resume_bundle.zip`):**
   - Tiếp tục train từ `reconstructed_last.pth` (epoch 11) lên epoch 40–60 trên bộ dữ liệu 41.645 mẫu của Hải để đưa độ hội tụ của checkpoint đạt và vượt mức của checkpoint `Historical` (epoch 32).

