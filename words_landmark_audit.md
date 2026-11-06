# BÁO CÁO CODE AUDIT: PIPELINE NHẬN DIỆN NGÔN NGỮ KÝ HIỆU (MODE `WORDS`)

> **Mục tiêu Audit:** Xác định nguyên nhân dữ liệu đầu vào mô hình `SSTCN_Attention` có shape `(30, 75, 3)` nhưng 29/33 pose landmarks có tọa độ cố định qua mọi khung hình, đặc biệt là hai vai tại `(0.65, 0.45)` và `(0.35, 0.45)`.  
> **Nguyên tắc thực hiện:** Chỉ đọc và phân tích mã nguồn (Read-only audit), tuyệt đối không chỉnh sửa code hiện hành, không can thiệp các chế độ `letters`, `digits`, `sentence`.

---

## TỔNG QUAN KẾT LUẬN (EXECUTIVE SUMMARY)

1. **Nguyên nhân cốt lõi:** Khi người dùng chuyển sang chế độ Từ vựng (`mode === 'words'`), hệ thống Frontend **vẫn duy trì chạy MediaPipe Hands** (mặc định của `recognitionScope = 'hands'`) thay vì `MediaPipe Holistic`.
2. Do MediaPipe Hands **chỉ phát hiện 21 khớp bàn tay và hoàn toàn không phát hiện cơ thể**, Frontend đã sử dụng một mảng hằng số giả lập `DEFAULT_POSE` gồm 33 điểm để "bù" vào cho đủ 75 khớp phục vụ mô hình SSTCN.
3. Trong 33 điểm mốc cơ thể (Pose):
   - **29/33 điểm bị HARDCODE CỐ ĐỊNH 100%:** Bao gồm 11 điểm đầu/mặt (0..10), 2 vai (11, 12) và 16 điểm thân dưới/chân (17..32). Tọa độ hai vai cố định vĩnh viễn tại `(0.65, 0.45, 0)` và `(0.35, 0.45, 0)`.
   - **Chỉ có 4/33 điểm biến thiên:** Cổ tay trái/phải (15, 16) lấy trực tiếp từ điểm mốc gốc bàn tay; Khuỷu tay trái/phải (13, 14) được **tính bằng công thức toán học nội suy trung điểm giữa vai giả lập và cổ tay**, hoàn toàn không đo từ camera.
4. **Dữ liệu huấn luyện gốc của SSTCN** được trích xuất từ video người thật bằng **MediaPipe Holistic** (33 khớp cơ thể thật cử động tự nhiên). Sự sai lệch phân phối (distribution mismatch) giữa cơ thể thật và khung xương giả lập này ảnh hưởng trực tiếp đến độ tin cậy nhận diện của mô hình trong môi trường thực tế.

---

## 1. TOÀN BỘ CODE TẠO, XỬ LÝ VÀ LƯU 75 LANDMARKS

Quy trình hình thành và luân chuyển của tensor `(30, 75, 3)` trải qua 4 giai đoạn:

```
[Webcam Frame]
      │
      ▼
MediaPipe Hands (hoặc Holistic)
      │
      ▼
extract75KeypointsFromHands() ───► Ghép 33 Fake Pose + 21 Left + 21 Right = 75 khớp
      │
      ▼
sequenceBufferRef.current ───────► Tích lũy 30 frame (Sliding Window bước nhảy 6)
      │
      ▼
sendSequenceForRecognition() ────► POST /api/recognize_sequence
      │
      ▼
backend.vision.recognizer ───────► SSTCN_Attention (Inference)
      │
      ▼
word_debugger.record_prediction ─► Lưu sequence_*.npy & events.jsonl
```

### 1.1. Frontend: Tạo và đóng gói 75 landmarks
- **File:** `src/components/CameraSignRecognizer.jsx`
- **Hàm:** `extract75KeypointsFromHands` (Dòng 202–254) và `extract75KeypointsFromHolistic` (Dòng 170–199).
- **Trích đoạn mã nguồn (`extract75KeypointsFromHands`):**
```javascript
// Dòng 202 - 254: src/components/CameraSignRecognizer.jsx
  const extract75KeypointsFromHands = (results) => {
    const keypoints = [];
    const DEFAULT_POSE = [
      [0.5, 0.2, 0],   // 0: Mũi
      [0.52, 0.18, 0], [0.54, 0.18, 0], [0.56, 0.18, 0],
      [0.48, 0.18, 0], [0.46, 0.18, 0], [0.44, 0.18, 0],
      [0.58, 0.22, 0], [0.42, 0.22, 0],
      [0.53, 0.26, 0], [0.47, 0.26, 0],
      [0.65, 0.45, 0], [0.35, 0.45, 0], // 11: Vai trái, 12: Vai phải
      [0.72, 0.65, 0], [0.28, 0.65, 0], // 13: Khuỷu trái, 14: Khuỷu phải
      [0.68, 0.80, 0], [0.32, 0.80, 0], // 15: Cổ tay trái, 16: Cổ tay phải
    ];

    for (let i = 0; i < 33; i++) {
      if (i < DEFAULT_POSE.length) keypoints.push(DEFAULT_POSE[i]);
      else keypoints.push([0.5, 0.8, 0]);
    }

    let leftHand = null;
    let rightHand = null;

    if (results.multiHandLandmarks) {
      results.multiHandLandmarks.forEach((lms, idx) => {
        const label = results.multiHandedness?.[idx]?.label || (idx === 0 ? 'Right' : 'Left');
        if (label === 'Left') leftHand = lms;
        else rightHand = lms;
      });
    }

    // Cập nhật tọa độ cổ tay và khuỷu tay tương quan với vị trí bàn tay phát hiện
    if (leftHand && leftHand[0]) {
      keypoints[15] = [leftHand[0].x || 0.68, leftHand[0].y || 0.80, leftHand[0].z || 0];
      keypoints[13] = [(0.65 + (leftHand[0].x || 0.68)) / 2 + 0.04, (0.45 + (leftHand[0].y || 0.80)) / 2, 0];
    }
    if (rightHand && rightHand[0]) {
      keypoints[16] = [rightHand[0].x || 0.32, rightHand[0].y || 0.80, rightHand[0].z || 0];
      keypoints[14] = [(0.35 + (rightHand[0].x || 0.32)) / 2 - 0.04, (0.45 + (rightHand[0].y || 0.80)) / 2, 0];
    }

    if (leftHand && leftHand.length >= 21) {
      for (let i = 0; i < 21; i++) keypoints.push([leftHand[i].x || 0, leftHand[i].y || 0, leftHand[i].z || 0]);
    } else {
      for (let i = 0; i < 21; i++) keypoints.push([0, 0, 0]);
    }

    if (rightHand && rightHand.length >= 21) {
      for (let i = 0; i < 21; i++) keypoints.push([rightHand[i].x || 0, rightHand[i].y || 0, rightHand[i].z || 0]);
    } else {
      for (let i = 0; i < 21; i++) keypoints.push([0, 0, 0]);
    }

    return keypoints;
  };
```

### 1.2. Frontend: Buffer và truyền dữ liệu
- **File:** `src/components/CameraSignRecognizer.jsx`
- **Hàm:** `handleMotionAndSequence` (Dòng 734–754)
- **Trích đoạn mã nguồn:**
```javascript
// Dòng 734 - 754: src/components/CameraSignRecognizer.jsx
      // Đẩy frame vào bộ đệm
      const frameKeypoints = getKeypointsFn();
      sequenceBufferRef.current.push(frameKeypoints);

      const currentLen = sequenceBufferRef.current.length;
      const progressPercent = Math.min(100, Math.round((currentLen / SEQ_LEN) * 100));
      updateHUDProgress(progressPercent, currentLen);

      // Khi đã đủ 30 frame và không có suy luận nào đang chạy
      if (currentLen >= SEQ_LEN && !isInferringRef.current) {
        isInferringRef.current = true;
        const currentGen = requestGenRef.current;
        const requestMode = currentConversionMode;
        const sequencePayload = sequenceBufferRef.current.slice(-SEQ_LEN);

        // SLIDING WINDOW: Loại bỏ 6 frame cũ nhất, giữ lại 24 frame để frame tiếp theo nhanh chóng đầy lại!
        sequenceBufferRef.current.splice(0, FRAME_STEP);

        // Gửi suy luận hoàn toàn không chặn luồng (Non-blocking Async)
        sendSequenceForRecognition(sequencePayload, confidenceThreshold, requestMode)
```

- **File:** `src/services/api.js` (Dòng 121–137):
```javascript
// Dòng 121 - 128: src/services/api.js
export async function sendSequenceForRecognition(sequence, threshold = 0.65, mode = 'words') {
  try {
    const res = await fetch(`${API_BASE_URL}/api/recognize_sequence`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sequence, threshold, mode })
    });
```

### 1.3. Backend: Nhận và đưa vào mô hình
- **File:** `backend/app.py`
```python
# Dòng 63 - 67: backend/app.py
class SequenceRequest(BaseModel):
    sequence: List[List[List[float]]]  # Chuỗi 30 frames x 75 joints x 3 coords
    threshold: Optional[float] = 0.65
    mode: Optional[str] = None

# Dòng 261 - 275: backend/app.py
@app.post("/api/recognize_sequence")
def recognize_sequence(req: SequenceRequest, request: Request):
    try:
        result = recognizer.predict_sequence(
            sequence_data=req.sequence,
            threshold=req.threshold,
            mode=req.mode or request.headers.get("x-conversion-mode")
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

- **File:** `backend/vision/recognizer.py`
```python
# Dòng 87 - 108: backend/vision/recognizer.py
            arr = np.array(sequence_data, dtype=np.float32)
            
            # Xử lý các định dạng input linh hoạt
            if arr.shape == (30, 75, 3):
                pass
            # ...
            # Chuyển đổi sang định dạng đầu vào của SSTCN: (Batch=1, Channels=3, Time=30, Joints=75)
            # arr: (30, 75, 3) -> transpose(2, 0, 1) -> (3, 30, 75)
            tensor_seq = arr.transpose(2, 0, 1)
            tensor_seq = tensor_seq[np.newaxis, ...]  # (1, 3, 30, 75)
            tensor_input = torch.tensor(tensor_seq, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                logits = self.sstcn_model(tensor_input)
```

### 1.4. Backend: Ghi file debug xuống đĩa
- **File:** `backend/vision/recognizer.py`
```python
# Dòng 174 - 176: backend/vision/recognizer.py
            from backend.vision.word_debugger import word_debugger
            if mode == "words":
                word_debugger.record_prediction(arr, top_k_list, max_prob, sign_info["term"], threshold=threshold)
```

- **File:** `backend/vision/word_debugger.py`
```python
# Dòng 223 - 227: backend/vision/word_debugger.py
                self.seq_counter += 1
                seq_idx = self.seq_counter
                filename = f"sequence_{seq_idx:03d}.npy"
                np.save(os.path.join(self.sequences_dir, filename), arr_input.astype(np.float32))
```

---

## 2. HỆ THỐNG ĐANG DÙNG MEDIAPIPE HANDS, POSE HAY HOLISTIC?

### 2.1. Kết luận và Bằng chứng chắc chắn (Fact)
Hệ thống **đang sử dụng MediaPipe Hands (`window.Hands`) làm động cơ mặc định** khi chạy mode `words`, **hoàn toàn không dùng Pose độc lập** và **chỉ dùng Holistic khi người dùng chủ động bấm đổi giao diện**.

Chi tiết bằng chứng:
1. **Khởi tạo state mặc định:**
   - **File:** `src/components/CameraSignRecognizer.jsx`
   ```javascript
   // Dòng 90 - 98: src/components/CameraSignRecognizer.jsx
   // TIÊU CHÍ A: Phạm vi nhận diện: 'hands' (Bàn tay - 60 FPS) | 'holistic' (Toàn thân - 75 Khớp)
   const [recognitionScope, setRecognitionScope] = useState('hands');

   // TIÊU CHÍ B: Chế độ chuyển đổi: 'letters_numbers' (Chữ cái & Số) | 'words' (Từ vựng) | 'sentence' (Ghép câu ngữ cảnh)
   const [conversionMode, setConversionMode] = useState('letters');

   // Trạng thái kỹ thuật đồng bộ với backend & pipeline
   const [recognitionMode, setRecognitionMode] = useState('instant'); // 'instant' | 'sequence'
   const [trackerEngine, setTrackerEngine] = useState('hands'); // 'hands' | 'holistic'
   ```

2. **Hàm khởi động Camera:**
   - **File:** `src/components/CameraSignRecognizer.jsx`
   ```javascript
   // Dòng 257 - 275: src/components/CameraSignRecognizer.jsx
   const startCamera = async (engine = trackerEngine) => {
     // ...
     if (engine === 'hands' && window.Hands) {
       tracker = new window.Hands({
         locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
       });
       tracker.setOptions({
         maxNumHands: 2,
         modelComplexity: 0, // Lite mode = 60 FPS
         minDetectionConfidence: 0.5,
         minTrackingConfidence: 0.4
       });
       tracker.onResults(onHandsResultsFast);
     } else if (window.Holistic) {
       tracker = new window.Holistic({
         locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
       });
       // ...
       tracker.onResults(onHolisticResultsFast);
     }
   ```

3. **Khi chuyển mode sang `words`:**
   - **File:** `src/components/CameraSignRecognizer.jsx`
   ```javascript
   // Dòng 405 - 418: src/components/CameraSignRecognizer.jsx
   const handleModeChange = (newMode) => {
     if (newMode === conversionMode) return;
     const newRecMode = (newMode === 'letters' || newMode === 'digits' || newMode === 'letters_numbers') ? 'instant' : 'sequence';

     conversionModeRef.current = newMode;
     recognitionModeRef.current = newRecMode;
     requestGenRef.current += 1;
     isInferringRef.current = false;

     setConversionMode(newMode);
     setRecognitionMode(newRecMode);
     sequenceBufferRef.current = [];
     updateHUDProgress(0, 0);
   ```
   **Nhận xét:** Trong `handleModeChange`, biến `trackerEngine` và `recognitionScope` **hoàn toàn không bị thay đổi**. Giá trị vẫn giữ nguyên là `'hands'`. Vì vậy, luồng frame camera vẫn chạy qua `tracker.onResults(onHandsResultsFast)` và gọi `extract75KeypointsFromHands`.

4. **UI Banner xác nhận thiết kế này:**
   - **File:** `src/components/CameraSignRecognizer.jsx`
   ```javascript
   // Dòng 490 - 497: src/components/CameraSignRecognizer.jsx
       } else {
         return {
           type: 'supported',
           badge: '⚡ Hỗ trợ tốt',
           badgeClass: 'badge-indigo',
           title: 'Bàn tay + 102 Từ vựng VSL',
           desc: 'Nhận diện 102 từ động qua 2 bàn tay (Khớp Pose thân trên được ước lượng tự động).'
         };
       }
   ```

---

## 3. TỌA ĐỘ HAI VAI: HARDCODE, GIẢ LẬP HAY FALLBACK?

### 3.1. Kết luận và Bằng chứng chắc chắn (Fact)
Tọa độ hai vai **ĐƯỢC HARDCODE TĨNH 100% (CỐ ĐỊNH HOÀN TOÀN)**, **KHÔNG PHẢI FALLBACK KHI MẤT DẤU**. Kể cả khi người dùng đứng trước camera với tư thế chuẩn và hai tay nhận diện hoàn hảo, tọa độ hai vai gửi vào SSTCN vẫn luôn luôn là:
- **Vai trái (Joint 11):** `(x = 0.65, y = 0.45, z = 0.0)`
- **Vai phải (Joint 12):** `(x = 0.35, y = 0.45, z = 0.0)`

### 3.2. Đoạn code liên quan
- **File:** `src/components/CameraSignRecognizer.jsx`
```javascript
// Dòng 204 - 218: src/components/CameraSignRecognizer.jsx
    const DEFAULT_POSE = [
      [0.5, 0.2, 0],   // 0: Mũi
      [0.52, 0.18, 0], [0.54, 0.18, 0], [0.56, 0.18, 0],
      [0.48, 0.18, 0], [0.46, 0.18, 0], [0.44, 0.18, 0],
      [0.58, 0.22, 0], [0.42, 0.22, 0],
      [0.53, 0.26, 0], [0.47, 0.26, 0],
      [0.65, 0.45, 0], [0.35, 0.45, 0], // 11: Vai trái, 12: Vai phải
      [0.72, 0.65, 0], [0.28, 0.65, 0], // 13: Khuỷu trái, 14: Khuỷu phải
      [0.68, 0.80, 0], [0.32, 0.80, 0], // 15: Cổ tay trái, 16: Cổ tay phải
    ];

    for (let i = 0; i < 33; i++) {
      if (i < DEFAULT_POSE.length) keypoints.push(DEFAULT_POSE[i]);
      else keypoints.push([0.5, 0.8, 0]);
    }
```
Tại các dòng tiếp theo (Dòng 231–239), mảng `keypoints` chỉ cập nhật lại các chỉ số 13, 14, 15, 16. **Không có bất kỳ dòng code nào gán lại `keypoints[11]` hoặc `keypoints[12]`**.

### 3.3. Dữ liệu thực chứng từ file `.npy` đã lưu trong `debug/words/`
Kiểm tra thực tế trên 1.051 file sequence được ghi nhận tại `debug/words/`:
```text
Shape: (30, 75, 3)
Pose idx 11 (Vai T): [[0.65, 0.45, 0.], [0.65, 0.45, 0.], ... (30 frames đồng nhất)]
Pose idx 12 (Vai P): [[0.35, 0.45, 0.], [0.35, 0.45, 0.], ... (30 frames đồng nhất)]
```
Trong số 33 điểm mốc Pose:
- Điểm 0..12: Đúng 13 điểm lấy từ `DEFAULT_POSE` và **bất biến qua 100% số frame**.
- Điểm 17..32: Đúng 16 điểm lấy từ fallback `[0.5, 0.8, 0]` và **bất biến qua 100% số frame**.
- Tổng số điểm mốc cơ thể bị đóng băng cố định: **13 + 16 = 29 điểm**.
- Chỉ duy nhất 4 điểm thay đổi: 13, 14, 15, 16.

---

## 4. TỌA ĐỘ KHUỶU TAY VÀ CỔ TAY: ĐO TRỰC TIẾP HAY SUY RA TỪ CÔNG THỨC?

### 4.1. Bằng chứng chắc chắn (Fact)
1. **Cổ tay (Joints 15, 16):**
   - **Đo trực tiếp từ gốc bàn tay:** Lấy tọa độ landmark thứ 0 (`leftHand[0]` và `rightHand[0]`) từ MediaPipe Hands.
   - Nếu frame đó không phát hiện được tay tương ứng, giá trị fallback tĩnh là `[0.68, 0.80, 0]` (tay trái) và `[0.32, 0.80, 0]` (tay phải).
2. **Khuỷu tay (Joints 13, 14):**
   - **HOÀN TOÀN KHÔNG ĐƯỢC ĐO TRỰC TIẾP TỪ CAMERA.**
   - Được **suy ra bằng công thức toán học nội suy trung điểm (midpoint)** giữa vai giả lập hardcode và cổ tay đo được, cộng thêm hệ số lệch (offset):

$$\begin{cases}
X_{\text{elbow\_L}} = \dfrac{0.65 + X_{\text{wrist\_L}}}{2} + 0.04 \\[8pt]
Y_{\text{elbow\_L}} = \dfrac{0.45 + Y_{\text{wrist\_L}}}{2} \\[8pt]
Z_{\text{elbow\_L}} = 0.0
\end{cases}
\qquad
\begin{cases}
X_{\text{elbow\_R}} = \dfrac{0.35 + X_{\text{wrist\_R}}}{2} - 0.04 \\[8pt]
Y_{\text{elbow\_R}} = \dfrac{0.45 + Y_{\text{wrist\_R}}}{2} \\[8pt]
Z_{\text{elbow\_R}} = 0.0
\end{cases}$$

### 4.2. Đoạn code liên quan
- **File:** `src/components/CameraSignRecognizer.jsx`
```javascript
// Dòng 231 - 240: src/components/CameraSignRecognizer.jsx
    // Cập nhật tọa độ cổ tay và khuỷu tay tương quan với vị trí bàn tay phát hiện
    if (leftHand && leftHand[0]) {
      keypoints[15] = [leftHand[0].x || 0.68, leftHand[0].y || 0.80, leftHand[0].z || 0];
      keypoints[13] = [(0.65 + (leftHand[0].x || 0.68)) / 2 + 0.04, (0.45 + (leftHand[0].y || 0.80)) / 2, 0];
    }
    if (rightHand && rightHand[0]) {
      keypoints[16] = [rightHand[0].x || 0.32, rightHand[0].y || 0.80, rightHand[0].z || 0];
      keypoints[14] = [(0.35 + (rightHand[0].x || 0.32)) / 2 - 0.04, (0.45 + (rightHand[0].y || 0.80)) / 2, 0];
    }
```
Hằng số `0.65` và `0.35` trong công thức trên chính là tọa độ X của vai trái và vai phải trong `DEFAULT_POSE`; hằng số `0.45` chính là tọa độ Y của hai vai. Tọa độ Z của khuỷu tay bị ép về 0.

---

## 5. ĐIỀU KIỆN `no_hands_detected`, `SEQUENCE_RESET` VÀ CƠ CHẾ THU THẬP 30 FRAME

### 5.1. Bằng chứng chắc chắn (Fact)
1. **Điều kiện kích hoạt `no_hands_detected`:**
   Xảy ra khi cả hai bàn tay đều không được MediaPipe Hands tìm thấy trong khung hình (`!hasHands`, tức `!leftHand && !rightHand`).
2. **Hành vi xử lý:**
   - Ngay lập tức xóa trắng bộ đệm: `sequenceBufferRef.current = []`.
   - Reset tiến trình trên giao diện về 0: `updateHUDProgress(0, 0)`.
   - Nếu đang ở `mode === 'words'`, bắn event `SEQUENCE_RESET` với lý do `'no_hands_detected'` lên backend để ghi nhật ký debug.

- **File:** `src/components/CameraSignRecognizer.jsx`
```javascript
// Dòng 722 - 732: src/components/CameraSignRecognizer.jsx
    // 2. CHẾ ĐỘ 1: CHUỖI ĐỘNG VSL (SSTCN-Attention 30-Frame Sliding Window)
    if (currentRecognitionMode === 'sequence') {
      if (!hasHands) {
        if (sequenceBufferRef.current.length > 0) {
          if (currentConversionMode === 'words') {
            logWordDebugReset(sequenceBufferRef.current.length, 'no_hands_detected');
          }
          sequenceBufferRef.current = [];
          updateHUDProgress(0, 0);
        }
        return;
      }
```

- **File:** `src/services/wordDebugApi.js`
```javascript
// Dòng 70 - 81: src/services/wordDebugApi.js
export function logWordDebugReset(framesBeforeReset, reason) {
  if (!wordDebugActive) return;
  fetch(`${API_BASE_URL}/api/debug/word/event`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      event_type: 'SEQUENCE_RESET',
      frames_before_reset: framesBeforeReset,
      reason
    })
  }).catch(() => {});
}
```

### 5.2. Cách thu thập 30 frame (Sliding Window Pipeline)
- **Tần số trích xuất:** Camera chạy qua vòng lặp `requestAnimationFrame(processCameraFrame)` với cơ chế throttling: gửi frame tối đa mỗi 30ms (~33 FPS) sang WebAssembly (Dòng 327–335).
- **Tích lũy:** Khi phát hiện có tay, mỗi frame trích xuất 75 điểm được `push` vào mảng `sequenceBufferRef.current`.
- **Cửa sổ trượt (Sliding Window):**
  - Khi buffer đạt đủ 30 frame (`currentLen >= SEQ_LEN`), hệ thống cắt đúng 30 frame cuối cùng: `sequenceBufferRef.current.slice(-SEQ_LEN)` để gửi đi suy luận.
  - Sau đó loại bỏ 6 frame cũ nhất: `sequenceBufferRef.current.splice(0, FRAME_STEP)` (với `FRAME_STEP = 6`).
  - Buffer còn lại 24 frame. Khoảng 6 frame tiếp theo (~180–200ms) buffer sẽ đầy lại 30 frame và thực hiện lượt dự đoán kế tiếp, tạo cảm giác nhận diện liên tục mà không phải chờ 1–2 giây từ đầu.

---

## 6. TRUY VẾT DỮ LIỆU TRAIN CỦA SSTCN_ATTENTION VÀ ĐỘ TƯƠNG THÍCH VỚI WEBCAM

### 6.1. Bằng chứng chắc chắn về dữ liệu huấn luyện (Training Data Contract)
1. **Nguồn gốc dữ liệu train:**
   - Được mô tả chi tiết tại `docs/sstcn/README.md` và `docs/sstcn/STATUS.md`.
   - Tập dữ liệu gồm 47.643 mẫu sequence từ 102 lớp cử chỉ VSL, trích xuất từ các video giáo trình/người khiếm thính ký hiệu thật.
2. **Công cụ trích xuất dữ liệu train:**
   - Sử dụng **`mediapipe.solutions.holistic`** (MediaPipe Holistic), không dùng MediaPipe Hands độc lập.
   - **File:** `backend/training/prepare_sstcn.py`
   ```python
   # Dòng 77 - 84: backend/training/prepare_sstcn.py
                   result = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                   parts = []
                   for field, count in (("pose_landmarks", 33), ("left_hand_landmarks", 21),
                                        ("right_hand_landmarks", 21)):
                       landmarks = getattr(result, field)
                       parts.append(np.asarray([[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
                                               dtype=np.float32) if landmarks else np.zeros((count, 3), np.float32))
                   frames.append(np.concatenate(parts))
   ```
3. **Quy cách tiền xử lý:**
   - **File:** `backend/training/sstcn_data.py`
   ```python
   # Dòng 17 - 23: backend/training/sstcn_data.py
   PREPROCESSING = {
       "version": 1, "shape": [3, 30, 75], "dtype": "float32",
       "joint_order": ["pose:0..32", "left_hand:0..20", "right_hand:0..20"],
       "coordinates": "mediapipe_image_normalized_xyz",
       "missing_landmarks": "zero", "normalization": "none",
       "temporal_sampling": "linear_resample_to_30_after_frame_stride",
   }
   ```
   **Đặc điểm:** Tọa độ được giữ nguyên định dạng chuẩn hóa ảnh `[0, 1]` của MediaPipe, không chuẩn hóa tâm theo vai hay cổ tay. Khi mất dấu khớp thì điền số `0`.

### 6.2. Kiến trúc mô hình `SSTCN_Attention`
- **File:** `backend/models/sstcn_model.py`
```python
# Dòng 36 - 42: backend/models/sstcn_model.py
        # --- Spatial Convolution ---
        self.spatial_conv = nn.Conv1d(in_channels, 128, kernel_size=1)
        # --- Multi-head Joint Attention riêng cho 2 bàn tay ---
        self.left_hand_attn = JointAttention(128, num_heads=4)
        self.right_hand_attn = JointAttention(128, num_heads=4)
# Dòng 92 - 93: backend/models/sstcn_model.py
        # --- 5. Global Average Pooling qua thời gian và các khớp ---
        x = x.mean(dim=2).mean(dim=2)  # [B, 512]
```
- Lớp `spatial_conv` quét qua toàn bộ 75 khớp (`J=75`).
- Mặc dù có cơ chế Attention riêng cho 2 bàn tay, nhưng ở bước cuối cùng, mô hình thực hiện **Global Average Pooling trên toàn bộ 75 khớp** (cả Pose lẫn Hands).

### 6.3. Đánh giá độ tương thích: Bằng chứng chắc chắn & Giả thuyết / Hệ quả

| Tiêu chí | Dữ liệu Train (`prepare_sstcn.py`) | Dữ liệu Webcam (`CameraSignRecognizer.jsx`) |
| :--- | :--- | :--- |
| **Công cụ nhận diện** | MediaPipe Holistic | MediaPipe Hands |
| **29 điểm Pose (đầu, mặt, vai, thân)** | Khung xương người thật cử động theo video | **Hardcode hằng số tĩnh vĩnh viễn** |
| **Hai vai (Khớp 11, 12)** | Di chuyển tự nhiên theo vóc dáng người | Cố định tại `(0.65, 0.45)` và `(0.35, 0.45)` |
| **Khuỷu tay (Khớp 13, 14)** | Khớp khuỷu tay 3D người thật | Suy ra từ trung điểm vai-cổ tay ($z=0$) |
| **Cổ tay (Khớp 15, 16)** | Khớp cổ tay từ MediaPipe Pose | Lấy từ landmark gốc ngón tay của Hands |
| **Bàn tay (Khớp 33–74)** | MediaPipe Holistic Hand landmarks | MediaPipe Hands landmarks |

#### Bằng chứng chắc chắn (Fact):
- Có sự **lệch phân phối dữ liệu nghiêm trọng (Data Distribution Mismatch)** giữa môi trường huấn luyện và môi trường suy luận thực tế.
- 38.7% số khớp đầu vào của mạng nơ-ron (29/75 khớp) trong runtime webcam là các hằng số chết, hoàn toàn không phản ánh bất kỳ thông tin cử động nào của cơ thể người dùng.

#### Giả thuyết và Hệ quả kỹ thuật (Hypothesis & Impact):
- Vì mô hình sử dụng Global Average Pooling trên toàn bộ 75 khớp, việc 29 khớp pose bị cố định sẽ tạo ra một vector đặc trưng nền (background bias) không đổi chiếm tỉ trọng đáng kể trong biểu diễn ẩn 512 chiều.
- Điều này giải thích nguyên nhân vì sao trong các phiên test Word Debug, mô hình SSTCN thường xuyên dự đoán sai lệch hoặc có độ tin cậy thấp đối với các từ có yếu tố chuyển động vai/thân (ví dụ các từ cử chỉ ở vị trí ngực, vai hoặc hai tay giao nhau), dù cử chỉ bàn tay thực hiện rất rõ ràng.

---

## 7. BẢNG TỔNG HỢP TRUY VẾT MÃ NGUỒN

| Thành phần | Vị trí File | Dòng code | Trạng thái kỹ thuật |
| :--- | :--- | :--- | :--- |
| `DEFAULT_POSE` (Vai 0.65 / 0.35) | `src/components/CameraSignRecognizer.jsx` | L204–218 | **Hardcode hằng số tĩnh** |
| Công thức tính khuỷu & gán cổ tay | `src/components/CameraSignRecognizer.jsx` | L231–239 | **Nội suy toán học (Khuỷu) / Gốc bàn tay (Cổ tay)** |
| Khởi tạo `recognitionScope = 'hands'` | `src/components/CameraSignRecognizer.jsx` | L91, L98 | **Mặc định Hands (không đổi khi sang Words)** |
| Nhánh Holistic đầy đủ (chưa kích hoạt mặc định) | `src/components/CameraSignRecognizer.jsx` | L170–199, L276–286 | **Có sẵn code nhưng cần chọn Scope thủ công** |
| Reset buffer khi mất tay | `src/components/CameraSignRecognizer.jsx` | L723–732 | **Xóa sạch buffer 30 frame** |
| Thu thập 30 frame & Sliding Window (bước 6) | `src/components/CameraSignRecognizer.jsx` | L743–754 | **Sliding window 30 frame** |
| API gửi chuỗi | `src/services/api.js` | L121–137 | **POST `/api/recognize_sequence`** |
| Backend Endpoint | `backend/app.py` | L261–275 | **FastAPI wrapper** |
| Suy luận SSTCN | `backend/vision/recognizer.py` | L77–195 | **Model forward pass** |
| Ghi file `.npy` thực tế | `backend/vision/word_debugger.py` | L223–227 | **Lưu ma trận `(30, 75, 3)` xuống đĩa** |
| Pipeline trích xuất dữ liệu train | `backend/training/prepare_sstcn.py` | L54–84 | **MediaPipe Holistic (Pose thật)** |

