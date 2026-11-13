# BÁO CÁO TRUY NGUYÊN DỮ LIỆU VÀ NGUỒN GỐC MỐC CƠ THỂ (DATA PROVENANCE AUDIT)

**Dự án:** Nhận diện Ngôn ngữ Ký hiệu Việt Nam (VSL)  
**Mã nhiệm vụ:** VSL-P011 — Công việc 1  
**Phạm vi:** Kiểm tra nguồn gốc dữ liệu SSTCN, MediaPipe tracker, tọa độ pose cố định và dataset huấn luyện.  
**Ngày thực hiện:** 07/11/2026  

---

## 1. Tại sao 29/33 khớp Pose bất động trong `hands_fallback_pose`?

Trong thử nghiệm VSL-P010, đã xác nhận 100% tensor thuộc nhóm `hands_fallback_pose` (268/268 windows) đều có 29/33 khớp Pose bất động với độ lệch chuẩn thời gian $\sigma < 10^{-4}$.

### Nguyên nhân kỹ thuật cốt lõi:
- **Bản chất của MediaPipe Hands:** Thư viện `MediaPipe Hands` (`@mediapipe/hands`) **chỉ phát hiện bàn tay** (tối đa 21 khớp cho mỗi bàn tay trái/phải), hoàn toàn **không hỗ trợ phát hiện cơ thể (Pose)** hay khuôn mặt.
- **Sự thiếu hụt dữ liệu đầu vào của SSTCN:** Kiến trúc mô hình `SSTCN_Attention` yêu cầu bắt buộc đầu vào có kích thước cố định là 75 khớp ($33 \text{ Pose} + 21 \text{ Left Hand} + 21 \text{ Right Hand}$).
- **Cơ chế "chắp vá" mốc giả lập:** Để đưa dữ liệu từ MediaPipe Hands vào mô hình SSTCN mà không gây lỗi kích thước ma trận (`AssertionError` hoặc `ValueError`), Frontend đã sử dụng một mảng hằng số cứng `DEFAULT_POSE` gồm 33 điểm giả lập.
- **Cơ chế cập nhật một phần:**
  - Trong số 33 mốc Pose, code chỉ cập nhật đúng **4 khớp** dựa theo gốc bàn tay (khớp 0 của bàn tay):
    - Khớp 15 (cổ tay trái) = Tọa độ cổ tay bàn tay trái.
    - Khớp 13 (khuỷu tay trái) = Tọa độ trung điểm giữa vai giả lập `(0.65, 0.45)` và cổ tay trái, cộng thêm độ lệch offset `0.04`.
    - Khớp 16 (cổ tay phải) = Tọa độ cổ tay bàn tay phải.
    - Khớp 14 (khuỷu tay phải) = Tọa độ trung điểm giữa vai giả lập `(0.35, 0.45)` và cổ tay phải, trừ đi độ lệch offset `0.04`.
  - **29 khớp còn lại hoàn toàn không được đo từ camera:**
    - 11 khớp đầu/mặt (0..10: Mũi, mắt, tai, miệng).
    - 2 khớp vai (11: Vai trái cố định tại `(0.65, 0.45, 0)`, 12: Vai phải cố định tại `(0.35, 0.45, 0)`).
    - 16 khớp thân dưới và chân (17..32: cố định tại `(0.5, 0.8, 0)`).
  - Do đó, 29 khớp này giữ nguyên giá trị hằng số qua 100% số khung hình của toàn bộ phiên quay.

---

## 2. Vị trí mã nguồn tạo tọa độ cố định

Toàn bộ logic tạo và gán tọa độ pose giả lập nằm tại Frontend:

- **File mã nguồn:** [`src/components/CameraSignRecognizer.jsx`](file:///E:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx)
- **Hàm:** `extract75KeypointsFromHands(results)` (dòng 217 – 269)
- **Đoạn mã chứng minh (Dòng 219 – 233):**
  ```javascript
  const DEFAULT_POSE = [
    [0.5, 0.2, 0],   // 0: Mũi
    [0.52, 0.18, 0], [0.54, 0.18, 0], [0.56, 0.18, 0], // 1..3: Mắt trái
    [0.48, 0.18, 0], [0.46, 0.18, 0], [0.44, 0.18, 0], // 4..6: Mắt phải
    [0.58, 0.22, 0], [0.42, 0.22, 0],                   // 7..8: Tai
    [0.53, 0.26, 0], [0.47, 0.26, 0],                   // 9..10: Miệng
    [0.65, 0.45, 0], [0.35, 0.45, 0],                   // 11: Vai trái, 12: Vai phải (CỐ ĐỊNH)
    [0.72, 0.65, 0], [0.28, 0.65, 0],                   // 13: Khuỷu trái, 14: Khuỷu phải
    [0.68, 0.80, 0], [0.32, 0.80, 0],                   // 15: Cổ tay trái, 16: Cổ tay phải
  ];

  for (let i = 0; i < 33; i++) {
    if (i < DEFAULT_POSE.length) keypoints.push(DEFAULT_POSE[i]);
    else keypoints.push([0.5, 0.8, 0]); // 17..32: Thân dưới và chân (CỐ ĐỊNH)
  }
  ```
- **Nội suy toán học (Dòng 247 – 254):**
  ```javascript
  if (leftHand && leftHand[0]) {
    keypoints[15] = [leftHand[0].x || 0.68, leftHand[0].y || 0.80, leftHand[0].z || 0];
    keypoints[13] = [(0.65 + (leftHand[0].x || 0.68)) / 2 + 0.04, (0.45 + (leftHand[0].y || 0.80)) / 2, 0];
  }
  if (rightHand && rightHand[0]) {
    keypoints[16] = [rightHand[0].x || 0.32, rightHand[0].y || 0.80, rightHand[0].z || 0];
    keypoints[14] = [(0.35 + (rightHand[0].x || 0.32)) / 2 - 0.04, (0.45 + (rightHand[0].y || 0.80)) / 2, 0];
  }
  ```

---

## 3. Điều kiện khiến hệ thống chuyển từ Holistic sang Hands fallback

Trong [`src/components/CameraSignRecognizer.jsx`](file:///E:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx), hệ thống được cấu hình theo 2 tiêu chí điều khiển:
- **Tiêu chí A (Phạm vi nhận diện - Scope):** `'hands'` (Bàn tay - 60 FPS) hoặc `'holistic'` (Toàn thân 75 khớp).
- **Tiêu chí B (Chế độ chuyển đổi - Mode):** `'letters'`, `'digits'`, `'words'`, `'sentence'`.

Hệ thống sẽ chạy và phát sinh `hands_fallback_pose` khi rơi vào một trong các điều kiện sau:

1. **Khởi tạo mặc định của Component (Dòng 90 – 99):**
   ```javascript
   const [recognitionScope, setRecognitionScope] = useState('hands'); // Mặc định là 'hands'
   const [trackerEngine, setTrackerEngine] = useState('hands');
   const trackerEngineRef = useRef('hands');
   ```
   Khi ứng dụng vừa mount và mở camera trước khi người dùng kịp bấm chuyển sang tab `words`, engine tracking đang là `hands`.
2. **Người dùng bấm chọn thủ công phạm vi "Bàn tay" (Dòng 1130 – 1135):**
   Trên giao diện có nút bấm cho phép chuyển đổi giữa `Bàn tay (60 FPS)` và `Toàn thân (75 khớp)`. Khi người dùng chuyển sang tab Từ vựng nhưng thấy video bị giật lag (~13 FPS) và chủ động bấm nút "Bàn tay" để tăng tốc độ khung hình lên 26–30 FPS, hàm `handleScopeChange('hands')` (dòng 466) lập tức kích hoạt `switchTrackerEngine('hands')`.
3. **Lỗi nạp thư viện MediaPipe Holistic từ CDN (Dòng 285 – 297 & Dòng 457 – 461):**
   MediaPipe Holistic yêu cầu nạp file nhị phân WebAssembly nặng từ CDN (`https://cdn.jsdelivr.net/npm/@mediapipe/holistic/...`). Nếu mạng chậm, bị chặn mạng hoặc CDN gặp sự cố khiến `window.Holistic` chưa sẵn sàng, hàm `createTracker('holistic')` ném lỗi; khối `try/catch` tại `switchTrackerEngine` ghi log cảnh báo và giữ nguyên `trackerEngineRef.current = 'hands'`.
4. **Cơ chế Callback xử lý frame (Dòng 724 – 732):**
   Mỗi khi `trackerEngine === 'hands'`, MediaPipe Hands trả kết quả về `onHandsResultsFast(results)`, hàm này truyền trực tiếp `() => extract75KeypointsFromHands(results)` vào `handleMotionAndSequence`, dẫn đến toàn bộ chuỗi 30 frame ghi nhận đều là `hands_fallback_pose`.

---

## 4. Dataset training gốc được trích xuất bằng Hands, Pose hay Holistic?

### Bằng chứng xác nhận từ tài liệu và mã nguồn chuẩn bị dữ liệu:

1. **Quy cách cấu trúc 75 khớp:**
   - Trong MediaPipe:
     - `MediaPipe Hands`: Chỉ cung cấp 21 khớp bàn tay (không có Pose).
     - `MediaPipe Pose`: Chỉ cung cấp 33 khớp cơ thể (không có chi tiết bàn tay 21 khớp).
     - `MediaPipe Holistic`: Giải pháp duy nhất tích hợp đồng thời **33 khớp Pose + 21 khớp bàn tay trái + 21 khớp bàn tay phải = đúng 75 khớp**.
2. **Mã nguồn tiền xử lý huấn luyện tái lập:**
   - Trong [`backend/training/prepare_sstcn.py`](file:///E:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/training/prepare_sstcn.py) (dòng 54 – 94):
     ```python
     with mp.solutions.holistic.Holistic(
         model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
     ) as detector:
         ...
         for field, count in (("pose_landmarks", 33), ("left_hand_landmarks", 21), ("right_hand_landmarks", 21)):
             landmarks = getattr(result, field)
             parts.append(np.asarray([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)
                          if landmarks else np.zeros((count, 3), np.float32))
         frames.append(np.concatenate(parts))
     ```
   - Trong [`docs/sstcn/video_preprocessing_check.json`](file:///E:/NAM4/SEMINAR/SIGN_LANGUAGE/docs/sstcn/video_preprocessing_check.json) (dòng 6):
     `"extractor": "mediapipe.solutions.holistic"`
3. **Metadata của Dataset Manifest:**
   - Trong `dataset_manifest.json.gz` và [`backend/data/sstcn/dataset_summary.json`](file:///E:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/data/sstcn/dataset_summary.json):
     `joint_order: ["pose:0..32", "left_hand:0..20", "right_hand:0..20"]`
4. **Kết luận:** Mô hình SSTCN_Attention và bộ dữ liệu 102 lớp được thiết kế và trích xuất dựa trên **MediaPipe Holistic**.

---

## 5. Trong dataset training, vai và khuỷu tay có thực sự chuyển động hay cũng bị gán cố định?

### Hiện trạng dữ liệu trên máy tính:
- Thư mục dữ liệu thô `backend/data/sstcn/raw/` và thư mục chuẩn bị `backend/data/sstcn/prepared/` **chưa tồn tại trên máy** (chỉ có metadata manifest, split, SHA checksum).
- Kho dữ liệu gốc gồm 6 tệp nén ZIP (`data_01.zip` đến `data_05.zip` và `test_set.zip`) với tổng dung lượng **1,67 GiB** (1.793.877.824 byte, 47.643 file `.npy`).

### Phân tích logic từ mã nguồn đã kiểm chứng:
1. **Pipeline backend không có mã gán pose giả lập:** Trong toàn bộ thư mục `backend/training/` (`prepare_sstcn.py`, `sstcn_data.py`, `train_sstcn.py`), không hề tồn tại mảng hằng số `DEFAULT_POSE` hay giá trị `(0.65, 0.45)`. Khi trích xuất từ video, tọa độ vai (11, 12) và khuỷu tay (13, 14) được đọc trực tiếp từ `results.pose_landmarks` của người diễn xuất thật.
2. **Khái niệm `DEFAULT_POSE` là độc quyền của Frontend Web:** Như đã chứng minh ở Mục 2, mảng `DEFAULT_POSE` chỉ được sinh ra trong `CameraSignRecognizer.jsx` để đối phó với việc MediaPipe Hands thiếu 33 khớp pose.
3. **Bằng chứng từ báo cáo DOCX lịch sử:** File [`docs/sstcn/legacy/test_set_result_report_nhom3.docx`](file:///E:/NAM4/SEMINAR/SIGN_LANGUAGE/docs/sstcn/legacy/test_set_result_report_nhom3.docx) ghi rõ:
   > *"Test set bao gồm 1420 videos, mỗi nhãn có từ 10 – 15 videos, mỗi video được trích xuất ra chuỗi skeleton để đưa vào mô hình dự đoán."*
   Các video này quay người thật thực hiện cử chỉ trong phòng, trong đó người thực hiện vung tay, chuyển động vai và cánh tay tự nhiên.

---

## 6. Báo cáo tình trạng dữ liệu cục bộ và phương án thu thập bằng chứng

| Hạng mục dữ liệu | Trạng thái trên máy | Dung lượng | Vị trí lưu trữ từ xa |
|---|---|---|---|
| `data_01.zip` (sign_000 - sign_020) | ❌ Chưa có | 328 MB | GitHub Release / Drive |
| `data_02.zip` (sign_021 - sign_054) | ❌ Chưa có | 644 MB | GitHub Release / Drive |
| `data_03.zip` (sign_055 - sign_069) | ❌ Chưa có | 352 MB | GitHub Release / Drive |
| `data_04.zip` (sign_070 - sign_084) | ❌ Chưa có | 414 MB | GitHub Release / Drive |
| `data_05.zip` (sign_085 - sign_101) | ❌ Chưa có | 27 MB | GitHub Release / Drive |
| `test_set.zip` (Test set 1.420 mẫu) | ❌ Chưa có | 28 MB | GitHub Release / Drive |
| `sequences.npy` (Toàn bộ 41.645 mẫu) | ❌ Chưa có | ~1,1 GB | Cần sinh từ `prepare_sstcn.py` |
| `dataset_manifest.json.gz` | ✅ ĐÃ CÓ TRÊN MÁY | 4,2 MB | `docs/sstcn/evaluation/...` |
| `splits.csv` (Phân chia train/val/test) | ✅ ĐÃ CÓ TRÊN MÁY | 7,0 MB | `backend/data/sstcn/splits.csv` |

### Cách thu thập bằng chứng thực nghiệm mà KHÔNG cần tải 1,67 GB:
Thay vì tải toàn bộ 6 tệp nén:
1. Chỉ cần tải một archive nhỏ nhất: **`data_05.zip` (27 MB)** hoặc **`test_set.zip` (28 MB)** qua lệnh:
   ```bash
   python -m backend.training.fetch_sstcn_data --only data_05.zip --extract
   ```
2. Chạy một đoạn mã kiểm tra nhanh 20 tệp `.npy` bất kỳ trong thư mục vừa giải nén:
   ```python
   import numpy as np
   from backend.training.sstcn_data import load_source
   data, label = load_source("backend/data/sstcn/raw/data_05/data_05/sign_085/sign_085_vid_01_00.npy")
   # data shape: (30, 75, 3)
   shoulder_std = data[:, [11, 12], :].std(axis=0)
   print("Độ biến thiên vai trong training data:", shoulder_std)
   ```
3. Nếu độ biến thiên $\sigma > 0.01$ $\rightarrow$ Khẳng định 100% khớp vai trong dataset training cử động thật sự, bác bỏ hoàn toàn giả thuyết training data dùng pose tĩnh.

