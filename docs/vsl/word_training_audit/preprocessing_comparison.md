# ĐỐI CHIẾU TIỀN XỬ LÝ DỮ LIỆU: HUẤN LUYỆN VS THỰC TẾ WEBCAM (PREPROCESSING COMPARISON)

**Dự án:** Nhận diện Ngôn ngữ Ký hiệu Việt Nam (VSL)  
**Mã nhiệm vụ:** VSL-P011 — Công việc 2  
**Phạm vi:** Đối chiếu 7 tiêu chí kỹ thuật giữa Training Pipeline, Runtime Webcam Holistic, và Runtime Webcam Hands Fallback.  
**Ngày thực hiện:** 07/11/2026  

---

## 1. Bảng đối chiếu tổng hợp 7 tiêu chí kỹ thuật

| Tiêu chí | (1) Training Pipeline | (2) Webcam Holistic (`holistic_raw`) | (3) Webcam Hands Fallback (`hands_fallback_pose`) | Kết luận & Đánh giá sai lệch |
| :--- | :--- | :--- | :--- | :--- |
| **1. Shape & Thứ tự 75 landmarks** | `(Batch, 3, 30, 75)`<br>Thứ tự: 33 Pose (0..32), 21 Tay trái (33..53), 21 Tay phải (54..74). | `(1, 3, 30, 75)`<br>Thứ tự: 33 Pose (0..32), 21 Tay trái (33..53), 21 Tay phải (54..74). | `(1, 3, 30, 75)`<br>Thứ tự: 33 Pose giả lập (0..32), 21 Tay trái (33..53), 21 Tay phải (54..74). | **Khớp Shape cấu trúc.**<br>Cả 3 luồng đều đưa về `(3, 30, 75)`. Riêng Fallback thì 33 khớp pose là mốc giả lập. *(Đã xác nhận bằng code & NPY)* |
| **2. Hệ tọa độ & Normalization** | `mediapipe_image_normalized_xyz`<br>x, y ∈ [0, 1] theo tỷ lệ khung hình.<br>z là chiều sâu tương đối.<br>**Normalization: None** (không căn chỉnh gốc cổ tay hay vai). | `mediapipe_image_normalized_xyz`<br>x, y ∈ [0, 1] từ canvas.<br>z từ MediaPipe.<br>**Normalization: None**. | `mediapipe_image_normalized_xyz`<br>Hand landmarks: x, y ∈ [0, 1].<br>Pose: Hằng số cứng trong [0, 1].<br>**Normalization: None**. | **Khớp hệ tọa độ.**<br>Không có bước zero-centering hay scale normalization ở bất kỳ luồng nào. *(Đã xác nhận bằng code)* |
| **3. Xử lý Missing Landmarks** | **Zero-fill:** Điền `0.0` cho toàn bộ 3 tọa độ (x, y, z) của các khớp không phát hiện được.<br>`np.zeros((count, 3))`. | **Zero-fill:** Điền `[0, 0, 0]` nếu thiếu bàn tay hoặc thiếu pose. | **Hỗn hợp sai lệch:**<br>- Bàn tay: Điền `[0, 0, 0]`.<br>- Pose: **ĐIỀN TỌA ĐỘ GIẢ LẬP** (không điền 0). | **SAI LỆCH NGHIÊM TRỌNG ở Fallback:**<br>Khi người dùng không vào khung hình hoặc mất pose, Training mong đợi số 0; nhưng Fallback lại đưa vào tọa độ giả `(0.65, 0.45)` khiến model bị đánh lừa. *(Đã xác nhận)* |
| **4. Cách tạo Sequence 30 frame** | **Cử chỉ độc lập trọn vẹn (Segmented Video):**<br>Lấy toàn bộ frame của video clip cử chỉ, nếu độ dài $T \ne 30$, dùng `np.linspace(0, T-1, 30)` để **nội suy tuyến tính (Linear Resampling) về đúng 30 frame**. | **Cửa sổ trượt thời gian thực (Sliding Window FIFO):**<br>Đệm trượt 30 frame liên tục trên luồng camera, bước nhảy `FRAME_STEP = 6` (hoặc 1). **Không có nội suy tuyến tính**. | **Cửa sổ trượt thời gian thực (Sliding Window FIFO):**<br>Đệm trượt 30 frame liên tục, bước nhảy `FRAME_STEP = 6`. **Không có nội suy tuyến tính**. | **SAI LỆCH CỐT LÕI (Domain Gap lớn nhất):**<br>Training học 30 frame trải đều từ đầu đến cuối một cử chỉ hoàn chỉnh. Webcam đưa vào 30 frame trượt ngẫu nhiên (chỉ chứa một giai đoạn nhỏ của cử chỉ, hoặc chứa cả đoạn nghỉ). *(Đã xác nhận)* |
| **5. Tốc độ khung hình (FPS) & Thời lượng window** | **Nội suy chuẩn hóa nhịp độ:**<br>Dù video quay ở 25 FPS, 30 FPS hay 60 FPS, sau khi nội suy đều thành 30 frame đại diện cho 1 cử chỉ chuẩn (~1.0s – 1.5s). | **FPS thấp & Bị kéo giãn:**<br>Client FPS: **13.6 FPS** (trung bình).<br>Cửa sổ 30 frame kéo dài **2.518 ms (~2.52 giây)**. | **FPS cao & Co ngắn:**<br>Client FPS: **26.1 FPS** (trung bình).<br>Cửa sổ 30 frame kéo dài **1.199 ms (~1.20 giây)**. | **BẤT NHẤT THỜI GIAN NGHIÊM TRỌNG:**<br>Holistic bị chậm gấp đôi so với Fallback. Cử chỉ trong Holistic bị làm chậm (slow-motion) đối với mắt mô hình SSTCN. *(Đã xác nhận bằng log thực nghiệm)* |
| **6. Class Mapping (102 lớp)** | 102 lớp (`0..101`) theo chuẩn giáo trình VSL:<br>- 0..28: Chữ cái<br>- 29..38: Chữ số<br>- 39..101: 63 từ vựng | Nhận đủ 102 logit; áp dụng Masking `logits[:, :39] = -inf` để chỉ lọc 63 lớp từ vựng (39..101). | Nhận đủ 102 logit; áp dụng Masking `logits[:, :39] = -inf` để chỉ lọc 63 lớp từ vựng (39..101). | **Khớp 100%.**<br>Class ID, nhãn từ điển và logic masking hoàn toàn đồng nhất. *(Đã xác nhận bằng code & unit tests)* |
| **7. Biến đổi dữ liệu trước khi đưa vào model** | Mảng NumPy trong bộ nhớ `sequences.npy` -> DataLoader nạp batch -> Cast sang GPU/CPU float32 tensor. | Mảng JS Array -> JSON serialize qua WebSocket/HTTP -> FastAPI parse -> `transpose(2,0,1)` -> PyTorch tensor. | Mảng JS Array -> JSON serialize -> FastAPI parse -> `transpose(2,0,1)` -> PyTorch tensor. | **Khớp logic chuyển đổi.**<br>Thao tác `transpose(2, 0, 1)` chuyển `(30, 75, 3)` thành `(3, 30, 75)` khớp hoàn toàn với pipeline huấn luyện. *(Đã xác nhận bằng VSL-P010 100% replay)* |

---

## 2. Chi tiết phân tích từng tiêu chí

### 2.1 Tiêu chí 1: Thứ tự và cấu trúc khớp (Joint Order)
- Trong [`backend/models/sstcn_model.py`](file:///E:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/models/sstcn_model.py) (dòng 14 – 16, 59 – 60):
  ```python
  POSE_START, POSE_END = 0, 33
  L_HAND_START, L_HAND_END = 33, 54
  R_HAND_START, R_HAND_END = 54, 75
  ```
  Khối Attention được áp dụng riêng cho `left_hand_joints = list(range(33, 54))` và `right_hand_joints = list(range(54, 75))`.
- Trong [`src/components/CameraSignRecognizer.jsx`](file:///E:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx), cả `extract75KeypointsFromHolistic` (dòng 185 – 213) và `extract75KeypointsFromHands` (dòng 217 – 268) đều đẩy mốc theo thứ tự: 33 Pose trước, sau đó 21 Left Hand, cuối cùng là 21 Right Hand.
- **Đánh giá:** Thứ tự chỉ mục của 75 khớp hoàn toàn chuẩn xác.

### 2.2 Tiêu chí 4: Sự khác biệt bản chất giữa "Linear Resampling" và "Sliding Window"
Đây là nguyên nhân kỹ thuật quan trọng nhất giải thích tại sao Accuracy trên tập Offline đạt ~88%, nhưng trên Webcam chỉ đạt ~6–15%:

```
[QUÁ TRÌNH HUẤN LUYỆN (OFFLINE TRAINING)]
Video clip cử chỉ mẫu (1-3s) ──► Trích xuất toàn bộ N frame ──► Linear Resample (linspace) ──► Đúng 30 frame
                                                                 (Đại diện ĐẦY ĐỦ: Bắt đầu -> Đỉnh -> Kết thúc)

[QUÁ TRÌNH NHẬN DIỆN THỰC TẾ (ONLINE WEBCAM)]
Camera liên tục ──► Buffer FIFO 30 frame ──► Đầy 30 frame gửi ngay ──► Sliding Window trượt 6 frame
                                               (Chỉ là một lát cắt thời gian ngẫu nhiên!)
```

1. **Trong tập train:** Mỗi mẫu dữ liệu là một **video đã được cắt biên (trimmed video)**, ghi lại từ lúc tay bắt đầu nâng lên, tạo hình cử chỉ, cho đến khi hạ tay xuống. Hàm `canonical_sequence()` trong [`backend/training/sstcn_data.py`](file:///E:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/training/sstcn_data.py) (dòng 89 – 95) dùng:
   ```python
   positions = np.linspace(0, len(a) - 1, 30)
   ```
   để co hoặc giãn toàn bộ cử chỉ thành đúng 30 bước thời gian.
2. **Trên webcam thời gian thực:** Người dùng bật camera, hệ thống liên tục tích lũy từng frame vào một mảng trượt 30 phần tử. Khi đủ 30 frame, hệ thống gửi đi nhận diện. Lúc này:
   - Nếu người dùng mới chỉ đang nhấc tay lên: 30 frame chỉ chứa pha "chuẩn bị", không có pha "thực hiện".
   - Nếu người dùng làm cử chỉ quá nhanh: cử chỉ có thể kết thúc trước frame thứ 20, 10 frame sau là tay đứng yên.
   - Nếu người dùng làm cử chỉ quá chậm: 30 frame chỉ bắt được nửa đầu cử chỉ.
3. **Hệ quả:** Mô hình SSTCN được huấn luyện để nhận dạng **hình thái toàn cục (global temporal pattern)** của một cử chỉ hoàn chỉnh. Khi gặp một lát cắt cục bộ (local slice) từ sliding window, xác suất phân loại giảm sút nghiêm trọng.

### 2.3 Tiêu chí 5: Tốc độ khung hình (FPS) và biến dạng thời gian
Từ số liệu log thực nghiệm của VSL-P010 trên 501 tensor:
- **`holistic_raw`:** Client FPS trung bình là **13.6 FPS** (nhanh nhất 16, chậm nhất 11 FPS). Do đó, để thu thập đủ 30 frame cần khoảng thời gian trung bình là **2.518 ms**.
- **`hands_fallback_pose`:** Client FPS trung bình là **26.1 FPS** (nhanh nhất 30, chậm nhất 22 FPS). 30 frame chỉ cần **1.199 ms**.
- **Sự sai lệch nhịp độ:** Một cử chỉ nếu người dùng thực hiện trong 1.2 giây:
  - Ở chế độ Hands Fallback: Bắt trọn trong ~31 frame.
  - Ở chế độ Holistic: Chỉ thu được ~16 frame, 14 frame còn lại bị lọt sang window tiếp theo!

---

## 3. Phân định mức độ bằng chứng khoa học

| Nội dung kết luận | Mức độ xác nhận | Căn cứ / Bằng chứng |
| :--- | :--- | :--- |
| Tọa độ vai/khuỷu tay cố định trong Fallback | **ĐÃ XÁC NHẬN 100%** | Code `extract75KeypointsFromHands` và dữ liệu thực nghiệm 268/268 NPY có $\sigma < 10^{-4}$. |
| Thứ tự 75 khớp đồng nhất | **ĐÃ XÁC NHẬN 100%** | Code `sstcn_model.py` và `CameraSignRecognizer.jsx`. |
| Class mapping 102 lớp khớp hoàn toàn | **ĐÃ XÁC NHẬN 100%** | Bảng `classes.json`, `SIGN_DICT_102`, 58 unit tests pytest pass. |
| Masking words `:39` hoạt động đúng | **ĐÃ XÁC NHẬN 100%** | Code `recognizer.py:128` và kết quả Replay 501/501 khớp. |
| Resampling vs Sliding Window mismatch | **ĐÃ XÁC NHẬN 100%** | So sánh giữa `sstcn_data.py:canonical_sequence` và `CameraSignRecognizer.jsx:sequenceBufferRef`. |
| Chênh lệch thời lượng window (2.5s vs 1.2s) | **ĐÃ XÁC NHẬN 100%** | Báo cáo `pose_temporal_audit.md` (VSL-P010). |
| FPS và thiết bị quay của tập video gốc lịch sử | **CHƯA ĐỦ BẰNG CHỨNG** | Tập dữ liệu gốc 47.643 npy chỉ lưu skeleton, video MP4 gốc không còn trong repo; báo cáo DOCX không ghi rõ FPS camera quay. |
| Mức độ ảnh hưởng tương đối giữa "Sai lệch Pose" vs "Sai lệch Segmentation" | **CẦN THỰC NGHIỆM THÊM** | Trên Holistic thật, accuracy đạt 15.0% (cao hơn Fallback 3.0%), nhưng vẫn thấp hơn rất nhiều so với Offline 88%. Điều này chứng minh **cả hai yếu tố đều gây lỗi**, trong đó lỗi Segmentation/Temporal đóng vai trò lớn trong phần chênh lệch còn lại. |

