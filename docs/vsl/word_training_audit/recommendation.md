# BÁO CÁO KHUYẾN NGHỊ VÀ ĐỀ XUẤT QUYẾT ĐỊNH (RECOMMENDATIONS)

**Dự án:** Nhận diện Ngôn ngữ Ký hiệu Việt Nam (VSL)  
**Mã nhiệm vụ:** VSL-P011 — Công việc 5  
**Phạm vi:** So sánh toàn diện 3 phương án phát triển mô hình SSTCN và xác lập lộ trình khắc phục nghịch lý độ chính xác Offline (88%) vs Webcam (6–15%).  
**Ngày thực hiện:** 07/11/2026  

---

## 1. So sánh 3 lựa chọn phát triển

### 🅰️ Lựa chọn A: Resume Reconstructed Last (Tiếp tục train từ Epoch 12 lên 30–40)

- **Mô tả:** Tải 1,67 GB dữ liệu thô, chuẩn bị thư mục `prepared/`, sau đó dùng lệnh `train_sstcn.py --resume last.pth` để tiếp tục huấn luyện mô hình Reconstructed từ epoch 12 lên epoch 30 hoặc 40.
- **Bằng chứng kỹ thuật hỗ trợ:**
  - Checkpoint `sign_sstcn_reconstructed_last.pth` có đầy đủ 100% trạng thái (Model, Optimizer, Scheduler, Scaler, RNG state).
  - Train loss tại epoch 12 vẫn đang giảm dốc (0.568), LR vẫn ở mức cao (0.001), mô hình còn dư địa học thêm để tăng điểm số benchmark.
- **Rủi ro & Hạn chế:**
  - ❌ **Tốn tài nguyên nhưng không giải quyết đúng bệnh:** Độ chính xác trên tập test offline có thể tăng từ 88.6% lên 90–92%, nhưng **độ chính xác webcam thực tế vẫn sẽ kẹt ở mức ~6–15%**.
  - Việc huấn luyện tiếp tục trên cùng một phân phối dữ liệu (video clip tĩnh cắt sẵn, co giãn tuyến tính về 30 frame) không giúp mô hình học được cách xử lý cửa sổ trượt (sliding window) hay đối phó với frame drop từ webcam.
  - Tốn thời gian tải 1,67 GB và chiếm dụng GPU tính toán ~30–45 phút mà không đem lại cải thiện trải nghiệm người dùng.
- **Điều kiện cần:**
  - Tải 6 archive ZIP (1,67 GB) từ GitHub Release hoặc Google Drive.
  - Khởi tạo thư mục `backend/data/sstcn/prepared/` khớp mã băm `e3340c0d...`.
- **Mức độ ưu tiên:** 🔴 **THẤP (KHÔNG KHUYẾN NGHỊ Ở GIAI ĐOẠN NÀY)**.

---

### 🅱️ Lựa chọn B: Warm-start Historical (Khởi động ấm từ Checkpoint lịch sử)

- **Mô tả:** Lấy trọng số `sign_sstcn_attention_model.pth` (epoch 32, SHA `2a8ddcc4...`) làm trọng số khởi đầu qua cờ `--init-from`, sau đó huấn luyện tinh chỉnh (fine-tune) trên tập dữ liệu mới với learning rate nhỏ hơn.
- **Bằng chứng kỹ thuật hỗ trợ:**
  - Trong thử nghiệm đối chứng VSL-P010, Historical đạt độ chính xác webcam cao hơn Reconstructed (15.0% vs 10.3% trên luồng Holistic thật). Các đặc trưng trích xuất của Historical có độ ổn định thực nghiệm tốt hơn Reconstructed.
- **Rủi ro & Hạn chế:**
  - ❌ **Mất tính minh bạch về ranh giới kiểm thử:** Chúng ta không có danh sách split của Historical cũ. Việc train tiếp có thể gây hiện tượng rò rỉ dữ liệu kiểm thử (data leakage) nếu tập test hiện tại từng nằm trong tập train cũ.
  - Không có trạng thái Optimizer và Scheduler cũ, phải khởi tạo lại từ đầu ở Epoch 0.
  - Vẫn chịu chung nhược điểm của Lựa chọn A: mô hình vẫn được tối ưu trên tập video tĩnh, không khớp với cơ chế sliding window của webcam.
- **Điều kiện cần:**
  - Tương tự Lựa chọn A: Cần tải và chuẩn bị đầy đủ 1,67 GB dữ liệu.
- **Mức độ ưu tiên:** 🟡 **TRUNG BÌNH THẤP**.

---

### 🅲 Lựa chọn C: Sửa và chuẩn hóa Pipeline / Dataset trước khi Train (KHUYẾN NGHỊ MẠNH NHẤT)

- **Mô tả:** Tạm dừng việc huấn luyện mô hình. Tập trung giải quyết triệt để 3 nguyên nhân cốt lõi gây ra khoảng cách miền dữ liệu (Domain Gap) giữa môi trường huấn luyện và môi trường thực tế, sau đó mới tính đến việc huấn luyện lại hoặc train bổ sung.
- **Bằng chứng kỹ thuật chứng minh tính cấp thiết:**
  1. **Bằng chứng về Lỗi Mốc Giả lập (Fake Pose):**
     - VSL-P010 đã chứng minh: 268/501 tensor (53.5%) dữ liệu webcam bị gán cứng 29/33 điểm mốc cơ thể do chạy MediaPipe Hands fallback.
     - Khi mốc vai bị gán cứng tại `(0.65, 0.45)`, độ chính xác của cả 2 mô hình sụp đổ xuống còn **1.9% – 3.0%**.
     - Bắt buộc phải **loại bỏ hoàn toàn cơ chế `hands_fallback_pose`** cho chế độ từ vựng; chỉ cho phép gửi inference khi có mốc Holistic thật.
  2. **Bằng chứng về Lỗi Phân đoạn Thời gian (Temporal Segmentation Mismatch):**
     - Training dùng video có điểm đầu và điểm cuối trọn vẹn, co giãn tuyến tính về 30 frame.
     - Webcam dùng sliding window 30 frame trượt liên tục (người dùng vừa đưa tay lên là hệ thống đã cắt 30 frame gửi đi dự đoán).
     - Cần bổ sung thuật toán **Gesture Spotting (Phát hiện điểm bắt đầu - kết thúc cử chỉ)**: Chỉ khi người dùng thực hiện xong một cử chỉ (dựa trên vận tốc di chuyển bàn tay hoặc trạng thái dừng động tác), hệ thống mới đóng gói chuỗi frame đó, resample về 30 frame rồi mới gửi tới SSTCN.
  3. **Bằng chứng về Nghẽn Tốc độ Khung hình (FPS Bottleneck):**
     - MediaPipe Holistic chạy WebAssembly trên trình duyệt chỉ đạt trung bình **13.6 FPS**, kéo dài cửa sổ 30 frame lên **2.52 giây** (làm biến dạng tốc độ cử động).
     - Cần tối ưu luồng xử lý camera (ví dụ giảm độ phân giải canvas đầu vào xuống 320×240 hoặc chuyển việc suy luận MediaPipe về backend thông qua WebRTC/WebSocket nếu cần FPS cao).
- **Rủi ro:** Cần can thiệp vào logic frontend và pipeline xử lý buffer thời gian thực.
- **Điều kiện cần:** Không cần tải 1,67 GB dữ liệu ngay; chỉ cần thao tác trên mã nguồn frontend và backend runtime.
- **Mức độ ưu tiên:** 🟢 **ƯU TIÊN CAO NHẤT (SỐ 1)**.

---

## 2. Bảng so sánh tổng hợp 3 phương án

| Tiêu chí so sánh | Phương án A (Resume Last) | Phương án B (Warm-start Historical) | Phương án C (Chuẩn hóa Pipeline - KHUYẾN NGHỊ) |
| :--- | :---: | :---: | :---: |
| **Mục tiêu chính** | Tăng điểm số Test Loss/Acc offline | Tận dụng trọng số 32 epoch của Historical | **Khắc phục triệt để lỗi nhận diện thực tế trên Webcam** |
| **Khả năng tăng Accuracy Webcam** | Gần như bằng 0 (vẫn kẹt ở ~10%) | Rất ít (< 2–3%) | **Rất cao (có thể đưa Accuracy lên > 60–75%)** |
| **Yêu cầu tải dữ liệu** | Phải tải 1,67 GB | Phải tải 1,67 GB | **Không cần tải dataset lớn** |
| **Thời gian triển khai** | ~1 giờ (tải + train) | ~1.5 giờ (tải + train) | ~2–3 giờ (tối ưu hóa pipeline runtime) |
| **Tính khoa học & Minh bạch** | Cao (đầy đủ manifest/RNG) | Trung bình (thiếu lịch sử gốc) | Rất cao (giải quyết trực diện root cause) |
| **Quyết định đề xuất** | ❌ **Bác bỏ** | ⏸️ **Tạm hoãn** |  **DUYỆT THỰC HIỆN NGAY** |

---

## 3. Lộ trình hành động chi tiết (Action Plan)

Để nâng cao độ chính xác nhận diện từ vựng VSL trong môi trường thực tế, đề xuất triển khai theo 3 giai đoạn:

### Giai đoạn 1: Chuẩn hóa Pipeline Thu thập & Nhận diện Runtime (Ưu tiên số 1)
1. **Khóa chặt Tracker Engine cho chế độ `words`:**
   - Trong `CameraSignRecognizer.jsx`, khi `conversionMode === 'words'`, vô hiệu hóa khả năng người dùng chuyển sang Scope `hands`.
   - Nếu MediaPipe Holistic chưa nạp xong hoặc gặp lỗi: Hiển thị thông báo chờ, tuyệt đối không âm thầm fallback sang mảng giả lập `DEFAULT_POSE`.
2. **Cải tiến Cơ chế Cắt Chuỗi Cử chỉ (Gesture Boundary Detection / Spotting):**
   - Thay vì gửi sliding window mù quáng mỗi 6 frame:
     - Giám sát độ dịch chuyển của 2 cổ tay (`velocity`).
     - Khi vận tốc bàn tay vượt ngưỡng kích hoạt $\rightarrow$ Bắt đầu ghi chuỗi frame.
     - Khi vận tốc hạ xuống vùng tĩnh trong 3–5 frame $\rightarrow$ Cử chỉ đã kết thúc.
     - Lấy chuỗi frame vừa thu thập, nội suy tuyến tính (Linear Resampling) về đúng 30 frame giống hệt hàm `canonical_sequence` lúc training, rồi mới gửi API nhận diện.

### Giai đoạn 2: Kiểm thử Thực nghiệm lại với Word Debug (VSL-P012)
1. Chạy lại bộ test đối chứng 14 session với pipeline đã chuẩn hóa.
2. So sánh tỷ lệ Top-1 Accuracy trên webcam trước và sau khi có Gesture Spotting.
3. Dự kiến: Accuracy thực tế sẽ tăng vọt vì hình thái dữ liệu đưa vào mô hình đã đồng nhất với phân phối huấn luyện.

### Giai đoạn 3: Huấn luyện Tinh chỉnh hoặc Huấn luyện Mới (Sau khi Pipeline đã chuẩn)
1. Sau khi pipeline runtime đã ổn định và xác định được định dạng chuẩn, nếu cần mở rộng thêm từ vựng hoặc nâng cao độ nhạy:
2. Tải 1,67 GB dataset và áp dụng các kỹ thuật Data Augmentation thời gian (Dynamic Time Warping, Temporal Jittering, Frame Dropping mô phỏng webcam 15–30 FPS) trong script huấn luyện `train_sstcn.py`.
3. Huấn luyện mô hình SSTCN_Attention thế hệ mới có khả năng miễn nhiễm cao với sự biến thiên FPS của webcam.

