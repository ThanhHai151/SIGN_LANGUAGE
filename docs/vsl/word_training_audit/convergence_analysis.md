# PHÂN TÍCH HỘI TỤ MÔ HÌNH SSTCN (CONVERGENCE ANALYSIS)

**Dự án:** Nhận diện Ngôn ngữ Ký hiệu Việt Nam (VSL)  
**Mã nhiệm vụ:** VSL-P011 — Công việc 3  
**Tài liệu phân tích:**  
- `docs/sstcn/evaluation/sstcn_reconstructed_full/training_log.txt`  
- `docs/sstcn/evaluation/sstcn_reconstructed_full/history.csv`  
- `docs/sstcn/evaluation/sstcn_reconstructed_full/run_config.json`  
- `docs/sstcn/evaluation/sstcn_reconstructed_full/model_card.json`  
- `docs/sstcn/legacy/checkpoint_metadata.json`  
- `backend/models/weights/sign_sstcn_attention_model.pth` (Historical)  
- `backend/models/weights/sign_sstcn_reconstructed_last.pth` (Reconstructed Last)  
**Ngày thực hiện:** 07/11/2026  

---

## 1. Dữ liệu trích xuất chi tiết theo từng Epoch (Reconstructed Full)

Quá trình huấn luyện mô hình `SSTCN_Attention` (102 lớp, batch size 32, optimizer AdamW, learning rate 0.001, dropout 0.3, mixed precision CUDA AMP) trên tập dữ liệu đầy đủ (38.818 mẫu train, 1.407 mẫu validation):

| Epoch (1-based) | Index (0-based) | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Learning Rate | Thời gian (s) | Trạng thái Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0 | 3.1723 | 18.03% | 2.1986 | 38.73% | 0.001 | 88.1s | Khởi tạo |
| 2 | 1 | 2.0484 | 39.31% | 1.4175 | 59.91% | 0.001 | 86.6s | Cải thiện |
| 3 | 2 | 1.5326 | 52.27% | 1.0334 | 69.79% | 0.001 | 86.6s | Cải thiện |
| 4 | 3 | 1.2413 | 60.88% | 0.8352 | 74.63% | 0.001 | 86.9s | Cải thiện |
| 5 | 4 | 1.0491 | 66.46% | 0.9109 | 72.64% | 0.001 | 86.7s | Val loss tăng nhẹ |
| 6 | 5 | 0.9130 | 70.72% | 0.6763 | 80.81% | 0.001 | 86.9s | Cải thiện |
| 7 | 6 | 0.8255 | 73.45% | 0.6643 | 79.96% | 0.001 | 87.0s | Val loss giảm |
| 8 | 7 | 0.7383 | 76.00% | 0.5501 | 84.65% | 0.001 | 86.6s | Cải thiện mạnh |
| 9 | 8 | 0.6874 | 78.04% | 0.5185 | 84.43% | 0.001 | 86.5s | Val loss giảm tiếp |
| 10 | 9 | 0.6492 | 79.00% | 0.5042 | 84.15% | 0.001 | 86.9s | Val loss giảm tiếp |
| **11 (Best)** | **10** | **0.6055** | **80.20%** | **0.4711** | **85.71%** | **0.001** | **86.6s** | ⭐ **BEST CHECKPOINT** |
| 12 (Last) | 11 | 0.5684 | 81.50% | 0.5040 | 84.22% | 0.001 | 86.6s | `bad_epochs = 1` |

### Đánh giá kết quả đánh giá cuối cùng (Final Evaluation sau Epoch 12):
- **Tập Validation (1.407 mẫu):**
  - Accuracy: **85.7854%**
  - Top-5 Accuracy: **98.6496%**
  - Weighted F1: **0.8561**
- **Tập Test độc lập (1.420 mẫu):**
  - Accuracy: **88.6620%**
  - Top-5 Accuracy: **99.5070%**
  - Weighted F1: **0.8844**
  - *So với Historical đo lại trên cùng tập test:* Cao hơn **+2.04%** (88.66% vs 86.62%).

---

## 2. Best Epoch và Nguyên nhân dừng Training

1. **Best Epoch:**
   - Điểm số chọn checkpoint tốt nhất: Theo tiêu chí `(validation_accuracy, -validation_loss)`.
   - Vị trí đạt đỉnh: **Epoch 11 (zero-based index 10)** với Validation Accuracy đạt **85.71%** và Validation Loss đạt **0.4711**.
   - Checkpoint này đã được xuất thành: [`backend/models/weights/sign_sstcn_reconstructed_model.pth`](file:///E:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/models/weights/sign_sstcn_reconstructed_model.pth).
2. **Nguyên nhân dừng Training:**
   - **Hoàn thành đúng số epoch chỉ định:** Trong `run_config.json`, tham số `epochs` được thiết lập cố định là **12** (`"epochs": 12`). Trainer chạy hết 12 epoch và tự động kết thúc.
   - **KHÔNG PHẢI do Early Stopping:** Tham số `patience` trong cấu hình là `5`. Tại epoch 12 (index 11), bộ đếm suy giảm hiệu năng mới chỉ ghi nhận `bad_epochs = 1` (do val loss tăng từ 0.4711 lên 0.5040). Do đó, tiến trình dừng là do hết epoch được giao, không phải bị trigger dừng sớm.

---

## 3. Đánh giá trạng thái hội tụ (Convergence Evaluation)

Dựa trên diễn biến các đường cong học tập:

### A. Mô hình có bị Overfitting không?
- **Không có hiện tượng Overfitting nghiêm trọng:**
  - Tại epoch 11: Train loss = 0.6055, Val loss = 0.4711. (Val loss thậm chí còn thấp hơn Train loss, một phần do Dropout 0.3 chỉ bật lúc train và tập Val loại bỏ bớt các bản data augmentation phức tạp).
  - Khoảng cách giữa Train Accuracy (80.20%) và Val Accuracy (85.71%) hoàn toàn lành mạnh.
  - Tại epoch 12, Val loss có tăng nhẹ từ 0.4711 lên 0.5040 nhưng vẫn thấp hơn Train loss (0.5684).

### B. Mô hình đã hội tụ hoàn toàn (Converged) chưa?
- **Mô hình CHƯA hội tụ hoàn toàn (Under-trained):**
  1. **Độ dốc Train Loss:** Train loss vẫn đang giảm dốc đều từ 0.738 (epoch 8) $\rightarrow$ 0.687 $\rightarrow$ 0.649 $\rightarrow$ 0.605 $\rightarrow$ 0.568 (epoch 12). Chưa hề xuất hiện dấu hiệu đi ngang (plateau).
  2. **Tốc độ học (Learning Rate):** Suốt 12 epoch, LR vẫn cố định ở giá trị khởi tạo `0.001`. Bộ điều chỉnh `ReduceLROnPlateau(patience=3, factor=0.5)` chưa từng kích hoạt lần nào để hạ LR xuống `0.0005` hay `0.00025`.
  3. **Tiềm năng lý thuyết:** Nếu tiếp tục huấn luyện lên 30–40 epoch kết hợp giảm dần learning rate, Train Accuracy hoàn toàn có thể đạt 90–95% và Test Accuracy có thể chạm ngưỡng 90–92%.

### C. Ý nghĩa thực tế đối với độ chính xác Webcam (QUAN TRỌNG NHẤT):
- **Phát hiện then chốt:** Mặc dù Reconstructed Full còn dư địa cải thiện trên tập dữ liệu offline, việc tiếp tục train thêm 20 epoch **KHÔNG THỂ giải quyết được nghịch lý Accuracy Webcam 6–9%**.
- **Lý do:**
  - Mô hình đạt **88.66% trên tập test**, nhưng khi đưa vào thực tế webcam chỉ nhận diện đúng **10.3% (Holistic)** và **1.9% (Fallback)**.
  - Ngay cả mô hình Historical (vốn đã được train tới **32 epoch**, train loss xuống sâu hơn) khi đưa vào webcam cũng chỉ đạt **15.0% (Holistic)** và **3.0% (Fallback)**.
  - Sự sụp đổ độ chính xác từ 88% xuống 10% bắt nguồn từ **khoảng cách miền dữ liệu (Domain Gap)** về phân đoạn cử chỉ (Segmentation) và tốc độ khung hình (Temporal Sampling), hoàn toàn **không phải do mô hình thiếu epoch huấn luyện**.

---

## 4. Kiểm tra dữ liệu lịch sử huấn luyện của Historical Model

Kiểm tra checkpoint [`backend/models/weights/sign_sstcn_attention_model.pth`](file:///E:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/models/weights/sign_sstcn_attention_model.pth):

- **Thông số đọc được trong checkpoint nhị phân:**
  - `epoch`: 32 (mô hình đã được huấn luyện qua 33 chu kỳ: 0..32).
  - `loss`: `1.0744346380233765` (đây là training loss của epoch cuối).
  - `model_state_dict`: Có đầy đủ trọng số các lớp Conv, BatchNorm, Attention, Linear.
  - `optimizer_state_dict`: Chứa trạng thái Adam (lr = 0.0005, weight_decay = 0.0001).
- **Những thông tin BỊ THIẾU HOÀN TOÀN ở Historical Model:**
  - ❌ **Không có file log dạng text hay JSON:** Không có bảng lịch sử train/val loss qua từng epoch 0..31.
  - ❌ **Không có metadata tiền xử lý (`preprocessing`):** Không có thông tin về cách lấy mẫu frame, chuẩn hóa hay nguồn trích xuất MediaPipe.
  - ❌ **Không có scheduler state, scaler state, RNG state:** Không thể khôi phục lại trạng thái ngẫu nhiên hay lịch trình giảm LR của lần train cũ.
  - ❌ **Không có manifest danh sách tệp:** Không thể biết chính xác 38.818 mẫu nào đã được đưa vào lần train lịch sử này, và liệu có bị rò rỉ dữ liệu (data leakage) giữa tập train cũ với tập test hay không.
- **Tài liệu lịch sử còn sót lại:**
  - Ảnh đồ thị: [`docs/sstcn/legacy/training_log.png`](file:///E:/NAM4/SEMINAR/SIGN_LANGUAGE/docs/sstcn/legacy/training_log.png) (chỉ biểu diễn đường cong trực quan, không có số liệu thô).
  - Báo cáo kết quả: [`docs/sstcn/legacy/test_set_result_report_nhom3.docx`](file:///E:/NAM4/SEMINAR/SIGN_LANGUAGE/docs/sstcn/legacy/test_set_result_report_nhom3.docx) (chỉ báo cáo điểm test set 0.8577).

