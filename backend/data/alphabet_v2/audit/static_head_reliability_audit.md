# BÁO CÁO KIỂM TOÁN ĐỘ TIN CẬY KẾT QUẢ HUẤN LUYỆN VSLStaticAlphabetHead (TASK 6B.1)

**Dự án:** VSL Connect AI  
**Ngày kiểm toán:** 30/10/2026  
**Chế độ thực hiện:** **READ-ONLY AUDIT (Không sửa code, không huấn luyện lại, không thay đổi checkpoint)**  
**Kết luận tổng thể:** **`NEEDS DATA VALIDATION` (CẦN BỔ SUNG XÁC MINH DỮ LIỆU TRƯỚC KHI TÍCH HỢP)**

---

## 1. Phân loại Kết luận Kiểm toán

| Hạng mục kiểm toán | Phân loại | Tóm tắt phát hiện |
| :--- | :---: | :--- |
| **Cách tính Holdout Accuracy 84.31%** | `VERIFIED` | Tính chính xác trên đúng 102 mẫu holdout (86 đúng / 102 mẫu →= 84.31\%→) |
| **Bản chất Bảng Accuracy từng chữ Task 6B** | `VERIFIED` | Bảng từng chữ trong Task 6B được tính trên **toàn bộ 551 mẫu** (Train + Holdout), không phải chỉ 102 mẫu holdout |
| **Rủi ro Rò rỉ Dữ liệu (Data Leakage)** | `WARNING` | Phân chia ngẫu nhiên từng frame từ 1 video làm 100% mẫu holdout có frame liền kề trong tập train (→\Delta_{\text{frame}} \le 2→) |
| **Độ tin cậy của 10 chữ cái mới** | `WARNING` | 9/10 chữ đạt tốt trên holdout nội bộ; riêng **chữ Q** độ tin cậy thấp, nhầm nặng sang **X** |
| **Bảo toàn Model và Checkpoint cũ** | `VERIFIED` | 100% mã băm SHA-256 của `weights.pt` và `SSTCN.pth` nguyên vẹn; 12 rules, 56 từ vựng không đổi |
| **Tính sẵn sàng tích hợp Camera** | `NEEDS DATA VALIDATION` | Cần hiệu chuẩn lại ngưỡng an toàn riêng cho chữ Q hoặc tạm thời tắt Q trước khi tích hợp |

---

## 2. Phần A — Kiểm tra Cách chia Dữ liệu và Phân tích Rò rỉ (Data Leakage)

### A. Cơ chế Phân chia
- **Phương pháp:** Stratified Random Split 80/20 được thực hiện **trên từng frame độc lập (frame-level)** trong file [`train_alphabet_static_head.py`](file:///d:/SIGN_LANGUAGE/backend/training/train_alphabet_static_head.py).
- **Cơ chế Seed:** Sử dụng `rng = np.random.RandomState(SEED + c_idx)` với `SEED = 42`.
- **Số lượng mẫu:** Tập huấn luyện (Train) gồm **449 mẫu**, Tập đánh giá nội bộ (Holdout) gồm **102 mẫu**. Tổng cộng: 551 mẫu.

### B. Nguy cơ Rò rỉ Dữ liệu Tạm thời (Temporal Data Leakage)
- **Bản chất Dataset V2:** Hiện tại mỗi chữ cái tĩnh chỉ có **1 video duy nhất** từ một diễn viên mẫu QIPEDC.
- **Phân tích khoảng cách khung hình:** Đối chiếu `frame_index` trong siêu dữ liệu `metadata_json` giữa tập Train và Holdout:
  - **100% số mẫu trong tập Holdout (102/102 mẫu)** đều có ít nhất một khung hình liền kề trong tập Train với khoảng cách →\Delta_{\text{frame}} \le 2→ (tương đương chênh lệch thời gian →\Delta t \le 0.08\text{s}→).
  - Độ dịch chuyển mốc xương giữa các frame liền kề này cực kỳ nhỏ (trung bình →\Delta < 0.035→).
- **Hệ quả kỹ thuật:** 
  - Mô hình học sâu nhận diện các frame holdout không phải vì nó đã học được đặc trưng khái quát của chữ cái cho mọi người dùng, mà vì nó nhận diện lại **cùng một bàn tay của cùng một người ở thời điểm cách đó 1/25 giây**.
  - Đây chính là lý do vì sao Holdout Accuracy đạt con số cao **84.31%**. Con số này là **độ chính xác nội bộ cực kỳ lạc quan (optimistic within-signer accuracy)**, không phản ánh độ chính xác thực tế khi gặp người dùng mới.

---

## 3. Phần B — Đối chiếu và Xác minh Accuracy

Kiểm toán đã tính toán độc lập 3 thước đo trên mô hình `alphabet_static_head_v1.pt`:

| Tập dữ liệu đánh giá | Số lượng mẫu | Số mẫu đoán đúng | Độ chính xác (Accuracy) | Ghi chú |
| :--- | :---: | :---: | :---: | :--- |
| **Tập Train thuần túy** | 449 mẫu | 419 | **93.32%** | Độ chính xác trên tập đã học |
| **Tập Holdout thuần túy** | 102 mẫu | 86 | **84.31%** | Trùng khớp 100% với báo cáo Task 6B |
| **Toàn bộ Dataset V2** | 551 mẫu | 505 | **91.65%** | **Nguồn gốc thực sự của bảng từng chữ trong Task 6B** |

### Xác minh sai lệch báo cáo Task 6B:
- **Phát hiện:** Trong báo cáo Task 6B, bảng accuracy từng chữ ghi số lượng mẫu là →N = 29→ cho G, →N = 19→ cho Q, →N = 64→ cho A,... Tổng số mẫu là **551 mẫu**.
- **Kết luận:** Bảng accuracy từng chữ trong Task 6B được đánh giá trên **toàn bộ 551 mẫu** (chứ không phải chỉ tính trên 102 mẫu holdout). Điều này giải thích tại sao accuracy từng lớp trong bảng Task 6B dao động từ 82% đến 100% (bởi vì 81.5% số mẫu trong đó nằm trong tập train).

---

## 4. Phần C — Kiểm tra Chi tiết 10 Chữ cái Mới trên Tập Holdout Thuần túy

Dưới đây là kết quả kiểm toán **hoàn toàn trên tập Holdout (102 mẫu)** cho 10 chữ cái mới:

| Chữ cái | Số mẫu Train | Số mẫu Holdout | Đúng / Tổng Holdout | Recall | Precision | F1-Score | Confidence TB | Nhầm sang chữ khác |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **G** | 24 | 5 | 4/5 | 80.0% | 80.0% | 80.0% | 0.7787 | Q (1) |
| **H** | 15 | 3 | 2/3 | 66.67% | 100.0% | 80.0% | 0.8747 | I (1) |
| **K** | 14 | 3 | 1/3 | 33.33% | 100.0% | 50.0% | 0.603 | C (1), G (1) |
| **M** | 16 | 3 | 3/3 | 100.0% | 60.0% | 75.0% | 0.803 | — |
| **N** | 17 | 4 | 2/4 | 50.0% | 100.0% | 66.67% | 0.6189 | M (2) |
| **P** | 16 | 4 | 3/4 | 75.0% | 100.0% | 85.71% | 0.8 | Q (1) |
| **Q** | 16 | 3 | 1/3 | 33.33% | 33.33% | 33.33% | 0.6417 | X (2) |
| **R** | 12 | 2 | 1/2 | 50.0% | 100.0% | 66.67% | 0.5202 | I (1) |
| **U** | 26 | 6 | 5/6 | 83.33% | 100.0% | 90.91% | 0.7793 | L (1) |
| **X** | 20 | 5 | 5/5 | 100.0% | 71.43% | 83.33% | 0.7143 | — |

*(Ghi chú đối chiếu: 12 chữ cái nền đã có rule cũng được kiểm toán trên holdout với kết quả tương tự: A đạt 12/12 (100%), B đạt 3/4 (75%), C đạt 3/4 (75%), D đạt 8/9 (88.9%), I đạt 3/3 (100%), L đạt 3/3 (100%), O đạt 9/9 (100%), V đạt 3/3 (100%), Y đạt 3/3 (100%)).*

---

## 5. Phân tích Sâu Chữ Q và Nguy cơ Nhầm lẫn với Chữ X

### A. Thực trạng nhận diện chữ Q
- Trong tập Holdout (3 mẫu):
  - 1 mẫu đoán đúng là **Q** (Confidence: →0.7831→).
  - 2 mẫu bị đoán nhầm sang **X** (Confidence: →0.6214→ và →0.5847→).
  - **Recall trên Holdout của Q chỉ đạt 33.33%**.
- Trên toàn bộ 551 mẫu:
  - Chữ Q có 19 mẫu, chỉ đoán đúng 12 mẫu (**63.16%**).
  - Có tới **5 mẫu Q bị nhận nhầm thành X**.
  - Ngược lại, có **2 mẫu X bị nhận nhầm thành Q**.

### B. Nguyên nhân Giải phẫu học và Thị giác Máy tính
- Trong bảng chữ cái VSL, chữ **Q** có tư thế ngón trỏ và ngón cái hướng chúc xuống phía dưới, đốt ngón trỏ hơi khum cong.
- Chữ **X** có ngón trỏ gập móc (hooked index).
- Khi MediaPipe trích xuất landmark 2D/3D từ góc camera chính diện, độ gập chúc xuống của ngón trỏ chữ Q bị chiếu vuông góc lên mặt phẳng ảnh, tạo ra vector hình học ngón trỏ gần như trùng khớp với ngón trỏ chữ X.

### C. Đánh giá Ngưỡng Confidence 0.60 cho Chữ Q
- Các trường hợp Q bị nhầm sang X có confidence dao động từ **→0.58→ đến →0.68→**.
- Do đó, **ngưỡng phẳng →0.60→ KHÔNG ĐỦ AN TOÀN để nhận diện chữ Q**. Nếu áp dụng ngưỡng 0.60 vào camera, hệ thống sẽ liên tục trả về kết quả sai giữa Q và X.

---

## 6. Phần D — Đánh giá Khả năng Tổng quát hóa và Dữ liệu Cần bổ sung

1. **Vì sao chia frame ngẫu nhiên tạo ra kết quả lạc quan?**
   - Các frame trong cùng 1 video cách nhau 0.04s mang cùng đặc điểm giải phẫu (chiều dài ngón tay, kích thước mu bàn tay, tông màu da, điều kiện ánh sáng). Mạng nơ-ron có thể dễ dàng "học vẹt" (memorize) các đặc trưng riêng của người mẫu này thay vì hình dạng ngón tay tổng quát.
2. **Hạn chế của dữ liệu 1 người ký:**
   - Người ký trong QIPEDC thực hiện tư thế rất chuẩn, nhưng người dùng thực tế sẽ có biên độ gập ngón khác nhau, góc nghiêng bàn tay khác nhau và tốc độ khác nhau.
3. **Dữ liệu cần bổ sung trước khi đánh giá khách quan:**
   - Cần tối thiểu **3 đến 5 người ký độc lập (multi-signer dataset)** thực hiện 29 chữ cái.
   - Khi đó, việc chia tập Train / Test bắt buộc phải thực hiện theo **Signer-Independent Split (Leave-One-Signer-Out)**: huấn luyện trên người A, B, C và kiểm thử trên người D.

---

## 7. Phần E — Kiểm tra Toàn vẹn Hệ thống

Kiểm toán xác nhận bằng chứng kỹ thuật:
- Checkpoint [`alphabet_static_head_v1.pt`](file:///d:/SIGN_LANGUAGE/backend/models/alphabet_static_head_v1.pt): Đúng **22 lớp**, đúng **72 chiều**, thứ tự nhãn trùng khớp →1:1→ với [`alphabet_labels_v2.json`](file:///d:/SIGN_LANGUAGE/backend/data/alphabet_v2/datasets/alphabet_labels_v2.json).
- Phiên bản Feature Extractor: Khớp hoàn toàn với hàm `normalize_hand_landmarks()` runtime.
- **Tính toàn vẹn Checkpoint cũ:**
  - `backend/models/weights.pt`: SHA-256 là `6965b4ffd9eed45c10317f0cc4eff453470ea4552c081f18ea997d74365907be` (**Khớp 100%**).
  - `sign_sstcn_attention_model.pth`: SHA-256 là `2a8ddcc4d06c4f23aa7910caf77371848034dcdf8f4749d134c5894764cf9697` (**Khớp 100%**).
- 12 Geometric Rules, Digit Classifier và luồng runtime `recognizer.py`: **Hoàn toàn không bị sửa đổi**.

---

## 8. Phần F — Kiểm tra Phương án Tích hợp và Kiến nghị

### A. Phân biệt UNKNOWN và NEUTRAL trong Codebase
- **`NEUTRAL`:** Là trạng thái người dùng thả lỏng tay, chưa ra hiệu hoặc camera chưa thấy bàn tay hợp lệ.
- **`UNKNOWN`:** Là trạng thái bàn tay đã đưa lên nhưng bộ phân loại không đủ bằng chứng tin cậy để gán nhãn, hoặc phát hiện xung đột hình học.

### B. Đánh giá Cơ chế Phân tầng (Hierarchical Fallback)
Phương án phân tầng đề xuất là **hoàn toàn hợp lý về mặt kiến trúc**, nhưng cần điều chỉnh quy tắc kích hoạt:
1. **Ưu tiên 1:** 12 Geometric Rules luôn chạy trước. Nếu `MATCH` →
ightarrow→ Chốt nhãn.
2. **Ưu tiên 2 (Fallback cho chữ thiếu):** Chỉ kích hoạt `VSLStaticAlphabetHead` khi rules trả về `NO_MATCH`.
3. **Điều chỉnh an toàn cho 10 chữ thiếu:**
   - Nhóm có độ tin cậy cao (**G, H, M, P, U, X**): Có thể cho phép fallback với ngưỡng →confidence \ge 0.70→.
   - Nhóm có độ tin cậy trung bình (**K, N, R**): Đặt ngưỡng an toàn cao →confidence \ge 0.80→.
   - **Riêng chữ Q:** **KHUYẾN NGHỊ TẠM THỜI CHƯA BẬT (HOẶC GIỮ Ở TRẠNG THÁI EXPERIMENTAL CÓ CẢNH BÁO)** cho đến khi có thêm dữ liệu video phân tách rõ ràng giữa Q và X.

---

## 9. KẾT LUẬN CUỐI CÙNG

→→\mathbf{NEEDS\ \ DATA\ \ VALIDATION}→→

*(CẦN BỔ SUNG XÁC MINH DỮ LIỆU TRƯỚC KHI TÍCH HỢP PRODUCTION)*

**Lý do:**
1. Module `VSLStaticAlphabetHead` đã hoàn thành xuất sắc về mặt kiến trúc, code sạch, độc lập, bảo toàn 100% hệ thống cũ.
2. Tuy nhiên, độ tin cậy 84.31% bị rò rỉ thông tin tạm thời do chia frame ngẫu nhiên từ 1 video đơn signer.
3. Chữ Q có nguy cơ xung đột thị giác rất cao với chữ X và chưa đạt độ tin cậy thực tế.
4. Cần bổ sung kiểm thử trên video thực tế hoặc điều chỉnh ngưỡng tin cậy phân tầng trước khi đưa vào luồng camera chính thức.
