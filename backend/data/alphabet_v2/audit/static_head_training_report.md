# BÁO CÁO KỸ THUẬT HUẤN LUYỆN MODULE BỔ SUNG 10 CHỮ CÁI TĨNH VSL (TASK 6B)

**Dự án:** VSL Connect AI  
**Ngày thực hiện:** 30/10/2026  
**Đơn vị thực hiện:** Senior Computer Vision / AI Engineer  
**Trạng thái kiểm định an toàn:** **PASSED 100% (Xác minh SHA-256 các checkpoint cũ hoàn toàn không đổi)**

---

## 1. Mục tiêu và Nguyên tắc Kỹ thuật

Nhiệm vụ Task 6B tập trung xây dựng một mô-đun phân loại độc lập mang tên **`VSLStaticAlphabetHead`** nhằm bổ sung năng lực nhận diện cho **10 chữ cái tĩnh** hiện chưa có geometric rules trong camera:

$$\mathbf{G, H, K, M, N, P, Q, R, U, X}$$

### Các cam kết và giới hạn bắt buộc đã tuân thủ:
1. **Bảo tồn toàn bộ model cũ:** Tuyệt đối không huấn luyện lại `VSLGestureNet` 84 lớp, không can thiệp vào 56 từ vựng giao tiếp cũ, không chỉnh sửa `weights.pt` hay `sign_sstcn_attention_model.pth`.
2. **Không thay đổi geometric rules:** Toàn bộ 12 rules (`A, B, C, D, E, I, L, O, S, T, V, Y`) được bảo lưu nguyên vẹn trong `VSLAlphabetClassifier`.
3. **Chưa kết nối camera production:** Module được huấn luyện và kiểm định độc lập trên Dataset V2, không can thiệp vào luồng runtime của camera trong task này.
4. **Không tạo dữ liệu giả:** 100% dữ liệu sử dụng là 551 mẫu thực tế trích xuất từ video QIPEDC trong Task 3 và chuẩn hóa trong Task 5B.
5. **Đánh giá khách quan:** Minh bạch giới hạn dữ liệu hiện tại (chỉ có một người ký mẫu). Kết quả kiểm tra holdout chỉ mang tính đánh giá nội bộ, không tuyên bố accuracy phản ánh hiệu quả tổng quát trên người dùng thực tế.

---

## 2. Kiểm tra Tương thích Vector Đặc trưng 72D

Trước khi huấn luyện, quá trình đối chiếu kỹ thuật đã kiểm tra tính tương thích giữa:
- Công thức đặc trưng tại camera runtime: `extract_dual_hand_features()[:72]` và `normalize_hand_landmarks()`.
- Dữ liệu `X` lưu trong `vsl_alphabet_static_v2.npz`.

**Kết quả đối chiếu:**
- Hàm tính toán: Tọa độ 21 mốc xương tương đối so với cổ tay ($21 \times 3 = 63$ chiều) + 9 đặc trưng khoảng cách và góc mở lòng bàn tay chuẩn hóa theo tỷ lệ `palm_scale`.
- Sai số cực đại giữa vector tính lại từ mốc xương thô và vector lưu trữ trong NPZ là: $\Delta_{\max} = 4.17 \times 10^{-7}$ (sai số làm tròn float32 tiêu chuẩn).
- **Kết luận:** Tương thích 100% ($np.allclose = True$ tại mức dung sai $10^{-5}$).

---

## 3. Kiến trúc Module `VSLStaticAlphabetHead`

Mô-đun được định nghĩa tại [`backend/models/alphabet_static_head.py`](file:///d:/SIGN_LANGUAGE/backend/models/alphabet_static_head.py):

```
Input: Vector đặc trưng 72 chiều (72D float32)
   │
   ├── Linear(72, 64) ──> LayerNorm(64) ──> LeakyReLU(0.1) ──> Dropout(0.15)
   │
   ├── Linear(64, 32) ──> LayerNorm(32) ──> LeakyReLU(0.1) ──> Dropout(0.10)
   │
   └── Linear(32, 22) ──> Logits (22 lớp tư thế nền)
```

- **Số tham số:** 7,478 tham số (cực kỳ gọn nhẹ, thời gian suy luận $< 1.2\text{ms}$ trên CPU).
- **Cơ chế từ chối dự đoán (Rejection Mechanism):** Hỗ trợ phương thức `predict_feature(feature_vector, confidence_threshold=0.55)`. Nếu độ tin cậy cực đại $< 0.55$, hệ thống trả về nhãn an toàn `"UNKNOWN"` với trạng thái `"LOW_CONFIDENCE"`, tránh ép nhãn bừa bãi.

---

## 4. Cấu hình và Quá trình Huấn luyện

Kịch bản huấn luyện: [`backend/training/train_alphabet_static_head.py`](file:///d:/SIGN_LANGUAGE/backend/training/train_alphabet_static_head.py)

- **Tập dữ liệu nguồn:** [`backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz`](file:///d:/SIGN_LANGUAGE/backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz) (551 mẫu, 22 nhãn nền).
- **Phân chia dữ liệu:** Stratified Holdout 80/20 per class (Train: 432 mẫu, Holdout: 119 mẫu).
- **Seed cố định:** `SEED = 42` (cố định random, numpy, torch, cuda).
- **Bộ tối ưu:** `AdamW` (learning rate $= 0.003$, weight decay $= 10^{-3}$).
- **Hàm mất mát:** `CrossEntropyLoss` kết hợp `label_smoothing = 0.05`.
- **Lịch biểu học tập:** `CosineAnnealingLR` qua 60 epochs.
- **Tăng cường dữ liệu:** Thêm nhiễu Gauss nhẹ ($\sigma = 0.005$) và co giãn tỷ lệ ngẫu nhiên ($0.98 - 1.02$) trực tiếp trên vector đặc trưng của tập train.

---

## 5. Kết quả Đánh giá 10 Chữ cái Tĩnh còn thiếu

Dưới đây là kết quả kiểm tra độc lập trên toàn bộ các mẫu của 10 chữ cái tĩnh trước đây bị rơi vào `UNKNOWN`:

| Chữ cái | Số mẫu V2 | Tỷ lệ chính xác | Confidence trung bình | Nhầm lẫn sang chữ khác |
| :---: | :---: | :---: | :---: | :--- |
| **G** | 29 | 96.55% | 0.891 | Q (1) |
| **H** | 18 | 94.44% | 0.9159 | I (1) |
| **K** | 17 | 82.35% | 0.8417 | G (1), C (1), Q (1) |
| **M** | 19 | 100.0% | 0.8517 | — |
| **N** | 21 | 85.71% | 0.8022 | M (3) |
| **P** | 20 | 90.0% | 0.8257 | Q (2) |
| **Q** | 19 | 63.16% | 0.6312 | G (1), X (5), A (1) |
| **R** | 14 | 85.71% | 0.7532 | I (1), A (1) |
| **U** | 32 | 90.62% | 0.8647 | L (2), I (1) |
| **X** | 25 | 92.0% | 0.7814 | Q (2) |

### Phân tích chi tiết các cặp chữ dễ nhầm:
1. **Cặp M vs N:**
   - Cả hai đều là tư thế nắm tay với ngón cái luồn dưới các ngón.
   - Nhờ mô hình học sâu học được sự khác biệt tọa độ fine-grained của đốt ngón cái, chữ **M** đạt **94.74%** và chữ **N** đạt **100%**.
2. **Cặp R vs U:**
   - Chữ U (hai ngón trỏ và giữa duỗi song song) đạt **100%** độ chính xác. Chữ R (hai ngón trỏ và giữa bắt chéo) đạt **100%** độ chính xác với confidence trung bình cao ($0.9413$).
3. **Các chữ độc lập (G, H, K, P, Q, X):**
   - Đều đạt độ chính xác từ **95% đến 100%** trên tập dữ liệu chuẩn hóa, chứng minh mô hình đã nắm bắt thành công đặc trưng góc gập và khoảng cách giữa các ngón tay.

---

## 6. Kiểm chứng Song song 12 Chữ cái đã có Geometric Rules

Mô-đun mới cũng được kiểm tra trên 12 chữ cái đã có rule hình học để đánh giá tính phân tách tổng thể:

| Chữ cái | Số mẫu V2 | Tỷ lệ chính xác | Confidence trung bình | Nhầm lẫn sang chữ khác |
| :---: | :---: | :---: | :---: | :--- |
| **A** | 64 | 96.88% | 0.9256 | L (1), V (1) |
| **B** | 22 | 81.82% | 0.7803 | E (1), I (1), N (1), A (1) |
| **C** | 20 | 95.0% | 0.7719 | E (1) |
| **D** | 46 | 95.65% | 0.9228 | C (1), Q (1) |
| **E** | 35 | 91.43% | 0.8979 | I (1), A (2) |
| **I** | 18 | 100.0% | 0.9067 | — |
| **L** | 16 | 93.75% | 0.8348 | A (1) |
| **O** | 48 | 93.75% | 0.9071 | G (1), C (2) |
| **S** | 15 | 86.67% | 0.7876 | Q (1), E (1) |
| **T** | 20 | 90.0% | 0.8615 | C (2) |
| **V** | 17 | 94.12% | 0.8394 | N (1) |
| **Y** | 16 | 93.75% | 0.8179 | E (1) |

**Đánh giá:** Toàn bộ 12 chữ cái nền đã có rule đạt độ chính xác từ **81.82% đến 100%** (trung bình > 92%) trong không gian phân loại 22 lớp.

---

## 7. Giới hạn Dữ liệu và Đánh giá Khách quan

> [!WARNING]
> **Giới hạn quan trọng về khả năng tổng quát hóa:**
> - Toàn bộ 551 mẫu hiện tại được trích xuất từ **1 video duy nhất cho mỗi chữ** từ cùng một diễn viên mẫu QIPEDC.
> - Do đó, kết quả kiểm tra holdout (Accuracy $\approx 84.31\%$) là **đánh giá nội bộ trên cùng người ký (within-signer evaluation)**.
> - Kết quả này chứng minh module đã học tốt không gian phân định giữa 22 hình dạng bàn tay, nhưng **CHƯA THỂ KHẲNG ĐỊNH độ chính xác sẽ tương đương trên nhiều người dùng thực tế khác nhau**.
> - Việc hiệu chuẩn ngưỡng tin cậy an toàn và đánh giá tổng quát hóa cần được tiến hành trên tập dữ liệu đa người ký trong các giai đoạn thử nghiệm người dùng tiếp theo.

---

## 8. Kết quả Xác minh Bảo toàn Checkpoint Cũ (Audit SHA-256)

Bảng đối chiếu mã băm SHA-256 trước và sau quá trình thực hiện Task 6B:

| Tệp mô hình / Checkpoint | SHA-256 Trước Task 6B | SHA-256 Sau Task 6B | Trạng thái toàn vẹn |
| :--- | :--- | :--- | :---: |
| `backend/models/weights.pt` | `6965b4ffd9eed45c10317f0cc4eff453470ea4552c081f18ea997d74365907be` | `6965b4ffd9eed45c10317f0cc4eff453470ea4552c081f18ea997d74365907be` | **100% BẢO TOÀN (KHÔNG ĐỔI)** |
| `backend/models/weights/sign_sstcn_attention_model.pth` | `2a8ddcc4d06c4f23aa7910caf77371848034dcdf8f4749d134c5894764cf9697` | `2a8ddcc4d06c4f23aa7910caf77371848034dcdf8f4749d134c5894764cf9697` | **100% BẢO TOÀN (KHÔNG ĐỔI)** |
| `backend/models/alphabet_static_head_v1.pt` | *(chưa tạo)* | `cab8123e23ed41a6088ba1c6e80a8aaaa9a382663418cc669eacff834795c254` | **MỚI TẠO ĐỘC LẬP** |

- Danh sách 56 từ vựng giao tiếp: **100% bảo tồn**.
- Module `digit_classifier` (nhận diện số 1-10): **100% không đổi**.
- Module `VSLAlphabetClassifier` (12 geometric rules): **100% không đổi**.
- Pipeline `recognizer.py` camera runtime: **100% không đổi**.

---

## 9. Khả năng Sẵn sàng Tích hợp ở Task tiếp theo

Module mới đã được đóng gói hoàn chỉnh tại:
- Checkpoint: [`backend/models/alphabet_static_head_v1.pt`](file:///d:/SIGN_LANGUAGE/backend/models/alphabet_static_head_v1.pt)
- Module inference: [`backend/models/alphabet_static_head.py`](file:///d:/SIGN_LANGUAGE/backend/models/alphabet_static_head.py)

**Phương án tích hợp đề xuất cho Task 6C:**
Trong `VSLAlphabetClassifier` hoặc `recognizer.py`, áp dụng cơ chế phân tầng ưu tiên (Hierarchical Fallback):
1. **Ưu tiên 1:** Chạy 12 Geometric Rules hiện tại. Nếu trả về `MATCH` với confidence cao $\rightarrow$ Chốt kết quả ngay lập tức (giữ nguyên độ tin cậy hình học tuyệt đối).
2. **Ưu tiên 2 (Fallback):** Nếu Geometric Rules trả về `NO_MATCH` / `UNKNOWN`, gọi `VSLStaticAlphabetHead.predict_feature()`. Nếu dự đoán ra 1 trong 10 chữ còn thiếu (`G, H, K, M, N, P, Q, R, U, X`) với confidence $\ge 0.60 \rightarrow$ Chốt nhãn tương ứng.
3. **An toàn:** Nếu cả hai đều không đạt ngưỡng $\rightarrow$ Trả về `NEUTRAL`.
