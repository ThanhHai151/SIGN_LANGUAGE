# BÁO CÁO BÀN GIAO DATASET V2 CHO 29 CHỮ CÁI VSL (TASK 5B)

**Dự án:** VSL Connect AI  
**Ngày thực hiện:** 30/10/2026  
**Đơn vị thực hiện:** Senior Computer Vision / AI Engineer  
**Trạng thái kiểm định:** **PASSED 100% (Xác minh không lỗi qua `validate_dataset_v2.py`)**

---

## 1. Tổng quan số lượng mẫu trước và sau đóng gói

| Thành phần dữ liệu | Trước đóng gói | Sau đóng gói | Ghi chú điều chỉnh kỹ thuật |
| :--- | :---: | :---: | :--- |
| **Mẫu tĩnh (22 chữ nền)** | 497 frames | **551 frames** | Lọc bỏ 59 frame ENTRY & 17 frame EXIT; Bổ sung **130 frame BASE_POSE** từ 7 chữ động |
| **Chuỗi động (7 chữ có dấu)** | 7 file rời | **1 file chuẩn (vsl_alphabet_motion_v2.json)** | 669 frames tổng, **359 frames hợp lệ**, phân tách rõ 4 pha ENTRY, BASE, DIACRITIC, EXIT |
| **Số chiều vector đặc trưng tĩnh** | 72 chiều | **72 chiều chuẩn hóa** | Tính toán trực tiếp qua `normalize_hand_landmarks()` chính thức của dự án |
| **Danh mục nhãn chuẩn** | 27 nhãn lỗi (có F,J,W,Z) | **22 nhãn nền + 7 nhãn động (đủ 29 chữ Việt)** | Tuyệt đối không chứa nhãn tiếng Anh ngoại lai hay từ vựng lẫn lộn |

---

## 2. Bổ sung các frame BASE_POSE vào 5 nhãn nền tương ứng

Toàn bộ **130 frame BASE_POSE** hợp lệ từ 7 chuỗi động đã được đưa vào dataset tĩnh với nhãn chữ nền tương ứng:

| Chữ cái động | Nhãn chữ nền được gán | Video nguồn | Số frame BASE_POSE bổ sung | Mốc frame trong video |
| :---: | :---: | :---: | :---: | :---: |
| **Ă** | **A** | `D0490B` | **19** | Frame 7 – 25 |
| **Â** | **A** | `D0491B` | **23** | Frame 27 – 49 |
| **Ê** | **E** | `D0497B` | **20** | Frame 8 – 27 |
| **Ô** | **O** | `D0507B` | **19** | Frame 16 – 34 |
| **Ơ** | **O** | `D0508N` | **9** | Frame 35 – 43 |
| **Ư** | **U** | `D0515N` | **14** | Frame 20 – 33 |
| **Đ** | **D** | `D0495` | **26** | Frame 38 – 63 |
| **Tổng cộng** | — | — | **130 frames** | *(Tất cả được gắn nhãn chữ nền, tuyệt đối không gán nhãn có dấu)* |

---

## 3. Phân bố mẫu của 22 nhãn tĩnh sau đóng gói (Tổng: 551 mẫu)

```
A: 64 mẫu (22 gốc + 19 từ Ă + 23 từ Â)       B: 22 mẫu          C: 20 mẫu
D: 46 mẫu (20 gốc + 26 từ Đ)                 E: 35 mẫu (15 gốc + 20 từ Ê)
G: 29 mẫu                                   H: 18 mẫu          I: 18 mẫu
K: 17 mẫu                                   L: 16 mẫu          M: 19 mẫu
N: 21 mẫu                                   O: 48 mẫu (20 gốc + 19 từ Ô + 9 từ Ơ)
P: 20 mẫu                                   Q: 19 mẫu          R: 14 mẫu
S: 15 mẫu                                   T: 20 mẫu          U: 32 mẫu (18 gốc + 14 từ Ư)
V: 17 mẫu                                   X: 25 mẫu          Y: 16 mẫu
```

---

## 4. Xử lý triệt để các nghi vấn đối chiếu số liệu (Mục 7)

1. **444 ảnh overlay trong Task 5A là gì?**
   - 444 là số lượng file ảnh đã được ghi xuống đĩa tại thời điểm lệnh kiểm tra tiến trình được thực thi khi tác vụ đang chạy.
   - Sau khi tiến trình hoàn tất, toàn bộ **669 ảnh overlay** (tương ứng 669 frame của cả 7 video động) đã được ghi trọn vẹn 100% vào thư mục `overlays/`.
2. **Số frame hợp lệ trong 7 chuỗi động:** Chính xác **359 frames**.
3. **Số frame BASE_POSE thực sự đưa vào dataset tĩnh:** Chính xác **130 frames** (tất cả đều có `quality_status: "ok"` và thuộc vùng ổn định).
4. **Số frame DIACRITIC_MOTION được giữ lại:** Chính xác **133 frames** (lưu trong `vsl_alphabet_motion_v2.json` để phục vụ huấn luyện máy trạng thái nét vẽ).
5. **Frame trùng lặp hoặc bỏ sót:** Không có frame nào bị bỏ sót. Trong metadata tĩnh, trường `is_representative` đã gắn cờ để phân biệt các frame gần như bất động ($\Delta < 0.015$) với các frame có độ biến thiên rõ nét.

---

## 5. Rủi ro Data Leakage và Khuyến nghị chia tập huấn luyện

- **Nguy cơ rò rỉ dữ liệu:** Do toàn bộ dataset hiện tại chỉ có **1 video duy nhất cho mỗi chữ** từ cùng một diễn viên, các frame liền kề có sự tương đồng giải phẫu rất lớn. Nếu chia ngẫu nhiên (Random Train/Test Split) theo frame, mô hình sẽ bị rò rỉ dữ liệu (Data Leakage) nghiêm trọng và báo cáo độ chính xác ảo 99%.
- **Khuyến nghị bắt buộc:**
  1. Giữ toàn bộ 551 mẫu tĩnh làm tập huấn luyện cơ sở kết hợp Data Augmentation có kiểm soát (xoay $\pm 10^\circ$, co giãn tỷ lệ lòng bàn tay).
  2. Tuyệt đối **không chia tập validation ngẫu nhiên từ cùng một video này**.
  3. Khi đánh giá hiệu năng (Evaluation), bắt buộc phải thu thập video từ người ký độc lập (người dùng thực tế hoặc diễn viên khác) để làm tập Test độc lập.

---

## 6. Danh mục tệp bàn giao hoàn chỉnh

1. [`backend/data/alphabet_v2/datasets/alphabet_labels_v2.json`](file:///d:/SIGN_LANGUAGE/backend/data/alphabet_v2/datasets/alphabet_labels_v2.json)  
   *(Định nghĩa chuẩn hóa 29 chữ cái, 22 nhãn nền và quy tắc ánh xạ)*
2. [`backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz`](file:///d:/SIGN_LANGUAGE/backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz)  
   *(551 mẫu vector đặc trưng 72D chuẩn hóa, 22 nhãn nền, kèm mốc xương 21 điểm và metadata)*
3. [`backend/data/alphabet_v2/datasets/vsl_alphabet_motion_v2.json`](file:///d:/SIGN_LANGUAGE/backend/data/alphabet_v2/datasets/vsl_alphabet_motion_v2.json)  
   *(7 chuỗi chuyển động hoàn chỉnh với 669 frames, 359 frames OK, phân đoạn 4 pha ENTRY/BASE/DIACRITIC/EXIT)*
4. [`backend/data/alphabet_v2/datasets/validate_dataset_v2.py`](file:///d:/SIGN_LANGUAGE/backend/data/alphabet_v2/datasets/validate_dataset_v2.py)  
   *(Kịch bản kiểm tra chất lượng và chống rò rỉ độc lập)*
5. [`backend/data/alphabet_v2/datasets/dataset_v2_report.json`](file:///d:/SIGN_LANGUAGE/backend/data/alphabet_v2/datasets/dataset_v2_report.json)  
   *(Báo cáo số liệu máy đọc định dạng JSON)*
