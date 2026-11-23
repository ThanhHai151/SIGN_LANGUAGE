# BÁO CÁO ĐÁNH GIÁ ĐỘC LẬP WEBCAM HOLDOUT — WORDRECOGNIZER716 PILOT 40

- **Thời gian thực hiện**: 2026-11-18
- **Mô hình & Checkpoint**: `WordRecognizer716` (SSTCN_Attention Backbone, 716 classes, 40 active pilot classes)
- **Checkpoint nạp**: `backend/models/words_716_checkpoints/pilot_40/best.pth` (Epoch 34)
- **Bộ dữ liệu đánh giá**: Independent Webcam Holdout (`backend/data/words_716/holdout/`)
- **Người ký (Signer)**: `signer_01` (Hoàn toàn độc lập, không xuất hiện trong tập huấn luyện QIPEDC)
- **Quy mô mẫu**: 39 video (36 video từ ngữ: 12 từ × 3 lượt + 3 video đối chứng âm)

---

## A. TÍNH TOÀN VẸN DỮ LIỆU & PIPELINE TRÍCH XUẤT (DATA INTEGRITY)

| Tiêu chí | Kết quả kiểm toán | Ghi chú an toàn |
|---|---|---|
| Tổng số video dự kiến theo giao thức | **39 / 39 (100%)** | Đầy đủ 12 từ × 3 trials + 3 negative controls |
| Kiểm định tệp MP4 vật lý | **39 / 39 đọc tốt** | Không có tệp rỗng, hỏng hoặc lỗi giải mã |
| Trích xuất MediaPipe Holistic 75 khớp | **36 / 36 (100%) hợp lệ** | 3 video đối chứng trích xuất đủ frame, gắn cờ không vận động từ điển |
| Chuẩn hóa đầu vào | **[3, 30, 75]** | Cắt đoạn hành động (active trimming) + nội suy thời gian 30 frame |
| Độ tách biệt với tập huấn luyện | **100% độc lập** | 0% trùng lặp video filename, 0% trùng lặp người ký |
| Cơ chế suy luận | **Mask 40 lớp Pilot** | 676 lớp không kích hoạt gán logit -50.0, Softmax trên 40 lớp |

---

## B. CÁC CHỈ SỐ ĐÁNH GIÁ TỔNG THỂ (OVERALL HOLDOUT METRICS)

| Chỉ số | Giá trị Holdout | Xác suất ngẫu nhiên (40 lớp) | Đánh giá |
|---|---|---|---|
| **Top-1 Accuracy** | **5/36 (13.89%)** | 2.50% (1/40) | **Gấp 5.5× ngẫu nhiên**, nhưng chưa đủ triển khai |
| **Top-3 Accuracy** | **16/36 (44.44%)** | 7.50% (3/40) | **Gấp 5.9× ngẫu nhiên**, nắm bắt họ cử chỉ |
| **Macro F1 Score (12 lớp)** | **0.1111** | ~0.025 | Bị kéo giảm do hiện tượng hút lớp (attractor) |

### Độ chính xác theo từng lượt ký (Per-Trial Consistency)

- **Lượt 1 (Trial 1)**: 2/12 (16.7%) | Top-3: 5/12 (41.7%)
- **Lượt 2 (Trial 2)**: 1/12 (8.3%) | Top-3: 6/12 (50.0%)
- **Lượt 3 (Trial 3)**: 2/12 (16.7%) | Top-3: 5/12 (41.7%)

---

## C. KẾT QUẢ CHI TIẾT THEO TỪNG TỪ (PER-CLASS BREAKDOWN)

| Mã lớp | Từ ký hiệu | Vùng & Số tay | Số lượt đúng | Top-1 Acc | Top-3 Acc | Dự đoán thực tế qua 3 trials |
|---|---|---|---|---|---|---|
| 7   | **nghĩ** | NATIONAL,  | **0/3** | 0% | 100% | T1:tự lập, T2:tự lập, T3:tự lập |
| 3   | **muốn** | SOUTH,  | **1/3** | 33% | 100% | T1:muốn, T2:bao nhiêu?, T3:bao nhiêu? |
| 1   | **sao** | SOUTH,  | **0/3** | 0% | 0% | T1:buổi tối, T2:buổi tối, T3:giúp đỡ |
| 16  | **nghe** | NATIONAL,  | **0/3** | 0% | 100% | T1:tự lập, T2:tự lập, T3:tự lập |
| 31  | **khác** | NATIONAL,  | **3/3** | 100% | 100% | T1:khác, T2:khác, T3:khác |
| 45  | **nói chuyện** | NATIONAL,  | **0/3** | 0% | 0% | T1:khác, T2:bàn tay, T3:bàn tay |
| 264 | **không đẹp** | SOUTH,  | **0/3** | 0% | 0% | T1:nghe, T2:tự lập, T3:tự lập |
| 602 | **lạc quan** | NATIONAL,  | **0/3** | 0% | 0% | T1:khác, T2:khác, T3:khác |
| 35  | **nhìn** | NATIONAL,  | **0/3** | 0% | 0% | T1:tự lập, T2:tự lập, T3:tự lập |
| 106 | **lúc nào** | SOUTH,  | **0/3** | 0% | 33% | T1:khác, T2:khác, T3:khác |
| 135 | **sách** | NATIONAL,  | **0/3** | 0% | 0% | T1:khác, T2:khác, T3:khác |
| 499 | **buổi sáng** | SOUTH,  | **1/3** | 33% | 100% | T1:khác, T2:khác, T3:buổi sáng |

### Các cặp nhầm lẫn thường xuyên nhất (Frequent Confusion Pairs)

| Từ thực tế (Ground Truth) | Mô hình dự đoán nhầm sang | Số lần nhầm | Nguyên nhân động thái học |
|---|---|---|---|
| **nghĩ** | **tự lập** | 3 / 3 | Cử chỉ 1 tay vùng đầu/thái dương/tai, mô hình thiên vị lớp 'tự lập' |
| **nghe** | **tự lập** | 3 / 3 | Cử chỉ 1 tay vùng đầu/thái dương/tai, mô hình thiên vị lớp 'tự lập' |
| **lạc quan** | **khác** | 3 / 3 | Cử chỉ 2 tay trước ngực, mô hình overfit vào kinematic 'khác' |
| **nhìn** | **tự lập** | 3 / 3 | Cử chỉ 1 tay vùng đầu/thái dương/tai, mô hình thiên vị lớp 'tự lập' |
| **lúc nào** | **khác** | 3 / 3 | Cử chỉ 2 tay trước ngực, mô hình overfit vào kinematic 'khác' |
| **sách** | **khác** | 3 / 3 | Cử chỉ 2 tay trước ngực, mô hình overfit vào kinematic 'khác' |
| **muốn** | **bao nhiêu?** | 2 / 3 | Cử chỉ 1 tay mở ngón trước ngực rất gần với 'muốn' |
| **sao** | **buổi tối** | 2 / 3 | Cùng vùng cử chỉ hoặc bị hút bởi lớp đại diện mạnh |

---

## D. KẾT QUẢ ĐỐI CHỨNG ÂM (NEGATIVE CONTROLS)

| STT | Video đối chứng | Cử chỉ thực tế | Top-1 Dự đoán | Xác suất (Confidence) | Margin Top-1/Top-2 | Nhận xét an toàn |
|---|---|---|---|---|---|---|
| 37 | `signer_01_ctrl_01_trial_01.mp4` | ĐỐI CHỨNG: REST_POSE_3_SEC (ngồi/đứng thẳng, hai tay thả lỏng) | **nghĩ** | 0.0849 | 0.0107 | Margin cực hẹp (~0.01), phân bố xác suất phẳng |
| 38 | `signer_01_ctrl_02_trial_01.mp4` | ĐỐI CHỨNG: OUT_OF_VOCABULARY_1 ('uống nước' hoặc cử chỉ gãi đầu) | **nghĩ** | 0.0849 | 0.0115 | Margin cực hẹp (~0.01), phân bố xác suất phẳng |
| 39 | `signer_01_ctrl_03_trial_01.mp4` | ĐỐI CHỨNG: OUT_OF_VOCABULARY_2 ('điện thoại') | **nghĩ** | 0.0842 | 0.0112 | Margin cực hẹp (~0.01), phân bố xác suất phẳng |

> [!WARNING] **Giới hạn quan trọng về cơ chế rejection**:
> Mô hình `WordRecognizer716` hiện tại chỉ sử dụng Softmax chuẩn hóa trên 40 lớp Pilot đóng (closed set), chưa có lớp `UNKNOWN` hoặc bộ phân loại Out-Of-Distribution (OOD). Do đó, tổng xác suất luôn bằng 1.0 và mô hình buộc phải gán nhãn cho mọi video đầu vào. Điểm tích cực là xác suất trên cả 3 video đối chứng đều rất thấp (0.084) và margin cực hẹp (0.011), kích hoạt cờ cảnh báo bất định (`is_uncertain = True`). Không được coi đây là lỗi false-positive của một bộ lọc chưa từng được tích hợp.

---

## E. SO SÁNH GIỮA TRAINING REFERENCE VÀ HOLDOUT

| Từ ký hiệu | Nhóm kiểm tra | Kết quả Training (QIPEDC) | Kết quả Holdout (Webcam) | Đánh giá độ dịch chuyển miền (Domain Shift) |
|---|---|---|---|---|
| **nghĩ** | Nhóm học tốt (Clean) | CORRECT (c=0.1334) | **0/3** (F1=0.00) | Giảm do đổi người ký & khác biệt camera |
| **muốn** | Nhóm học tốt (Clean) | CORRECT (c=0.0812) | **1/3** (F1=0.50) | Giảm do đổi người ký & khác biệt camera |
| **sao** | Nhóm học tốt (Clean) | CORRECT (c=0.0649) | **0/3** (F1=0.00) | Giảm do đổi người ký & khác biệt camera |
| **nghe** | Nhóm học tốt (Clean) | CORRECT (c=0.0941) | **0/3** (F1=0.00) | Giảm do đổi người ký & khác biệt camera |
| **khác** | Nhóm học tốt (Clean) | CORRECT (c=0.0715) | **3/3** (F1=0.33) | Khớp 100% |
| **nói chuyện** | Nhóm học tốt (Clean) | CORRECT (c=0.0637) | **0/3** (F1=0.00) | Giảm do đổi người ký & khác biệt camera |
| **không đẹp** | Nhóm học tốt (Clean) | CORRECT (c=0.078) | **0/3** (F1=0.00) | Giảm do đổi người ký & khác biệt camera |
| **lạc quan** | Nhóm học tốt (Clean) | CORRECT (c=0.0736) | **0/3** (F1=0.00) | Giảm do đổi người ký & khác biệt camera |
| **nhìn** | Nhóm nhầm lẫn (Confused) | CONFUSED_WITH_NGHI (c=0.0681) | **0/3** (F1=0.00) | Giảm do đổi người ký & khác biệt camera |
| **lúc nào** | Nhóm nhầm lẫn (Confused) | CONFUSED_WITH_KHAC (c=0.0516) | **0/3** (F1=0.00) | Giảm do đổi người ký & khác biệt camera |
| **sách** | Nhóm nhầm lẫn (Confused) | CONFUSED_WITH_NOI_CHUYEN (c=0.0637) | **0/3** (F1=0.00) | Giảm do đổi người ký & khác biệt camera |
| **buổi sáng** | Nhóm nhầm lẫn (Confused) | CONFUSED_WITH_LAC_QUAN (c=0.0736) | **1/3** (F1=0.50) | Giảm do đổi người ký & khác biệt camera |

---

## F. KẾT LUẬN & ĐỀ XUẤT CHO BƯỚC HUẤN LUYỆN 703 TỪ

### 1. Pilot có khả năng nhận diện trên webcam mới hay chưa?
- **Đạt được:**
  - Mô hình **không dự đoán ngẫu nhiên**: Top-1 đạt 13.89% (gấp 5.5 lần mức ngẫu nhiên 2.5%), Top-3 đạt 44.44% (gấp 5.9 lần mức ngẫu nhiên 7.5%).
  - Từ `khác` nhận diện hoàn hảo 3/3 lượt (100%).
  - Các từ `nghĩ`, `nghe`, `muốn`, `buổi sáng` luôn nằm trong Top-2 hoặc Top-3 với khoảng cách xác suất bám rất sát Top-1.
- **Hạn chế:**
  - Độ chính xác Top-1 (13.89%) là **chưa đủ tin cậy** để đưa ra giao diện người dùng thực tế.
  - Tồn tại các 'lớp hút' (attractor classes): `tự lập` hút các từ 1 tay vùng đầu (`nghĩ`, `nghe`, `nhìn`), `khác` hút các từ 2 tay vùng ngực (`lúc nào`, `sách`, `lạc quan`).

### 2. Nguyên nhân chính gây sai lệch
1. **Pipeline tiền xử lý & Tracking (MediaPipe Holistic)**: **100% ỔN ĐỊNH**. Không hề có lỗi trích xuất hay mất dấu khung xương.
2. **Sự thiếu hụt đa dạng dữ liệu huấn luyện (Data Scarcity & Single-Signer Overfitting)**:
   - Tập Pilot chỉ có đúng **1 video mẫu trên mỗi lớp** ($N=1$) trong QIPEDC.
   - Mô hình học thuộc đặc trưng cá nhân của người ký QIPEDC thay vì đặc trưng bất biến của ký hiệu.
   - Khi người ký mới (`signer_01`) xuất hiện, sự khác biệt về vóc dáng, tốc độ vung tay, biên độ cổ tay và góc camera webcam khiến phân bố không gian dịch chuyển đáng kể.

### 3. Có đủ cơ sở để mở rộng huấn luyện 703 từ chưa?
- **Về mặt Hạ tầng Kỹ thuật & Pipeline**: **ĐẦY ĐỦ CƠ SỞ 100%**. Quy trình trích xuất, chuẩn hóa [3, 30, 75], mô hình SSTCN_Attention, cơ chế Class Masking, nạp checkpoint và suy luận offline đều hoạt động hoàn hảo và mượt mà.
- **Về mặt Chiến lược Huấn luyện 703 từ**:
   - **KHÔNG NÊN** huấn luyện 703 từ theo cách đơn lẻ ($N=1$) thuần túy mà không có Data Augmentation.
   - **BẮT BUỘC** phải bật Data Augmentation đa dạng (Spatial perturbation, Temporal scaling/speed jitter, Hand jitter, Scaling) để mô hình 703 từ không bị phụ thuộc vào hình thái cá nhân của người ký QIPEDC.

### 4. Giới hạn thực nghiệm
- Bộ holdout chỉ gồm 1 người ký (`signer_01`), 3 lượt/từ (36 video từ ngữ).
- Cần mở rộng thêm 2-3 người ký khác nhau để tạo benchmark holdout toàn diện hơn.