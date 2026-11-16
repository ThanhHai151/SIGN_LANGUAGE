# HƯỚNG DẪN THU THẬP VÀ ĐÓNG GÓI DỮ LIỆU ĐA NGƯỜI KÝ (MULTI-SIGNER BENCHMARK)

**Dự án:** VSL Connect AI  
**Mục tiêu:** Thu thập tập dữ liệu kiểm thử độc lập từ 3 – 5 người ký mới (chưa từng xuất hiện trong tập huấn luyện QIPEDC) nhằm đánh giá năng lực tổng quát hóa thực tế của checkpoint [`backend/models/alphabet_static_head_v1.pt`](file:///d:/SIGN_LANGUAGE/backend/models/alphabet_static_head_v1.pt).

---

## 1. Yêu cầu Người Ký và Thiết lập Ghi hình

### A. Đối tượng người ký (Signers)
- **Số lượng mục tiêu:** 3 đến 5 người tham gia độc lập (ví dụ: `signer_01`, `signer_02`, `signer_03`).
- **Đặc điểm:** Có sự đa dạng tự nhiên về kích thước bàn tay (nam/nữ), độ dài ngón tay và thói quen ra ký hiệu.
- **Tư thế chuẩn:** Thực hiện đúng theo quy chuẩn Bảng chữ cái Ngôn ngữ Ký hiệu Việt Nam (VSL) cho **22 chữ cái tĩnh nền**:
  $$\mathbf{A, B, C, D, E, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, X, Y}$$

### B. Thiết lập Môi trường và Camera
- **Khoảng cách tới camera:** Biến thiên tự nhiên trong khoảng $0.5\text{m} - 1.2\text{m}$ (đủ nhìn rõ từ ngực lên đến đỉnh đầu và trọn vẹn bàn tay).
- **Góc nghiêng bàn tay:** Để bàn tay tự nhiên, góc nghiêng tự nhiên trong khoảng $\pm 15^\circ$ (không ép buộc giữ tay cứng nhắc $90^\circ$).
- **Điều kiện ánh sáng:** Thu thập ở điều kiện ánh sáng phòng bình thường (ánh sáng ban ngày tự nhiên, đèn huỳnh quang văn phòng, hoặc đèn học).
- **Camera:** Webcam máy tính hoặc camera điện thoại tiêu chuẩn (tối thiểu 720p, 30 FPS).

---

## 2. Quy trình Thu thập Độc lập (Independent Trial Protocol)

> [!IMPORTANT]
> **Nguyên tắc chống rò rỉ dữ liệu:**
> Tuyệt đối **KHÔNG** giữ tay một lần rồi cắt hàng trăm frame liên tiếp! Làm như vậy chỉ thu được các frame trùng lặp và gây rò rỉ thông tin.
> Thay vào đó, mỗi chữ cái phải được thực hiện qua **nhiều lần độc lập (Independent Trials)**.

### Quy trình cho từng chữ cái:
1. **Lượt 1 (Trial 1):** Người ký đưa tay lên từ vị trí nghỉ, tạo dáng chữ cái chuẩn, giữ ổn định trong **2 – 3 giây**, sau đó **hạ tay xuống hoàn toàn**.
2. **Nghỉ ngắn 2 giây.**
3. **Lượt 2 (Trial 2):** Đưa tay lên lại, thay đổi góc nghiêng hoặc khoảng cách nhẹ, giữ ổn định **2 – 3 giây**, hạ tay xuống.
4. **Lượt 3 (Trial 3):** Lặp lại lượt thứ 3 độc lập.

**Chỉ tiêu định lượng:**
- Mỗi signer: 22 chữ cái $\times$ 3 lượt độc lập $= \mathbf{66\ \text{video clip ngắn}}$ (hoặc 66 đoạn hành động trong 1 video có đánh dấu timestamp).
- Với 3 – 5 signers: Tổng cộng khoảng **200 – 330 lượt thực hiện độc lập**.
- Từ mỗi lượt thực hiện chỉ trích xuất **3 – 5 frame đại diện ổn định nhất** (vùng giữa lượt, bỏ hoàn toàn lúc tay đưa lên và hạ xuống).
- Tổng số mẫu kiểm thử mục tiêu: khoảng **600 – 1,000 vector đặc trưng 72D độc lập**.

---

## 3. Cấu trúc Đặt tên và Thư mục

```
backend/data/alphabet_v2/multi_signer/
├── README.md                                # Hướng dẫn này
├── collection_schema.json                   # JSON Schema định dạng chuẩn
├── raw_videos/                              # Thư mục chứa video quay thô
│   ├── signer_01/
│   │   ├── signer01_A_trial1.mp4
│   │   ├── signer01_A_trial2.mp4
│   │   ├── signer01_A_trial3.mp4
│   │   └── ... (đủ 22 chữ)
│   ├── signer_02/
│   └── signer_03/
└── extracted_data/                          # Dữ liệu đặc trưng sau trích xuất
    ├── signer_01_extracted.json
    ├── signer_02_extracted.json
    └── vsl_multisigner_benchmark.npz        # File NPZ tổng hợp dùng cho kịch bản đánh giá
```

---

## 4. Trích xuất Landmark và Vector Đặc trưng 72D

Khi xử lý các video thu thập mới, **bắt buộc phải sử dụng chính xác hàm trích xuất đặc trưng hiện hành của dự án**:

```python
from backend.vision.feature_extractor import normalize_hand_landmarks

# landmarks: danh sách 21 điểm [x, y, z] do MediaPipe Hands trả về
feature_vector_72d, finger_states = normalize_hand_landmarks(landmarks)

# feature_vector_72d có shape (72,) float32
# - 63 giá trị đầu: tọa độ 21 điểm mốc relative với cổ tay (wrist)
# - 9 giá trị sau: các đặc trưng hình học khoảng cách và góc ngón tay chuẩn hóa
```

---

## 5. Chạy Kịch bản Đánh giá Độc lập

Sau khi đóng gói dữ liệu vào file NPZ hoặc JSON theo đúng cấu trúc, chạy công cụ kiểm thử:

```bash
# Kiểm tra kịch bản ở chế độ Dry-run (xác minh pipeline và checkpoint)
python backend/training/evaluate_alphabet_static_head_multisigner.py --dry_run

# Chạy đánh giá chính thức trên tập dữ liệu đa người ký
python backend/training/evaluate_alphabet_static_head_multisigner.py --dataset_path backend/data/alphabet_v2/multi_signer/extracted_data/vsl_multisigner_benchmark.npz
```

Kết quả đánh giá sẽ tự động được xuất ra:
- `backend/data/alphabet_v2/audit/multisigner_evaluation_report.json`
- `backend/data/alphabet_v2/audit/multisigner_evaluation_report.md`

