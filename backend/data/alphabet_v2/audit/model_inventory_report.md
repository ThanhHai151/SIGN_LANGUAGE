# BÁO CÁO TOÀN DIỆN TASK 6A: KIỂM KÊ MODEL HIỆN TẠI VÀ XÁC ĐỊNH CHỮ CÁI VSL CÒN THIẾU

**Dự án:** VSL Connect AI  
**Ngày thực hiện:** 30/10/2026  
**Đơn vị thực hiện:** Senior Computer Vision / AI Engineer  
**Định hướng chiến lược bắt buộc:** **GIỮ NGUYÊN MODEL HIỆN TẠI, BẢO TOÀN TOÀN BỘ 56 TỪ VỰNG VÀ 10 CHỮ SỐ, CHỈ BỔ SUNG KIẾN THỨC CÒN THIẾU CHO ĐỦ 29 CHỮ CÁI TIẾNG VIỆT.**

---

## 1. Tóm tắt điều hành (Executive Summary)

Sau khi kiểm tra trực tiếp mã nhị phân checkpoint, cấu trúc mạng nơ-ron, tập dữ liệu huấn luyện và toàn bộ đường ống camera thực tế, chúng tôi xác nhận các phát hiện cốt lõi sau:

1. **Mô hình tĩnh `weights.pt` (`VSLGestureNet`):**
   - Checkpoint có đúng **84 output classes** (27 chữ cái A-Z, Đ; 56 từ vựng giao tiếp; 1 lớp NEUTRAL).
   - **Thực trạng dữ liệu cũ:** Toàn bộ 12,600 mẫu trong `vsl_dataset.npz` (150 mẫu/lớp) đều là **dữ liệu tổng hợp nhân tạo (synthetic heuristic)** do `download_dataset.py` sinh ra bằng toán học và nhiễu ngẫu nhiên, hoàn toàn không phải landmark MediaPipe trích xuất từ video người Điếc thật.
   - **Sự thiếu hụt:** Model tĩnh hoàn toàn **KHÔNG CÓ 6 chữ cái tiếng Việt có dấu**: `Ă, Â, Ê, Ô, Ơ, Ư`. Ngoài ra, 4 chữ cái tiếng Anh ngoại lai (`F, J, W, Z`) đang tồn tại trong danh mục output class.
2. **Mô hình động `sign_sstcn_attention_model.pth` (`SSTCN_Attention`):**
   - Checkpoint pretrained có **102 output classes** theo giáo trình VSL chuẩn (29 chữ cái, 10 chữ số, 63 từ vựng đời sống).
   - **Giới hạn thực tế:** Mạng yêu cầu đầu vào chuỗi **30 frames x 75 joints** (33 mốc cơ thể Pose + 21 mốc tay trái + 21 mốc tay phải). Trong khi đó, giao diện camera hiện tại ở chế độ bảng chữ cái (`letters`) chỉ chạy MediaPipe Hands trích xuất **21 mốc bàn tay** trên từng frame độc lập. Do đó, **SSTCN_Attention hoàn toàn KHÔNG được kích hoạt trong luồng nhận diện camera chữ cái trực tiếp**.
3. **Đường nhận diện camera thực tế (`recognizer.py`):**
   - Khi người dùng chọn chế độ `letters`, hệ thống **bỏ qua hoàn toàn mạng nơ-ron MLP `VSLGestureNet`** (để tránh lỗi mạng MLP đoán đè `Z` lên chữ `A`), và cũng **không gọi SSTCN**.
   - Thay vào đó, luồng camera gọi trực tiếp vào bộ phân loại hình học độc lập `VSLAlphabetClassifier`.
   - **Lỗ hổng lớn:** `VSLAlphabetClassifier` hiện chỉ định nghĩa quy tắc hình học cho **12 chữ cái**: `A, B, C, D, E, I, L, O, S, T, V, Y`. Toàn bộ **17 chữ cái còn lại** (10 chữ tĩnh `G, H, K, M, N, P, Q, R, U, X` và 7 chữ động `Ă, Â, Ê, Ô, Ơ, Ư, Đ`) đều bị rơi vào trạng thái `UNKNOWN`/`NO_MATCH` và không thể nhận diện được.
4. **Giải pháp kiến trúc:** Tận dụng **Dataset V2** (đã chuẩn hóa 551 mẫu tĩnh thật và 7 chuỗi động thật) để bổ sung trọn vẹn 29 chữ cái thông qua cơ chế phân lớp mô-đun (Modular Classifier Head & Diacritic Motion Engine) mà **tuyệt đối không huấn luyện lại từ đầu**, bảo tồn 100% trọng số của 56 từ vựng cũ.

---

## 2. Kiểm kê Model Tĩnh (`weights.pt` & `VSLGestureNet`)

### A. Cấu trúc Checkpoint và Lớp Ngõ Ra
- **Vị trí file:** [`backend/models/weights.pt`](file:///d:/SIGN_LANGUAGE/backend/models/weights.pt) (Kích thước: 369,179 bytes).
- **Kiến trúc:** `VSLGestureNet` với đầu vào 72 chiều (63 tọa độ chuẩn hóa + 9 đặc trưng hình học), 2 khối `ResidualBlock` (128 chiều), tầng phân loại cuối `Linear(64, 84)`.
- **Trọng số thực tế:** Ma trận trọng số cuối `classifier.2.weight` có kích thước chính xác là `torch.Size([84, 64])` và bias `torch.Size([84])`.
- **Tổng số lớp:** Đúng **84 classes**, bao gồm:
  - **Index 0 – 26 (27 lớp):** Bảng chữ cái A-Z và chữ `Đ`:
    `A, B, C, D, Đ, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z`
  - **Index 27 – 82 (56 lớp):** Từ vựng giao tiếp VSL:
    `XIN_CHAO, CAM_ON, XIN_VUI_LONG, TAM_BIET, XIN_LOI, DONG_Y, KHONG, YEU_THUONG, GIUP_DO, TOI, BAN, AN_COM, UONG_NUOC, BENH_VIEN, GIA_DINH, HOC_BAI, NAU_AN, LAU_NHA, DI_NGU, RUA_TAY, DOC_SACH, VIET_BAI, LAM_VIEC, DI_HOC, TAM, KHOE_MANH, MET_MOI, VUI_VE, BUON, HOI, TRA_LOI, CHO, XIN, CUOI_TUAN, LICH_TRINH, PHU_THUOC, BAN_RON, TRONG_TUAN, THOI_GIAN, IT, DU_AN, CA_NHAN, DAM_BAO, DU, THU_GIAN, NEU, CO_THE, THICH, GAP_GO, BAN_BE, KHAM_PHA, DIA_DIEM, MOI, CO_HOI, THOI_QUEN, HANG_NGAY`
  - **Index 83 (1 lớp):** Trạng thái chờ `NEUTRAL`.

### B. Bản chất của tập dữ liệu huấn luyện cũ `backend/data/vsl_dataset.npz`
- File `backend/data/vsl_dataset.npz` chứa đúng 12,600 mẫu (mỗi lớp trong 84 lớp có đúng 150 mẫu).
- **Phát hiện quan trọng:** Phân tích mã nguồn [`backend/training/download_dataset.py`](file:///d:/SIGN_LANGUAGE/backend/training/download_dataset.py) cho thấy toàn bộ các mẫu này được sinh ra bởi hàm toán học `generate_base_hand_landmarks()` kết hợp với việc cộng nhiễu Gauss ($\sigma = 0.015$), xoay góc ($\pm 15^\circ$) và co giãn tỷ lệ (0.9 – 1.1).
- **Hậu quả:** Mô hình MLP `VSLGestureNet` khi huấn luyện trên dữ liệu nhân tạo này có hiện tượng không tương thích với mốc xương người thật từ MediaPipe, dẫn đến suy luận sai lệch khi camera đưa tay chữ `A` lại dự đoán thành `Z`.

---

## 3. Kiểm kê Model Động (`sign_sstcn_attention_model.pth` & `SSTCN_Attention`)

- **Vị trí file:** [`backend/models/weights/sign_sstcn_attention_model.pth`](file:///d:/SIGN_LANGUAGE/backend/models/weights/sign_sstcn_attention_model.pth) (8,186,546 bytes).
- **Kiến trúc:** `SSTCN_Attention` (Separable Spatial-Temporal Convolution Network kết hợp Multihead Joint Attention riêng cho 2 bàn tay).
- **Đầu vào yêu cầu:** Tensor kích thước `(Batch, Channels=3, Time=30, Joints=75)`.
  - 75 khớp gồm: 33 khớp thân trên (MediaPipe Pose 0-32), 21 khớp bàn tay trái (33-53), 21 khớp bàn tay phải (54-74).
- **Số lớp đầu ra:** **102 lớp** được định nghĩa trong [`backend/models/vsl_102_dictionary.py`](file:///d:/SIGN_LANGUAGE/backend/models/vsl_102_dictionary.py):
  - Index 0 – 28: Đầy đủ **29 chữ cái tiếng Việt** (A, Ă, Â, B, C, D, Đ, E, Ê, G, H, I, K, L, M, N, O, Ô, Ơ, P, Q, R, S, T, U, Ư, V, X, Y).
  - Index 29 – 38: 10 chữ số ("Số 1" đến "Số 10").
  - Index 39 – 101: 63 từ vựng giáo trình VSL.
- **Đánh giá riêng cho 7 chữ động (`Ă, Â, Ê, Ô, Ơ, Ư, Đ`):**
  - Trong danh mục 102 lớp, 7 chữ này đều có tên và chỉ mục (Ă: 1, Â: 2, Đ: 6, Ê: 8, Ô: 17, Ơ: 18, Ư: 25).
  - Tuy nhiên, checkpoint này chỉ hoạt động khi có đủ chuỗi 30 frame của toàn thân (75 joints). Luồng camera chữ cái hiện tại **không thu thập pose**, do đó SSTCN không thể được sử dụng trực tiếp để nhận diện chữ cái trong chế độ camera hiện thời.

---

## 4. Đường nhận diện Camera thực tế (Runtime Inference Flow)

Sơ đồ luồng xử lý từ Camera đến kết quả giao diện:

```mermaid
flowchart TD
    Cam[Webcam Browser] --> MP[MediaPipe Hands WASM]
    MP --> |21 Landmark 3D| API["POST /api/recognize_landmarks"]
    API --> ModeCheckKiểm tra Chế độ (mode)
    
    ModeCheck --> |mode == 'digits'| DigDet["VSL10GestureDetector + digit_classifier"]
    DigDet --> OutDig["Số 1 - 10"]
    
    ModeCheck --> |mode == 'letters'| AlphaClf["VSLAlphabetClassifier (Chỉ dùng Quy tắc Hình học)"]
    AlphaClf --> |12 chữ hỗ trợ| MatchRuleKết quả Khớp?
    MatchRule --> |MATCH| Smooth["Temporal Smoothing (prediction_history: 10 frames)"]
    MatchRule --> |NO_MATCH / CONFLICT| Neut["Trả về NEUTRAL (conf = 0.0)"]
    Smooth --> OutLetter["Hiển thị Chữ cái (A, B, C, D, E, I, L, O, S, T, V, Y)"]
    
    ModeCheck --> |mode == 'words'| Hybrid["Ensemble: Dual-hands + classify_geometry + MLP (weights.pt)"]
    Hybrid --> OutWord["Từ vựng 56 lớp"]

    Note1["LƯU Ý: Trong mode 'letters', MLP VSLGestureNet và SSTCN_Attention HOÀN TOÀN KHÔNG ĐƯỢC GỌI!"]
    AlphaClf -.-> Note1
```

### Chi tiết các hạn chế trong đường nhận diện:
1. **Chế độ `letters` không gọi MLP:** Mã nguồn [`backend/vision/recognizer.py`](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py) (dòng 650-721) quy định rõ: nếu `mode == "letters"`, chỉ gọi duy nhất `self.alphabet_classifier.classify_detailed(p_prim)`. Thuộc tính `has_mlp` bị gán `False` và `nn_sign = "NEUTRAL"`.
2. **Quy tắc hình học chỉ có 12 chữ:** File [`backend/vision/alphabet_classifier.py`](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py) chỉ cài đặt logic cho 12 chữ (`A, B, C, D, E, I, L, O, S, T, V, Y`).
3. **17 chữ cái bị tê liệt trong camera:**
   - 10 chữ cái tĩnh nền: `G, H, K, M, N, P, Q, R, U, X` luôn trả về `NO_MATCH` -> camera báo `NEUTRAL`.
   - 7 chữ cái động: `Ă, Â, Ê, Ô, Ơ, Ư, Đ` không có cơ chế bắt nét vẽ chuyển động nên không bao giờ nhận diện được.

---

## 5. Bảng kiểm kê toàn diện 29 Chữ cái Tiếng Việt

| Chữ cái | Trong weights.pt (84)? | Trong vsl_dataset.npz? | Trong SSTCN (102)? | Trong alphabet_classifier? | Trong Dataset V2? | Đánh giá hiện trạng kỹ thuật |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **A** | Có (idx 0) | Có (150 mẫu ảo) | Có (idx 0) | **Có (Rule MATCH)** | Static (64 mẫu) | Nhóm 2: Đã nhận diện tốt, có 64 mẫu thật V2 để củng cố |
| **Ă** | **KHÔNG** | **KHÔNG** | Có (idx 1) | **KHÔNG (UNKNOWN)** | Motion (53 frames OK) | **Nhóm 4:** Chữ động; Base A (19 frame), Diacritic (23 frame) |
| **Â** | **KHÔNG** | **KHÔNG** | Có (idx 2) | **KHÔNG (UNKNOWN)** | Motion (57 frames OK) | **Nhóm 4:** Chữ động; Base A (23 frame), Diacritic (25 frame) |
| **B** | Có (idx 1) | Có (150 mẫu ảo) | Có (idx 3) | **Có (Rule MATCH)** | Static (22 mẫu) | Nhóm 2: Đã nhận diện tốt qua quy tắc 4 ngón duỗi |
| **C** | Có (idx 2) | Có (150 mẫu ảo) | Có (idx 4) | **Có (Rule MATCH)** | Static (20 mẫu) | Nhóm 2: Đã nhận diện tốt qua quy tắc vòm cong |
| **D** | Có (idx 3) | Có (150 mẫu ảo) | Có (idx 5) | **Có (Rule MATCH)** | Static (46 mẫu) | Nhóm 2: Đã nhận diện tốt qua quy tắc trỏ duỗi đơn |
| **Đ** | Có (idx 4) | Có (150 mẫu ảo) | Có (idx 6) | **KHÔNG (UNKNOWN)** | Motion (47 frames OK) | **Nhóm 4:** Chữ động; Base D (26 frame), Diacritic gạch ngang (9 frame) |
| **E** | Có (idx 5) | Có (150 mẫu ảo) | Có (idx 7) | **Có (Rule MATCH)** | Static (35 mẫu) | Nhóm 2: Đã nhận diện tốt qua quy tắc 4 ngón tì ngón cái |
| **Ê** | **KHÔNG** | **KHÔNG** | Có (idx 8) | **KHÔNG (UNKNOWN)** | Motion (52 frames OK) | **Nhóm 4:** Chữ động; Base E (20 frame), Diacritic (20 frame) |
| **G** | Có (idx 7) | Có (150 mẫu ảo) | Có (idx 9) | **KHÔNG (UNKNOWN)** | Static (29 mẫu) | **Nhóm 3:** Chưa có rule camera; Đã có 29 mẫu thật V2 |
| **H** | Có (idx 8) | Có (150 mẫu ảo) | Có (idx 10) | **KHÔNG (UNKNOWN)** | Static (18 mẫu) | **Nhóm 3:** Chưa có rule camera; Đã có 18 mẫu thật V2 |
| **I** | Có (idx 9) | Có (150 mẫu ảo) | Có (idx 11) | **Có (Rule MATCH)** | Static (18 mẫu) | Nhóm 2: Đã nhận diện tốt qua quy tắc ngón út duỗi đơn |
| **K** | Có (idx 11) | Có (150 mẫu ảo) | Có (idx 12) | **KHÔNG (UNKNOWN)** | Static (17 mẫu) | **Nhóm 3:** Chưa có rule camera; Đã có 17 mẫu thật V2 |
| **L** | Có (idx 12) | Có (150 mẫu ảo) | Có (idx 13) | **Có (Rule MATCH)** | Static (16 mẫu) | Nhóm 2: Đã nhận diện tốt qua quy tắc L vuông góc |
| **M** | Có (idx 13) | Có (150 mẫu ảo) | Có (idx 14) | **KHÔNG (UNKNOWN)** | Static (19 mẫu) | **Nhóm 3:** Chưa có rule camera; Đã có 19 mẫu thật V2 |
| **N** | Có (idx 14) | Có (150 mẫu ảo) | Có (idx 15) | **KHÔNG (UNKNOWN)** | Static (21 mẫu) | **Nhóm 3:** Chưa có rule camera; Đã có 21 mẫu thật V2 |
| **O** | Có (idx 15) | Có (150 mẫu ảo) | Có (idx 16) | **Có (Rule MATCH)** | Static (48 mẫu) | Nhóm 2: Đã nhận diện tốt qua quy tắc vòng tròn kín O |
| **Ô** | **KHÔNG** | **KHÔNG** | Có (idx 17) | **KHÔNG (UNKNOWN)** | Motion (58 frames OK) | **Nhóm 4:** Chữ động; Base O (19 frame), Diacritic (20 frame) |
| **Ơ** | **KHÔNG** | **KHÔNG** | Có (idx 18) | **KHÔNG (UNKNOWN)** | Motion (45 frames OK) | **Nhóm 4:** Chữ động; Base O (9 frame), Diacritic râu (24 frame) |
| **P** | Có (idx 16) | Có (150 mẫu ảo) | Có (idx 19) | **KHÔNG (UNKNOWN)** | Static (20 mẫu) | **Nhóm 3:** Chưa có rule camera; Đã có 20 mẫu thật V2 |
| **Q** | Có (idx 17) | Có (150 mẫu ảo) | Có (idx 20) | **KHÔNG (UNKNOWN)** | Static (19 mẫu) | **Nhóm 3:** Chưa có rule camera; Đã có 19 mẫu thật V2 |
| **R** | Có (idx 18) | Có (150 mẫu ảo) | Có (idx 21) | **KHÔNG (UNKNOWN)** | Static (14 mẫu) | **Nhóm 3:** Chưa có rule camera; Đã có 14 mẫu thật V2 |
| **S** | Có (idx 19) | Có (150 mẫu ảo) | Có (idx 22) | **Có (Rule MATCH)** | Static (15 mẫu) | Nhóm 2: Đã nhận diện tốt qua quy tắc nắm tay ngón cái vắt ngang |
| **T** | Có (idx 20) | Có (150 mẫu ảo) | Có (idx 23) | **Có (Rule MATCH)** | Static (20 mẫu) | Nhóm 2: Đã nhận diện tốt qua quy tắc ngón cái kẹp trỏ/giữa |
| **U** | Có (idx 21) | Có (150 mẫu ảo) | Có (idx 24) | **KHÔNG (UNKNOWN)** | Static (32 mẫu) | **Nhóm 3:** Chưa có rule trong alphabet_clf; Đã có 32 mẫu V2 |
| **Ư** | **KHÔNG** | **KHÔNG** | Có (idx 25) | **KHÔNG (UNKNOWN)** | Motion (47 frames OK) | **Nhóm 4:** Chữ động; Base U (14 frame), Diacritic râu (12 frame) |
| **V** | Có (idx 22) | Có (150 mẫu ảo) | Có (idx 26) | **Có (Rule MATCH)** | Static (17 mẫu) | Nhóm 2: Đã nhận diện tốt qua quy tắc 2 ngón xòe chữ V |
| **X** | Có (idx 24) | Có (150 mẫu ảo) | Có (idx 27) | **KHÔNG (UNKNOWN)** | Static (25 mẫu) | **Nhóm 3:** Chưa có rule camera; Đã có 25 mẫu thật V2 |
| **Y** | Có (idx 25) | Có (150 mẫu ảo) | Có (idx 28) | **Có (Rule MATCH)** | Static (16 mẫu) | Nhóm 2: Đã nhận diện tốt qua quy tắc Shaka (ngón cái + út) |

---

## 6. Phân tích Khả năng Bổ sung mà KHÔNG Huấn luyện lại từ đầu

### A. Bài toán Cốt lõi: Bảo tồn 56 Từ vựng và Tránh Catastrophic Forgetting
- Mô hình `VSLGestureNet` trong [`backend/models/weights.pt`](file:///d:/SIGN_LANGUAGE/backend/models/weights.pt) hiện đang lưu giữ trọng số cho 56 từ vựng giao tiếp (`XIN_CHAO`, `CAM_ON`, `TAM_BIET`, `GIA_DINH`, ...).
- **Nguy cơ lớn:** Nếu mang toàn bộ mạng 84 lớp ra để huấn luyện lại (retrain) chỉ với 551 mẫu chữ cái của Dataset V2 (trong khi không có dữ liệu thật của 56 từ vựng), hiện tượng **Catastrophic Forgetting** (Quên thảm họa) chắc chắn sẽ xảy ra: mạng sẽ hoàn toàn mất khả năng nhận diện 56 từ vựng này.

### B. Phương án Kỹ thuật Tối ưu (Modular Extension Architecture)
Thay vì huấn luyện lại toàn bộ mô hình cũ, kiến trúc được đề xuất sẽ sử dụng cơ chế **Mở rộng Mô-đun Độc lập (Modular Pipeline)**:

1. **Bảo tồn nguyên vẹn mô hình tĩnh cũ:**
   - Giữ nguyên checkpoint `backend/models/weights.pt` làm bộ nhận diện từ vựng cho chế độ `words`.
2. **Xây dựng Mô-đun Chuyên biệt cho 22 Chữ Tĩnh (VSL Static Alphabet Head):**
   - Huấn luyện một mạng phân loại nhẹ (Linear Classifier Head hoặc Light MLP 72 -> 64 -> 22) trên tập dữ liệu chuẩn **551 mẫu tĩnh V2**.
   - Mô-đun này được tích hợp vào `VSLAlphabetClassifier` như một tầng bổ trợ học máy (ML-backed Alphabet Classifier) để giải quyết dứt điểm 10 chữ cái khó (`G, H, K, M, N, P, Q, R, U, X`).
3. **Xây dựng Bộ Máy Trạng thái Chuyển động Nét vẽ (Diacritic Motion Engine):**
   - Sử dụng dữ liệu từ `vsl_alphabet_motion_v2.json` (7 chuỗi động).
   - Khi camera nhận diện được tư thế nền ổn định (`A, D, E, O, U`), bộ đệm thời gian (temporal buffer 15-25 frames) sẽ theo dõi quỹ đạo của đầu ngón tay/cổ tay:
     - Nếu có chuyển động vẽ mũ đi lên rồi đi xuống $
→$ **Â** (từ A), **Ê** (từ E), **Ô** (từ O).
     - Nếu có chuyển động uốn cong đáy thuyền $
→$ **Ă** (từ A).
     - Nếu có chuyển động vẽ móc râu sang phải/lên $
→$ **Ơ** (từ O), **Ư** (từ U).
     - Nếu có chuyển động lắc ngang gạch nét $
→$ **Đ** (từ D).
   - Nếu không có chuyển động nét vẽ $
→$ Chốt chữ cái nền (`A, D, E, O, U`).

---

## 7. Phân chia 4 Nhóm và Kế hoạch Triển khai từng bước

### Nhóm 1 — Đã có và có thể giữ nguyên (Bảo tồn 100%)
- **Thành phần:** 56 lớp từ vựng trong `weights.pt` (từ `XIN_CHAO` đến `HANG_NGAY`), 10 chữ số (1 - 10) trong `digit_classifier`, và 102 lớp trong pretrained SSTCN.
- **Hành động:** Khóa cố định (Freeze), không can thiệp, không sửa đổi weights.

### Nhóm 2 — Đã có rule hình học nhưng cần kiểm chứng bằng Dataset V2
- **Thành phần (12 chữ):** `A, B, C, D, E, I, L, O, S, T, V, Y`.
- **Hành động:** Sử dụng 551 mẫu V2 để kiểm tra độ nhạy và độ đặc hiệu của các ngưỡng hình học hiện tại, bảo đảm độ chính xác $> 95\%$.

### Nhóm 3 — Chưa có rule trong camera (Đang bị UNKNOWN)
- **Thành phần (10 chữ):** `G, H, K, M, N, P, Q, R, U, X`.
- **Hành động:** Bổ sung quy tắc hình học dựa trên landmark V2 hoặc tích hợp bộ phân loại tĩnh V2 (72 chiều) vào `VSLAlphabetClassifier`.

### Nhóm 4 — Cần xử lý bằng chuyển động (7 Chữ cái Động có dấu)
- **Thành phần (7 chữ):** `Ă, Â, Ê, Ô, Ơ, Ư, Đ`.
- **Hành động:** Xây dựng Diacritic Motion Engine theo dõi vector vận tốc $vec{v}(t) = (dx, dy)$ của đầu ngón tay cái/trỏ từ các frame của `vsl_alphabet_motion_v2.json`.

---

## 8. Danh sách tệp liên quan cho Task tiếp theo (Task 6B)

1. [`backend/vision/alphabet_classifier.py`](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py): Nâng cấp bộ phân loại hỗ trợ đủ 22 chữ cái tĩnh.
2. [`backend/vision/diacritic_detector.py`](file:///d:/SIGN_LANGUAGE/backend/vision/diacritic_detector.py) *(tạo mới)*: Bộ máy trạng thái nhận diện nét vẽ 7 chữ có dấu.
3. [`backend/vision/recognizer.py`](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py): Kết nối Diacritic Motion Engine vào luồng `mode == "letters"`.
4. [`backend/tests/test_alphabet_classifier.py`](file:///d:/SIGN_LANGUAGE/backend/tests/test_alphabet_classifier.py): Bổ sung kiểm thử tích hợp cho trọn vẹn 29 chữ cái.
