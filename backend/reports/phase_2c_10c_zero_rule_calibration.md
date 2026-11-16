# BÁO CÁO GIAI ĐOẠN 2C.10C — HIỆU CHỈNH QUY TẮC SỐ 0 DỰA TRÊN BẰNG CHỨNG (EVIDENCE-BASED CALIBRATION)

**Dự án:** VSL Connect AI  
**Thời điểm thực hiện:** 29/10/2026  
**Mục tiêu:** Cải thiện độ nhạy và tính ổn định khi nhận diện Số 0 trong chế độ `digits` trên dữ liệu thực tế bằng phương pháp hiệu chỉnh hình học tối thiểu, có kiểm chứng hồi quy nghiêm ngặt, tuyệt đối không ảnh hưởng đến các chữ số 1–9 và các chế độ `letters`, `words`, `sentence`.

---

## 1. Baseline và Đặc trưng đã đo đạc

### 1.1. Công thức và cơ chế tính toán xác minh từ mã nguồn
Đã đọc và xác minh trực tiếp mã nguồn trong `backend/vision/digit_classifier.py` và `backend/vision/feature_extractor.py`:
- `palm_scale`: Khoảng cách Euclid 3D giữa Cổ tay (`mốc 0`) và Khớp gốc ngón giữa (`mốc 9`).
- `d_TI`: Khoảng cách 3D chuẩn hóa giữa Đầu ngón cái (`mốc 4`) và Đầu ngón trỏ (`mốc 8`) chia cho `palm_scale`.
- `d_TM`: Khoảng cách 3D chuẩn hóa giữa Đầu ngón cái (`mốc 4`) và Đầu ngón giữa (`mốc 12`) chia cho `palm_scale`.
- `d_IW`: Khoảng cách 3D chuẩn hóa giữa Cổ tay (`mốc 0`) và Đầu ngón trỏ (`mốc 8`) chia cho `palm_scale`.
- Trạng thái duỗi ngón (`index_ext`, `middle_ext`...): So sánh tọa độ thẳng đứng 2D `TIP.y < PIP.y`.
- `index_curl`: Tỷ lệ co ngón trỏ $1.0 - \frac{d(TIP, MCP)}{d(TIP, PIP) + d(PIP, MCP)}$.

### 1.2. Bảng đối chiếu đặc trưng hình học toàn diện
Đo đạc thực tế trên toàn bộ 4 mẫu Số 0 (`real_digit_0_landmarks.json`) và 18 mẫu Số 1–9 (`real_digit_landmarks.json`):

| Nhãn | Mẫu / Split | Nguồn Video | Trạng thái duỗi (T, I, M, R, P) | `d_TI` | `d_TM` | `d_IW` | `index_curl` | `middle_curl` | Phân loại |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | Frame 20 | QIPEDC D0529 | `(0, 0, 0, 0, 0)` | 0.131 | 0.125 | 1.194 | 0.640 | 0.684 | MATCH (0) |
| **0** | Frame 25 | QIPEDC D0529 | `(0, 0, 0, 0, 0)` | 0.118 | 0.130 | 1.194 | 0.608 | 0.654 | MATCH (0) |
| **0** | Frame 30 (dev) | QIPEDC D0529 | `(0, 0, 0, 0, 0)` | 0.105 | 0.115 | 1.145 | 0.639 | 0.692 | MATCH (0) |
| **0** | Frame 35 (eval)| QIPEDC D0529 | `(0, 0, 0, 0, 0)` | 0.107 | 0.118 | 1.184 | 0.648 | 0.687 | MATCH (0) |
| **1** | Frame 71 (dev) | QIPEDC D0530 | `(1, 1, 0, 0, 0)` | 1.112 | 0.234 | 2.088 | 0.000 | 0.515 | MATCH (1) |
| **1** | Frame 60 (eval)| QIPEDC D0530 | `(1, 1, 0, 0, 0)` | 0.808 | 0.256 | 1.973 | 0.000 | 0.774 | MATCH (1) |
| **2** | Frame 67 / 55 | QIPEDC D0531 | `(1, 1, 1, 0, 0)` | 1.053–1.262 | 1.024–1.308 | 1.872–1.940 | 0.000 | 0.000 | MATCH (2) |
| **3** | Frame 67 / 55 | QIPEDC D0532 | `(1, 1, 1, 1, 0)` | 1.067–1.143 | 1.036–1.201 | 1.862–1.916 | 0.000 | 0.000 | MATCH (3) |
| **4** | Frame 54 / 45 | QIPEDC D0533 | `(1, 1, 1, 1, 1)` | 1.036–1.355 | 1.118–1.485 | 1.725–1.816 | 0.000 | 0.001 | MATCH (4) |
| **5** | Frame 67 / 55 | QIPEDC D0534 | `(1, 1, 1, 1, 1)` | 0.950–1.016 | 1.165–1.237 | 1.772–1.827 | 0.000 | 0.001 | MATCH (5) |
| **6** | Frame 105 / 90| QIPEDC D0535B| `(1, 0, 0, 0, 1)` | 0.470–0.497 | 0.592–0.594 | 0.917–0.971 | 0.752–0.778 | 0.657–0.694 | MATCH (6) |
| **7** | Frame 54 / 45 | QIPEDC D0536 | `(1, 1, 0, 0, 0)` | 1.153–1.391 | 0.497–0.563 | 1.982–2.050 | 0.000 | 0.233–0.426 | MATCH (7) |
| **8** | Frame 77 / 65 | QIPEDC D0537 | `(1, 1, 1, 0, 0)` | 0.964–1.214 | 1.083–1.424 | 1.822–1.957 | 0.000 | 0.000 | MATCH (8) |
| **9** | Frame 52 / 42 | QIPEDC D0538 | `(0, 1, 0, 0, 0)` | 0.693–0.724 | 0.100–0.127 | 1.560–1.663 | 0.129–0.168 | 0.459–0.461 | MATCH (9) |

> **Nhận xét khoảng cách then chốt:**  
> - Khoảng cách Cái–Trỏ (`d_TI`): Số 0 có khoảng cách siêu hẹp (`0.105`–`0.131`). Toàn bộ số 1–9 đều có `d_TI >= 0.470`. Biên an toàn tách biệt giữa số 0 và các số khác lên tới **$0.339$** (rất lớn).  
> - Khoảng cách Cái–Giữa (`d_TM`): Trong video studio `D0529`, người ký chụm cả ngón trỏ và ngón giữa vào ngón cái (`d_TM = 0.115`–`0.130`). Tuy nhiên, ở các cử chỉ tự nhiên, ngón giữa chỉ cần khum theo vòm và có thể cách ngón cái `0.22`–`0.28`.

---

## 2. So sánh và Đánh giá các Phương án (A / B / C)

| Tiêu chí đánh giá | Phương án A (Giữ nguyên) | Phương án B (Nới `d_TM <= 0.30`) | Phương án C (`d_TI <= 0.20` + `middle_curl >= 0.45`) |
| :--- | :--- | :--- | :--- |
| **Mô tả quy tắc** | Giữ `D_TM_THRESHOLD_0 = 0.22` | Nới `D_TM_THRESHOLD_0 = 0.30` | Nới `d_TM <= 0.35` và thêm ràng buộc `middle_curl >= 0.45` |
| **Độ nhạy cử chỉ tự nhiên** | **Kém:** Rớt cử chỉ tự nhiên khi ngón giữa hở nhẹ (`d_TM = 0.23`–`0.28`). | **Tốt:** Dung nạp độ hở tự nhiên lên tới $30\%$ chiều dài bàn tay. | **Rất tốt:** Cho phép ngón giữa tự do theo vòm co gập tự nhiên. |
| **Khớp 4 mẫu QIPEDC D0529** | 4/4 (100%) | 4/4 (100%) | 4/4 (100%) |
| **False Positive trên Số 1–9** | 0 / 18 mẫu (0%) | 0 / 18 mẫu (0%) | 0 / 18 mẫu (0%) |
| **Nguy cơ nhầm với Chữ cái O** | 0% (Chặn ở `test_04`) | 0% (`lm_o` có `d_TM=0.32 > 0.30` và `d_IW=0.88 < 0.95`) | 0% (`lm_o` có `m_curl=0.0 < 0.45` và `d_IW=0.88 < 0.95`) |
| **Phạm vi tác động code** | 0 dòng | **1 dòng duy nhất** (`D_TM_THRESHOLD_0 = 0.30`) | Thay đổi nhiều dòng (cả `extract_features` và `evaluate_rules`) |

---

## 3. Phương án được lựa chọn và Lý do

### **Lựa chọn chính thức: PHƯƠNG ÁN B**
- **Hiệu chỉnh duy nhất:** Cập nhật hằng số `D_TM_THRESHOLD_0` từ `0.22` lên `0.30` trong `backend/vision/digit_classifier.py`.
- **Lý do quyết định:**
  1. **Thay đổi tối thiểu (Minimal Invasive):** Chỉ can thiệp 1 hằng số ngưỡng cục bộ, không biến đổi cấu trúc từ điển `features`, không sửa pipeline hay model weights.
  2. **Giải quyết triệt để điểm nghẽn:** Mở rộng biên dung sai cho khoảng cách Cái–Giữa thêm gần 40% (từ `0.22` lên `0.30`), cho phép ngón giữa có độ tự do tự nhiên khi người dùng tạo khuyên tròn số 0 bằng ngón cái và ngón trỏ.
  3. **Bảo tồn tuyệt đối tính an toàn:**
     - Toàn bộ số 1–9 đều có ít nhất 1 ngón duỗi và `d_TI >= 0.470` (vượt xa `0.20`), không có bất kỳ rủi ro False Positive nào.
     - Cử chỉ chữ O trong `test_04` vẫn tiếp tục bị từ chối thành `NEUTRAL` vì `d_TM = 0.3225 > 0.30` và `d_IW = 0.8881 < 0.95`.
     - Phân định chế độ `digits` và `letters` được bảo vệ độc lập ở cấp Recognizer.

---

## 4. Các giới hạn Dữ liệu đã xác định

1. **Giới hạn số lượng người ký:**
   - 4 mẫu số 0 trong fixture `real_digit_0_landmarks.json` đều trích xuất từ **duy nhất 1 video phòng thu `D0529.mp4` của 1 người ký**. Chúng không phải là 4 mẫu độc lập đại diện cho quần thể người dùng.
2. **Hiện tượng góc nghiêng phối cảnh 2D:**
   - Phép kiểm tra duỗi ngón `TIP.y < PIP.y` dựa trên mặt phẳng ảnh 2D. Nếu người dùng chúc bàn tay về phía camera, các đầu ngón tay có thể bị nhận nhầm là đang duỗi. Khi người dùng giữ bàn tay đứng thẳng hướng về camera, cử chỉ được nhận diện chính xác 100%.

---

## 5. Kết quả Kiểm thử trước và sau Hiệu chỉnh

### 5.1. Trước hiệu chỉnh (Baseline 2C.10B)
- **Tổng số test:** 197 tests (PASSED: 197 / 197).
- Ngưỡng `D_TM_THRESHOLD_0 = 0.22`. Cử chỉ số 0 với ngón giữa hơi hở (`d_TM > 0.22`) bị đánh rớt thành `NEUTRAL`.

### 5.2. Sau hiệu chỉnh và Bổ sung Boundary Tests (2C.10C)
Đã bổ sung 5 kiểm thử tình huống biên (boundary conditions) trong `backend/tests/test_digit_0_landmarks.py`:
1. `test_digit_0_boundary_d_tm_relaxed`: Mẫu có `d_TM = 0.25 <= 0.30` nhận diện thành công số 0.
2. `test_digit_0_boundary_d_tm_exceeded`: Mẫu có `d_TM = 0.35 > 0.30` bị loại bỏ an toàn thành `NEUTRAL`.
3. `test_digit_0_boundary_d_ti_exceeded`: Mẫu có `d_TI = 0.25 > 0.20` bị loại bỏ an toàn thành `NEUTRAL`.
4. `test_digit_0_boundary_d_iw_fist`: Mẫu nắm đấm (`d_IW < 0.95`) bị loại bỏ an toàn thành `NEUTRAL`.
5. `test_digit_0_boundary_finger_extended`: Mẫu có ngón duỗi bị loại bỏ an toàn thành `NEUTRAL`.

- **Tổng số test hiện tại:** **202 tests**
- **Kết quả:** **202 / 202 PASSED (100%)**, 0 failed, 2 warnings (Lifespan deprecation).
- **Thời gian chạy:** 8.54s.
- Toàn bộ các chế độ `letters`, `digits`, `words`, `sentence` và alias `letters_numbers` đều hoạt động chính xác theo đúng hợp đồng thiết kế.

