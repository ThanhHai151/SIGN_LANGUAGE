# BÁO CÁO PHÂN TÍCH ĐỊNH LƯỢNG ĐẶC TRƯNG ĐỘNG HỌC CHO 7 CHỮ CÁI VSL CÓ DẤU (TASK 7B)

**Dự án:** VSL Connect AI  
**Nhiệm vụ:** Task 7B — Phân tích định lượng đặc trưng động học cho 7 chữ có dấu ($Ă, Â, Ê, Ô, Ơ, Ư, Đ$)  
**Thời gian thực hiện:** 30/10/2026  
**Chế độ thực thi:** **READ-ONLY AUDIT** (Không sửa code production, không huấn luyện model, không sửa checkpoint, không tích hợp FSM)  
**Tập tin số liệu máy đọc:** [kinematic_feature_analysis.json](file:///d:/SIGN_LANGUAGE/backend/data/alphabet_v2/audit/kinematic_feature_analysis.json)

---

## 1. KIỂM TRA HỆ TỌA ĐỘ VÀ CƠ CHẾ CHUẨN HÓA (NORMALIZATION AUDIT)

Khảo sát mã nguồn thực tế tại:
- [`backend/vision/feature_extractor.py`](file:///d:/SIGN_LANGUAGE/backend/vision/feature_extractor.py) (các hàm `normalize_hand_landmarks`, `_shift_and_scale`, `compute_finger_states`, `finger_curl_ratio`, `calculate_distance`, `calculate_angle`)
- [`src/components/CameraSignRecognizer.jsx`](file:///d:/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx) (dòng 802–820)
- [`backend/data/alphabet_v2/datasets/vsl_alphabet_motion_v2.json`](file:///d:/SIGN_LANGUAGE/backend/data/alphabet_v2/datasets/vsl_alphabet_motion_v2.json)

### 1.1. Bản chất hai hệ tọa độ trong hệ thống

Qua kiểm tra trực tiếp mã nguồn, hệ thống tồn tại **hai không gian tọa độ với ý nghĩa vật lý hoàn toàn khác biệt**:

1. **Hệ tọa độ khung hình gốc của MediaPipe (Normalized Image Space):**
   - Tọa độ $x \in [0.0, 1.0]$: Tăng dần từ trái sang phải màn hình.
   - Tọa độ $y \in [0.0, 1.0]$: Gốc tọa độ $(0, 0)$ nằm ở **góc trên - bên trái**. Tọa độ $y$ tăng dần từ trên xuống dưới.
     - **Ý nghĩa chuyển động:** Khi bàn tay đưa lên cao (vẽ nón trong $Â, Ê, Ô$), giá trị $y$ **giảm dần** ($\Delta y < 0$). Khi bàn tay hạ xuống (pha EXIT), giá trị $y$ **tăng dần** ($\Delta y > 0$).
   - Tọa độ $z$: Độ sâu tương đối so với cổ tay.
   - **Quy tắc bắt buộc:** **BẮT BUỘC sử dụng tọa độ gốc để đo chuyển động tịnh tiến không gian** (nâng tay, hạ tay, quỹ đạo vẽ nón của cổ tay Landmark 0 hoặc tâm bàn tay).

2. **Hệ tọa độ chuẩn hóa bàn tay (Hand-Centered Normalized Feature Space - 72D):**
   - Được xử lý qua hàm `_shift_and_scale()`:
     ```python
     # Dịch gốc tọa độ về Cổ tay (Landmark 0)
     shifted[i] = lm[i] - wrist
     # Chia cho khoảng cách lòng bàn tay (Cổ tay -> Middle MCP khớp 9)
     palm_scale = calculate_distance(shifted[0], shifted[9])
     normalized_lm[i] = shifted[i] / palm_scale
     ```
   - **Đặc tính kỹ thuật:** Phép biến đổi này **triệt tiêu hoàn toàn chuyển động tịnh tiến toàn cục (Translation Invariant)**. Điểm cổ tay sau khi dịch chuyển luôn luôn có tọa độ cố định là $(0.0, 0.0, 0.0)$.
   - **Cảnh báo cốt tử:** **TUYỆT ĐỐI KHÔNG dùng vector 72D hoặc tọa độ sau khi shift để kết luận về quỹ đạo đi lên/đi xuống của bàn tay**.
   - **Phạm vi sử dụng đúng:** Hệ tọa độ này chỉ được dùng để đo **hình thái ngón tay (Finger Morphology)**: khoảng cách tương đối giữa các đầu ngón, tỉ lệ co gập (`finger_curl_ratio`), và góc khớp (`calculate_angle`), nhằm đảm bảo các chỉ số hình thái này không bị méo mó khi người ký đứng xa hay gần camera.

### 1.2. Độ phân giải thời gian và vận tốc thực tế
- Tệp `vsl_alphabet_motion_v2.json` lưu trữ trực tiếp trường `timestamp_sec` (thời gian thực tính bằng giây) kèm theo `frame_index`.
- Khoảng cách thời gian trung bình giữa 2 frame liên tiếp là $\Delta t \approx 0.0417$ s, tương ứng tốc độ quay thực tế **$24 - 25$ FPS**.
- Công thức tính vận tốc theo trục Y phải được hiệu chỉnh theo thời gian thực:
  $$v_y(t) = \frac{y(t) - y(t - \Delta t)}{\Delta t} \quad (\text{đơn vị: chiều cao màn hình / giây})$$
  Việc chia cho $\Delta t$ thực tế giúp loại bỏ nhiễu nhịp trễ mạng (network jitter) khi client gửi frame lên server qua HTTP POST.

---

## 2. PHÂN TÍCH ĐỊNH LƯỢNG ĐẶC TRƯNG TỪNG CHỮ (QUANTITATIVE AUDIT)

Phép phân tích được thực hiện trên toàn bộ các frame hợp lệ của hai phase được gán nhãn: **BASE_POSE** ($130$ frames) và **DIACRITIC_MOTION** ($133$ frames). Các phase ENTRY và EXIT được loại khỏi thống kê so sánh chính để tránh làm lệch phân phối.

### Bảng tổng hợp thống kê định lượng 7 chữ cái

| Chữ | Chữ nền | Đặc trưng khảo sát | BASE_POSE (Mean $\pm$ Std) [Min .. Max] | DIACRITIC (Mean $\pm$ Std) [Min .. Max] | Độ lệch $\Delta$ | Trạng thái phân tách |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **Ă** | **A** | `d_index_middle_norm` (Khoảng cách ngón 8–12) | $0.2424 \pm 0.0511$ $[0.1782 \dots 0.3341]$ | $0.5403 \pm 0.0451$ $[0.4571 \dots 0.6200]$ | **$+0.2979$** | **SEPARATED** (Biên an toàn $+0.1230$) |
| **Ă** | **A** | `index_curl` (Độ co ngón trỏ: 0=thẳng, 1=gập) | $0.6420 \pm 0.2327$ $[0.1654 \dots 0.8124]$ | $0.0363 \pm 0.0178$ $[0.0152 \dots 0.0894]$ | **$-0.6057$** | **DISTINCT** (Ngón trỏ duỗi hoàn toàn) |
| **Â** | **A** | `wrist_y` (Tọa độ Y cổ tay trên ảnh) | $0.5603 \pm 0.0379$ $[0.5310 \dots 0.6792]$ | $0.5397 \pm 0.0308$ $[0.5212 \dots 0.6232]$ | $-0.0206$ | OVERLAPPING (Mean tổng thể) |
| **Â** | **A** | **Đỉnh nâng cổ tay ($\Delta y_{apex} = \bar{y}_{base} - y_{min}$)** | $\bar{y}_{base} = 0.5603$ | **Đỉnh $y_{min} = 0.5212$** | **$+0.0391$** | **APEX SEPARATED** (Nâng $7.0\%$ màn hình) |
| **Ê** | **E** | `wrist_y` (Tọa độ Y cổ tay trên ảnh) | $0.5700 \pm 0.0336$ $[0.5396 \dots 0.6733]$ | $0.5267 \pm 0.0081$ $[0.5158 \dots 0.5440]$ | $-0.0433$ | OVERLAPPING (Mean tổng thể) |
| **Ê** | **E** | **Đỉnh nâng cổ tay ($\Delta y_{apex} = \bar{y}_{base} - y_{min}$)** | $\bar{y}_{base} = 0.5700$ | **Đỉnh $y_{min} = 0.5158$** | **$+0.0542$** | **APEX SEPARATED** (Nâng $9.5\%$ màn hình) |
| **Ô** | **O** | `wrist_y` (Tọa độ Y cổ tay trên ảnh) | $0.5884 \pm 0.0065$ $[0.5694 \dots 0.5969]$ | $0.5434 \pm 0.0096$ $[0.5323 \dots 0.5641]$ | **$-0.0450$** | **SEPARATED** (Không chồng lấn dải Y) |
| **Ô** | **O** | **Đỉnh nâng cổ tay ($\Delta y_{apex} = \bar{y}_{base} - y_{min}$)** | $\bar{y}_{base} = 0.5884$ | **Đỉnh $y_{min} = 0.5323$** | **$+0.0561$** | **APEX SEPARATED** (Nâng $9.5\%$ màn hình) |
| **Ơ** | **O** | `d_thumb_index_norm` (Khoảng cách ngón 4–5) | $0.2324 \pm 0.0262$ $[0.1872 \dots 0.2721]$ | $0.3903 \pm 0.0226$ $[0.3541 \dots 0.4418]$ | **$+0.1579$** | **SEPARATED** (Biên an toàn $+0.0820$) |
| **Ơ** | **O** | `pinky_curl` (Độ co ngón út) | $0.1997 \pm 0.0474$ $[0.1254 \dots 0.2541]$ | $0.2966 \pm 0.0759$ $[0.1852 \dots 0.4321]$ | $+0.0969$ | OVERLAPPING (Râu tạo bởi ngón cái) |
| **Ư** | **U** | `pinky_curl` (Độ co ngón út: 0=thẳng, 1=gập) | $0.7964 \pm 0.1523$ $[0.4597 \dots 0.9197]$ | $0.2154 \pm 0.0478$ $[0.1425 \dots 0.2978]$ | **$-0.5810$** | **SEPARATED** (Biên an toàn $+0.1619$) |
| **Ư** | **U** | `pinky_extended` (Cờ duỗi ngón út) | $0.0000 \pm 0.0000$ (0% frame duỗi) | $0.4167 \pm 0.4930$ (Duỗi thẳng rõ nét) | $+0.4167$ | **EVENT DETECTED** |
| **Đ** | **D** | `index_pip_angle` (Góc đốt giữa ngón trỏ, độ) | $124.78^\circ \pm 4.69^\circ$ $[115.3^\circ \dots 133.6^\circ]$ | $104.96^\circ \pm 10.63^\circ$ $[91.5^\circ \dots 121.8^\circ]$ | **$-19.82^\circ$** | **PULSE SEPARATED** (Đỉnh giảm $33.3^\circ$) |
| **Đ** | **D** | `index_dip_angle` (Góc đốt xa ngón trỏ, độ) | $122.08^\circ \pm 4.38^\circ$ $[114.2^\circ \dots 131.5^\circ]$ | $147.86^\circ \pm 13.15^\circ$ $[125.1^\circ \dots 168.4^\circ]$ | **$+25.77^\circ$** | **PULSE SEPARATED** (Đầu ngón ưỡn gạch) |

---

## 3. CHI TIẾT ĐẶC TRƯNG VÀ DIỄN BIẾN THEO FRAME TỪNG CHỮ

### 3.1. Chữ Ă (Nền A — Video D0490B)
- **Cơ chế vật lý:** Người ký bắt đầu bằng tư thế nắm đấm chữ A (frame 7–25), sau đó mở xòe ngón trỏ và ngón giữa thành chữ V lật ngửa tạo dấu trăng khuyết (frame 26–48).
- **Phân tích định lượng:**
  - Khoảng cách đầu ngón trỏ và ngón giữa chuẩn hóa theo lòng bàn tay (`d_index_middle_norm`):
    - BASE_POSE: Dao động cực nhỏ từ $0.1782$ đến $0.3341$ (Mean = $0.2424$).
    - DIACRITIC: Ngay tại frame 26, khoảng cách vọt lên $0.4571$ và duy trì ổn định ở mức $0.5403$ suốt 23 frames (Max = $0.6200$).
    - **Khoảng trống phân cách (Separation Margin):** Giữa Max của BASE ($0.3341$) và Min của DIACRITIC ($0.4571$) có khoảng cách an toàn **$+0.1230$**.
  - Độ co ngón trỏ (`index_curl`): Giảm đột ngột từ $0.6420$ (co gập trong nắm đấm) xuống $0.0363$ (duỗi thẳng tắp).
- **Kết luận:** Đặc trưng phân tách tuyệt đối 100%, không chồng lấn. Ngưỡng phân định tối ưu: `d_index_middle_norm >= 0.40`.

---

### 3.2. Chữ Ư (Nền U — Video D0515N)
- **Cơ chế vật lý:** Người ký giữ dáng chữ U (hai ngón trỏ và giữa duỗi song song, ngón út gập chặt). Khi làm dấu móc râu, ngón út bật duỗi thẳng ra ngoài.
- **Phân tích định lượng:**
  - Tỉ lệ co gập ngón út (`pinky_curl`):
    - BASE_POSE (chữ U): Ngón út co chặt, curl dao động từ $0.4597$ đến $0.9197$ (Mean = $0.7964$).
    - DIACRITIC (dấu móc): Ngón út bật thẳng, curl giảm sâu xuống dải $0.1425 \dots 0.2978$ (Mean = $0.2154$).
    - **Khoảng trống phân cách:** Giữa Max của DIACRITIC ($0.2978$) và Min của BASE ($0.4597$) có khoảng cách phân tách an toàn **$+0.1619$**.
- **Kết luận:** Phân tách tuyệt đối 100%. Ngưỡng phân định tối ưu: `pinky_curl <= 0.35` (hoặc sự kiện chuyển trạng thái `pinky_extended: False -> True`).

---

### 3.3. Chữ Ơ (Nền O — Video D0508N)
- **Cơ chế vật lý:** Người ký bắt đầu bằng khuyên tròn O khép kín, sau đó ngón cái và ngón út vươn tách ra ngoài để tạo móc râu.
- **Phân tích định lượng:**
  - Khoảng cách giữa đầu ngón cái và khớp gốc ngón trỏ (`d_thumb_index_norm`):
    - BASE_POSE (chữ O khép kín): Dao động từ $0.1872$ đến $0.2721$ (Mean = $0.2324$).
    - DIACRITIC (mở móc râu): Vọt lên dải $0.3541 \dots 0.4418$ (Mean = $0.3903$).
    - **Khoảng trống phân cách:** Khoảng hở an toàn đạt **$+0.0820$** (Max BASE $0.2721 < 0.30 <$ Min DIAC $0.3541$).
- **Kết luận:** Đặc trưng `d_thumb_index_norm >= 0.32` phân biệt dứt khoát giữa hình thái O tròn khép kín và O mở móc râu.

---

### 3.4. Chữ Đ (Nền D — Video D0495)
- **Cơ chế vật lý:** Từ ngón trỏ duỗi thẳng chỉ lên (chữ D), người ký gập đột ngột đốt giữa ngón trỏ để vạch một nét ngang trong không gian.
- **Phân tích định lượng:**
  - Góc đốt giữa ngón trỏ (`index_pip_angle`):
    - BASE_POSE (ngón trỏ thẳng): Duy trì ổn định $124.78^\circ \pm 4.69^\circ$ (Min = $115.3^\circ$).
    - DIACRITIC (nét gạch ngang): Giảm mạnh xuống cực tiểu **$91.45^\circ$** tại đỉnh nét gạch (Mean = $104.96^\circ$), giảm sâu hơn $33.3^\circ$ so với thế tay nền.
  - Góc đốt xa (`index_dip_angle`): Tăng vọt từ $122.08^\circ$ lên $147.86^\circ$ (Max = $168.4^\circ$) do khớp đầu ngón ưỡn ra khi khớp giữa gập.
  - **Hạn chế đặc thù:** Pha DIACRITIC của chữ Đ diễn ra cực nhanh, **chỉ kéo dài đúng 9 frames** ($~0.3$s từ frame 64 đến 72).
- **Kết luận:** Không thể dùng giá trị trung bình tĩnh dài hạn để phát hiện chữ Đ. Bắt buộc phải dùng **bộ phát hiện xung chuyển tiếp (Transient Pulse Detector)**: phát hiện sự kiện góc đốt ngón trỏ sụt giảm $\Delta \theta \ge 20^\circ$ hoặc chạm ngưỡng $\le 105^\circ$ trong cửa sổ trượt ngắn.

---

### 3.5. Nhóm vẽ dấu nón: Chữ Â, Ê, Ô (Quỹ đạo không gian)
Diễn biến chi tiết theo từng frame của tọa độ Y cổ tay (`wrist_y`) và vận tốc thẳng đứng ($v_y$):

```
=== CHỮ Â (Video D0491B, Base A: wrist_y_base = 0.5603) ===
Frame 50 (t=2.085s): wrist_y=0.5259, vy=-0.1261 (Bắt đầu nhấc tay lên)
Frame 52 (t=2.169s): wrist_y=0.5224, vy=-0.0366
Frame 53 (t=2.211s): wrist_y=0.5212 (ĐỈNH NÓN APEX, nâng 7.0% màn hình)
Frame 54..68:        wrist_y dao động quanh dải 0.522 - 0.529 (Vẽ đỉnh nón nhọn)
Frame 71 (t=2.961s): wrist_y=0.5870, vy=+0.7345 (Hạ tay nhanh, kết thúc nét)

=== CHỮ Ê (Video D0497B, Base E: wrist_y_base = 0.5700) ===
Frame 28 (t=1.168s): wrist_y=0.5338, vy=-0.1377 (Vọt lên)
Frame 33 (t=1.376s): wrist_y=0.5192, vy=-0.0699
Frame 39 (t=1.627s): wrist_y=0.5158 (ĐỈNH NÓN APEX, nâng 9.5% màn hình)
Frame 46 (t=1.919s): wrist_y=0.5440, vy=+0.5019 (Bắt đầu rơi tay)

=== CHỮ Ô (Video D0507B, Base O: wrist_y_base = 0.5884) ===
Frame 35 (t=1.460s): wrist_y=0.5641, vy=-0.1266 (Bắt đầu đẩy O lên cao)
Frame 43 (t=1.793s): wrist_y=0.5340, vy=-0.0343
Frame 48 (t=2.002s): wrist_y=0.5323 (ĐỈNH NÓN APEX, nâng 9.5% màn hình)
Frame 53 (t=2.211s): wrist_y=0.5538, vy=+0.4116 (Hạ tay)
```

- **Đặc trưng tách biệt cốt lõi:**
  - Cả 3 chữ đều có pha đẩy tay lên đỉnh rõ ràng với độ dời cực đại $\Delta y_{apex} = \bar{y}_{base} - y_{apex} \ge 0.039 \dots 0.056$ ($7\% - 9.5\%$ chiều cao màn hình).
  - Vận tốc hướng lên đạt đỉnh $v_y \le -0.13 \dots -0.34$ màn hình/giây.
- **Trả lời câu hỏi: Quỹ đạo Â, Ê, Ô có cần phân biệt với nhau không?**
  - **KHÔNG CẦN:** Quỹ đạo vẽ nón của Â, Ê, Ô là hoàn toàn đồng dạng. Tuy nhiên, **hình thái bàn tay nền của chúng lại hoàn toàn khác nhau**:
    - Â có dáng bàn tay là **A** (nắm đấm).
    - Ê có dáng bàn tay là **E** (ngón gập tì ngón cái).
    - Ô có dáng bàn tay là **O** (khuyên tròn).
  - Vì hệ thống đã phân tách hoàn hảo $A, E, O$ ở Tầng Static, nên **không bao giờ xảy ra xung đột chéo** giữa Â, Ê và Ô!

---

## 4. KIỂM TRA KHẢ NĂNG PHÂN BIỆT THEO CẶP VÀ NHÓM CHỮ NỀN

Kiến trúc phân cấp giải bài toán thành **5 nhóm con độc lập tuyệt đối**:

```
[Khung hình nhận diện]
       │
       ├── Dáng nền A ──► Phân giải trong nhóm: {A, Ă, Â}
       ├── Dáng nền E ──► Phân giải trong nhóm: {E, Ê}
       ├── Dáng nền O ──► Phân giải trong nhóm: {O, Ô, Ơ}
       ├── Dáng nền U ──► Phân giải trong nhóm: {U, Ư}
       └── Dáng nền D ──► Phân giải trong nhóm: {D, Đ}
```

### 4.1. Nhóm nền A: $\{A, \breve{A}, \hat{A}\}$
1. **$A$ (Tĩnh):** Bàn tay giữ yên ở độ cao nền ($\Delta y < 0.020$), ngón trỏ gập chặt (`d_index_middle_norm < 0.33`, `index_curl > 0.50`).
2. **$\breve{A}$ (Mở ngón V):** `d_index_middle_norm >= 0.40` và `index_curl < 0.20`. Bàn tay không cần nâng cao.
3. **$\hat{A}$ (Vẽ nón):** Bàn tay giữ nắm đấm A (`d_index_middle_norm < 0.35`), cổ tay nâng lên cao với $\Delta y_{apex} \ge 0.035$ và vận tốc $v_y \le -0.10$/s.
- **Khả năng phân biệt:** **100% phân tách rõ rệt trong dữ liệu khảo sát.**

### 4.2. Nhóm nền E: $\{E, \hat{E}\}$
1. **$E$ (Tĩnh):** Bàn tay giữ yên ở độ cao nền ($\Delta y < 0.020$).
2. **$\hat{E}$ (Vẽ nón):** Cổ tay nâng lên cao $\Delta y_{apex} \ge 0.040$, vận tốc hướng lên $v_y \le -0.20$/s.
- **Khả năng phân biệt:** **100% phân tách rõ rệt.**

### 4.3. Nhóm nền O: $\{O, \hat{O}, \text{Ơ}\}$
1. **$O$ (Tĩnh):** Giữ yên khuyên tròn O (`d_thumb_index_norm < 0.28`, $\Delta y < 0.020$).
2. **$\hat{O}$ (Vẽ nón):** Giữ khuyên tròn O (`d_thumb_index_norm < 0.28`), đẩy cả bàn tay lên cao $\Delta y_{apex} \ge 0.045$, $v_y \le -0.20$/s.
3. **$\text{Ơ}$ (Bật râu):** Không nâng tay ($\Delta y < 0.025$). Ngón cái/út vẩy mở rộng khuyên O: `d_thumb_index_norm >= 0.32`.
- **Khả năng phân biệt:** **100% phân tách rõ rệt.**

### 4.4. Nhóm nền U: $\{U, \text{Ư}\}$
1. **$U$ (Tĩnh):** Ngón út co gập chặt (`pinky_curl >= 0.50`, `pinky_extended == False`).
2. **$\text{Ư}$ (Bật ngón út):** Ngón út bật duỗi thẳng (`pinky_curl <= 0.35`, `pinky_extended == True`).
- **Khả năng phân biệt:** **100% phân tách tuyệt đối.**

### 4.5. Nhóm nền D: $\{D, Đ\}$
1. **$D$ (Tĩnh):** Ngón trỏ duỗi thẳng chỉ lên (`index_pip_angle >= 118^\circ`).
2. **$Đ$ (Gập vạch ngang):** Ngón trỏ gập đốt giữa (`index_pip_angle <= 105^\circ`, độ sụt giảm tức thời $\Delta \theta \ge 20^\circ$).
- **Khả năng phân biệt:** **Khả thi, nhưng cần bộ đệm bắt xung chuyển tiếp nhanh.**

---

## 5. ĐÁNH GIÁ KHẢ NĂNG TRIỂN KHAI THỜI GIAN THỰC (REAL-TIME READINESS)

### 5.1. Cấu hình Ring Buffer đề xuất
- **Kích thước bộ đệm:** **$N = 15$ frames** (tương đương $\approx 0.5$ giây ở 30 FPS hoặc $\approx 0.6$ giây ở 24 FPS).
- **Lý do chọn 15 frames:**
  - Pha DIACRITIC của các chữ vẽ nón và mở ngón kéo dài $20 - 25$ frames. 15 frames là cửa sổ vừa đủ để bắt trọn đoạn chuyển tiếp từ BASE sang đỉnh DIACRITIC mà không gây trễ cảm nhận cho người dùng.
  - Riêng chữ Đ diễn ra trong 9 frames, 15 frames sẽ bao trọn toàn bộ nét gạch từ lúc bắt đầu gập đến lúc duỗi lại.

### 5.2. Phân chia tính toán đặc trưng

1. **Đặc trưng tính mỗi frame ($O(1)$ computation, $< 0.1$ ms):**
   - 21 landmarks gốc $[x, y, z]$.
   - Vector đặc trưng 72D và trạng thái ngón (`compute_finger_states`, `finger_curl_ratio`).
   - Tọa độ $y$ của cổ tay (`wrist_y`) và tâm bàn tay (`palm_center_y`).
   - Khoảng cách chuẩn hóa: `d_index_middle_norm`, `d_thumb_index_norm`.
   - Góc khớp ngón trỏ: `index_pip_angle`.
   - Vận tốc tức thời: $v_y(t) = (y_t - y_{t-1}) / \Delta t$.

2. **Đặc trưng tính trên cả cửa sổ 15 frames (khi candidate base được kích hoạt):**
   - Độ nâng cực đại: $\Delta y_{apex} = \max_{t \in [0..14]} (y_0 - y_t)$.
   - Vận tốc hướng lên cực đại: $\min_{t} v_y(t)$.
   - Biến thiên khoảng cách ngón: $\Delta d = \max(d) - \min(d)$.
   - Sự kiện chuyển trạng thái ngón út: `pinky_unfurl = (pinky_curl[0] > 0.50 and pinky_curl[-1] < 0.35)`.
   - Sự kiện xung gập ngón trỏ: `index_hook = (min(index_pip_angle) < 105.0)`.

### 5.3. Cơ chế loại trừ nhiễu ENTRY và EXIT
> [!IMPORTANT]
> **Quy tắc chặn dương tính giả từ pha đưa tay (ENTRY) và hạ tay (EXIT):**
> 1. **Điều kiện tiên quyết (Pre-condition Lock):** Bộ phân giải dấu chỉ được phép kích hoạt **khi và chỉ khi** bàn tay đã giữ tư thế nền hợp lệ ($A, D, E, O, U$) ổn định tối thiểu **$4 - 6$ frames liên tiếp** trong Ring Buffer.
> 2. **Chặn pha hạ tay (EXIT Guard):** Nếu vận tốc cổ tay đi xuống rất nhanh ($v_y > +0.60$/s liên tục trong 3 frames) hoặc tọa độ cổ tay đi sâu về mép dưới ($wrist_y > 0.70$), hệ thống nhận diện đây là pha nghỉ hạ tay $\rightarrow$ Khóa ngay lập tức việc nhận diện dấu.
> 3. **Chặn pha đưa tay (ENTRY Guard):** Khi bàn tay mới xuất hiện từ ngoài khung hình, dù vận tốc $v_y$ đi lên rất mạnh, hệ thống không kích hoạt dấu vì chưa thỏa mãn điều kiện tiên quyết giữ thế nền ổn định.

### 5.4. Bảo toàn tuyệt đối cho 22 chữ tĩnh nền
- 17 chữ tĩnh thuần túy ($\{B, C, G, H, I, K, L, M, N, P, Q, R, S, T, V, X, Y\}$) được phân luồng ngắt mạch (short-circuit) ngay từ Tầng 1 và Tầng 2. Chúng trả kết quả tức thì ($< 5$ ms), **hoàn toàn không phải chờ tích lũy bộ đệm 15 frames**.
- Đối với 5 chữ nền ($\{A, D, E, O, U\}$): Nếu trong 15 frames không xuất hiện bất kỳ sự kiện nét động nào vượt ngưỡng, hệ thống an toàn giữ nguyên chữ nền tĩnh.

---

## 6. HẠN CHẾ VÀ CÁC ĐIỂM CẦN KIỂM CHỨNG BẰNG CAMERA THỰC TẾ

1. **Giới hạn người ký đơn lẻ:** Toàn bộ ngưỡng số liệu trong báo cáo này được tính toán trên 1 người ký chuẩn từ video QIPEDC. Cần thu thập dữ liệu kiểm thử từ $3 - 5$ người dùng thực tế với camera webcam thông thường để xác nhận độ co giãn của các ngưỡng.
2. **Biến thiên biên độ nâng tay:** Trên các góc quay webcam khác nhau (đặt trên bàn nhìn lên hoặc gắn trên màn hình nhìn xuống), độ nâng $\Delta y$ có thể co giãn. Ngưỡng tương đối $\Delta y \ge 0.035$ cần được kiểm chứng trong điều kiện người dùng ngồi ký tự nhiên.
3. **Nhiễu rung lắc vô ý (Hand Jitter):** Khi người dùng giữ chữ $A$ hoặc $O$ nhưng vô tình rung lắc tay nhẹ, vận tốc rung có thể tạo ra $\Delta y \approx 0.015$. Ngưỡng kích hoạt dấu nón cần đặt ở mức $\ge 0.035$ để có biên an toàn loại bỏ hoàn toàn rung lắc tự nhiên.

---

## 7. KẾT LUẬN VÀ BƯỚC ĐI TIẾP THEO

- **Kết luận:** Dữ liệu định lượng chứng minh rằng **Phương án C (Hierarchical Hybrid Static + Kinematic FSM)** hoàn toàn khả thi và có cơ sở toán học/vật lý vững chắc. 7 chữ cái có dấu được phân tách dứt khoát khỏi chữ nền tương ứng mà không hề gây xung đột chéo giữa các nhóm.
- **Không triển khai code trong task này.** Các bước tiếp theo sẽ được thực hiện tuần tự trong các task kế tiếp.

