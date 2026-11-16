# BÁO CÁO GIAI ĐOẠN 2C.9B — KIỂM CHỨNG DỮ LIỆU TRƯỚC KHI TRIỂN KHAI QUY ƯỚC KÉP 6–9

**Dự án:** VSL Connect AI — Hệ thống hỗ trợ giao tiếp Ngôn ngữ ký hiệu Việt Nam  
**Ngày thực hiện:** 29/10/2026  
**Người thực hiện:** Antigravity AI Coding Assistant  
**Trạng thái:** Hoàn thành kiểm toán dữ liệu và phân tích tính khả thi. Không sửa mã nguồn production, không thay đổi fixture hay checkpoint.

---

## 1. Baseline Test và Hiện trạng Hệ thống

Trước khi thực hiện kiểm toán, toàn bộ test suite backend đã được chạy lại trên môi trường hiện tại:

- **Lệnh thực thi:**
  ```powershell
  C:\Users\PC\AppData\Local\Programs\Python\Python312\python.exe scratch\run_all_tests.py
  ```
- **Kết quả tổng hợp:**
  ```
  ======================= 195 passed, 2 warnings in 6.70s =======================
  ```
  - **195/195 tests đạt 100% PASSED.**
  - **Số 0:** Hoạt động chính xác trong chế độ `digits` (`final_sign = "0"`, `label = "Số 0"`, `type = "digit"`), với 4 mẫu fixture thật đạt độ tin cậy $\ge 0.90$.
  - **Cách ly chế độ:** Chế độ `letters` tuyệt đối không rò rỉ Số 0 (cử chỉ khum tròn được nhận diện là Chữ cái `'O'` hoặc `'NEUTRAL'`).
  - **Các chế độ khác:** `words`, `sentence` và alias `letters_numbers` giữ nguyên 100% logic và không có bất kỳ hồi quy (regression) nào.

---

## 2. Bảng dữ liệu theo Số và Quy ước (Quy ước A vs Quy ước B)

Đối chiếu toàn diện giữa hai trường phái ký hiệu cho các số từ 6 đến 9:

- **Quy ước A (Chuẩn Quốc gia — Thông tư 17/2020/TT-BGDĐT):** Hệ ký hiệu tiếp xúc ngón tay kiểu ASL/quốc tế chuẩn hóa (ngón cái chạm lần lượt vào ngón út, áp út, giữa, trỏ).
- **Quy ước B (Dữ liệu Video QIPEDC hiện có trong dự án):** Hệ đếm tay thị giác Đông Á (Counting gestures: Shaka, súng ngắn L-shape, súng 3 nòng, móc câu).

| Chữ số | Tiêu chí đánh giá | Quy ước A (Chuẩn Thông tư 17) | Quy ước B (Video QIPEDC hiện có) | Đánh giá so sánh |
| :---: | :--- | :--- | :--- | :--- |
| **6** | **Mô tả cử chỉ** | Ngón cái gập chạm vào đầu ngón út; 3 ngón (trỏ, giữa, áp út) duỗi thẳng xòe lên | Ký hiệu Shaka: Ngón cái và ngón út duỗi thẳng sang hai bên; 3 ngón giữa gập chặt | Hai cử chỉ có cấu trúc ngón tay **hoàn toàn trái ngược nhau** |
| | **Nguồn dữ liệu** | Văn bản Thông tư 17/2020/TT-BGDĐT | Video `backend/data/media_cache/videos/D0535B.mp4` | Quy ước B có video thật, Quy ước A chỉ có văn bản |
| | **Số mẫu landmark thật** | **0 mẫu** (Chưa có trong repo) | **2 mẫu** (`dev`: f105, `eval`: f90) trong `real_digit_landmarks.json` | Thiếu dữ liệu thực nghiệm cho Quy ước A |
| | **Đặc trưng đo đạc** | *Kỳ vọng:* `I=True, M=True, R=True, P=False, dist(T, P) < 0.20` | `T=True, P=True, I=False, M=False, R=False, ang_TP=51.5°–52.8°` | Đặc trưng Quy ước B đã được định lượng chính xác |
| | **Nguy cơ nhầm lẫn** | Trùng với cấu trúc chữ W nếu ngón cái không chạm út | **Trùng lặp 100% với Chữ cái Y** trong bảng chữ cái VSL | Quy ước B xung đột nghiêm trọng với Chữ Y |
| **7** | **Mô tả cử chỉ** | Ngón cái gập chạm vào đầu ngón áp út; 3 ngón (trỏ, giữa, út) duỗi thẳng xòe lên | Dạng chữ L góc nhọn: Ngón cái và ngón trỏ duỗi mở; ngón giữa, áp út, út gập | Cấu trúc ngón tay khác biệt hoàn toàn |
| | **Nguồn dữ liệu** | Văn bản Thông tư 17/2020/TT-BGDĐT | Video `backend/data/media_cache/videos/D0536.mp4` | Quy ước B có video thật |
| | **Số mẫu landmark thật** | **0 mẫu** (Chưa có trong repo) | **2 mẫu** (`dev`: f54, `eval`: f45) trong `real_digit_landmarks.json` | Thiếu dữ liệu thực nghiệm cho Quy ước A |
| | **Đặc trưng đo đạc** | *Kỳ vọng:* `I=True, M=True, P=True, R=False, dist(T, R) < 0.20` | `T=True, I=True, M=False, R=False, P=False, ang_TI=28.1°–37.8°` | Quy ước B có góc Cái–Trỏ khá hẹp |
| | **Nguy cơ nhầm lẫn** | Ít nhầm lẫn với bảng chữ cái | **Dễ nhầm với Chữ cái L** nếu người dùng mở rộng ngón cái | Quy ước B nhạy cảm góc quay camera |
| **8** | **Mô tả cử chỉ** | Ngón cái gập chạm vào đầu ngón giữa; 3 ngón (trỏ, áp út, út) duỗi thẳng xòe lên | 3 ngón Cái + Trỏ + Giữa mở rộng xòe; ngón áp út và út gập | Cấu trúc ngón tay khác biệt hoàn toàn |
| | **Nguồn dữ liệu** | Văn bản Thông tư 17/2020/TT-BGDĐT | Video `backend/data/media_cache/videos/D0537.mp4` | Quy ước B có video thật |
| | **Số mẫu landmark thật** | **0 mẫu** (Chưa có trong repo) | **2 mẫu** (`dev`: f77, `eval`: f65) trong `real_digit_landmarks.json` | Thiếu dữ liệu thực nghiệm cho Quy ước A |
| | **Đặc trưng đo đạc** | *Kỳ vọng:* `I=True, R=True, P=True, M=False, dist(T, M) < 0.20` | `T=True, I=True, M=True, R=False, P=False, thumb_lat=0.52–0.55` | Quy ước B xòe 3 ngón cái-trỏ-giữa |
| | **Nguy cơ nhầm lẫn** | Ít nhầm lẫn với bảng chữ cái | **Xung đột với Chữ V và Số 2** nếu ngón cái khép nhẹ | Quy ước B dễ nhiễu khi cổ tay xoay |
| **9** | **Mô tả cử chỉ** | Ngón cái gập chạm đầu ngón trỏ tạo khuyên tròn; 3 ngón (giữa, áp út, út) duỗi thẳng | Ngón trỏ co hình móc câu (hook); ngón cái ép vào ngón giữa; các ngón khác gập | Cấu trúc ngón tay khác biệt hoàn toàn |
| | **Nguồn dữ liệu** | Văn bản Thông tư 17/2020/TT-BGDĐT | Video `backend/data/media_cache/videos/D0538.mp4` | Quy ước B có video thật |
| | **Số mẫu landmark thật** | **0 mẫu** (Chưa có trong repo) | **2 mẫu** (`dev`: f52, `eval`: f42) trong `real_digit_landmarks.json` | Thiếu dữ liệu thực nghiệm cho Quy ước A |
| | **Đặc trưng đo đạc** | *Kỳ vọng:* `M=True, R=True, P=True, I=False, dist(T, I) < 0.20` | `T=False, I=True, M=False, R=False, P=False, index_curl=0.13–0.17` | Quy ước B phụ thuộc góc nghiêng ngón trỏ |
| | **Nguy cơ nhầm lẫn** | Trùng hình thái với Chữ cái F trong ASL | Dễ rớt về `NEUTRAL` khi quay trực diện camera | Quy ước B bị lỗi phối cảnh rút ngắn 2D |

---

## 3. Bằng chứng hình học và Phân tích các cặp dễ nhầm lẫn

### 3.1. Cặp Số 6 (Quy ước B) vs Chữ cái Y
- **Phân tích hình thái:** Cả hai cử chỉ đều duỗi ngón cái và ngón út (`thumb_ext=True, pinky_ext=True`), gập chặt 3 ngón giữa (`index_ext=False, middle_ext=False, ring_ext=False`).
- **Góc chiếu 2D:** Trong mẫu thực tế của số 6, `ang_TP = 51.5°–52.8°`, `thumb_lat = 0.67`. Ở Chữ Y, các chỉ số này cũng nằm trong dải tương tự (`ang_TP > 45°`, `thumb_lat > 0.50`).
- **Kết luận:** **Trong không gian đặc trưng landmark tĩnh 2D/3D đơn khung hình, Số 6 kiểu QIPEDC và Chữ cái Y trùng nhau 100%.**
  - *Giải pháp an toàn hiện tại:* Tách biệt chế độ (`mode="digits"` trả về 6, `mode="letters"` trả về Y).
  - *Giải pháp triệt để:* Khi hỗ trợ Quy ước A (Số 6 là chạm cái vào út, 3 ngón giữa xòe), xung đột với chữ Y biến mất hoàn toàn!

### 3.2. Cặp Số 7 (Quy ước B) vs Chữ cái L và Chữ cái D
- **Với Chữ cái L:** Cả hai đều duỗi ngón cái và ngón trỏ (`T=True, I=True`), gập 3 ngón còn lại.
  - Chữ L: Hai ngón vuông góc $\approx 90^\circ$.
  - Số 7 (Quy ước B): Góc mở nhọn hơn, trong mẫu thực tế `ang_TI = 28.06°` (eval) đến `37.80°` (dev).
  - *Rủi ro thực tế:* Khi người ký đứng trước webcam thực tế, phản xạ mở ngón cái tự nhiên thường tạo góc $50^\circ - 65^\circ$, rơi vào khoảng trống giữa ngưỡng `< 50°` của số 7 và `> 65°` của chữ L, dẫn đến việc rớt nhận diện (`NO_MATCH`).
- **Với Chữ cái D:** Chữ D có ngón trỏ duỗi thẳng, ngón cái chạm đầu ngón giữa. Khác biệt rõ rệt vì ngón cái chữ D khép sát (`thumb_lat < 0.20`), trong khi Số 7 có `thumb_lat = 0.43 – 0.52`.

### 3.3. Cặp Số 8 (Quy ước B) vs Số 3, Chữ cái V và Chữ cái U
- **Với Số 3 (QIPEDC & TT17):**
  - Số 3 duỗi ngón Trỏ, Giữa, Áp út (`I=True, M=True, R=True`), ngón Út gập, ngón cái khép (`thumb_lat = 0.17`).
  - Số 8 duỗi Cái, Trỏ, Giữa (`T=True, I=True, M=True`), áp út và út gập, ngón cái xòe (`thumb_lat = 0.52 – 0.55`).
  - Phân biệt rõ nhờ trạng thái ngón áp út (`ring_ext`).
- **Với Chữ cái V và U:**
  - Chữ V và U đều duỗi ngón trỏ và ngón giữa, gập áp út và út (`I=True, M=True, R=False, P=False`), tương tự cấu trúc 2 ngón trên của Số 8.
  - Ở Chữ V, ngón cái gập đè lên ngón áp út (`thumb_lat <= 0.30`, `ang_TP < 25°`). Số 8 QIPEDC yêu cầu ngón cái xòe rộng (`thumb_lat > 0.30`, `ang_TP >= 25°`).
  - *Rủi ro:* Nếu người ký làm chữ V nhưng ngón cái hơi thả lỏng xòe ra ngoài, góc `ang_TP` có thể vượt $25^\circ$, gây nhầm lẫn sang Số 8 trong chế độ lai.

### 3.4. Cặp Số 9 (Quy ước B) vs Chữ cái D
- Số 9 (QIPEDC) là ngón trỏ móc câu (`index_curl = 0.13 – 0.17`).
- Chữ D là ngón trỏ duỗi thẳng (`index_curl \approx 0.0`), ngón cái chạm đầu ngón giữa (`dist(T, M) < 0.15`).
- *Rủi ro trên camera:* Khi hướng thẳng ngón trỏ móc câu vào ống kính camera (axial perspective), hình chiếu 2D làm khoảng cách từ TIP đến MCP co ngắn lại, MediaPipe có xu hướng ước lượng ngón trỏ đã gập hoàn toàn (`index_curl > 0.35`) hoặc ngộ nhận là ngón trỏ thẳng, gây mất nhận diện số 9.

---

## 4. Giới hạn của Dữ liệu Hiện có

1. **Hoàn toàn vắng bóng dữ liệu thực nghiệm cho Quy ước A (Thông tư 17):**
   - Trong toàn bộ repository, không có bất kỳ video, frame ảnh hay mẫu landmark thực tế nào ghi hình cử chỉ ngón cái chạm lần lượt vào ngón út (6), áp út (7), giữa (8), và trỏ (9).
   - Mọi giả định về vector đặc trưng của Quy ước A hiện nay đều chỉ là suy luận lý thuyết từ văn bản.
2. **Hạn chế người ký đơn lẻ (Single-signer) của Quy ước B:**
   - Cả 8 mẫu của các số 6, 7, 8, 9 trong `real_digit_landmarks.json` đều được trích xuất từ 4 video (`D0535B`, `D0536`, `D0537`, `D0538`) với cùng **1 người ký duy nhất**, cùng góc quay chính diện, cùng tỷ lệ bàn tay.
   - Các mẫu `dev` và `eval` tuy ở frame khác nhau nhưng **không độc lập về mặt giải phẫu và không có giá trị chứng minh khả năng tổng quát hóa (generalization)**.
3. **Hiệu ứng phối cảnh 2D (Perspective foreshortening):**
   - Bộ phân loại hiện tại tính toán góc và khoảng cách trên mặt phẳng 2D chuẩn hóa (`sh[4]`, `sh[8]`, `sh[0]`). Khi người dùng thay đổi góc nghiêng cổ tay, các góc chiếu 2D bị co giãn mạnh, dẫn đến trượt ra ngoài các ngưỡng cửa sổ hẹp (như `25° <= ang_TI < 50°` của số 7 hay `0.08 <= index_curl <= 0.35` của số 9).

---

## 5. Kết luận Kiểm toán

> [!CAUTION]
> **KẾT LUẬN: CHƯA ĐỦ ĐIỀU KIỆN ĐỂ TRIỂN KHAI PRODUCTION CHO QUY ƯỚC KÉP 6–9.**
>
> **Lý do có bằng chứng:**
> 1. Vi phạm nguyên tắc nền tảng *"Ground-truth First"*: Không thể đưa luật phân loại vào production khi chưa có ít nhất một bộ fixture landmark thực tế độc lập (dev và eval) để kiểm chứng.
> 2. Nếu tự ý viết luật cho Quy ước A lúc này, mã nguồn sẽ chỉ dựa trên phỏng đoán hình học, không có bài test nào chứng minh được tính đúng đắn trên dữ liệu thực tế.
> 3. Cần bảo vệ sự ổn định tuyệt đối của hệ thống sau khi vừa đạt mốc 195/195 tests passed.

---

## 6. Danh sách bước tiếp theo và Tiêu chí kiểm chứng cụ thể

Để đủ điều kiện triển khai Quy ước Kép trong giai đoạn sau, lộ trình bắt buộc phải gồm các bước chuẩn bị dữ liệu:

### Bước 1: Thu thập Dữ liệu Landmark thực tế cho Quy ước A (Giai đoạn chuẩn bị)
- Thu thập hoặc ghi hình tối thiểu 4 clip ngắn (mỗi clip cho 1 số 6, 7, 8, 9 theo chuẩn Thông tư 17: ngón cái chạm lần lượt vào ngón út, áp út, giữa, trỏ).
- Trích xuất 21 điểm landmark chuẩn hóa MediaPipe qua pipeline headless Chrome hiện có (`extract_video_landmarks`).
- Tạo tập tin fixture độc lập: `backend/tests/fixtures/real_digit_convention_a_landmarks.json` chứa tối thiểu 8 mẫu (mỗi số có 1 dev sample và 1 eval sample).

### Bước 2: Khảo sát và Xác lập Ngưỡng Hình học Khách quan
- Dùng feature extractor đo đạc khoảng cách tiếp xúc thực tế:
  - Số 6: `dist(thumb_tip, pinky_tip) / palm_scale`
  - Số 7: `dist(thumb_tip, ring_tip) / palm_scale`
  - Số 8: `dist(thumb_tip, middle_tip) / palm_scale`
  - Số 9: `dist(thumb_tip, index_tip) / palm_scale`
- Xác định khoảng cách phân tách an toàn (margin) với các cử chỉ lân cận.

### Bước 3: Xây dựng Test Suite độc lập trước khi sửa Code
- Viết test `test_digit_convention_a.py` kiểm chứng cấu trúc fixture và tính toàn vẹn.
- Chỉ khi bộ test này sẵn sàng, mới tiến hành tích hợp nhánh Quy ước A vào `VSLDigitClassifier`.

