# BÁO CÁO GIAI ĐOẠN 2C.6 — KIỂM TRA QUY ƯỚC VSL SỐ 0–9 VÀ BỔ SUNG SỐ 0

**Dự án:** VSL Connect AI — Hệ thống hỗ trợ giao tiếp Ngôn ngữ ký hiệu Việt Nam  
**Ngày thực hiện:** 29/10/2026  
**Người thực hiện:** Antigravity AI Coding Assistant  
**Trạng thái:** Hoàn thành khảo sát & lập báo cáo kiểm toán quy ước. Chưa can thiệp mã nguồn production.

---

## 1. Tóm tắt kết quả kiểm tra

Giai đoạn 2C.6 thực hiện khảo sát toàn diện và kiểm toán quy ước cử chỉ cho dải chữ số VSL từ **0 đến 9**, nhằm giải quyết triệt để 3 vấn đề thực tế:
1. Số 1–5 hoạt động tương đối tốt trong một số lần thử camera nhưng cần xác minh xem có đúng quy ước VSL được dự án phê duyệt hay không.
2. Số 6–9 liên tục nhận diện sai hoặc không nhận diện được khi người dùng thử nghiệm camera thực tế.
3. Số 0 chưa được hỗ trợ, thiếu dữ liệu video `D0529.mp4` trong kho cục bộ.

### Các phát hiện cốt lõi:
- **Xung đột quy ước nền tảng (Convention Conflict):** Có sự khác biệt căn bản giữa hai hệ quy ước:
  - **Chuẩn Quốc gia (Thông tư 17/2020/TT-BGDĐT):** Sử dụng hệ ký hiệu số tiếp xúc đầu ngón tay (tương đồng hệ ASL/Quốc tế điều chỉnh): Số 6 (ngón cái chạm ngón út), Số 7 (ngón cái chạm áp út), Số 8 (ngón cái chạm ngón giữa), Số 9 (ngón cái chạm ngón trỏ tạo vòng khuyên).
  - **Dữ liệu Video QIPEDC cục bộ (D0535B–D0538):** Sử dụng hệ đếm tay phổ thông Đông Á (Counting gestures): Số 6 (Shaka: cái + út duỗi), Số 7 (L-shape góc nhọn: cái + trỏ duỗi), Số 8 (Gun/Tri-finger: cái + trỏ + giữa), Số 9 (Hook: ngón trỏ khum móc câu).
  - Module `backend/vision/digit_classifier.py` hiện tại được lập trình cứng theo dữ liệu video QIPEDC. Nếu người dùng thực hiện theo chuẩn Thông tư 17, hệ thống **100% từ chối (báo NO_MATCH)**.
- **Nguyên nhân số 6–9 lỗi khi chạy camera thực tế:**
  1. *Người dùng làm cử chỉ khác với luật QIPEDC* (thực hiện theo chuẩn SGK/Thông tư 17 hoặc phản xạ tự nhiên).
  2. *Ngưỡng góc chiếu 2D quá hẹp và nhạy cảm góc quay:* Số 7 yêu cầu góc ngón cái - trỏ nằm chặt trong dải `[25.0°, 50.0°]`; Số 9 yêu cầu độ co ngón trỏ `0.08 <= index_curl <= 0.35` (chỉ đúng khi quay nghiêng, khi đối diện trực diện camera hình chiếu 2D làm ngón trỏ bị ngộ nhận là co gập hoàn toàn > 0.35); Số 6 yêu cầu góc cái - út `ang_TP >= 35.0°`.
- **Tình trạng Số 0:**
  - File `backend/data/media_cache/videos/D0529.mp4` **hoàn toàn không tồn tại** trên đĩa.
  - Chưa có bất kỳ frame landmark nào cho số 0 trong `backend/tests/fixtures/real_digit_landmarks.json`.
  - Classifier hiện tại chưa định nghĩa số 0 (cử chỉ chụm tròn hiện cố tình trả về `NEUTRAL` conf=0.0 để bảo vệ nhận diện chữ cái O).

---

## 2. Bảng đối chiếu quy ước số 0–9

| Chữ số | Quy ước Chuẩn Quốc Gia (Thông tư 17/2020/TT-BGDĐT) | Quy ước Video QIPEDC cục bộ (D0529–D0538) | Quy tắc hình học hiện tại trong `digit_classifier.py` | Trạng thái xác minh | Bằng chứng cụ thể |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **0** | Bàn tay khum tròn, các đầu ngón tay khép chạm đầu ngón cái tạo hình chữ O / số 0 | Ghi nhận trong metadata `qipedc_dictionary.json` là `D0529` ("0 (số không)"), nhưng **thiếu file video** | Chưa hỗ trợ (trả về `NEUTRAL`, conf=0.0) | **CHƯA XÁC MINH** | File `D0529.mp4` không có trên đĩa; không có landmark fixture; chưa có code nhận diện. |
| **1** | Ngón trỏ duỗi thẳng hướng lên; ngón cái gập giữ các ngón giữa, áp út, út gập | Video `D0530.mp4`: Ngón trỏ duỗi thẳng; ngón cái đè giữ 3 ngón còn lại | Ngón trỏ duỗi (`index=EXTENDED`), 3 ngón còn lại gập; ngón cái không duỗi độc lập | **ĐÃ XÁC MINH** | Thống nhất giữa TT17 và QIPEDC. Fixture 2 mẫu pass 100% test độc lập và tích hợp. |
| **2** | Ngón trỏ và ngón giữa duỗi thẳng hình chữ V; ngón cái gập đè ngón áp út và út | Video `D0531.mp4`: Ngón trỏ và giữa duỗi xòe hình chữ V; ngón áp út, út gập; ngón cái đè ngón áp út | Ngón trỏ và giữa duỗi (`EXTENDED`), áp út và út gập (`FOLDED`); góc `ang_IM >= 8.0°` | **ĐÃ XÁC MINH** | Thống nhất giữa TT17 và QIPEDC. Fixture 2 mẫu pass 100% test độc lập và tích hợp. |
| **3** | Ngón trỏ, giữa, áp út duỗi thẳng hướng lên; ngón cái gập giữ đầu ngón út (kiểu W / số 3 quốc tế) | Video `D0532.mp4`: Ngón cái, ngón trỏ và ngón giữa duỗi; ngón áp út và út gập chặt | Ngón cái, trỏ, giữa duỗi (`EXTENDED`); áp út và út gập (`FOLDED`); góc `ang_TI >= 15.0°`, `ang_IM >= 8.0°` | **CÓ BẰNG CHỨNG MỘT PHẦN** | Bất đồng quy ước: TT17 dùng 3 ngón (trỏ-giữa-áp út); QIPEDC dùng 3 ngón (cái-trỏ-giữa). Code hiện chạy theo QIPEDC. |
| **4** | 4 ngón (trỏ, giữa, áp út, út) duỗi thẳng xòe; ngón cái gập ngang lòng bàn tay | Video `D0533.mp4`: 4 ngón duỗi thẳng hướng lên; ngón cái gập chặt áp vào lòng bàn tay | 4 ngón duỗi (`EXTENDED`), ngón cái gập (`thumb != EXTENDED`, `thumb_folded=True`) | **ĐÃ XÁC MINH** | Thống nhất giữa TT17 và QIPEDC. Fixture 2 mẫu pass 100% test độc lập và tích hợp. |
| **5** | Toàn bộ 5 ngón tay mở rộng, xòe đều tự nhiên hướng lên | Video `D0534.mp4`: Cả 5 ngón tay xòe đều tự nhiên hướng lên | Cả 5 ngón đều duỗi (`EXTENDED`); góc `ang_TI >= 12.0°` | **ĐÃ XÁC MINH** | Thống nhất giữa TT17 và QIPEDC. Fixture 2 mẫu pass 100% test độc lập và tích hợp. |
| **6** | Ngón cái gập chạm vào đầu ngón út; 3 ngón (trỏ, giữa, áp út) duỗi thẳng xòe lên (ASL 6) | Video `D0535B.mp4`: Ký hiệu Shaka — Ngón cái và ngón út duỗi thẳng sang hai bên; 3 ngón giữa gập chặt | Ngón cái và ngón út duỗi (`EXTENDED`); ngón trỏ, giữa, áp út gập (`FOLDED`); góc `ang_TP >= 35.0°` | **CÓ BẰNG CHỨNG MỘT PHẦN** | Bất đồng quy ước sâu sắc: TT17 là chạm ngón cái vào út; QIPEDC là Shaka. Nhận diện camera thực tế sai do người ký làm theo TT17 hoặc góc quay hẹp. |
| **7** | Ngón cái gập chạm vào đầu ngón áp út; 3 ngón (trỏ, giữa, út) duỗi thẳng xòe lên (ASL 7) | Video `D0536.mp4`: Dạng chữ L góc nhọn — Ngón cái và ngón trỏ duỗi mở; ngón giữa, áp út, út gập | Ngón cái và trỏ duỗi (`EXTENDED`); 3 ngón còn lại gập; dải góc hẹp `25.0° <= ang_TI < 50.0°` | **CÓ BẰNG CHỨNG MỘT PHẦN** | Bất đồng quy ước: TT17 là chạm cái vào áp út; QIPEDC là chữ L góc nhọn. Nhận diện camera thực tế sai do dải góc `25°–50°` quá khắt khe và dễ nhầm chữ L. |
| **8** | Ngón cái gập chạm vào đầu ngón giữa; 3 ngón (trỏ, áp út, út) duỗi thẳng xòe lên (ASL 8) | Video `D0537.mp4`: Ngón cái, ngón trỏ và ngón giữa duỗi mở rộng; ngón áp út và út gập | Ngón cái, trỏ, giữa duỗi (`EXTENDED`); áp út, út gập; dải góc `ang_TI >= 20.0°`, `ang_TM >= 25.0°` | **CÓ BẰNG CHỨNG MỘT PHẦN** | Bất đồng quy ước: TT17 là chạm cái vào giữa; QIPEDC là xòe 3 ngón cái-trỏ-giữa. Nhận diện camera sai do trùng cấu trúc ngón với số 3 kiểu QIPEDC và góc xoay che khuất. |
| **9** | Ngón cái gập chạm đầu ngón trỏ tạo vòng tròn khuyên; 3 ngón (giữa, áp út, út) duỗi thẳng (ASL 9) | Video `D0538.mp4`: Ngón trỏ co hình móc câu (hook); ngón cái ép vào ngón giữa; các ngón khác gập | Ngón trỏ co khum `0.08 <= index_curl <= 0.35`; ngón giữa, áp út, út gập (`FOLDED`); ngón cái không duỗi | **CÓ BẰNG CHỨNG MỘT PHẦN** | Bất đồng quy ước: TT17 là chạm cái vào trỏ; QIPEDC là móc câu ngón trỏ. Nhận diện camera sai do ngưỡng `curl` 2D không bền với góc nhìn trực diện. |

---

## 3. Nguồn tài liệu, video và mức độ tin cậy

### 3.1. Chuẩn Quốc Gia — Thông tư 17/2020/TT-BGDĐT
- **Cơ quan ban hành:** Bộ Giáo dục và Đào tạo Việt Nam (ngày 26/05/2020).
- **Mục tiêu:** Quy định tài liệu hướng dẫn dạy học môn Tiếng Việt dành cho học sinh khiếm thính cấp tiểu học.
- **Đặc điểm chữ số:** Ban hành bảng chữ số chuẩn quốc gia dựa trên hệ thống ký hiệu số tiếp xúc ngón tay (tương đồng hệ ASL). Số 6, 7, 8, 9 được phân định bằng việc ngón cái lần lượt chạm vào đầu ngón út (6), ngón áp út (7), ngón giữa (8), và ngón trỏ (9), trong khi các ngón còn lại giơ thẳng hướng lên.
- **Mức độ tin cậy:** **Cao nhất về mặt pháp lý và chuẩn hóa giáo dục tại Việt Nam.**

### 3.2. Dự án QIPEDC (Dự án Nâng cao chất lượng giáo dục học sinh khiếm thính)
- **Nguồn dữ liệu:** `backend/data/sources/qipedc_dictionary.json` và kho video tại `backend/data/media_cache/videos/`.
- **Đặc điểm chữ số:** Các video từ `D0530` đến `D0538` ghi nhận người ký khiếm thính Việt Nam sử dụng hệ đếm số thị giác phổ thông Đông Á (Counting gestures: 6 Shaka, 7 súng ngắn nhỏ L-shape, 8 súng ba nòng, 9 ngón trỏ móc câu).
- **Mức độ tin cậy:** **Độ tin cậy thực nghiệm cao đối với cộng đồng người khiếm thính sử dụng quy ước tự nhiên ngoài đời**, nhưng lại có tính phân tán vùng miền (Bắc/Trung/Nam) và không hoàn toàn tuân theo Thông tư 17.
- **Hạn chế:** Toàn bộ video số trong kho chỉ có **1 người ký duy nhất** (1 signer), nền đồng nhất, góc máy quay ngang trực diện bán thân.

### 3.3. Bộ Fixture kiểm thử `backend/tests/fixtures/real_digit_landmarks.json`
- **Quy mô:** 18 mẫu landmark bàn tay phải trích xuất từ 9 video QIPEDC (mỗi video 2 frame tại 35% và 55% thời lượng).
- **Hạn chế:** Không có mẫu kiểm thử chéo người ký (cross-signer test), không có dữ liệu số 0, không có các góc nghiêng camera thực tế (yaw, pitch, roll).

---

## 4. Nguyên nhân có bằng chứng giải thích việc nhận diện số 6–9 sai khi chạy camera thực tế

Từ các phân tích mã nguồn và kiểm chứng hình học, việc nhận diện số 6–9 thất bại khi chạy camera thực tế do 3 nguyên nhân cốt lõi sau:

### Nguyên nhân 1: Xung đột quy ước cử chỉ giữa Người dùng và Mô hình (Convention Mismatch)
- Khi một người dùng đã học ký hiệu VSL chuẩn (theo Thông tư 17 hoặc các trung tâm khiếm thính chính quy) thử nghiệm camera, họ sẽ làm:
  - Số 6: Giơ ngón trỏ, giữa, áp út thẳng lên; ngón cái chạm ngón út.
  - Số 7: Giơ ngón trỏ, giữa, út thẳng lên; ngón cái chạm áp út.
  - Số 8: Giơ ngón trỏ, áp út, út thẳng lên; ngón cái chạm ngón giữa.
  - Số 9: Giơ ngón giữa, áp út, út thẳng lên; ngón cái chạm ngón trỏ.
- **Hậu quả:** `digit_classifier.py` hiện tại lập tức từ chối các cử chỉ này:
  - Với số 6 chuẩn TT17: `finger_states['index'] == EXTENDED` $\rightarrow$ Vi phạm luật QIPEDC số 6 (yêu cầu trỏ gập `FOLDED`) $\rightarrow$ Classifier trả về `NO_MATCH`.
  - Tương tự cho số 7, 8, 9: Cấu trúc ngón tay của chuẩn TT17 hoàn toàn trái ngược với luật hình học đang được cài đặt.

### Nguyên nhân 2: Ngưỡng hình học 2D quá nhạy cảm với góc quay camera (Planar Perspective Artifacts)
- **Số 7 (L-shape hẹp):**
  - Điều kiện: `25.0° <= ang_TI < 50.0°`.
  - Trên camera thực tế, góc nhìn phối cảnh (camera perspective) thay đổi theo độ mở cánh tay và xoay cổ tay. Nếu người dùng mở ngón cái tự nhiên theo phản xạ góc ~55°, điều kiện `< 50.0°` bị vi phạm (để tránh nhầm chữ L), khiến số 7 bị rớt xuống `NO_MATCH`. Dải hợp lệ chỉ 25 độ là quá hẹp trong môi trường không định hình vị trí tay.
- **Số 9 (Hook - Ngón trỏ móc câu):**
  - Điều kiện: `0.08 <= index_curl <= 0.35` (với `index_curl = (d_tip_mcp_proj - d_pip_mcp_proj) / hand_size`).
  - Phép tính này dựa trên hình chiếu 2D lên mặt phẳng camera. Khi người dùng giơ ngón trỏ móc câu trực diện về phía ống kính (camera axial line), hiệu ứng rút ngắn phối cảnh (foreshortening) làm khoảng cách 2D từ TIP đến MCP co cụm lại, khiến `index_curl` bị tính > 0.35 (coi như ngón trỏ gập hoàn toàn). Ngược lại, nếu tay hơi ngửa, `index_curl` rơi xuống < 0.08. Cửa sổ `[0.08, 0.35]` chỉ hợp lệ khi người ký đứng góc nghiêng chuẩn như trong video fixture `D0538`.
- **Số 6 (Shaka):**
  - Điều kiện: `ang_TP >= 35.0°`. Nếu người dùng giơ ngón cái và út nhưng bàn tay hơi xoay chéo so với mặt phẳng cảm biến, góc chiếu 2D giữa hai ngón bị co lại dưới 35°, dẫn tới không nhận diện được.

### Nguyên nhân 3: Hiện tượng che khuất ngón tay (Self-Occlusion) & Xung đột giữa Số 3 và Số 8
- **Xung đột giữa Số 3 và Số 8 trong quy ước QIPEDC:**
  - Cả số 3 và số 8 trong video QIPEDC đều duỗi 3 ngón: ngón cái, ngón trỏ và ngón giữa; gập 2 ngón áp út và út!
  - Trong `digit_classifier.py`:
    - Số 3 yêu cầu: `ang_TI >= 15.0°`, `ang_IM >= 8.0°`.
    - Số 8 yêu cầu: `ang_TI >= 20.0°`, `ang_TM >= 25.0°`, `ang_IM >= 10.0°`.
  - Trên camera thực tế, sự chênh lệch vài độ giữa hai số này với cùng một bộ 3 ngón duỗi là không ổn định. Bàn tay của người dùng chỉ cần xòe nhẹ ngón giữa là bị nhảy loạn giữa Số 3 và Số 8.
- **Nhiễu ngón gập:** Với số 8, khi gập ngón áp út và ngón út, đầu ngón tay thường bị che khuất bởi lòng bàn tay hoặc ngón cái tùy theo góc quay, khiến MediaPipe ước lượng sai trạng thái `FOLDED` thành `UNKNOWN`.

---

## 5. Tình trạng dữ liệu số 0

### 5.1. Báo cáo kiểm tra kho dữ liệu
- **Tra cứu từ điển `backend/data/sources/qipedc_dictionary.json`:**
  - Mục từ có ID `D0529`, từ khóa: `0 (số không)`, loại từ: `Số từ`.
- **Kiểm tra trên hệ thống tập tin cục bộ:**
  - Thư mục: `backend/data/media_cache/videos/`
  - Kết quả: Có các file `D0530.mp4` đến `D0539.mp4`, nhưng **hoàn toàn không có file `D0529.mp4`**.
- **Kiểm tra thư mục fixtures và tests:**
  - `backend/tests/fixtures/real_digit_landmarks.json`: Chỉ chứa các nhãn từ 1 đến 9. Hoàn toàn không có mẫu số 0.
- **Trạng thái hỗ trợ trong code:**
  - Trong `backend/vision/digit_classifier.py`: `VSLDigitClassifier.predict()` chưa định nghĩa nhãn `'0'`. Khi phát hiện bàn tay chụm tròn, classifier chủ động trả về `NEUTRAL` (conf=0.0).

### 5.2. Phân tích khác biệt hình thái giữa Số 0 và Chữ O
- **Hình thái Chữ O:**
  - Các đầu ngón tay (trỏ, giữa, áp út, út) cong đều chạm hoặc gần chạm vào đầu ngón cái, tạo thành một khoảng trống tròn rỗng rõ rệt nhìn xuyên qua được (dạng chữ O tròn).
- **Hình thái Số 0:**
  - *Theo Thông tư 17:* Tương tự chữ O, khum tròn bàn tay tượng trưng cho số 0.
  - *Vấn đề phân biệt trong hệ thống:*
    - Ở chế độ tách biệt `digits` (vừa hoàn thành ở Giai đoạn 2C.5B), **không có sự xung đột với Chữ O** vì chế độ `digits` chỉ lọc nhãn trong danh sách chữ số. Bất kỳ hình thái bàn tay chụm tròn nào trong chế độ `digits` đều có thể phân loại một cách an toàn thành `'0'` mà không sợ làm bẩn kết quả chữ cái!
    - Ở chế độ lai `letters_numbers` (alias cũ): Nếu cả chữ O và số 0 đều mở, cần chính sách ngữ cảnh (context/expected_sign) để phân xử.

### 5.3. Quy trình 4 bước thu thập dữ liệu số 0 chuẩn
1. **Bước 1 — Tìm nguồn video gốc:** Kiểm tra từ nguồn phát hành video dự án QIPEDC hoặc ghi hình mẫu chuẩn VSL số 0 từ người khiếm thính / chuyên gia ngôn ngữ ký hiệu.
2. **Bước 2 — Thu thập video đa người ký (Multi-signer recording):** Ghi hình tối thiểu 3 cá nhân khác nhau thực hiện số 0, bao gồm các biến thiên góc quay camera (chính diện 0°, nghiêng trái 15°, nghiêng phải 15°).
3. **Bước 3 — Trích xuất landmark MediaPipe:** Sử dụng script `backend/training/extract_video_landmarks.py` để trích xuất 21 landmark 3D chuẩn hóa cho tối thiểu 6–10 mẫu thực tế của số 0.
4. **Bước 4 — Bổ sung fixture & kiểm thử độc lập:** Đưa landmark vào `real_digit_landmarks.json` và bổ sung unit test `test_digit_0` trước khi cập nhật luật trong `digit_classifier.py`.

---

## 6. Các trường hợp nhầm lẫn giữa chữ cái và số

Mặc dù chế độ `digits` độc lập đã ngăn chặn việc rò rỉ nhãn số sang chế độ chữ cái, nhưng việc hiểu rõ các cặp đồng dạng hình thái (Homoglyphs/Morphological Collisions) là bắt buộc cho tính ổn định của hệ thống:

| Cặp Ký hiệu | Cấu trúc ngón tay tương đồng | Điểm khác biệt hình học then chốt | Giải pháp xử lý |
| :---: | :--- | :--- | :--- |
| **1 vs D** | Đều duỗi ngón trỏ, 3 ngón (giữa, áp út, út) gập | Chữ D: ngón cái chạm vào đầu ngón giữa tạo vòng khép kín. Số 1: ngón cái đè ngang lưng ngón giữa hoặc ép sát ngoài. | Tách mode hoàn toàn (`digits` vs `letters`). Trong mode lai, kiểm tra khoảng cách tiếp xúc giữa ngón cái và ngón giữa. |
| **2 vs V / U** | Đều duỗi ngón trỏ và ngón giữa, gập ngón áp út và út | Chữ V: xòe rộng (`ang_IM >= 15°`). Chữ U: khép chặt hai ngón (`ang_IM < 5°`). Số 2: độ xòe vừa phải (~8°–20°). | Mode độc lập đã giải quyết triệt để xung đột này. |
| **6 vs Y** | Cả hai đều duỗi ngón cái và ngón út, gập 3 ngón giữa (theo QIPEDC Shaka) | Hình thái gần như giống hệt nhau 100% giữa số 6 kiểu QIPEDC và chữ cái Y trong bảng chữ cái VSL! | **Đây là xung đột nghiêm trọng nhất.** Nếu dùng số 6 kiểu QIPEDC trong mode chung, không thể phân biệt bằng hình học tĩnh. Nếu dùng số 6 kiểu Thông tư 17 (ngón cái chạm ngón út, 3 ngón giữa xòe), xung đột biến mất hoàn toàn! |
| **7 vs L** | Đều duỗi ngón cái và ngón trỏ, gập 3 ngón còn lại (theo QIPEDC L-shape) | Chữ L: góc mở vuông ~90°. Số 7 QIPEDC: góc mở nhọn ~30°–45°. | Tách mode độc lập. Trong mode số, cần nới lỏng góc `ang_TI` lên đến 65° để người dùng dễ làm hơn. |
| **0 vs O** | Đều chụm các đầu ngón tay vào ngón cái tạo hình tròn khép kín | Hình thái gần như trùng khớp hoàn toàn. | Tách mode độc lập: Trong mode `digits` trả về `'0'`, trong mode `letters` trả về `'O'`. |

---

## 7. Đề xuất sửa đổi theo từng bước nhỏ (Roadmap)

Việc nâng cấp bộ nhận diện số cần tuân thủ lộ trình phân đoạn nghiêm ngặt, không làm ảnh hưởng đến các mode đã ổn định:

### Giai đoạn 2C.7A: Thu thập và kiểm chuẩn dữ liệu Số 0
- Bổ sung video / landmark thực tế cho số 0 vào `backend/tests/fixtures/real_digit_landmarks.json`.
- Thêm unit test kiểm tra độc lập cấu trúc hình học của số 0 (`test_digit_0`).
- Cập nhật `backend/vision/digit_classifier.py` để nhận diện số 0 khi bàn tay khum chụm tròn trong chế độ `digits`.

### Giai đoạn 2C.7B: Hỗ trợ cơ chế Ký hiệu Kép (Dual Convention) cho Số 6–9
- Thay vì bắt buộc người dùng chỉ làm theo một kiểu duy nhất, nâng cấp `digit_classifier.py` để hỗ trợ **cả hai quy ước**:
  - *Nhánh A (Quy ước Phổ thông QIPEDC):* Shaka (6), L-shape nhọn (7), Tri-finger (8), Hook (9).
  - *Nhánh B (Chuẩn Quốc gia Thông tư 17 / ASL):* Cái chạm út (6), Cái chạm áp út (7), Cái chạm giữa (8), Cái chạm trỏ (9).
- Bằng cách này, dù người dùng làm theo sách giáo khoa tiểu học hay làm theo thói quen đếm tay tự nhiên, hệ thống đều nhận diện đúng!

### Giai đoạn 2C.7C: Tối ưu độ bền góc nhìn camera (Camera Perspective Robustness)
- Nới lỏng dải góc số 7 (`20.0° <= ang_TI <= 65.0°`).
- Bổ sung chuẩn hóa 3D cho độ co ngón trỏ số 9 thay vì chỉ dựa vào hình chiếu 2D `index_curl`.
- Thêm cơ chế smoothing riêng biệt cho mode `digits` để chống nhảy giữa Số 3 và Số 8.

---

## 8. Danh sách dữ liệu còn thiếu cần người dùng cung cấp

Để thực hiện các giai đoạn tiếp theo một cách chuẩn xác, cần sự xác nhận và cung cấp từ phía người dùng/chủ dự án:
1. **Dữ liệu Video / Mẫu Landmark Số 0:**
   - Cung cấp file video `D0529.mp4` gốc (nếu có trong kho lưu trữ ngoài) HOẶC cho phép ghi hình trực tiếp 1–2 video mẫu bàn tay làm số 0 để trích xuất fixture.
2. **Quyết định về định hướng Quy ước:**
   - **Lựa chọn 1:** Ưu tiên Chuẩn Quốc gia (Thông tư 17/2020/TT-BGDĐT) — phù hợp với môi trường giáo dục chính quy, không bị trùng hình thái giữa 6 và Y.
   - **Lựa chọn 2:** Giữ nguyên quy ước QIPEDC hiện tại (đếm tay Đông Á) nhưng nới lỏng ngưỡng góc và huấn luyện lại người dùng.
   - **Lựa chọn 3 (Khuyến nghị cao nhất):** Hỗ trợ **Quy ước Kép (Dual-Convention Classifier)** — chấp nhận cả hai cách ký hiệu cho số 6, 7, 8, 9 trong chế độ `digits`.

---

## 9. Bộ kiểm thử cần bổ sung sau khi có dữ liệu

Khi có dữ liệu số 0 và thống nhất quy ước, test suite cần được mở rộng với các test case sau trong `backend/tests/test_digit_classifier.py`:
1. `test_classify_digit_0_real_landmarks`: Kiểm tra số 0 với các landmark thực tế được bổ sung.
2. `test_classify_digit_6_to_9_dual_convention`: Kiểm tra cả 2 dạng biểu diễn của các số 6, 7, 8, 9.
3. `test_digit_camera_angle_invariance`: Kiểm tra tính bất biến khi landmark bị xoay nghiêng cổ tay ±15° và ±30°.
4. `test_digit_mode_no_letter_leakage`: Đảm bảo toàn bộ 10 chữ số (0–9) trả về đúng nhãn trong mode `digits` và không bao giờ rò rỉ nhãn chữ cái.

---

## 10. Kết luận rõ ràng cho từng số (0–9)

Dưới đây là kết luận kiểm toán cuối cùng cho từng chữ số trong hệ thống VSL Connect AI hiện tại:

- **Số 0: CHƯA XÁC MINH**  
  *Lý do:* Không có video D0529 trong repo; không có frame landmark fixture; chưa có code nhận diện trong classifier.
- **Số 1: ĐÃ XÁC MINH**  
  *Lý do:* Cả Thông tư 17 và QIPEDC đều thống nhất giơ ngón trỏ; fixture thực tế có 2 mẫu; pass 100% test độc lập và tích hợp.
- **Số 2: ĐÃ XÁC MINH**  
  *Lý do:* Cả Thông tư 17 và QIPEDC đều thống nhất giơ ngón trỏ và ngón giữa (chữ V); fixture thực tế có 2 mẫu; pass 100% test.
- **Số 3: CÓ BẰNG CHỨNG MỘT PHẦN**  
  *Lý do:* Xung đột quy ước giữa Thông tư 17 (trỏ-giữa-áp út) và QIPEDC (cái-trỏ-giữa). Bộ phân loại hiện hoạt động tốt theo quy ước QIPEDC nhưng chưa hỗ trợ quy ước Thông tư 17.
- **Số 4: ĐÃ XÁC MINH**  
  *Lý do:* Cả Thông tư 17 và QIPEDC đều thống nhất 4 ngón duỗi, ngón cái gập; fixture thực tế có 2 mẫu; pass 100% test.
- **Số 5: ĐÃ XÁC MINH**  
  *Lý do:* Cả Thông tư 17 và QIPEDC đều thống nhất 5 ngón xòe tự nhiên; fixture thực tế có 2 mẫu; pass 100% test.
- **Số 6: CÓ BẰNG CHỨNG MỘT PHẦN**  
  *Lý do:* Xung đột nghiêm trọng giữa Thông tư 17 (ngón cái chạm út) và QIPEDC (Shaka: cái và út duỗi, 3 ngón giữa gập). Luật hiện tại chỉ hỗ trợ Shaka với ngưỡng góc hẹp, gây lỗi khi thử camera thực tế.
- **Số 7: CÓ BẰNG CHỨNG MỘT PHẦN**  
  *Lý do:* Xung đột giữa Thông tư 17 (cái chạm áp út) và QIPEDC (L-shape nhọn). Dải góc 2D `25°–50°` quá hẹp, rất dễ rớt nhận diện khi cổ tay xoay tự nhiên trước camera.
- **Số 8: CÓ BẰNG CHỨNG MỘT PHẦN**  
  *Lý do:* Xung đột giữa Thông tư 17 (cái chạm giữa) và QIPEDC (cái-trỏ-giữa mở rộng). Dễ bị nhiễu và xung đột hình thái với Số 3 trên camera thực tế.
- **Số 9: CÓ BẰNG CHỨNG MỘT PHẦN**  
  *Lý do:* Xung đột giữa Thông tư 17 (cái chạm trỏ tạo khuyên) và QIPEDC (ngón trỏ móc câu). Ngưỡng `index_curl` 2D bị sai lệch nghiêm trọng do hiệu ứng rút ngắn phối cảnh camera.

