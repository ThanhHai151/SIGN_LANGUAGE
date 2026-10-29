# BÁO CÁO GIAI ĐOẠN 2C.8 — XÁC MINH VÀ THU THẬP LANDMARK THẬT CHO SỐ 0

**Dự án:** VSL Connect AI — Hệ thống hỗ trợ giao tiếp Ngôn ngữ ký hiệu Việt Nam  
**Ngày thực hiện:** 29/10/2026  
**Người thực hiện:** Antigravity AI Coding Assistant  
**Trạng thái:** Hoàn thành xác minh nguồn, thu thập video gốc D0529, trích xuất landmark thật, tạo fixture và kiểm thử độc lập. Không can thiệp mã nguồn production.

---

## 1. Kết quả xác minh URL và nguồn video

Dựa trên metadata từ [vsl_lexicon.json](file:///d:/SIGN_LANGUAGE/backend/data/vsl_lexicon.json) (mục từ ID `q-0-so-khong`), hệ thống đã thăm dò trực tiếp máy chủ QIPEDC của Bộ Giáo dục và Đào tạo:

| Thành phần tài nguyên | URL nguồn từ xa | Trạng thái HTTP | Content-Type | Kích thước | Magic Bytes / Header |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Video gốc** | `https://qipedc.moet.gov.vn/videos/D0529.mp4` | **200 OK** | `video/mp4` | 376.467 bytes (~367.6 KB) | `000000206674797069736f6d` (`ftypisom` MP4 container) |
| **Thumbnail** | `https://qipedc.moet.gov.vn/thumbs/D0529.png` | **200 OK** | `image/png` | 26.394 bytes | `89504e470d0a1a0a` (Chuẩn PNG) |
| **Ảnh minh họa** | `https://qipedc.moet.gov.vn/Anh/D0529.png` | **200 OK** | `image/png` | 5.592 bytes | `89504e470d0a1a0a` (Chuẩn PNG) |

**Kết luận xác minh mạng:** Máy chủ phản hồi nội dung thực, không chuyển hướng (redirect), không trả về trang lỗi HTML hay cơ chế chặn truy cập bot/WAF.

---

## 2. Bằng chứng xác nhận nội dung video

Sau khi tải video và giải mã bằng OpenCV, các thông số kỹ thuật và hình thái cử chỉ của video `D0529.mp4` được ghi nhận như sau:

- **Độ phân giải:** 1280 x 720 (Chuẩn HD 720p, hoàn toàn đồng nhất với các video `D0530`–`D0538`).
- **Tốc độ khung hình (FPS):** 29.97 fps.
- **Tổng số khung hình:** 81 frames.
- **Thời lượng:** 2.70 giây.
- **Khoảng thời gian cử chỉ thực hiện (Active Interval):** Từ giây 0.67 đến 1.40 (Frame 20 đến 42).
- **Phân tích hình thái cử chỉ thực tế:**
  - Bàn tay phải đưa lên ngang ngực.
  - Toàn bộ 5 ngón tay khum cong lại thành hình vòm tròn.
  - Đầu các ngón trỏ, giữa, áp út chụm chạm sát vào đầu ngón cái, tạo thành một vòng khuyên tròn mở rỗng rõ ràng (hình dạng chuẩn số 0 / chữ O theo quy ước VSL).
  - Khoảng cách chuẩn hóa giữa đầu ngón trỏ và đầu ngón cái trong dải ổn định đạt mức rất nhỏ: `0.105` đến `0.131` lần kích thước lòng bàn tay (`palm_scale`).

---

## 3. Thông tin điều kiện sử dụng dữ liệu

- **Cơ quan chủ quản & Bản quyền nguồn:**
  - Dự án Nâng cao chất lượng giáo dục học sinh khiếm thính cấp tiểu học (QIPEDC - *Quality Improvement of Primary Education for Deaf Children*).
  - Chủ quản: Bộ Giáo dục và Đào tạo Việt Nam (MOET), tài trợ bởi Quỹ Phát triển Xã hội Nhật Bản (JSDF) thông qua Ngân hàng Thế giới (World Bank).
- **Mục đích sử dụng:**
  - Kho ngữ liệu video mở được MOET ban hành phục vụ học tập, nghiên cứu và hỗ trợ giao tiếp cho người khiếm thính tại Việt Nam.
  - Dự án VSL Connect AI sử dụng phi thương mại cho mục đích nghiên cứu, phát triển công cụ trợ tiếp cận và xây dựng bộ kiểm chuẩn (Ground Truth Test Fixtures).
  - Không phân phối lại trái phép hoặc biến đổi sai lệch nội dung bản quyền.

---

## 4. Vị trí lưu video và checksum xác thực

- **Đường dẫn lưu trữ cục bộ:** [`backend/data/media_cache/videos/D0529.mp4`](file:///d:/SIGN_LANGUAGE/backend/data/media_cache/videos/D0529.mp4)
- **Kích thước tệp:** 376.467 bytes
- **Mã băm toàn vẹn SHA-256:**  
  `29bb754be325199584cca2edcee23d5fcda3b54f670840b5948fcf8c176f8c3b`
- **Ngày thu thập:** 29/10/2026.

---

## 5. Thống kê trích xuất Landmark MediaPipe Hands

Trích xuất được thực hiện qua pipeline MediaPipe Hands chuẩn của dự án:

- **Tổng số frame được kiểm tra chi tiết:** 11 frames (từ frame 15 đến 65).
- **Số mẫu landmark trích xuất thành công:** 8 frames (frames 20, 25, 30, 35, 40, 45, 50, 55).
- **Độ tin cậy nhận diện bàn tay (Handedness Score):** Đạt từ **0.9849 đến 0.9971** (trung bình > 0.991).
- **Phân loại mẫu:**
  - **Mẫu ổn định đại diện (Được đưa vào Fixture chính thức):** 4 mẫu (Frame 20, 25, 30, 35).
    - `dev` sample: Frame 30 (t=1.00s, conf=0.9912) — Điểm giữa tư thế ổn định nhất, đầu ngón cái và trỏ tiếp xúc sát nhất (`d_norm = 0.105`).
    - `eval` sample: Frame 35 (t=1.17s, conf=0.9917) — Mẫu kiểm thử độc lập trong pha giữ tư thế (`d_norm = 0.107`).
    - `aux_entry` sample: Frame 20 (t=0.67s, conf=0.9902) — Pha bắt đầu khum tay hoàn chỉnh.
    - `aux_hold` sample: Frame 25 (t=0.83s, conf=0.9932) — Pha giữ tư thế mở đầu.
  - **Mẫu bị loại khỏi fixture chính thức:** 4 mẫu (Frames 45, 50, 55 và các frame ngoài dải) do người ký bắt đầu xòe tay hoặc hạ tay xuống kết thúc cử chỉ, không phản ánh tư thế tĩnh của số 0.

---

## 6. Thông tin Fixture mới và khả năng truy vết nguồn

Để bảo đảm không làm ảnh hưởng hay phá vỡ các bài kiểm tra hiện hữu của `real_digit_landmarks.json` (vốn đang khóa chặt kiểm tra cho dải số 1–9), một tập tin fixture riêng biệt đã được khởi tạo:

- **Tập tin Fixture:** [`backend/tests/fixtures/real_digit_0_landmarks.json`](file:///d:/SIGN_LANGUAGE/backend/tests/fixtures/real_digit_0_landmarks.json)
- **Cấu trúc:** Tuân thủ schema chuẩn `1.0` của dự án với đầy đủ siêu dữ liệu truy vết:
  - `digit`: `"0"`
  - `source_video_id`: `"D0529"`
  - `source_video_file`: `"backend/data/media_cache/videos/D0529.mp4"`
  - `remote_url`: `"https://qipedc.moet.gov.vn/videos/D0529.mp4"`
  - `sha256`: `"29bb754be325199584cca2edcee23d5fcda3b54f670840b5948fcf8c176f8c3b"`
  - `total_samples`: 4
  - Mỗi mẫu chứa: 21 điểm `[x, y, z]`, `split`, `frame_idx`, `timestamp_sec`, `handedness`, `confidence`, `quality_status: "ok"`.

---

## 7. Kết quả kiểm thử cấu trúc và kiểm thử hồi quy

### 7.1. Bộ kiểm thử mới cho Số 0: `test_digit_0_landmarks.py`
Tập tin kiểm thử độc lập [`backend/tests/test_digit_0_landmarks.py`](file:///d:/SIGN_LANGUAGE/backend/tests/test_digit_0_landmarks.py) được tạo với 5 test cases chuyên biệt:
1. `test_digit_0_video_exists_and_checksum`: **PASSED** — Kiểm tra file video và mã băm SHA-256.
2. `test_digit_0_fixture_structure`: **PASSED** — Kiểm tra schema, định dạng 21 điểm, tọa độ hữu hạn trong phạm vi chuẩn hóa.
3. `test_digit_0_dev_eval_distinction`: **PASSED** — Xác nhận tính độc lập tuyệt đối giữa dev frame (f30) và eval frame (f35).
4. `test_digit_0_morphological_properties`: **PASSED** — Kiểm tra đặc tính hình thái (các ngón khum, không xòe thẳng, khoảng cách đầu ngón cái - ngón trỏ < 0.25).
5. `test_digit_classifier_leaves_digit_0_neutral_for_now`: **PASSED** — Xác nhận bộ phân loại hiện tại không nhận diện tùy tiện và giữ an toàn trả về `NEUTRAL` (conf=0.0).

### 7.2. Kết quả kiểm thử toàn diện toàn hệ thống
- **Lệnh chạy:**
  ```powershell
  C:\Users\PC\AppData\Local\Programs\Python\Python312\python.exe scratch\run_all_tests.py
  ```
- **Kết quả tổng kết:**
  ```
  ====================== 189 passed, 2 warnings in 10.07s =======================
  ```
  - **184/184 tests cũ:** **100% PASSED** (Không có bất kỳ sự hồi quy nào).
  - **5/5 tests mới:** **100% PASSED**.
  - **Tổng cộng:** **189/189 tests passed.**

---

## 8. Hạn chế về số lượng mẫu và tính độc lập của dữ liệu

Mặc dù việc thu thập và trích xuất đã thành công mỹ mãn về mặt kỹ thuật, hệ thống ghi nhận các hạn chế sau:
1. **Phụ thuộc 1 người ký (Single Signer Limitation):** Video `D0529.mp4` chỉ có 1 người ký duy nhất (cùng diễn viên thực hiện các video `D0530`–`D0538`). Các mẫu dev và eval tuy ở hai thời điểm khác nhau nhưng không độc lập về mặt giải phẫu bàn tay.
2. **Góc máy cố định:** Video được quay ở góc nhìn chính diện bán thân, chưa bao phủ các góc quay nghiêng của camera người dùng (tilt, yaw).
3. **Cần kiểm thử phân định với Chữ O:** Mặc dù chế độ `digits` đã độc lập, nhưng khi triển khai cần bảo đảm luật số 0 chặt chẽ để không vô tình nhận diện các cử chỉ thả lỏng tự nhiên của bàn tay.

---

## 9. Những việc cần thực hiện ở giai đoạn tiếp theo (Giai đoạn 2C.9)

Trước khi có thể kích hoạt nhận diện Số 0 trong pipeline chính:
1. **Thiết kế luật hình học cho Số 0 trong `digit_classifier.py`:**
   - Điều kiện dương: Tất cả các ngón không duỗi (`index_ext=False, middle_ext=False, ring_ext=False, pinky_ext=False`), khoảng cách tiếp xúc `calculate_distance(thumb_tip, index_tip) / palm_scale < 0.22`.
   - Điều kiện âm: Lòng bàn tay mở rỗng (`d_pip_wrist` và khoảng không giữa các ngón phải đủ lớn để không nhầm với nắm đấm chữ S hay chữ A).
2. **Bổ sung tích hợp vào `VSLRecognizer`:**
   - Cho phép trả về nhãn `'0'` khi ở chế độ `digits` hoặc `letters_numbers`.
   - Bảo đảm chế độ `letters` tuyệt đối không bao giờ trả về `'0'`.
3. **Cập nhật bộ fixture chính thức và registry:**
   - Sau khi người dùng duyệt, có thể đồng bộ cập nhật vào `digit_video_registry.json`.

---

## 10. Xác nhận phạm vi thay đổi

- **Mã nguồn production (`backend/vision/`, `backend/app.py`, `src/`):** **Tuyệt đối không sửa đổi bất kỳ dòng mã nào.**
- **Mô hình & Checkpoint:** **Không huấn luyện lại, không thay đổi weights.**
- **Fixture cũ (`real_digit_landmarks.json`, `digit_video_registry.json`):** **Giữ nguyên 100%, không sửa đổi.**
- **Các thành phần được bổ sung trong giai đoạn này:**
  1. File video: [`backend/data/media_cache/videos/D0529.mp4`](file:///d:/SIGN_LANGUAGE/backend/data/media_cache/videos/D0529.mp4)
  2. File fixture mới: [`backend/tests/fixtures/real_digit_0_landmarks.json`](file:///d:/SIGN_LANGUAGE/backend/tests/fixtures/real_digit_0_landmarks.json)
  3. File test mới: [`backend/tests/test_digit_0_landmarks.py`](file:///d:/SIGN_LANGUAGE/backend/tests/test_digit_0_landmarks.py)
  4. File báo cáo: [`backend/reports/phase_2c_8_zero_landmark_collection.md`](file:///d:/SIGN_LANGUAGE/backend/reports/phase_2c_8_zero_landmark_collection.md)

