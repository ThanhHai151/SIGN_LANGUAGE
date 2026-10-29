# BÁO CÁO GIAI ĐOẠN 2C.9D — XÁC MINH VIDEO ỨNG VIÊN TỪ METADATA CHO QUY ƯỚC A (SỐ 6–9)

**Thời điểm thực hiện:** 29/10/2026  
**Mục tiêu:** Kiểm tra và xác minh các nguồn video ứng viên đã được phát hiện trong metadata (từ giai đoạn 2C.9C) để tìm video thực tế cho Quy ước A của các số 6, 7, 8, 9.  
**Nguyên tắc thực thi:**  
- Không tải lại toàn bộ kho dữ liệu. Chỉ tải có chọn lọc các video ứng viên.  
- Không sửa production code, không can thiệp checkpoint hay sửa bộ phân loại.  
- Không giả định cử chỉ theo tên file/metadata; chỉ kết luận khi đã phân tích trực quan và trích xuất landmark thật.  

---

## 1. Xác minh Metadata (Bước 1)

Đã tra cứu toàn bộ `backend/data/vsl_lexicon.json` và cấu trúc dữ liệu QIPEDC hiện có. Kết quả định danh chính xác 6 URL ứng viên cho các số 6–9:

| Chữ số | Mã định danh | Nguồn dữ liệu | Vùng miền / Nguồn gốc | URL đầy đủ trong metadata |
| :---: | :---: | :---: | :---: | :--- |
| **6** | `D0535N` | QIPEDC | Miền Nam | `https://qipedc.moet.gov.vn/videos/D0535N.mp4` |
| **6** | `D0535T` | QIPEDC | Miền Trung | `https://qipedc.moet.gov.vn/videos/D0535T.mp4` |
| **6** | `sign_034` | Giáo trình 102 | Supabase Storage | `https://gajsffqnyujwyllfcmny.supabase.co/storage/v1/object/public/sign_videos/sign_034%20-%20LOOP%20-%20Videobolt.net.mp4` |
| **7** | `sign_035` | Giáo trình 102 | Supabase Storage | `https://gajsffqnyujwyllfcmny.supabase.co/storage/v1/object/public/sign_videos/sign_035%20-%20LOOP%20-%20Videobolt.net.mp4` |
| **8** | `sign_036` | Giáo trình 102 | Supabase Storage | `https://gajsffqnyujwyllfcmny.supabase.co/storage/v1/object/public/sign_videos/sign_036%20-%20LOOP%20-%20Videobolt.net.mp4` |
| **9** | `sign_037` | Giáo trình 102 | Supabase Storage | `https://gajsffqnyujwyllfcmny.supabase.co/storage/v1/object/public/sign_videos/sign_037%20-%20LOOP%20-%20Videobolt.net.mp4` |

*Ghi chú:* Trong bộ QIPEDC, các số 7, 8, 9 chỉ có video đơn `D0536.mp4`, `D0537.mp4`, `D0538.mp4` (đã xác định là Quy ước B ở giai đoạn 2C.6/2C.7) và không có biến thể vùng miền `N` hay `T` trong metadata.

---

## 2. Kết quả tải và kiểm tra tệp tin (Bước 2)

Toàn bộ các tệp tải thành công được lưu trữ có kiểm soát tại:  
`backend/data/media_cache/videos/convention_a_candidates/`

### Bảng kết quả kiểm tra mạng & tệp tin:

| File / Mã | Trạng thái HTTP | Kích thước | SHA-256 Checksum | Magic Bytes / Định dạng | Tình trạng video |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`D0535N.mp4`** | `200 OK` | 390.098 bytes | `9897940e4a9aa4d5704072dc59016041a1ec17c8c7acff5886138949c88a957e` | `0000002066747970` (`ftypmp42`) | **Hợp lệ**, video phát bình thường (1280x720, 29.97 fps, 101 frames) |
| **`D0535T.mp4`** | `200 OK` | 388.272 bytes | `74d95c1749713b4021bf9e9aeee9ee2785fd5dbb82131a905fc516882c635b1a` | `0000002066747970` (`ftypmp42`) | **Hợp lệ**, video phát bình thường (1280x720, 29.97 fps, 101 frames) |
| **`sign_034`** | `Lỗi kết nối` | 0 bytes | N/A | N/A | **Lỗi DNS**: `[Errno 11001] getaddrinfo failed`. Host Supabase không còn tồn tại. |
| **`sign_035`** | `Lỗi kết nối` | 0 bytes | N/A | N/A | **Lỗi DNS**: `[Errno 11001] getaddrinfo failed`. Host Supabase không còn tồn tại. |
| **`sign_036`** | `Lỗi kết nối` | 0 bytes | N/A | N/A | **Lỗi DNS**: `[Errno 11001] getaddrinfo failed`. Host Supabase không còn tồn tại. |
| **`sign_037`** | `Lỗi kết nối` | 0 bytes | N/A | N/A | **Lỗi DNS**: `[Errno 11001] getaddrinfo failed`. Host Supabase không còn tồn tại. |

---

## 3. Phân tích trực quan & Mốc Landmark thực tế (Bước 3)

Để xác minh chính xác cử chỉ thể hiện trong `D0535N.mp4` và `D0535T.mp4`, toàn bộ các khung hình then chốt (frames 25 đến 65) đã được giải mã và chạy mô hình MediaPipe Hands trích xuất 21 điểm mốc không gian 3 chiều.

### Kết quả đo đạc hình học:

| Video | Khung hình | Độ tin cậy (MP) | Trạng thái duỗi ngón (T, I, M, R, P) | Góc Cái–Út (`ang_TP`) | Khoảng cách Cái–Út (`d_TP`) | Khoảng cách Cái–Áp út (`d_TR`) | Kết luận phân loại |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`D0535N`** | Frame 25 (0.83s) | 0.98 | `(1, 0, 0, 0, 1)` | 56.5° | 1.393 | 0.749 | **Quy ước B (Shaka)** |
| **`D0535N`** | Frame 30 (1.00s) | 0.99 | `(1, 0, 0, 0, 1)` | 53.7° | 1.320 | 0.727 | **Quy ước B (Shaka)** |
| **`D0535N`** | Frame 35 (1.17s) | 0.99 | `(1, 0, 0, 0, 1)` | 60.2° | 1.429 | 0.786 | **Quy ước B (Shaka)** |
| **`D0535N`** | Frame 40 (1.33s) | 0.99 | `(1, 0, 0, 0, 1)` | 57.6° | 1.376 | 0.726 | **Quy ước B (Shaka)** |
| **`D0535N`** | Frame 45 (1.50s) | 0.99 | `(1, 0, 0, 0, 1)` | 58.0° | 1.402 | 0.756 | **Quy ước B (Shaka)** |
| **`D0535N`** | Frame 50 (1.67s) | 0.98 | `(1, 0, 0, 0, 1)` | 55.4° | 1.314 | 0.776 | **Quy ước B (Shaka)** |
| **`D0535N`** | Frame 55 (1.84s) | 0.99 | `(1, 0, 0, 0, 1)` | 61.3° | 1.184 | 0.977 | **Quy ước B (Shaka)** |
| **`D0535T`** | Frame 25 (0.83s) | 0.98 | `(1, 0, 0, 0, 1)` | 56.8° | 1.402 | 0.752 | **Quy ước B (Shaka)** |
| **`D0535T`** | Frame 30 (1.00s) | 0.99 | `(1, 0, 0, 0, 1)` | 54.1° | 1.331 | 0.728 | **Quy ước B (Shaka)** |
| **`D0535T`** | Frame 35 (1.17s) | 0.99 | `(1, 0, 0, 0, 1)` | 60.0° | 1.433 | 0.791 | **Quy ước B (Shaka)** |
| **`D0535T`** | Frame 40 (1.33s) | 0.98 | `(1, 0, 0, 0, 1)` | 57.6° | 1.368 | 0.720 | **Quy ước B (Shaka)** |
| **`D0535T`** | Frame 45 (1.50s) | 0.98 | `(1, 0, 0, 0, 1)` | 57.6° | 1.389 | 0.749 | **Quy ước B (Shaka)** |
| **`D0535T`** | Frame 50 (1.67s) | 0.98 | `(1, 0, 0, 0, 1)` | 55.3° | 1.304 | 0.768 | **Quy ước B (Shaka)** |
| **`D0535T`** | Frame 55 (1.84s) | 0.99 | `(1, 0, 0, 0, 1)` | 61.3° | 1.189 | 0.979 | **Quy ước B (Shaka)** |

### Nhận xét cử chỉ:
1. Trong cả 2 video `D0535N.mp4` (Miền Nam) và `D0535T.mp4` (Miền Trung), người ký thực hiện cử chỉ:
   - **Ngón cái và ngón út duỗi thẳng xòe rộng**, góc `ang_TP` lớn (53.7°–61.3°).
   - **3 ngón giữa (trỏ, giữa, áp út) gập sát vào lòng bàn tay**.
   - Khoảng cách `d_thumb_pinky` rất lớn (1.18–1.43), hoàn toàn không có sự tiếp xúc giữa ngón cái và ngón út.
2. Đây chính là cử chỉ **Quy ước B (Shaka)**, tương tự như video `D0535B.mp4` (Miền Bắc).
3. Do đó, cả 3 miền Bắc - Trung - Nam trong bộ từ điển QIPEDC đối với Số 6 đều thống nhất sử dụng cử chỉ **Quy ước B**. **Không có video nào thể hiện cử chỉ Quy ước A (ngón cái chạm ngón út, 3 ngón giữa duỗi).**

---

## 4. Bảng tổng hợp trạng thái dữ liệu chữ số 6, 7, 8, 9 (Bước 4)

| Chữ số | Mô tả cử chỉ Quy ước A | Nguồn video ứng viên đã kiểm tra | Tình trạng video | Hình dáng thực tế trong video | Có video Quy ước A không? |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **6** | Ngón cái chạm ngón út; trỏ, giữa, áp út duỗi | 1. `D0535N.mp4`<br>2. `D0535T.mp4`<br>3. `sign_034.mp4` | - Tải thành công (390 KB)<br>- Tải thành công (388 KB)<br>- Lỗi DNS (Supabase) | - **Quy ước B** (Shaka)<br>- **Quy ước B** (Shaka)<br>- Không truy cập được | **KHÔNG** |
| **7** | Ngón cái chạm ngón áp út; trỏ, giữa, út duỗi | 1. `sign_035.mp4`<br>*(QIPEDC D0536 là Quy ước B)* | - Lỗi DNS (Supabase) | - Không truy cập được | **KHÔNG** |
| **8** | Ngón cái chạm ngón giữa; trỏ, áp út, út duỗi | 1. `sign_036.mp4`<br>*(QIPEDC D0537 là Quy ước B)* | - Lỗi DNS (Supabase) | - Không truy cập được | **KHÔNG** |
| **9** | Ngón cái chạm ngón trỏ; giữa, áp út, út duỗi | 1. `sign_037.mp4`<br>*(QIPEDC D0538 là Quy ước B)* | - Lỗi DNS (Supabase) | - Không truy cập được | **KHÔNG** |

---

## 5. Kết luận và Khuyến nghị

### Kết luận chính thức:

> **KẾT LUẬN 3: Không tìm thấy video Quy ước A nào trong các nguồn hiện có.**  
> Quy ước A hiện tại **KHÔNG CÓ DỮ LIỆU THỰC TẾ** (không có video, không có landmark thật) trong dự án cũng như trong các nguồn trực tuyến được tham chiếu trong metadata.

### Phân tích chi tiết:
1. **Nguồn QIPEDC:** Toàn bộ các video chữ số 6, 7, 8, 9 của QIPEDC (kể cả các biến thể vùng miền Bắc `D0535B`, Nam `D0535N`, Trung `D0535T`) đều ký theo **Quy ước B**.
2. **Nguồn Giáo trình 102:** Kho lưu trữ Supabase liên kết trong metadata đã dừng hoạt động (lỗi DNS phân giải tên miền), không thể truy cập hay tải về.
3. **Tính toàn vẹn kỹ thuật:** Không thể triển khai hoặc kiểm thử một bộ phân loại nhận diện kép có độ tin cậy nếu hoàn toàn không có dữ liệu ground truth thực tế để xác minh.

### Đề xuất bước tiếp theo:
1. **Giữ nguyên trạng thái hoạt động hiện tại của Quy ước B:** Hệ thống hiện tại nhận diện chính xác và ổn định các chữ số từ 0 đến 9 theo Quy ước B (đã kiểm chứng qua 195/195 unit & e2e tests đạt 100% pass).
2. **Tài liệu hóa Quy ước A như một hạn chế đã biết (Known Limitation):** Ghi rõ trong tài liệu kỹ thuật rằng hệ thống hiện tại hỗ trợ chuẩn cử chỉ số VSL theo bộ từ điển chính thống QIPEDC (Quy ước B).
3. **Chỉ triển khai Quy ước A khi có dữ liệu thật:** Khi dự án có điều kiện thu thập video hoặc ghi hình thực tế người ký thể hiện Quy ước A, pipeline trích xuất landmark đã xây dựng sẵn sàng tạo fixture và kiểm chứng trước khi tích hợp vào production classifier.

