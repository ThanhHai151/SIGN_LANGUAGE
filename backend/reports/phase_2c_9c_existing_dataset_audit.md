# BÁO CÁO GIAI ĐOẠN 2C.9C — KIỂM TRA LẠI KHO DỮ LIỆU QIPEDC ĐÃ LƯU

**Dự án:** VSL Connect AI — Hệ thống hỗ trợ giao tiếp Ngôn ngữ ký hiệu Việt Nam  
**Ngày thực hiện:** 29/10/2026  
**Người thực hiện:** Antigravity AI Coding Assistant  
**Trạng thái:** Hoàn thành kiểm toán toàn bộ kho ngữ liệu QIPEDC cục bộ và từ điển metadata. Không tải video mới, không sửa mã nguồn production, không thay đổi checkpoint/fixture.

---

## 1. Tổng quan Kiểm toán và Phương pháp Rà soát

Nhằm trả lời dứt khoát câu hỏi: *"Liệu trong kho dữ liệu QIPEDC và danh mục từ điển hiện có của dự án đã có sẵn nguồn video nào thể hiện Quy ước A (Chuẩn Thông tư 17: ngón cái chạm lần lượt vào các ngón) cho các số 6, 7, 8, 9 hay chưa?"*, hệ thống đã tiến hành rà soát 3 lớp dữ liệu:

1. **Lớp 1 — Kho video cục bộ (`backend/data/media_cache/videos/`):** Quét toàn bộ 760 tệp video hiện lưu trên đĩa, đối chiếu mã tệp và thời lượng.
2. **Lớp 2 — Từ điển và Danh mục Metadata:** Rà soát 4.362 mục từ trong `qipedc_dictionary.json` và 3.363 mục từ trong `vsl_lexicon.json`, trích xuất toàn bộ biến thể (variants), nhãn vùng miền (Bắc/Trung/Nam) và liên kết video.
3. **Lớp 3 — Giải mã hình ảnh và kiểm tra Landmark thực tế:** Sử dụng OpenCV và MediaPipe Hands phân tích từng phân đoạn thời gian của các video liên quan để kiểm chứng trực quan cử chỉ người ký.

---

## 2. Bảng Đối chiếu Chi tiết Video Cục bộ liên quan đến Số 6–9

Trong tổng số 760 tệp video cục bộ, có đúng **4 tệp video** tương ứng với các số từ 6 đến 9:

| Số | Mã tệp cục bộ | Kích thước | Tổng số frame | Thời lượng | Phân tích hình thái cử chỉ qua hình ảnh thực tế | Quy ước xác định |
| :---: | :--- | :---: | :---: | :---: | :--- | :---: |
| **6** | `D0535B.mp4` | 1.494.462 bytes | 279 frames | 9.31 giây | **Pha 1 (t=3.0s–3.5s):** Ngón cái và ngón út duỗi sang hai bên, 3 ngón giữa gập chặt (Ký hiệu Shaka).<br>**Pha 2 (t=6.5s–7.5s):** Ngón cái giơ đơn (thumbs-up), không có động tác chạm ngón cái vào ngón út với 3 ngón xòe. | **Quy ước B (100%)** |
| **7** | `D0536.mp4` | 488.279 bytes | 105 frames | 3.50 giây | Ngón cái và ngón trỏ mở rộng hình chữ L góc nhọn (`ang_TI = 28°–38°`), 3 ngón giữa-áp út-út gập chặt. Không chạm ngón cái vào ngón áp út. | **Quy ước B (100%)** |
| **8** | `D0537.mp4` | 662.667 bytes | 121 frames | 4.04 giây | Ngón cái, ngón trỏ và ngón giữa xòe mở rộng (súng 3 nòng), ngón áp út và út gập. Không chạm ngón cái vào ngón giữa. | **Quy ước B (100%)** |
| **9** | `D0538.mp4` | 535.537 bytes | 113 frames | 3.77 giây | Ngón trỏ co khum hình móc câu (`index_curl = 0.13–0.17`), 3 ngón giữa-áp út-út gập. Không tạo khuyên tròn chạm ngón cái vào ngón trỏ. | **Quy ước B (100%)** |

> [!NOTE]
> Video `D0535B.mp4` có thời lượng dài (9.31s) do người ký thực hiện 2 lần lặp lại (repetition), trong đó pha 1 là Shaka và pha 2 là cử chỉ phụ ngón cái giơ đơn. Toàn bộ video **hoàn toàn không chứa cử chỉ Quy ước A** (ngón cái chạm ngón út, 3 ngón giữa xòe).

---

## 3. Rà soát Danh mục Từ điển và Biến thể Vùng miền

Kiểm tra chi tiết các trường `variants` trong [`vsl_lexicon.json`](file:///d:/SIGN_LANGUAGE/backend/data/vsl_lexicon.json) và [`qipedc_dictionary.json`](file:///d:/SIGN_LANGUAGE/backend/data/sources/qipedc_dictionary.json):

### 3.1. Đối với Số 6 (Mục từ `q-6`):
- `video_url`: `https://qipedc.moet.gov.vn/videos/D0535B.mp4` (Miền Bắc — ĐÃ CÓ CỤC BỘ)
- Danh mục biến thể:
  1. `Region: 'B'`, Label: 'Miền Bắc', URL: `https://qipedc.moet.gov.vn/videos/D0535B.mp4` $\rightarrow$ Đã lưu cục bộ (Quy ước B - Shaka).
  2. `Region: 'T'`, Label: 'Miền Trung', URL: `https://qipedc.moet.gov.vn/videos/D0535T.mp4` $\rightarrow$ **Chưa có trên đĩa cục bộ.**
  3. `Region: 'N'`, Label: 'Miền Nam', URL: `https://qipedc.moet.gov.vn/videos/D0535N.mp4` $\rightarrow$ **Chưa có trên đĩa cục bộ.**
  4. `Region: 'GT'`, Label: 'Giáo trình 102', URL: `https://.../sign_034...mp4` $\rightarrow$ **Chưa có trên đĩa cục bộ.**

### 3.2. Đối với Số 7, 8, 9 (Mục từ `q-7`, `q-8`, `q-9`):
- Trong `qipedc_dictionary.json`, các số 7, 8, 9 chỉ có duy nhất 1 mục từ mã số không có hậu tố vùng miền (`D0536`, `D0537`, `D0538`), không có mã `D0536B/N/T`, `D0537B/N/T` hay `D0538B/N/T`.
- Trong `vsl_lexicon.json`, mỗi số chỉ có thêm 1 biến thể từ xa thuộc nguồn Giáo trình 102 (Supabase):
  - Số 7: `sign_035...mp4` $\rightarrow$ **Chưa có trên đĩa cục bộ.**
  - Số 8: `sign_036...mp4` $\rightarrow$ **Chưa có trên đĩa cục bộ.**
  - Số 9: `sign_037...mp4` $\rightarrow$ **Chưa có trên đĩa cục bộ.**

---

## 4. Phân loại 4 Nhóm Dữ liệu theo Yêu cầu

Tuân thủ nghiêm ngặt nguyên tắc kiểm toán khách quan và không phỏng đoán:

### Nhóm 1: Video QIPEDC đã có trong kho cục bộ
- Gồm đúng 4 video: `D0535B.mp4`, `D0536.mp4`, `D0537.mp4`, `D0538.mp4`.
- Tình trạng: Đã lưu tại `backend/data/media_cache/videos/`, đã trích xuất 8 mẫu landmark vào `real_digit_landmarks.json`.
- Quy ước: **100% thuộc Quy ước B (Counting gestures Đông Á).**

### Nhóm 2: Video có thể tải từ URL nhưng chưa lưu cục bộ
- Các biến thể QIPEDC vùng miền:
  - `https://qipedc.moet.gov.vn/videos/D0535T.mp4` (Số 6 Miền Trung)
  - `https://qipedc.moet.gov.vn/videos/D0535N.mp4` (Số 6 Miền Nam)
- Các video thuộc nguồn Giáo trình 102:
  - `sign_034...mp4` (Số 6), `sign_035...mp4` (Số 7), `sign_036...mp4` (Số 8), `sign_037...mp4` (Số 9).
- Tình trạng: Có URL trong metadata nhưng tệp nhị phân MP4 chưa tồn tại trên đĩa cục bộ.

### Nhóm 3: Video thể hiện Quy ước A đã được xác minh
- **Số lượng: 0 video.**
- Trong toàn bộ repository và kho lưu trữ cục bộ, **chưa có bất kỳ video nào được xác minh thể hiện Quy ước A**.

### Nhóm 4: Video chưa thể xác định quy ước
- Gồm toàn bộ các video thuộc Nhóm 2 (`D0535T.mp4`, `D0535N.mp4`, và 4 video Giáo trình 102 `sign_034`–`sign_037`).
- Lý do: Tệp chưa có trên đĩa cục bộ, chưa được giải mã khung hình và kiểm tra trực quan. Theo đúng quy tắc kiểm toán: *Không suy luận quy ước chỉ từ tên file hay số thứ tự*.

---

## 5. Kết luận Kiểm toán

> [!IMPORTANT]
> **KẾT LUẬN CUỐI CÙNG:**
> 1. Kho dữ liệu QIPEDC cục bộ hiện tại **HOÀN TOÀN KHÔNG CHỨA NGUỒN VIDEO PHÙ HỢP VỚI QUY ƯỚC A** cho các số 6, 7, 8, 9.
> 2. Toàn bộ 4 video số 6–9 đang lưu trên đĩa đều là video Quy ước B (Shaka, súng ngắn L, súng 3 nòng, móc câu).
> 3. Không thể triển khai Quy ước A nếu chỉ trông cậy vào dữ liệu cục bộ sẵn có.

---

## 6. Danh sách Dữ liệu Cụ thể Còn thiếu

Để có thể triển khai nhận diện kép cho Số 6–9 trong các giai đoạn tiếp theo, hệ thống còn thiếu chính xác các dữ liệu sau:

1. **Dữ liệu Video / Mẫu thực tế cho Quy ước A (4 số x 2 split = tối thiểu 8 mẫu landmark):**
   - **Số 6 (TT17):** Video/hình ảnh ngón cái chạm ngón út, 3 ngón giữa (trỏ, giữa, áp út) xòe thẳng.
   - **Số 7 (TT17):** Video/hình ảnh ngón cái chạm ngón áp út, 3 ngón (trỏ, giữa, út) xòe thẳng.
   - **Số 8 (TT17):** Video/hình ảnh ngón cái chạm ngón giữa, 3 ngón (trỏ, áp út, út) xòe thẳng.
   - **Số 9 (TT17):** Video/hình ảnh ngón cái chạm ngón trỏ tạo khuyên tròn, 3 ngón (giữa, áp út, út) xòe thẳng.
2. **Khảo sát mở rộng các URL từ xa (Tùy chọn nếu được phê duyệt):**
   - Kiểm tra hai tệp từ xa `D0535N.mp4` và `D0535T.mp4` trên cổng QIPEDC xem biến thể Miền Nam / Miền Trung có phải là ký hiệu Quy ước A hay không.
   - Kiểm tra nguồn "Giáo trình 102" (`sign_034`–`sign_037`) trên Supabase xem các video này thể hiện Quy ước nào.

---

> [!TIP]
> **Cam kết ranh giới:** Không có mã nguồn production nào bị sửa đổi. Hệ thống dừng lại tại đây và chờ chỉ đạo tiếp theo từ phía người dùng.

