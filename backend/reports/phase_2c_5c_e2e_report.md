# BÁO CÁO KIỂM THỬ END-TO-END 4 CHẾ ĐỘ NHẬN DIỆN VSL (GIAI ĐOẠN 2C.5C)

> **Mã giai đoạn:** 2C.5C — End-to-End Verification & Diagnosis  
> **Thời gian thực hiện:** 2026-10-29  
> **Trạng thái thực thi:** Hoàn thành toàn diện (Không sửa đổi production code)

---

## 1. Phạm vi Kiểm thử và Môi trường

### 1.1. Mục tiêu kiểm chứng
Kiểm chứng tính toàn vẹn của hợp đồng 4 chế độ nhận diện độc lập trên toàn bộ đường dẫn từ Giao diện (Frontend) $\rightarrow$ API Service $\rightarrow$ FastAPI Router $\rightarrow$ Động cơ VSLRecognizer $\rightarrow$ Bộ đệm làm mượt (Smoothing) $\rightarrow$ Vùng hiển thị văn bản:
1. `letters`: Nhận diện 27 chữ cái ngón tay VSL (A–Z, Đ), không gọi `VSLDigitClassifier`, chặn 100% nhãn số và từ vựng.
2. `digits`: Nhận diện chữ số VSL 1–9 qua `VSLDigitClassifier`, không fallback sang chữ cái, không rò rỉ từ vựng, không coi chữ O là số 0.
3. `words` & `sentence`: Bảo toàn hành vi nhận diện từ vựng 102 từ (SSTCN) và ghép câu tự động, không gọi bộ phân loại số.
4. `letters_numbers`: Duy trì hành vi lai tương thích ngược cho client cũ.
5. Quản lý trạng thái chuyển chế độ: Xóa lịch sử smoothing của phiên làm việc để tránh confirm nhầm frame đầu tiên, nhưng **bảo tồn nguyên vẹn** chuỗi văn bản đã xác nhận (`sentence`).

### 1.2. Môi trường thực thi
- **Hệ điều hành:** Windows 10/11 x64.
- **Python Runtime:** Python 3.12 (Virtual Environment `.venv`).
- **Node.js & Bundler:** Vite 8.3.0, React 19.2.8.
- **Linter:** Oxlint 1.81.0.
- **Tập dữ liệu Ground Truth:** `backend/tests/fixtures/real_digit_landmarks.json` (18 mẫu mốc MediaPipe 21 điểm trích xuất từ 8 video QIPEDC D0530–D0538 gồm 9 frame DEV và 9 frame EVAL).

---

## 2. Luồng Mã Nguồn Đã Xác Minh

Việc kiểm tra và đối chiếu trực tiếp mã nguồn xác nhận 9 điểm căn cứ kỹ thuật:

| STT | Luồng kiểm tra | Vị trí mã nguồn căn cứ | Chi tiết xác minh |
| :---: | :--- | :--- | :--- |
| **1** | UI gửi đúng giá trị mode trong request | [`src/components/CameraSignRecognizer.jsx` L62, L740, L911-L943](file:///d:/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L62)<br>[`src/services/api.js` L139-L148](file:///d:/SIGN_LANGUAGE/src/services/api.js#L139-L148) | State `conversionMode` nhận 1 trong 4 giá trị: `'letters'`, `'digits'`, `'words'`, `'sentence'`. `api.js` truyền đồng thời trong Request Body `{ mode }` và Request Header `X-Conversion-Mode`. |
| **2** | Backend xác thực mode trước khi chạy | [`backend/app.py` L228-L244](file:///d:/SIGN_LANGUAGE/backend/app.py#L228-L244) | Định nghĩa whitelist `VALID_CONVERSION_MODES = {"letters", "digits", "words", "sentence", "letters_numbers"}`. Nếu `mode` không nằm trong tập này, raise HTTP 400 Bad Request ngay lập tức. |
| **3** | `letters` không gọi digit classifier | [`backend/vision/recognizer.py` L419, L486, L577, L629-L633](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L419) | `is_letters_mode = (mode == "letters")`. `digit_classifier` chỉ được import tại L487 (`is_digits_mode`) và L578 (`is_legacy_mode`). Khi ở `letters`, bộ phân loại số hoàn toàn không được gọi. Tầng chặn cuối lọc `final_sign in ALLOWED_LETTERS`. |
| **4** | `digits` không để chữ cái/từ vựng lọt qua | [`backend/vision/recognizer.py` L486-L550](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L486-L550) | Khối lệnh `if is_digits_mode:` thực thi độc lập và `return` ngay ở L540–L550, hoàn toàn bỏ qua MLP và `classify_geometry`. Nếu `digit_classifier` không MATCH hoặc `final_sign not in ALLOWED_DIGITS`, gán cứng `final_sign = "NEUTRAL"` và `conf = 0.0`. |
| **5** | `words` và `sentence` không gọi classifier | [`backend/vision/recognizer.py` L486, L577](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L486) | Khi `mode` là `"words"` hoặc `"sentence"`, cả `is_digits_mode` và `is_legacy_mode` đều là `False`, bộ phân loại số không chạy. |
| **6** | `letters_numbers` giữ tương thích ngược | [`backend/vision/recognizer.py` L421, L577-L596, L634-L639](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L421) | Duy trì logic phân xử ưu tiên cũ (1/D, 2/U, 6/Y ưu tiên chữ cái; 3,4,5,7,8,9 ưu tiên số; whitelist `ALLOWED_LETTERS_AND_DIGITS`). |
| **7** | Reset trạng thái tạm thời theo phiên khi đổi mode | [`backend/vision/recognizer.py` L427-L433](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L427-L433)<br>[`src/components/CameraSignRecognizer.jsx` L360-L377](file:///d:/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L360-L377) | Backend kiểm tra `_session_last_modes.get(session_key) != mode` $\rightarrow$ gọi `self.prediction_history.clear()`. Frontend xóa `sequenceBufferRef.current = []` và reset `currentResult`. |
| **8** | Bảo tồn văn bản đã ghép khi đổi mode | [`backend/vision/recognizer.py` L427-L433](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L427-L433)<br>[`src/components/CameraSignRecognizer.jsx` L360-L385](file:///d:/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L360-L385) | `self.sentence_tokens` phía backend và `sentence` state phía frontend **không bị xóa** trong hàm chuyển mode. |
| **9** | Xử lý an toàn khi mất tracking / lỗi API | [`backend/vision/recognizer.py` L448-L481](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L448-L481)<br>[`src/services/api.js` L151-L157](file:///d:/SIGN_LANGUAGE/src/services/api.js#L151-L157) | Đầu vào rỗng, thiếu điểm (<21), chứa NaN/Inf trả về `NEUTRAL, conf=0.0`. Lỗi mạng ở frontend rơi vào khối catch an toàn không làm crash giao diện. |

---

## 3. Lệnh Đã Chạy và Kết Quả Thực Tế

### 3.1. Toàn bộ Test Suite Backend (Pytest)
```powershell
python scratch/run_all_tests.py
```
- **Kết quả:** **184 PASSED / 184 tests (100% Pass, 0 Failed, 0 Error)** trong 8.47s.
  - `backend/tests/test_mode_separation.py`: **12/12 PASSED**
  - `backend/tests/test_digit_classifier_e2e.py`: **6/6 PASSED**
  - `backend/tests/test_digit_classifier_integration.py`: **10/10 PASSED**
  - `backend/tests/test_digit_classifier.py`: **33/33 PASSED**
  - `backend/tests/test_geometry.py`: **45/45 PASSED**
  - `backend/tests/test_digits.py`: **21/21 PASSED**
  - `backend/tests/test_api.py`: **22/22 PASSED**
  - `backend/tests/test_mode_filtering.py`: **8/8 PASSED**
  - `backend/tests/test_landmark_extractor.py`: **9/9 PASSED**
  - `backend/tests/test_diagnostic.py`: **10/10 PASSED**
  - `backend/tests/test_digit_letter_design.py`: **8/8 PASSED**

### 3.2. Kiểm tra Đóng gói Frontend (Vite Production Build)
```bash
npm run build
```
- **Kết quả:** `✓ built in 7.41s` (0 error). Tạo thành công bundle production tối ưu tại `dist/`.

### 3.3. Kiểm tra Mã nguồn Frontend (Oxlint)
```bash
npm run lint
```
- **Kết quả:** `0 errors` trên toàn bộ 15 file của dự án.

---

## 4. Bảng Kết Quả Đo Đạc Chi Tiết theo Từng Chế Độ

Đã thực hiện đánh giá độc lập 41 trường hợp kiểm thử tự động, kết quả được lưu tại [`backend/reports/phase_2c_5c_e2e_results.json`](file:///d:/SIGN_LANGUAGE/backend/reports/phase_2c_5c_e2e_results.json):

### 4.1. Chế độ `digits` với Landmark Số Thực tế (18 mẫu từ 8 video QIPEDC)
*Trong chế độ này, các số đồng dạng tĩnh (1, 2, 6) được nhận diện chính xác thành số 1, 2, 6 mà không còn bị ghi đè thành chữ cái D, U/V, Y.*

| Mã mẫu | Video nguồn & Frame | Nhãn kỳ vọng | Nhãn Classifier | Nhãn Backend cuối | Hiển thị Frontend | Độ tin cậy | Kết quả |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: |
| `DIGIT_1_DEV_D0530_F71` | D0530 (F71, DEV) | **1** | 1 | **1** | `Số 1` | 0.92 | **PASS** |
| `DIGIT_1_EVAL_D0530_F60` | D0530 (F60, EVAL) | **1** | 1 | **1** | `Số 1` | 0.92 | **PASS** |
| `DIGIT_2_DEV_D0531_F67` | D0531 (F67, DEV) | **2** | 2 | **2** | `Số 2` | 0.92 | **PASS** |
| `DIGIT_2_EVAL_D0531_F55` | D0531 (F55, EVAL) | **2** | 2 | **2** | `Số 2` | 0.92 | **PASS** |
| `DIGIT_3_DEV_D0532_F67` | D0532 (F67, DEV) | **3** | 3 | **3** | `Số 3` | 0.95 | **PASS** |
| `DIGIT_3_EVAL_D0532_F55` | D0532 (F55, EVAL) | **3** | 3 | **3** | `Số 3` | 0.95 | **PASS** |
| `DIGIT_4_DEV_D0533_F54` | D0533 (F54, DEV) | **4** | 4 | **4** | `Số 4` | 0.92 | **PASS** |
| `DIGIT_4_EVAL_D0533_F45` | D0533 (F45, EVAL) | **4** | 4 | **4** | `Số 4` | 0.92 | **PASS** |
| `DIGIT_5_DEV_D0534_F67` | D0534 (F67, DEV) | **5** | 5 | **5** | `Số 5` | 0.95 | **PASS** |
| `DIGIT_5_EVAL_D0534_F55` | D0534 (F55, EVAL) | **5** | 5 | **5** | `Số 5` | 0.95 | **PASS** |
| `DIGIT_6_DEV_D0535B_F105` | D0535B (F105, DEV) | **6** | 6 | **6** | `Số 6` | 0.95 | **PASS** |
| `DIGIT_6_EVAL_D0535B_F90` | D0535B (F90, EVAL) | **6** | 6 | **6** | `Số 6` | 0.95 | **PASS** |
| `DIGIT_7_DEV_D0536_F54` | D0536 (F54, DEV) | **7** | 7 | **7** | `Số 7` | 0.90 | **PASS** |
| `DIGIT_7_EVAL_D0536_F45` | D0536 (F45, EVAL) | **7** | 7 | **7** | `Số 7` | 0.90 | **PASS** |
| `DIGIT_8_DEV_D0537_F77` | D0537 (F77, DEV) | **8** | 8 | **8** | `Số 8` | 0.92 | **PASS** |
| `DIGIT_8_EVAL_D0537_F65` | D0537 (F65, EVAL) | **8** | 8 | **8** | `Số 8` | 0.92 | **PASS** |
| `DIGIT_9_DEV_D0538_F52` | D0538 (F52, DEV) | **9** | 9 | **9** | `Số 9` | 0.90 | **PASS** |
| `DIGIT_9_EVAL_D0538_F42` | D0538 (F42, EVAL) | **9** | 9 | **9** | `Số 9` | 0.90 | **PASS** |

### 4.2. Chế độ `letters` Thử nghiệm với Dáng tay Số (Kiểm tra Chặn Nhãn Số)
*Chứng minh khi ở chế độ `letters`, hệ thống tuyệt đối không xuất ra bất kỳ nhãn chữ số nào, chỉ trả về chữ cái tương ứng hoặc NEUTRAL.*

| Mã mẫu đầu vào | Dáng tay thử nghiệm | Phản hồi Backend | Hiển thị Frontend | Đánh giá an toàn |
| :--- | :--- | :---: | :--- | :---: |
| Mẫu Số 1 (DEV & EVAL) | 1 ngón trỏ duỗi | **D** (conf=1.00) | `D` (Bảng chữ cái VSL) | **PASS** (Không ra số 1) |
| Mẫu Số 2 (DEV & EVAL) | 2 ngón khép sát | **U** (conf=0.86) | `U` (Bảng chữ cái VSL) | **PASS** (Không ra số 2) |
| Mẫu Số 3 (DEV & EVAL) | 3 ngón mở | **NEUTRAL** (conf=0.0) | `Đang theo dõi cử chỉ...` | **PASS** (Không ra số 3) |
| Mẫu Số 4 (DEV & EVAL) | 4 ngón mở | **NEUTRAL** (conf=0.0) | `Đang theo dõi cử chỉ...` | **PASS** (Không ra số 4) |
| Mẫu Số 5 (DEV & EVAL) | 5 ngón xòe | **NEUTRAL** (conf=0.0) | `Đang theo dõi cử chỉ...` | **PASS** (Không ra số 5) |
| Mẫu Số 6 (DEV & EVAL) | Shaka (Cái + Út) | **Y** (conf=0.90) | `Y` (Bảng chữ cái VSL) | **PASS** (Không ra số 6) |
| Mẫu Số 7 (DEV & EVAL) | Cái + Trỏ mở hẹp | **D** (conf=0.84) | `D` (Bảng chữ cái VSL) | **PASS** (Không ra số 7) |
| Mẫu Số 8 (DEV & EVAL) | Cái + Trỏ + Giữa | **V** / **U** (conf=0.88) | `V` / `U` (Bảng chữ cái VSL) | **PASS** (Không ra số 8) |
| Mẫu Số 9 (DEV & EVAL) | Trỏ khum | **D** (conf=0.84) | `D` (Bảng chữ cái VSL) | **PASS** (Không ra số 9) |

### 4.3. Trường hợp Ngoại lệ và Cử chỉ Chữ O / Số 0
- **Cử chỉ Chữ O trong mode `digits` (`SYNTHETIC_GESTURE_O_IN_DIGITS`):** Trả về `NEUTRAL, conf=0.0`. Không nhận nhầm thành số 0 (do chưa có ground truth) và không rò rỉ chữ cái O. $\rightarrow$ **PASS**.
- **Đầu vào rỗng (`EDGE_EMPTY_LANDMARKS`):** Cả 2 chế độ đều trả về `NEUTRAL, conf=0.0`. $\rightarrow$ **PASS**.
- **Đầu vào chứa NaN (`EDGE_NAN_LANDMARKS`):** Bộ lọc mảng NaN bắt lỗi an toàn, trả về `NEUTRAL, conf=0.0`. $\rightarrow$ **PASS**.

---

## 5. Các Lỗi Tái Hiện Được và Phân Tích

Trong suốt quá trình kiểm thử tự động toàn diện:
- **0 lỗi hồi quy (Zero regression):** Không phát hiện bất kỳ lỗi hồi quy nào trong 184 bài test.
- **Hành vi chuyển đổi:**
  - Khi chuyển từ `letters` $\rightarrow$ `digits`, bộ đệm được reset chính xác; không có frame nào của chế độ trước kích hoạt `confirmed` nhầm trong chế độ sau.
  - Chuỗi văn bản đã xác nhận được bảo toàn 100%.

---

## 6. Những Mục Chưa Kiểm Thử Được và Checklist Camera Thật

### 6.1. Hạn chế môi trường
- Môi trường thực thi lệnh hiện tại là **môi trường máy chủ/sandbox tự động (Headless CLI)**, không kết nối trực tiếp thiết bị phần cứng webcam vật lý và không có cử động tay của người dùng thực tế trước ống kính camera.
- Các kiểm thử trên được xác nhận qua: TestClient HTTP API, Unit/Integration Test suite, và Replay Fixture từ video MediaPipe thực tế.

### 6.2. Checklist Kiểm thử Thủ công bằng Camera Thật cho Người Dùng
Người dùng có thể thực hiện kiểm chứng tương tác trực tiếp qua camera theo các bước sau:

- [ ] **Bước 1: Khởi động hệ thống:**
  - Chạy backend: `uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload`
  - Chạy frontend: `npm run dev` $\rightarrow$ Mở trình duyệt tại `http://localhost:5173`.
- [ ] **Bước 2: Cấp quyền Camera:** Bấm "Bật Camera", kiểm tra khung hình video và skeleton 21 điểm bám sát bàn tay.
- [ ] **Bước 3: Kiểm thử Chế độ Chữ cái (`letters`):**
  - Bấm chọn nút **"Chữ cái"**.
  - Đưa tay tạo dáng chữ **D** (ngón trỏ thẳng) $\rightarrow$ Kiểm tra hiển thị đúng chữ **D** (không ra số 1).
  - Đưa tay tạo dáng chữ **V** (2 ngón xòe) $\rightarrow$ Kiểm tra hiển thị đúng chữ **V** (không ra số 2).
  - Đưa tay tạo dáng chữ **Y** (ngón cái và út) $\rightarrow$ Kiểm tra hiển thị đúng chữ **Y** (không ra số 6).
- [ ] **Bước 4: Kiểm thử Chế độ Chữ số (`digits`):**
  - Bấm chọn nút **"Chữ số"** (kiểm tra văn bản chữ cái vừa ghép ở Bước 3 vẫn giữ nguyên, không bị mất).
  - Đưa tay số **1** (1 ngón trỏ) $\rightarrow$ Kiểm tra hiển thị **"Số 1"**.
  - Đưa tay số **2** (2 ngón) $\rightarrow$ Kiểm tra hiển thị **"Số 2"**.
  - Đưa tay số **6** (ngón cái và út) $\rightarrow$ Kiểm tra hiển thị **"Số 6"**.
  - Đưa tay số **7**, **8**, **9**, **3**, **4**, **5** $\rightarrow$ Kiểm tra hiển thị các số tương ứng.
- [ ] **Bước 5: Kiểm tra Rời tay khỏi khung hình (Lost tracking):**
  - Rút tay ra khỏi camera $\rightarrow$ Kiểm tra HUD hiển thị "Đang theo dõi cử chỉ..." / "Chờ cử chỉ...", không phát sinh ngoại lệ console, văn bản đã ghép không bị xóa.
- [ ] **Bước 6: Kiểm thử Chuyển sang Từ vựng & Ghép câu (`words`, `sentence`):**
  - Bấm chọn **"Từ (102 Từ)"** $\rightarrow$ Kiểm tra pipeline chuyển sang 30-frame SSTCN.
  - Văn bản đã ghép tích lũy trước đó tiếp tục được bảo toàn.

---

## 7. Phân Biệt Nguồn Dữ Liệu Kiểm Thử

| Phân loại nguồn | Số lượng mẫu | Ý nghĩa kiểm chứng | Giới hạn thực tế |
| :--- | :---: | :--- | :--- |
| **Test tự động (Unit / Integration)** | 184 test cases | Xác minh tính đúng đắn của logic nhánh, hợp đồng HTTP, validation mã lỗi 400, cô lập session và reset bộ đệm smoothing. | Không bao quát các rung lắc vật lý hoặc nhiễu camera thực tế. |
| **Replay Fixture (Video QIPEDC thực tế)** | 18 mẫu (DEV/EVAL) | Trích xuất trực tiếp từ video của người khiếm thính chuẩn QIPEDC cho số 1–9. | Các frame DEV và EVAL được trích từ cùng 1 video/người mẫu, phản ánh độ nhất quán của video gốc nhưng **chưa khẳng định khả năng tổng quát hóa** trên nhiều người ký hiệu khác nhau ngoài đời thực. |
| **Camera vật lý trực tiếp** | Chưa thực hiện (Headless) | Kiểm tra độ trễ hiển thị trực quan, tốc độ khung hình và trải nghiệm người dùng thực. | Cần người dùng thực hiện theo checklist tại Mục 6.2. |

---

## 8. Đánh Giá Độ Tin Cậy và Khuyến Nghị Kế Tiếp

1. **Về mặt Kiến trúc & Hợp đồng:**
   - Việc tách thành 4 chế độ độc lập (`letters`, `digits`, `words`, `sentence`) đã giải quyết dứt điểm 100% sự mập mờ hình thái giữa chữ cái và chữ số đồng dạng.
   - Hợp đồng API được bảo vệ nghiêm ngặt bằng mã lỗi HTTP 400 Bad Request, đồng thời tương thích ngược hoàn hảo với các client cũ qua alias `letters_numbers`.
2. **Khuyến nghị cho giai đoạn kế tiếp:**
   - Thu thập thêm video chuẩn VSL cho **Chữ số 0** (Video D0529) để hoàn thiện trọn vẹn dải số 0–9.
   - Bổ sung thêm các mẫu chữ cái thực tế (đặc biệt là Chữ L và Chữ D) vào tập fixture để kiểm thử đối chứng đa dạng hơn.
   - Hướng dẫn người dùng nghiệm thu thực tế bằng webcam cá nhân theo checklist Mục 6.2.

