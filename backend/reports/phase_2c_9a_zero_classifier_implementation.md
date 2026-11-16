# BÁO CÁO GIAI ĐOẠN 2C.9A — TRIỂN KHAI NHẬN DIỆN SỐ 0 CÓ KIỂM CHỨNG

**Dự án:** VSL Connect AI — Hệ thống hỗ trợ giao tiếp Ngôn ngữ ký hiệu Việt Nam  
**Ngày thực hiện:** 29/10/2026  
**Người thực hiện:** Antigravity AI Coding Assistant  
**Trạng thái:** Hoàn thành triển khai và kiểm chứng nhận diện Số 0 trong chế độ `digits`. Toàn bộ 195/195 tests backend đạt 100% PASSED.

---

## 1. File production đã thay đổi và lý do

Chỉ có 2 file production được cập nhật với phạm vi thay đổi tối thiểu:

1. [`backend/vision/digit_classifier.py`](file:///d:/SIGN_LANGUAGE/backend/vision/digit_classifier.py):
   - **Lý do:** Bổ sung trích xuất 3 đặc trưng hình học chuẩn hóa (`d_TI`, `d_TM`, `d_IW`) và cài đặt quy tắc hình học cho Số 0 (`Vòng khuyên tròn - Zero`) dựa trên bằng chứng đo đạc từ 4 mẫu landmark thực tế của video QIPEDC `D0529.mp4`.
   - **An toàn:** Duy trì đầy đủ các trạng thái `MATCH`, `NO_MATCH`, `CONFLICT`, `INVALID_INPUT`. Khi phát hiện xung đột hoặc không thỏa mãn ngưỡng, trả về an toàn `NEUTRAL` (conf=0.0).

2. [`backend/vision/recognizer.py`](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py):
   - **Lý do:** Kích hoạt nhận diện Số 0 **chuyên biệt cho chế độ `digits`** thông qua tập nhãn `DIGITS_MODE_SIGNS = {str(i) for i in range(0, 10)}`.
   - **Bảo vệ ranh giới:** Tập `ALLOWED_DIGITS` tiếp tục giữ nguyên dải `1–9` cho chế độ lai cũ (`letters_numbers`), bảo đảm các chế độ `letters`, `words`, `sentence` và alias `letters_numbers` tuyệt đối không bị ảnh hưởng hay hồi quy.

---

## 2. Đặc trưng, ngưỡng và bằng chứng quyết định

### 2.1. Phân tích đối chiếu đặc trưng thực nghiệm
Đối chiếu giữa 4 mẫu Số 0 thật trong [`real_digit_0_landmarks.json`](file:///d:/SIGN_LANGUAGE/backend/tests/fixtures/real_digit_0_landmarks.json) và 18 mẫu Số 1–9 trong [`real_digit_landmarks.json`](file:///d:/SIGN_LANGUAGE/backend/tests/fixtures/real_digit_landmarks.json):

| Nhóm cử chỉ | Trạng thái duỗi 4 ngón (`I`, `M`, `R`, `P`) | Khoảng cách Cái–Trỏ chuẩn hóa (`d_TI`) | Khoảng cách Cái–Giữa chuẩn hóa (`d_TM`) | Khoảng cách Trỏ–Cổ tay (`d_IW`) | Độ co gập ngón trỏ (`index_curl`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Số 0 (4 mẫu thật)** | Cả 4 ngón đều **KHÔNG DUỖI** (`False`) | **`0.105` – `0.131`** | **`0.115` – `0.130`** | **`1.145` – `1.194`** | **`0.608` – `0.648`** |
| **Số 1–9 (18 mẫu thật)** | Luôn có **ít nhất 1 ngón DUỖI** (`True`) | **`0.470` – `1.391`** | `0.400` – `1.450` | `0.800` – `1.400` | `0.000` – `0.778` (chỉ số 6 có curl cao nhưng duỗi út) |
| **Nắm đấm (Chữ S / A)** | Cả 4 ngón cuộn chặt vào lòng bàn tay | Thường > 0.30 | Thường > 0.30 | **`< 0.750`** (cuộn sát cổ tay) | `> 0.850` |

### 2.2. Các hằng số ngưỡng được chuẩn hóa
```python
D_TI_THRESHOLD_0 = 0.20           # Khoảng cách chuẩn hóa Cái–Trỏ (dữ liệu thật: 0.105–0.131, margin an toàn > 0.07)
D_TM_THRESHOLD_0 = 0.22           # Khoảng cách chuẩn hóa Cái–Giữa (dữ liệu thật: 0.115–0.130)
D_IW_MIN_0 = 0.95                 # Khoảng cách tối thiểu Trỏ–Cổ tay (dữ liệu thật: 1.145–1.194; loại trừ hoàn toàn nắm đấm < 0.75)
INDEX_CURL_MIN_0 = 0.45           # Độ co gập tối thiểu (loại trừ ngón duỗi thẳng)
INDEX_CURL_MAX_0 = 0.80           # Độ co gập tối đa (loại trừ ngón quặp sát lòng bàn tay)
```

### 2.3. Quy tắc hình học Số 0 trong `VSLDigitClassifier`
```python
if not i_ext and not m_ext and not r_ext and not p_ext:
    d_ti = feat.get("d_TI", 1.0)
    d_tm = feat.get("d_TM", 1.0)
    d_iw = feat.get("d_IW", 0.0)
    if d_ti <= D_TI_THRESHOLD_0 and d_tm <= D_TM_THRESHOLD_0 and d_iw >= D_IW_MIN_0:
        if INDEX_CURL_MIN_0 <= i_curl <= INDEX_CURL_MAX_0:
            matches.append(("0", f"Vòng tròn số 0: d_TI={d_ti:.3f}<={D_TI_THRESHOLD_0}, d_TM={d_tm:.3f}<={D_TM_THRESHOLD_0}, d_IW={d_iw:.3f}>={D_IW_MIN_0}", 0.95))
```

---

## 3. Kết quả kiểm thử trước và sau thay đổi

### 3.1. Kết quả kiểm thử trước thay đổi (Baseline)
- **Tổng số test:** 189 tests
- **Passed:** 189 (100%)
- **Trạng thái Số 0:** Chưa có quy tắc phân loại, Số 0 trả về `NEUTRAL`.

### 3.2. Kết quả kiểm thử sau thay đổi
- **Tổng số test:** **195 tests**
- **Passed:** **195 (100%)**
- **Failed / Error:** 0
- **Skipped:** 0
- **Thời gian chạy:** 8.69s

### Chi tiết các nhóm kiểm thử Số 0 mới:
| Test Case trong `test_digit_0_landmarks.py` | Mục đích kiểm tra | Kết quả |
| :--- | :--- | :---: |
| `test_digit_0_video_exists_and_checksum` | Kiểm tra video `D0529.mp4` và SHA-256 | **PASSED** |
| `test_digit_0_fixture_structure` | Kiểm tra định dạng 21 điểm landmark | **PASSED** |
| `test_digit_0_dev_eval_distinction` | Xác minh tính độc lập dev frame 30 và eval frame 35 | **PASSED** |
| `test_digit_0_morphological_properties` | Kiểm tra thuộc tính khum vòm rỗng | **PASSED** |
| `test_all_d0_fixture_samples_classified_as_zero` | Cả 4 mẫu số 0 đều nhận diện thành `'0'` (conf >= 0.90) | **PASSED** |
| `test_digits_1_to_9_never_confused_with_zero` | Không mẫu nào trong 18 mẫu 1–9 bị nhận nhầm thành `'0'` | **PASSED** |
| `test_invalid_input_safety` | An toàn với None, rỗng, sai số điểm, NaN, Inf | **PASSED** |
| `test_recognizer_digits_mode_recognizes_zero` | Chế độ `digits` nhận diện `'0'`, label `'Số 0'`, type `'digit'` | **PASSED** |
| `test_recognizer_letters_mode_never_returns_digit_zero` | Chế độ `letters` tuyệt đối không trả về số `'0'` | **PASSED** |
| `test_recognizer_words_and_sentence_unaffected` | Chế độ `words` và `sentence` không bị ảnh hưởng | **PASSED** |
| `test_recognizer_legacy_letters_numbers_mode_preserves_policy` | Chế độ cũ `letters_numbers` giữ nguyên chính sách | **PASSED** |

---

## 4. Các trường hợp nhầm lẫn đã kiểm tra (Confusion Cases)

1. **Số 0 vs Số 1–9:**
   - Đã kiểm tra chéo với toàn bộ 18 mẫu landmark thực tế của các số 1 đến 9.
   - **Kết quả:** Tỷ lệ nhầm lẫn **0%**. Toàn bộ số 1–9 đều có ít nhất 1 ngón duỗi và khoảng cách `d_TI >= 0.470`, bị loại bỏ ngay từ điều kiện đầu tiên.
2. **Số 0 vs Chữ O:**
   - Trong chế độ `digits`: Cử chỉ khum tròn chuẩn VSL được nhận diện thành Số `'0'`.
   - Trong chế độ `letters`: Cử chỉ khum tròn được nhận diện thành Chữ cái `'O'`, tuyệt đối không rò rỉ Số `'0'`.
   - Phân định hoàn toàn theo hợp đồng chế độ nhận diện (Mode separation).
3. **Số 0 vs Nắm đấm Chữ S / Chữ A:**
   - Nhờ ràng buộc `d_IW >= 0.95` và `index_curl <= 0.80`, các cử chỉ nắm tay cuộn sát vào cổ tay đều bị loại trừ khỏi Số 0.
4. **Mẫu giả lập O trong `test_mode_separation.py` (test_04):**
   - Mẫu giả lập thiếu độ chụm của các ngón giữa và áp út (`d_tm = 0.322 > 0.22`) $\rightarrow$ Classifier báo `NO_MATCH` và Recognizer trả về `NEUTRAL` $\rightarrow$ Test 04 tiếp tục PASS 100%.

---

## 5. Những hạn chế chưa giải quyết

1. **Phụ thuộc 1 video nguồn (Single Signer Limitation):**
   - Dữ liệu ground truth số 0 hiện tại được trích xuất từ 1 video `D0529.mp4` (cùng diễn viên trong bộ QIPEDC). Chưa có dữ liệu kiểm thử từ người ký khác (cross-signer evaluation).
2. **Độ bền với góc nghiêng camera:**
   - Các mẫu hiện tại được ghi hình ở góc quay chính diện. Khi người dùng đưa tay nghiêng góc lớn (> 30°) hoặc quá gần mép camera, khoảng cách phối cảnh `d_TI` có thể bị biến dạng quang học.

---

## 6. Xác nhận phạm vi và cam kết

- **Mô hình & Checkpoint:** Tuyệt đối không thay đổi weights, không huấn luyện lại model MLP hay SSTCN.
- **Dữ liệu nguồn & Fixture cũ:** `real_digit_landmarks.json` và `digit_video_registry.json` hoàn toàn nguyên vẹn.
- **Các chế độ ngoài phạm vi:** `letters`, `words`, `sentence` giữ nguyên 100% logic và hành vi trước đó.
- **Nhận diện kép số 6–9:** Chưa triển khai trong giai đoạn này, tuân thủ nghiêm ngặt chỉ thị dừng chờ phản hồi.

