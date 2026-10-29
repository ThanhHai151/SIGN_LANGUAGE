# BÁO CÁO KỸ THUẬT: TÍCH HỢP VSLAlphabetClassifier VÀO LUỒNG NHẬN DIỆN CHỮ CÁI (TASK DIAG-A2)

**Dự án:** VSL Connect AI — Hệ thống giao tiếp hai chiều Ngôn ngữ Ký hiệu Việt Nam  
**Giai đoạn:** Task DIAG-A2 — Tích hợp `VSLAlphabetClassifier` vào luồng nhận diện chữ cái  
**Ngày thực hiện:** 30/10/2026  
**Kỹ sư phụ trách:** Senior Computer Vision / AI Engineer  
**Trạng thái kiểm thử:** 251/251 tests passed (9 tests tích hợp mới + 242 tests regression, 0 failures, 0 regressions)  

---

## 1. Danh sách các File đã Sửa đổi và Lý do Kỹ thuật

| File | Thay đổi chính | Lý do kỹ thuật |
| :--- | :--- | :--- |
| [`backend/vision/alphabet_classifier.py`](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py#L404-L414) | Khởi tạo singleton instance `alphabet_classifier = VSLAlphabetClassifier()` và hàm `classify_vsl_alphabet`. | Đảm bảo bộ phân loại chữ cái độc lập được khởi tạo sẵn ở module-level, tránh việc cấp phát lại object ở mỗi frame camera. |
| [`backend/vision/recognizer.py`](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L23-L41) | 1. Import `alphabet_classifier` và gán `self.alphabet_classifier = alphabet_classifier` trong `VSLRecognizer.__init__`.<br>2. Thêm nhánh xử lý chuyên biệt cho `if is_letters_mode:` trong `predict_landmarks`. | 1. Tích hợp trực tiếp bộ phân loại hình học độc lập vào luồng camera cho `mode="letters"`.<br>2. Ưu tiên tuyệt đối kết quả `MATCH` từ `VSLAlphabetClassifier`, ngăn chặn mô hình MLP (`VSLGestureNet`) ghi đè kết quả chữ cái sang `Z` hoặc các nhãn sai lệch. |
| [`backend/tests/test_alphabet_recognition_integration.py`](file:///d:/SIGN_LANGUAGE/backend/tests/test_alphabet_recognition_integration.py) | Tạo mới 9 test case tích hợp toàn diện (Trường hợp A, B, C, D, E). | Đảm bảo kiểm chứng logic tích hợp, bảo vệ an toàn các trạng thái biên (`NO_MATCH`, `CONFLICT`, `INVALID_INPUT`) và kiểm tra hồi quy toàn diện. |

---

## 2. Luồng Xử lý Trước và Sau Tích hợp

### 2.1. Trước tích hợp (Bị lỗi ghi đè)
```
[Frontend (mode="letters")] 
       |
       v (POST /api/recognize_landmarks)
[backend/vision/recognizer.py: predict_landmarks]
       |
       +---> classify_geometry (heuristic cũ) -> Trả về "A" (conf = 0.84)
       |
       +---> Kiểm tra Ensemble: conf 0.84 < 0.85 -> TRƯỢT NGƯỠNG ƯU TIÊN HÌNH HỌC!
       |
       +---> Kích hoạt MLP (VSLGestureNet) -> MLP dự đoán nhãn "Z" (conf = 1.00)
       |
       v
[Quyết định cuối cùng]: "Z" (MLP ghi đè "A" thành "Z"!)
```

### 2.2. Sau tích hợp (Sửa lỗi triệt để)
```
[Frontend (mode="letters")]
       |
       v (POST /api/recognize_landmarks)
[backend/vision/recognizer.py: predict_landmarks]
       |
       +---> if is_letters_mode:
       |        |
       |        +---> self.alphabet_classifier.classify_detailed(p_prim)
       |        |
       |        +---> status == "MATCH" & letter in ALLOWED_LETTERS:
       |        |        ├── final_sign = alpha_res["letter"] (Ví dụ: "A")
       |        |        ├── final_conf = alpha_res["confidence"] (0.95)
       |        |        └── KHÓA CHẶT: MLP HOÀN TOÀN KHÔNG ĐƯỢC PHÉP GHI ĐÈ!
       |        |
       |        +---> status in ("NO_MATCH", "CONFLICT", "INVALID_INPUT"):
       |                 ├── final_sign = "NEUTRAL"
       |                 └── final_conf = 0.0 (Không đoán mò, không ép nhãn)
       |
       v
[Quyết định cuối cùng]: "A" (conf = 0.95, type = "alphabet")
```

---

## 3. Bảng Kết quả Đối chiếu 4 Frame Chữ A Trước và Sau Sửa

Kiểm tra trực tiếp trên 4 frame chữ A thực tế từ video QIPEDC chuẩn (`backend/tests/fixtures/real_alphabet_landmarks.json` — nguồn gốc video `D0489.mp4` của Bộ GD&ĐT):

| Frame ID | Pha cử chỉ | Kết quả Trước Tích hợp (`predict_landmarks`) | Kết quả `VSLAlphabetClassifier` | Kết quả Sau Tích hợp (`predict_landmarks`) | Trạng thái |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Frame 23** | transition | ❌ `sign = "Z"`, `conf = 1.00` | `A` (conf = 0.95, MATCH) | ✅ **`sign = "A"`, `conf = 0.95`** | ĐÃ SỬA |
| **Frame 37** | stable_pose | ❌ `sign = "Z"`, `conf = 0.99` | `A` (conf = 0.95, MATCH) | ✅ **`sign = "A"`, `conf = 0.95`** | ĐÃ SỬA |
| **Frame 46** | stable_pose | ❌ `sign = "Z"`, `conf = 0.99` | `A` (conf = 0.95, MATCH) | ✅ **`sign = "A"`, `conf = 0.95`** | ĐÃ SỬA |
| **Frame 55** | stable_pose | ❌ `sign = "Z"`, `conf = 0.99` | `A` (conf = 0.95, MATCH) | ✅ **`sign = "A"`, `conf = 0.95`** | ĐÃ SỬA |

**Lý do lựa chọn kết quả:** Khi `alphabet_classifier` xác nhận cử chỉ hình học là `A` với `status: MATCH` và độ tin cậy $0.95$, nhánh `mode="letters"` gán trực tiếp `final_sign = "A"` và bỏ qua việc tham vấn mô hình MLP, loại bỏ hoàn toàn khả năng ghi đè sai lệch.

---

## 4. Xử lý An toàn các Trạng thái Biên (Edge Cases)

Trong thiết kế mới, không có việc ngầm định mọi cử chỉ đều là `MATCH`. Từng trường hợp được định nghĩa tường minh:
1. **Trạng thái `NO_MATCH`:**
   - Khi cử chỉ không thỏa mãn bất kỳ luật hình học nào (ví dụ tư thế tay lạ, hoặc cử chỉ của chữ chưa hỗ trợ).
   - Hệ thống gán: `final_sign = "NEUTRAL"`, `confidence = 0.0`, `type = "unknown"`.
   - Tuyệt đối không cho phép mô hình MLP tự động gán bừa một chữ cái khác chỉ vì confidence của MLP cao.
2. **Trạng thái `CONFLICT`:**
   - Khi có từ 2 quy tắc cạnh tranh nhau thỏa mãn đồng thời (ví dụ hình thái nằm ở vùng ranh giới giữa 2 chữ cái).
   - Hệ thống gán: `final_sign = "NEUTRAL"`, `confidence = 0.0`.
3. **Trạng thái `INVALID_INPUT`:**
   - Khi đầu vào là `None`, rỗng, sai số điểm khác 21, chứa giá trị `NaN`, `Inf` hoặc dữ liệu mốc bị phẳng/suy biến (`palm_scale < 1e-5`).
   - Hệ thống từ chối an toàn: `final_sign = "NEUTRAL"`, `confidence = 0.0`. Không gây crash ứng dụng.

---

## 5. Kết quả Kiểm thử Toàn diện

### 5.1. Unit Test Tích hợp Mới (`backend/tests/test_alphabet_recognition_integration.py`)
- Tổng số test: **9 test cases**.
- Kết quả: **9/9 PASSED (100%)** trong 8.15s.
- Chi tiết bao phủ:
  - `test_case_a_letter_a_not_overridden_by_mlp_z`: Giả lập MLP trả về `Z` với xác suất cực đại (100.0), kiểm chứng `predict_landmarks` vẫn kiên định trả về `A`.
  - `test_case_b_no_match_returns_neutral_safely`: Mock `NO_MATCH` trả về `NEUTRAL` (conf = 0.0).
  - `test_case_b_conflict_returns_neutral_safely`: Mock `CONFLICT` trả về `NEUTRAL` (conf = 0.0).
  - `test_case_c_invalid_inputs_safety`: Kiểm tra 5 dạng đầu vào lỗi (`None`, rỗng, 5 điểm, `NaN`, `Inf`).
  - `test_case_d_digits_mode_completely_unaffected`: Kiểm tra số 1 vẫn nhận diện đúng số 1 trong `digits` mode.
  - `test_case_d_words_and_sentence_modes_unaffected`: Kiểm tra `words` và `sentence` không bị ảnh hưởng.
  - `test_case_d_legacy_letters_numbers_mode_unaffected`: Kiểm tra mode lai cũ `letters_numbers` hoạt động bình thường.
  - `test_case_e_api_response_schema_compatibility`: Kiểm tra đầy đủ 8 trường API frontend: `sign`, `label`, `type`, `confidence`, `confirmed`, `new_token`, `sentence`, `description`.
  - `test_case_e_other_supported_letters`: Kiểm tra các chữ cái tĩnh khác (`B, D, L, O, S, T, V, Y`) trong fixture thật đều nhận diện chính xác.

### 5.2. Regression Test Toàn Dự án
- **Tổng số test cases:** **251 items** (242 items trước đó + 9 items mới).
- **Kết quả:** **251 passed (100%), 0 failed, 0 regressions** trong 34.81s.

---

## 6. Xác nhận Tính Độc lập của các Mode ngoài `letters`

Đã kiểm tra và chứng minh bằng test suite:
1. **Chế độ `digits`:** Hoàn toàn tách biệt, sử dụng nhánh `is_digits_mode` gọi `digit_classifier`. Không nhận diện chữ cái, không bị ảnh hưởng.
2. **Chế độ `words` và `sentence`:** Sử dụng chuỗi mô hình động SSTCN_Attention (30-frame), hoàn toàn không đi vào nhánh `is_letters_mode`.
3. **Chế độ lai tương thích ngược `letters_numbers` (legacy):** Chạy tiếp xuống nhánh ensemble cũ bên dưới, duy trì 100% khả năng tương thích ngược cho các client cũ.
4. **Trường hợp không truyền `mode` (`mode=None`):** Giữ nguyên hành vi mặc định cũ.

---

## 7. Các Giới hạn Kỹ thuật Chưa Giải quyết

1. **Giới hạn Single-Signer:** Cả 4 frame của chữ A được kiểm chứng đều trích xuất từ video nguồn `D0489.mp4` của cùng một người ký QIPEDC. Đây là bằng chứng xác nhận việc sửa lỗi logic tích hợp trên tập mẫu thực tế, nhưng chưa phải là phép thử độc lập chứng minh độ bao phủ tổng quát trên mọi người dùng với các kích thước và thói quen bàn tay khác nhau.
2. **Phạm vi Chữ cái Hỗ trợ Hiện tại:** Module `VSLAlphabetClassifier` hiện cung cấp luật hình học chắc chắn cho 12 chữ cái tĩnh (A, B, C, D, E, I, L, O, S, T, V, Y). 10 chữ cái tĩnh còn lại (G, H, K, M, N, P, Q, R, U, X) và 7 chữ cái có dấu động (Ă, Â, Ê, Ô, Ơ, Ư, Đ) đang được bảo lưu an toàn ở trạng thái `NEUTRAL`/`UNKNOWN` để tránh false positive, và sẽ tiếp tục được mở rộng trong các task tiếp theo (Task 3B.2 và 3B.3).

