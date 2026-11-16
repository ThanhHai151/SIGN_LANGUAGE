# VSL — KẾ HOẠCH CHẨN ĐOÁN VÀ SỬA 6 CHỮ H, O, Q, R, M, N
*(Vietnamese Sign Language Alphabet 6 Letters Repair & Verification Plan)*

- **Đường dẫn tài liệu:** [`docs/VSL_6_LETTERS_REPAIR_PLAN.md`](file:///d:/SIGN_LANGUAGE/docs/VSL_6_LETTERS_REPAIR_PLAN.md)
- **Trạng thái:** `DRAFT — AWAITING GATE 0 APPROVAL` (Mọi implementation kỹ thuật vẫn bị khóa)
- **Ngày cập nhật:** 10/11/2026
- **Nhánh làm việc đối chiếu:** `feat/controlled-pilot-29-letters` (HEAD `67ad75e3`, working tree sạch 100%)
- **Phạm vi kiểm soát:** 6 chữ cái chưa ổn định trong tập 22 chữ cái tĩnh không dấu: **H, O, Q, R, M, N**; bảo vệ tuyệt đối 16 chữ còn lại (Zero Regression Baseline).
- **Tài liệu SSOT liên quan:**
  - [`docs/VSL_8_LETTERS_FIX_PLAN.md`](file:///d:/SIGN_LANGUAGE/docs/VSL_8_LETTERS_FIX_PLAN.md) *(SSOT gốc đợt trước — BẢO LƯU NGUYÊN VẸN, KHÔNG GHI ĐÈ)*
  - [`backend/data/alphabet_v2/audit/task_7f1_static_letters_audit_report.md`](file:///d:/SIGN_LANGUAGE/backend/data/alphabet_v2/audit/task_7f1_static_letters_audit_report.md) *(Báo cáo kiểm toán 551 mẫu Dataset V2)*
  - Báo cáo kiểm toán đọc độc lập: *VSL Six Letters Diagnostic Audit Report — Read-Only Empirical Audit (10/11/2026)*.

---

## 0. VAI TRÒ CỦA TÀI LIỆU VÀ RANH GIỚI CÔNG VIỆC

1. **Bản điều phối kiểm soát thay đổi (Change Control Document):** Tài liệu này là kế hoạch chi tiết, ràng buộc quy trình kỹ thuật cho vòng sửa 6 chữ H/O/Q/R/M/N. Nó **không phải là bằng chứng** rằng bất kỳ lỗi nào đã được sửa.
2. **Đối chiếu hiện trạng mã nguồn thực tế:**
   - Phiên kiểm thử webcam chính: `pilot_20261110_023915_0150a8` (16 trials, 474 frames) lưu tại [`backend/data/diagnostics/pilot_eval_29_pilot_20261110_023915_0150a8_20261110_024454.json`](file:///d:/SIGN_LANGUAGE/backend/data/diagnostics/pilot_eval_29_pilot_20261110_023915_0150a8_20261110_024454.json).
   - Nhánh Git: `feat/controlled-pilot-29-letters`, commit HEAD `67ad75e3`.
3. **Nguyên tắc bảo toàn SSOT lịch sử:** Không ghi đè hoặc làm mất nội dung của `docs/VSL_8_LETTERS_FIX_PLAN.md`. Tài liệu này đóng vai trò là kế hoạch triển khai cụ thể kế thừa SSOT, chỉ chuyển sang trạng thái `APPROVED` sau khi người dùng phê duyệt Gate 0 bằng văn bản.
4. **Kỷ luật thực thi bắt buộc:**
   - Tuyệt đối **KHÔNG** sửa mã nguồn nhận diện, model weights, thresholds, datasets, HUD hoặc cấu hình trong giai đoạn lập kế hoạch.
   - Tuyệt đối **KHÔNG** tự ý đổi nhánh Git, force push, reset hard, clean, commit hoặc push.
   - Khi người dùng chỉ yêu cầu cập nhật docs, chỉ được phép chỉnh sửa duy nhất file tài liệu này.
   - Tuyệt đối **KHÔNG** mở rộng phạm vi sửa đổi sang 16 chữ cái đã ổn định, nhưng bắt buộc kiểm thử hồi quy trên toàn bộ 22 chữ cái tĩnh ở mọi checkpoint.

---

## 1. ĐỐI CHIẾU KIẾN TRÚC HỆ THỐNG HIỆN HÀNH

Hệ thống nhận diện chữ cái tĩnh VSL Connect AI hoạt động theo kiến trúc đường ống 3 tầng nối tiếp (3-tier pipeline):

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                       KIẾN TRÚC ĐƯỜNG ỐNG NHẬN DIỆN 3 TẦNG                    │
│                                                                               │
│  Landmarks (21 điểm MediaPipe)                                                │
│        │                                                                      │
│        ▼                                                                      │
│  [TẦNG 1: GEOMETRIC RULES] (VSLAlphabetClassifier.evaluate_rules)             │
│        │                                                                      │
│        ├─► MATCH: Chữ cái A..Y (12 luật hình học cơ sở + 7 luật bổ sung)      │
│        │          recognition_source = "geometric_rule", conf = 0.95          │
│        │          (Static Head HOÀN TOÀN BỊ BỎ QUA)                           │
│        │                                                                      │
│        └─► NO_MATCH / UNKNOWN                                                 │
│                  │                                                            │
│                  ▼                                                            │
│  [TẦNG 2: STATIC ALPHABET HEAD (MLP 72D) + GATING] (VSLRecognizer.recognize)  │
│        │                                                                      │
│        ├─► Nhóm Fallback Base: ALLOWED_FALLBACK_BASE_LETTERS = {"A", "D", "E"}│
│        │   Nếu head_conf >= 0.70 ──► recognition_source = "static_head_fallback"
│        │                                                                      │
│        ├─► Nhóm Supplemental: ALLOWED_SUPPLEMENTAL_LETTERS                   │
│        │   {"G", "H", "K", "M", "N", "P", "Q", "R", "U", "X"}                │
│        │   Nếu head_conf >= req_threshold ──► recognition_source = "static_head"
│        │                                                                      │
│        └─► Dưới ngưỡng / Ngoài danh mục                                       │
│            final_sign = "NEUTRAL", gating_status = "rejected"                │
│                  │                                                            │
│                  ▼                                                            │
│  [TẦNG 3: TEMPORAL SMOOTHING BUFFER] (VSLRecognizer)                          │
│        │                                                                      │
│        ├─► sliding window deque(maxlen=10), lọc frame conf >= 0.70           │
│        ├─► kích hoạt bầu chọn khi len(recent_signs) >= 5 (window // 2)        │
│        ├─► nhãn dẫn đầu phải có số phiếu count >= 5 (window // 2)             │
│        └─► debounce cooldown_sec = 1.2s (backend/vision/recognizer.py#L128)   │
│                  │                                                            │
│                  ▼                                                            │
│  [HUD CLIENT & PROTOCOL EVALUATION] (PilotTestingHUD & pilotProtocol.js)      │
│        │                                                                      │
│        └─► activeFrames = matchFrames + mismatchFrames                        │
│            Nếu activeFrames === 0 ──► trial status = "NO_HAND"               │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Các phát hiện kiểm toán kiến trúc then chốt:

1. **Chữ M & Chữ N — Hoàn toàn vắng bóng luật hình học `[SOURCE VERIFIED]`:**
   - Trong [`backend/vision/alphabet_classifier.py`](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py), hàm `evaluate_rules()` **không có bất kỳ dòng code nào kiểm tra M hoặc N**.
   - Cả hai chữ phụ thuộc 100% vào Tầng 2 (Static Head MLP 72D). Khi MLP cho ra confidence dưới ngưỡng ($0.70$ cho M, $0.75$ cho N), 100% frame rơi vào `NEUTRAL`.
2. **Chữ O — Bất đối xứng trong cơ chế Fallback `[SOURCE VERIFIED]`:**
   - Trong [`backend/vision/recognizer.py#L78`](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L78):
     `ALLOWED_FALLBACK_BASE_LETTERS = {"A", "D", "E"}` $\rightarrow$ **Chữ O không có tên trong danh sách này!**
     Khi Rule O hình học trả về `NO_MATCH` (do khe hở ngón cái - ngón trỏ $d_{TI} > 0.23$), Static Head dự đoán dáng tay nắm tròn thành `A`. Do `A` nằm trong danh sách fallback, nhãn `A` được phát ra ngay lập tức. Ngược lại, nếu Static Head đoán đúng `O`, hệ thống từ chối tại dòng 1510 vì O không thuộc fallback!
3. **Chữ Q — Bị áp đặt ngưỡng nhân tạo bất thường `[SOURCE VERIFIED]`:**
   - Trong [`backend/vision/recognizer.py#L90`](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L90):
     `DEFAULT_SUPPLEMENTAL_THRESHOLDS["Q"] = 0.85` (trong khi các chữ khác dùng 0.70 hoặc 0.75). Ngưỡng này loại bỏ hơn 75% các mẫu đúng thực tế của Q từ Static Head.
4. **Chữ H — Bị chiếm nhãn chéo (Rule Hijacking) `[SOURCE VERIFIED]`:**
   - Rule V tại dòng 630–645 được kiểm tra trước Rule H (dòng 840–875). Chỉ cần hai ngón tay duỗi ngang hơi mở góc $ang_{IM} \ge 12^\circ$, Rule V kích hoạt cướp nhãn.
   - Rule R tại dòng 1055–1085 kích hoạt khi ngón trỏ và ngón giữa chồng lấn trên hình chiếu 2D của camera ($cross\_indicator \le 0.0060$), biến H thành R.
5. **Cơ chế Temporal Smoothing & Cooldown trong mã nguồn thực tế `[SOURCE VERIFIED]`:**
   - Trích dẫn trực tiếp mã nguồn tại [`backend/vision/recognizer.py#L124-L128`](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L124-L128) và dòng 1648–1669:
     ```python
     # recognizer.py#L126-L128:
     confidence_threshold = 0.70
     smoothing_window = 10
     cooldown_sec = 1.2
     
     # recognizer.py#L1651-L1664:
     recent_signs = [s for s, c in self.prediction_history if c >= self.confidence_threshold]
     if len(recent_signs) >= self.smoothing_window // 2:  # Cần tối thiểu 5 frames đạt conf >= 0.70
         most_common, count = collections.Counter(recent_signs).most_common(1)[0]
         if count >= (self.smoothing_window // 2) and most_common in ALLOWED_LETTERS:
             if most_common != self.last_confirmed_sign or (current_time_clock - self.last_confirmed_time > self.cooldown_sec):
                 self.last_confirmed_sign = most_common
                 self.last_confirmed_time = current_time_clock
                 confirmed = True
     ```
   - **Quy tắc vận hành temporal smoothing thực tế:**
     1. `window`: Kích thước tối đa là 10 frames (`deque(maxlen=10)`).
     2. `Lọc frame`: Chỉ các frame có `confidence >= self.confidence_threshold` (mặc định là `0.70`) mới được đưa vào danh sách bầu chọn `recent_signs`.
     3. `Xử lý khi buffer chưa đủ 10 frames`: Hệ thống **không** bắt buộc phải chờ đủ 10 frames mới bắt đầu bầu chọn! Chỉ cần danh sách `recent_signs` tích lũy được tối thiểu **5 frames** (`self.smoothing_window // 2 = 5`) là khối bầu chọn được kích hoạt. Nếu `len(recent_signs) < 5`, hệ thống không bầu chọn (`confirmed = False`).
     4. `Số phiếu tối thiểu (Vote threshold)`: Nhãn phổ biến nhất (`most_common`) phải có số lần xuất hiện `count >= 5` (`self.smoothing_window // 2 = 5`) và phải thuộc `ALLOWED_LETTERS`.
     5. `Debounce & Cooldown`: Giá trị mặc định là **`cooldown_sec = 1.2s`** (không phải 1.5s). Nếu nhãn vừa bầu chọn khác với nhãn trước đó (`most_common != last_confirmed_sign`), token được xác nhận ngay mà không cần đợi cooldown. Nếu trùng nhãn trước đó, phải cách nhau $> 1.2s$ mới được xác nhận lặp lại.
6. **HUD gán nhãn `NO_HAND` không đồng nghĩa camera mất tay `[SOURCE VERIFIED]`:**
   - Tại [`src/components/pilotProtocol.js#L223-L224`](file:///d:/SIGN_LANGUAGE/src/components/pilotProtocol.js#L223-L224): Khi toàn bộ frame trong trial đều là `NEUTRAL`, `activeFrames = matchFrames + mismatchFrames = 0`. HUD tự động gán nhãn `NO_HAND`. Đây là cơ chế phân loại của HUD, không phải camera mất tracking landmark.

---

## 2. QUY TẮC BẤT BIẾN — CHỌN NGUỒN QIPEDC VÀ PHÂN TÁCH BỘ ĐÁNH GIÁ

### 2.1. Thứ tự ưu tiên bắt buộc (Strict Invariant Hierarchy)

| Thứ tự ưu tiên | Quy ước nhận dạng nguồn | Vai trò kỹ thuật | Quyết định tuyển chọn |
| :---: | :---| :---| :---: |
| **Ưu tiên 1 (Cao nhất)** | Mã QIPEDC **KHÔNG có hậu tố** vùng miền trong mã file (ví dụ: `D0504.mp4`, `D0505.mp4`, `D0506.mp4`, `D0510.mp4`, `D0511.mp4`). Trong quy ước SSOT, không hậu tố = **Nguồn chuẩn dùng chung toàn quốc**. Tuyệt đối không tự hạ ưu tiên. | **Chuẩn Toàn quốc (Canonical Baseline)** | `SELECT_NATIONWIDE` |
| **Ưu tiên 2 (Dự phòng)** | File có **hậu tố `N`** (ví dụ: `D0500N.mp4`, `D0508N.mp4`, `D0515N.mp4`). **Chỉ chọn khi KHÔNG CÓ mẫu toàn quốc hợp lệ** cho đúng chữ cái/biến thể cần xét, và bắt buộc ghi rõ nguyên nhân chọn nguồn miền Nam. | **Dự phòng chuẩn Miền Nam** | `SELECT_SOUTH` (Chỉ khi vắng nguồn #1) |
| **LOẠI KHỎI TẬP CHUẨN** | File có **hậu tố `B`** (Miền Bắc, ví dụ `D0500B.mp4`) hoặc **hậu tố `T`** (Miền Trung). Giữ các bản này để đối chiếu, phân tích biến thể vùng miền nếu cần. **Tuyệt đối không đưa vào tập chuẩn, calibration, benchmark hoặc training để sửa 6 chữ.** | **Chỉ tham khảo / đối chiếu biến thể** | `EXCLUDE_REGIONAL_VARIANT` (Cấm làm canonical) |
| **DỪNG KIỂM TRA** | Thiếu file, không rõ filename, thiếu mapping provenance hoặc sai lệch nhãn. | **Thiếu dữ liệu** | `HOLD_FOR_VERIFICATION` |

### 2.2. Kiểm chứng nguồn dữ liệu Chữ H và Kế hoạch xử lý dữ liệu sai nguồn

1. **Kiểm chứng sự vắng mặt của file toàn quốc `[SOURCE VERIFIED]`:**
   - Tìm kiếm toàn bộ hệ thống file trong repository và cache: **File `D0500.mp4` (không hậu tố) KHÔNG TỒN TẠI** trong bất kỳ thư mục nào (`MISSING_CANONICAL_NATIONWIDE`).
2. **Xác minh nguồn Miền Nam dự phòng `[SOURCE VERIFIED]`:**
   - File video `D0500N.mp4` tồn tại tại [`backend/data/media_cache/videos/D0500N.mp4`](file:///d:/SIGN_LANGUAGE/backend/data/media_cache/videos/D0500N.mp4).
   - File landmark JSON `D0500N.json` tồn tại tại [`backend/data/alphabet_v2/static_landmarks/D0500N.json`](file:///d:/SIGN_LANGUAGE/backend/data/alphabet_v2/static_landmarks/D0500N.json), chứa 65 frames có nhãn `letter: "H"`, `variant: "Southern"` hợp lệ.
   - $\rightarrow$ Theo Ưu tiên 2, **`D0500N` là nguồn chuẩn dự phòng hợp lệ duy nhất (`SELECT_SOUTH`)**.
3. **Phát hiện dữ liệu sai nguồn lịch sử `[SOURCE VERIFIED]`:**
   - File [`backend/data/alphabet_v2/static_landmarks/H.json`](file:///d:/SIGN_LANGUAGE/backend/data/alphabet_v2/static_landmarks/H.json) và tập dữ liệu [`backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz`](file:///d:/SIGN_LANGUAGE/backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz) hiện đang chứa 18 mẫu trích xuất từ **`D0500B` (Miền Bắc)**!
   - $\rightarrow$ `D0500B` bị cấm làm chuẩn theo quy tắc dự án (`EXCLUDE_REGIONAL_VARIANT`).
4. **Kế hoạch xử lý dữ liệu sai nguồn (Chỉ lập kế hoạch — TUYỆT ĐỐI KHÔNG SỬA DATASET TRONG LƯỢT NÀY):**
   - Trong giai đoạn lập kế hoạch này, **không chạm vào file `vsl_alphabet_static_v2.npz` hay `H.json`**.
   - Phân tách rạch ròi 2 bộ dữ liệu đánh giá (xem Mục 2.3 dưới đây).

### 2.3. Phân tách hai bộ đánh giá độc lập (Two Evaluation Suites)

Để đảm bảo tính nhất quán khoa học và không gây xáo trộn tập dữ liệu hiện hành, hệ thống phân tách rạch ròi:

1. **Bộ 1 — Legacy Regression Suite (`vsl_alphabet_static_v2.npz` — 551 mẫu lịch sử):**
   - **Nguyên tắc:** Giữ nguyên 100%, không chỉnh sửa, không xóa dữ liệu, không đổi nhãn.
   - **Mục đích:** Dùng làm bài kiểm tra hồi quy lịch sử (Historical Behavioral Regression) để phát hiện sự thay đổi ngoài ý muốn trên 16 chữ cái đã ổn định và các mẫu tĩnh cũ.
   - **Báo cáo nguồn:** Báo cáo rõ ràng rằng 18 mẫu H trong tập này thuộc nguồn Miền Bắc `D0500B`. Sự biến động của H trên bộ này chỉ mang tính tham khảo đối chiếu, không dùng làm tiêu chí đánh giá PASS/FAIL của Chữ H.
2. **Bộ 2 — Canonical Evaluation Suite (QIPEDC Canonical Standards):**
   - **Nguyên tắc:** Chỉ sử dụng các nguồn dữ liệu hợp lệ tuyệt đối: Toàn quốc trước (không hậu tố: `D0504`, `D0505`, `D0506`, `D0510`, `D0511`), Miền Nam sau (`D0500N` cho Chữ H).
   - **Đối với Chữ H:** Sử dụng duy nhất file landmarks `static_landmarks/D0500N.json` và video `D0500N.mp4`. Tuyệt đối không dùng `D0500B` làm chuẩn đánh giá.
   - **Mục đích:** Dùng làm thước đo chính thức duy nhất cho việc hiệu chuẩn (calibration), tuning ngưỡng và nghiệm thu năng lực nhận diện của các chữ cái.

---

## 3. BẢNG PROVENANCE VÀ NGUỒN DỮ LIỆU DỰ KIẾN CHO TỪNG CHỮ

Bảng kiểm kê nguồn dữ liệu chính thức phục vụ calibration, replay và offline testing cho 6 chữ cái:

| Chữ cái | Video ID | Tên file video | Hậu tố | Phân loại vùng miền | Quyết định tuyển chọn | Lý do tuyển chọn / Loại trừ | Đường dẫn Landmarks hiện có | Số mẫu hợp lệ | Trạng thái xác minh |
| :---: | :---: | :---| :---: | :---: | :---: | :---| :---| :---: | :---: |
| **H** | `D0500N` | `D0500N.mp4` | `N` | Miền Nam | **`SELECT_SOUTH`** | Không có file toàn quốc; chọn Miền Nam làm chuẩn dự phòng theo Ưu tiên 2. | `backend/data/alphabet_v2/static_landmarks/D0500N.json` | 65 frames (12 stable) | `[SOURCE VERIFIED]` |
| **H** | `D0500B` | `D0500B.mp4` | `B` | Miền Bắc | **`EXCLUDE_REGIONAL_VARIANT`** | Hậu tố B bị cấm làm chuẩn; chỉ giữ để đối chiếu biến thể phương ngữ Bắc trong Legacy Suite. | `backend/data/alphabet_v2/static_landmarks/H.json` | 36 frames (22 valid) | `[SOURCE VERIFIED]` |
| **H** | `D0500` | *(không tồn tại)* | *(none)* | Toàn quốc | **`MISSING_CANONICAL_NATIONWIDE`** | Đã kiểm chứng toàn bộ repo: không có file D0500 không hậu tố. | Không có | 0 | `[SOURCE VERIFIED]` |
| **O** | `D0506` | `D0506.mp4` | *(none)* | Toàn quốc | **`SELECT_NATIONWIDE`** | File không hậu tố, chuẩn toàn quốc theo Ưu tiên 1 tuyệt đối. | `backend/data/alphabet_v2/static_landmarks/O.json` | 35 frames (28 stable) | `[SOURCE VERIFIED]` |
| **O** | `D0507B` | `D0507B.mp4` | `B` | Miền Bắc | **`EXCLUDE_REGIONAL_VARIANT`** | Video chữ Ô phương ngữ Bắc; loại khỏi chuẩn của chữ O. | Nằm trong dataset static_v2 (10 frames) | 10 frames | `[SOURCE VERIFIED]` |
| **O** | `D0508N` | `D0508N.mp4` | `N` | Miền Nam | **`EXCLUDE_REGIONAL_VARIANT`** | Video chữ Ơ phương ngữ Nam; loại khỏi chuẩn của chữ O vì đã có D0506 toàn quốc. | Nằm trong dataset static_v2 (10 frames) | 10 frames | `[SOURCE VERIFIED]` |
| **Q** | `D0510` | `D0510.mp4` | *(none)* | Toàn quốc | **`SELECT_NATIONWIDE`** | File không hậu tố, chuẩn toàn quốc theo Ưu tiên 1 tuyệt đối. | `backend/data/alphabet_v2/static_landmarks/Q.json` | 36 frames (19 stable) | `[SOURCE VERIFIED]` |
| **R** | `D0511` | `D0511.mp4` | *(none)* | Toàn quốc | **`SELECT_NATIONWIDE`** | File không hậu tố, chuẩn toàn quốc theo Ưu tiên 1 tuyệt đối. | `backend/data/alphabet_v2/static_landmarks/R.json` | 36 frames (14 stable) | `[SOURCE VERIFIED]` |
| **M** | `D0504` | `D0504.mp4` | *(none)* | Toàn quốc | **`SELECT_NATIONWIDE`** | File không hậu tố, chuẩn toàn quốc theo Ưu tiên 1 tuyệt đối. Không tự chế D0504N. | `backend/data/alphabet_v2/static_landmarks/M.json` | 36 frames (19 stable) | `[SOURCE VERIFIED]` |
| **N** | `D0505` | `D0505.mp4` | *(none)* | Toàn quốc | **`SELECT_NATIONWIDE`** | File không hậu tố, chuẩn toàn quốc theo Ưu tiên 1 tuyệt đối. Không tự chế D0505N. | `backend/data/alphabet_v2/static_landmarks/N.json` | 36 frames (21 stable) | `[SOURCE VERIFIED]` |

---

## 4. BASELINE HIỆU NĂNG THỰC TẾ VÀ QUY TẮC MẪU SỐ

### 4.1. Bảng số liệu định lượng phiên Webcam `pilot_20261110_023915_0150a8`

Dữ liệu kiểm toán từ file log [`pilot_eval_29_pilot_20261110_023915_0150a8_20261110_024454_frames.jsonl`](file:///d:/SIGN_LANGUAGE/backend/data/diagnostics/pilot_eval_29_pilot_20261110_023915_0150a8_20261110_024454_frames.jsonl):

| Chữ cái | Tổng frames logged | Match (Đúng) | Mismatch (Sai) | NEUTRAL | Active Frames (Match+Mismatch) | Accuracy trên All Frames (%) | Non-Neutral Precision (%) | Trial PASS / Total | Nhầm lẫn chính | Trạng thái HUD |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---| :---: |
| **H** | 88 | 0 | 20 | 68 | 20 | **0.0%** | **0.0%** | 0/3 (0%) | V (13), R (5), D (1), U (1) | `NEEDS_FIX` |
| **O** | 57 | 35 | 21 | 1 | 56 | **61.4%** | **62.5%** | 1/2 (50%) | A (21) | `NEEDS_FIX` |
| **Q** | 29 | 0 | 1 | 28 | 1 | **0.0%** | **0.0%** | 0/1 (0%) | P (1), còn lại 96.5% NEUTRAL | `NEEDS_FIX` |
| **R** | 109 | 16 | 3 | 90 | 19 | **14.7%** | **84.2%** | 2/4 (50%) | H (1), D (1), U (1) | `PASS` |
| **M** | 88 | 0 | 0 | 88 | 0 | **0.0%** | **N/A (0 active)** | 0/3 (0%) | 100% NEUTRAL | `NO_HAND` |
| **N** | 103 | 0 | 0 | 103 | 0 | **0.0%** | **N/A (0 active)** | 0/3 (0%) | 100% NEUTRAL | `NO_HAND` |

### 4.2. Quy tắc chuẩn hóa mẫu số (Mẫu số minh bạch — Zero Statistical Fallacy)

Để chống lại hiện tượng "thổi phồng độ chính xác" hoặc nhầm lẫn giữa các tỷ lệ:
1. **Tỷ lệ đúng toàn cục (Overall Frame Accuracy):**
   $$\text{Overall Accuracy} = \frac{\text{Correct Frames}}{\text{Total Logged Frames}}$$
   *Ví dụ:* Chữ R đạt $16 / 109 = 14.7\%$ trên toàn bộ frame ghi nhận. Tuyệt đối **không** lấy con số $84.2\%$ để tuyên bố "chữ R đã đạt $84\%$ độ chính xác trên webcam"!
2. **Độ chính xác trên các frame hoạt động (Non-Neutral Active Precision):**
   $$\text{Active Precision} = \frac{\text{Correct Frames}}{\text{Correct Frames} + \text{Wrong Mismatch Frames}}$$
   *Ví dụ:* Chữ R đạt $16 / (16 + 3) = 84.2\%$ trên các frame nhận diện được nhãn (loại bỏ 90 frame NEUTRAL trong pha chuyển động/giơ tay).
3. **Tỷ lệ đạt cấp Trial (Trial Pass Rate):**
   $$\text{Trial Pass Rate} = \frac{\text{Passed Trials}}{\text{Total Valid Completed Trials}}$$
   Một trial chỉ đạt `PASS` khi thỏa mãn điều kiện protocol: $\text{Active Precision} \ge 60.0\%$ và số active frames $> 0$.
4. **Phân biệt rạch ròi NEUTRAL và NO_HAND:**
   - `NEUTRAL` là kết quả phân loại của pipeline khi không khớp luật hình học và bị gating từ chối (bàn tay vẫn có landmark đầy đủ).
   - `NO_HAND` là nhãn do HUD gán khi `activeFrames === 0`, hoặc do MediaPipe không phát hiện bàn tay. Phải kiểm tra trường `landmarks` trong log frame để kết luận, không đánh đồng hai khái niệm.

### 4.3. Bảng số liệu Baseline Đo đạc Thực tế (Kiểm toán TASK 0.D)

Dữ liệu đo đạc thực nghiệm độc lập thực hiện tại TASK 0.D trước khi can thiệp thuật toán:

#### A. Baseline Bộ Kế thừa — Legacy Regression Suite (`vsl_alphabet_static_v2.npz`, 551 mẫu):
- **Độ chính xác toàn cục (Overall Accuracy):** **451/551 (81.85%)**.
- **Hiệu năng 6 chữ cái mục tiêu trong tập kế thừa:**
  - **H:** 16/18 (88.9%) — *Ghi chú nguồn:* Thuộc video phương ngữ Bắc `D0500B`, nhận diện tốt theo dáng Bắc nhưng không đại diện cho dáng chuẩn miền Nam.
  - **O:** 41/48 (85.4%), 7 frame NEUTRAL.
  - **Q:** 14/19 (73.7%), 5 frame NEUTRAL.
  - **R:** 11/14 (78.6%), 3 frame NEUTRAL.
  - **M:** 17/19 (89.5%), 2 frame NEUTRAL.
  - **N:** 15/21 (71.4%), 5 frame NEUTRAL, 1 frame nhầm sang M.
- **16 chữ cái nền còn lại:** Duy trì độ chính xác cao: A (93.8%), D (93.5%), E (91.4%), G (89.7%), U (84.4%), X (80.0%), v.v.

#### B. Baseline Bộ Chuẩn tắc — Canonical Evaluation Suite (QIPEDC Canonical Standards):
Đo đạc trên 100% frames hợp lệ (quality_status='ok', có landmark đầy đủ) từ nguồn chuẩn tắc:

| Chữ cái | Nguồn Canonical áp dụng | Tổng frames | Frames hợp lệ | Nhận diện đúng | Độ chính xác Baseline (%) | Phân bố dự đoán & Nhầm lẫn chính |
| :---: | :---| :---: | :---: | :---: | :---: | :---|
| **H** | `D0500N.json` (Miền Nam - Chuẩn dự phòng) | 65 | 37 | **24** | **64.9%** | **H: 24**, NEUTRAL: 13 |
| **O** | `D0506.json` (Toàn quốc) | 37 | 20 | **13** | **65.0%** | **O: 13**, NEUTRAL: 7 |
| **Q** | `D0510.json` (Toàn quốc) | 41 | 19 | **14** | **73.7%** | **Q: 14**, NEUTRAL: 5 |
| **R** | `D0511.json` (Toàn quốc) | 32 | 16 | **11** | **68.8%** | **R: 11**, NEUTRAL: 5 |
| **M** | `D0504.json` (Toàn quốc) | 48 | 19 | **17** | **89.5%** | **M: 17**, NEUTRAL: 2 |
| **N** | `D0505.json` (Toàn quốc) | 38 | 25 | **16** | **64.0%** | **N: 16**, NEUTRAL: 8, M: 1 |

#### C. Đánh giá Hạn chế của Dữ liệu Telemetry Webcam Hiện có:
- **Hiện trạng log webcam:** Trong file tổng kết webcam gần nhất (`pilot_eval_29_..._frames.jsonl`, 474 frames), trường `landmarks` được lưu là `None` do cấu trúc payload gửi từ client chỉ tóm tắt nhãn dự đoán.
- **Hệ quả kỹ thuật:** Không thể trích xuất trực tiếp raw MLP confidence hoặc tái hiện chi tiết các điều kiện con của bộ trích xuất đặc trưng hình học từ file tóm tắt này cho H, Q, R, M, N. Riêng chữ O có các file deep log đầy đủ (`pilot_pilot_*.jsonl` với hơn 2,000 frames có landmarks).
- **Giải pháp bảo đảm tính khoa học:** Bắt buộc sử dụng 100% landmarks thực tế từ **Canonical Evaluation Suite** (`D0510`, `D0500N`, `D0506`, `D0511`, `D0504`, `D0505`) để mô phỏng offline trước khi đề xuất thay đổi code tại các TASK 1–5. Tuyệt đối không dùng video đã tham gia training để tuyên bố độ chính xác trên tập holdout độc lập.

---

## 5. ĐỐI CHIẾU ROOT CAUSE VERIFIED VS HYPOTHESIS (RẠCH RÒI BẰNG CHỨNG)

| Chữ cái | Hiện tượng quan sát được | Root Cause Verified (Đã xác minh từ mã nguồn & dữ liệu thực tế) | Hypothesis / Unverified (Giả thuyết chưa kiểm chứng — Cấm khẳng định chắc chắn) |
| :---: | :---| :---| :---|
| **H** | 13 frame thành V, 5 frame thành R, 68 frame NEUTRAL trong phiên 024454; 28/30 frame sai thành R trong pilot 08/11. | **`[ROOT CAUSE VERIFIED]`**<br>1. Rule V (dòng 630) chạy trước Rule H và cướp nhãn khi $ang_{IM} \ge 12^\circ$.<br>2. Rule R (dòng 1055) cướp nhãn khi 2 ngón duỗi ngang chồng lấn trên hình chiếu 2D làm $cross\_indicator \le 0.0060$.<br>3. Điều kiện góc ngang của Rule H bị trượt khi tay hơi nghiêng về phía camera hoặc hai ngón tay chưa song song tuyệt đối. | **`[HYPOTHESIS / UNVERIFIED]`**<br>1. Ngưỡng V-guard ứng viên ($idx\_horiz \ge 0.50$, $|idx\_dx| > |idx\_dy|$) có bảo toàn 100% các mẫu chữ V thật hay không.<br>2. Nới lỏng góc H ngang từ 28° lên 35° có an toàn trên webcam đa người ký hay không. |
| **O** | 21 frame O thành A trong phiên 024454; 81 frame A trong pilot 070345; 17 frame A trong pilot 062957. | **`[ROOT CAUSE VERIFIED]`**<br>1. Ngưỡng hình học $d_{TI} \le 0.23$ quá ngặt khi tay hơi lỏng.<br>2. `recognizer.py#L78` vắng mặt O trong Fallback list: khi Rule O trượt, Static Head đoán A được chấp nhận ngay, nhưng đoán đúng O lại bị từ chối thành NEUTRAL. | **`[HYPOTHESIS / UNVERIFIED]`**<br>1. Giá trị ngưỡng nới lỏng ứng viên $d_{TI} \le 0.28$ có gây False Positive sang S hoặc C hay không.<br>2. Cơ chế chặn Fallback A có triệt tiêu hoàn toàn O $\rightarrow$ A mà không gây phụ thuộc quá mức vào heuristics hay không. |
| **Q** | 28/29 frame NEUTRAL; 1 frame nhầm P. Pilot 2141 có 300/303 frame NEUTRAL. Replay V2 chỉ đạt 26.32%. | **`[ROOT CAUSE VERIFIED]`**<br>Ngưỡng nhân tạo `DEFAULT_SUPPLEMENTAL_THRESHOLDS["Q"] = 0.85` tại `recognizer.py#L90` loại bỏ toàn bộ các frame đúng của Q có confidence từ 0.62 đến 0.81 từ Static Head. | **`[HYPOTHESIS / UNVERIFIED]`**<br>Hạ ngưỡng ứng viên $0.85 \rightarrow 0.70$ có làm tăng False Positive sang P hoặc X trên người dùng thực tế hay không. (CẤM khẳng định 0.70 là an toàn khi chưa có thực nghiệm đối chứng). |
| **R** | 2/4 trials PASS, 16/19 non-neutral frames đúng (84.2%). Nhầm 1 H, 1 D, 1 U. | **`[ROOT CAUSE VERIFIED]`**<br>Chữ R đạt 84.2% active precision trên webcam nhưng nhạy cảm với góc xoay cổ tay do $cross\_indicator$ đo trên 2D MediaPipe bị che khuất ngón giữa. | **`[HYPOTHESIS / UNVERIFIED]`**<br>Giả thuyết "Rule R sẽ tự ổn định hoàn toàn sau khi sửa H (triệt tiêu H $\rightarrow$ R)" là giả thuyết cần kiểm chứng lại ở TASK 5, chưa thể kết luận trước. |
| **M** | 88/88 frame NEUTRAL trong phiên 024454; HUD báo NO_HAND. Pilot 2141 có 97/99 frame MLP đúng bị reject. | **`[ROOT CAUSE VERIFIED]`**<br>Hoàn toàn KHÔNG CÓ Rule M trong `alphabet_classifier.py`. Toàn bộ frame phụ thuộc vào Static Head MLP 72D và bị gating từ chối khi conf < 0.70. | **`[HYPOTHESIS / UNVERIFIED]`**<br>1. Giả thuyết "Mô hình MLP bị overfitting video đơn lẻ D0504" là GIẢ THUYẾT, chưa được chứng minh bằng thực nghiệm đa người ký.<br>2. Khả năng xây dựng Anchor Rule hình học phân biệt M vs N trên landmark 2D là giả thuyết thiết kế, chưa có độ phân tách kiểm chứng. |
| **N** | 103/103 frame NEUTRAL trong phiên 024454; HUD báo NO_HAND. Pilot 2141 có 291/295 frame MLP đúng bị reject. | **`[ROOT CAUSE VERIFIED]`**<br>Hoàn toàn KHÔNG CÓ Rule N trong `alphabet_classifier.py`. Ngưỡng gating N đặt ở 0.75 trong khi conf thực tế webcam tập trung ở 0.55–0.72. | **`[HYPOTHESIS / UNVERIFIED]`**<br>1. Giả thuyết "Hạ ngưỡng N $0.75 \rightarrow 0.70$ là an toàn" là GIẢ THUYẾT CANDIDATE; chưa có dữ liệu đo đạc tỷ lệ nhầm lẫn chéo M $\leftrightarrow$ N.<br>2. Khoảng cách phân tách hình học giữa N (2 ngón trùm) và M (3 ngón trùm) trên 2D landmarks chưa được chứng minh. |

---

## 6. MA TRẬN PHỤ THUỘC VÀ ĐÁNH GIÁ RỦI RO HỒI QUY (DEPENDENCIES & REGRESSION RISKS)

### 6.1. Đồ thị phụ thuộc kỹ thuật và Thứ tự thực thi chuẩn (Execution Sequence)

Thứ tự xử lý các Task được thống nhất tuần tự và chặt chẽ:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 SƠ ĐỒ PHỤ THUỘC VÀ THỨ TỰ THỰC THI CHUẨN                    │
│                                                                             │
│  [TASK 0: READ-ONLY BASELINE & PROVENANCE]                                  │
│         │                                                                   │
│         ▼                                                                   │
│  [TASK 1: FIX Q] (P0 — Hạ ngưỡng nhân tạo 0.85 -> 0.70, đối kháng P/X)      │
│         │                                                                   │
│         ▼                                                                   │
│  [TASK 2: FIX O] (P1 — Kích hoạt Fallback O, nới lỏng d_TI <= 0.28, chặn A) │
│         │                                                                   │
│         ▼                                                                   │
│  [TASK 3: FIX H] (P1 — V-guard chặn V cướp H, chặn H -> R, giữ H ngang)    │
│         │                                                                   │
│         ▼                                                                   │
│  [TASK 4: FIX M & N] (P2/P3 — Gating tuning & Geometric Anchor Rules M/N)   │
│         │                                                                   │
│         ▼                                                                   │
│  [TASK 5: ĐÁNH GIÁ HOẶC SKIP R] (P2 — TÙY CHỌN, CHỈ SỬA KHI R THỰC SỰ CẦN)  │
│         │                                                                   │
│         ▼                                                                   │
│  [TASK 6: FINAL REGRESSION & OFFICIAL WEBCAM HOLDOUT] (P3 — Nghiệm thu)     │
└─────────────────────────────────────────────────────────────────────────────┘
```

1. **TASK 0 (Baseline & Provenance):** Chạy kiểm chứng toàn diện môi trường ở chế độ read-only.
2. **TASK 1 (Q - P0):** Độc lập nhất, sửa tối thiểu ngưỡng gating và đối chứng P/X.
3. **TASK 2 (O - P1):** Độc lập hình thái nắm tròn, kích hoạt Fallback O và chặn O $\rightarrow$ A.
4. **TASK 3 (H - P1):** Thiết lập V-guard chặn V cướp H và bảo vệ H ngang. Phải hoàn thành H trước để triệt tiêu nguyên nhân gây nhầm lẫn sang R.
5. **TASK 4 (M & N - P2/P3):** Xử lý đồng thời cặp đối kháng M và N sau khi các chữ hình học cơ bản đã ổn định.
6. **TASK 5 (R - P2 - TÙY CHỌN):** **Chỉ thực hiện khi R thực sự cần sửa**. Sau khi TASK 3 (H) hoàn tất, hệ thống sẽ đo đạc lại R trên webcam holdout mới. Nếu R thỏa mãn đầy đủ 4 tiêu chí SKIP R (xem Mục 6.2), **CHÍNH THỨC SKIP TASK 5 (Không sửa code Rule R)**. Chỉ sửa khi R phát sinh lỗi.
7. **TASK 6 (Final Regression - P3):** Nghiệm thu chính thức trên toàn bộ 22 chữ cái tĩnh.

### 6.2. Tiêu chuẩn định lượng để SKIP TASK 5 (Điều kiện SKIP R)

Để đưa ra quyết định **SKIP TASK 5 (Không sửa Rule R)** một cách khoa học, hệ thống bắt buộc phải kiểm chứng và thỏa mãn đồng thời **4 tiêu chí định lượng** trên phiên Webcam Holdout mới (tối thiểu 10 trials, mỗi trial 5.0s):

1. **Tỷ lệ Trial Đạt chuẩn (Trial Pass Rate):**
   $$\text{Trial Pass Rate} = \frac{\text{Passed Trials}}{\text{Total Valid Trials}} \ge \frac{7}{10} \ (70.0\%)$$
   (Có tối thiểu 7 trong 10 trials của R đạt chuẩn PASS theo protocol).
2. **Độ chính xác trên các frame hoạt động trong pha Stable Hold (Active Precision in Stable Hold):**
   $$\text{Active Precision}_{\text{Stable Hold}} = \frac{\text{Match Frames}}{\text{Match Frames} + \text{Mismatch Frames}} \ge 75.0\%$$
   (Đạt tối thiểu 75.0% trên các frame có nhãn nhận diện được trong pha giữ cử chỉ ổn định).
3. **Kiểm soát tỷ lệ rơi vào NEUTRAL trong pha Stable Hold (Neutral Ratio in Stable Hold):**
   $$\text{Neutral Ratio}_{\text{Stable Hold}} = \frac{\text{Neutral Frames}}{\text{Total Frames in Stable Hold}} \le 30.0\%$$
   (Trong vùng Stable Hold của người dùng, tỷ lệ rơi vào NEUTRAL không được vượt quá 30.0%, đảm bảo nhận diện liên tục, không bị giật/chập chờn).
4. **Triệt tiêu nhầm lẫn chéo hai chiều H $\leftrightarrow$ R (Bidirectional H/R Zero Regression):**
   - Trên 10 trials của Chữ H: Số frame bị Rule R cướp nhãn ($H \rightarrow R$) phải bằng $0$ frame (hoặc $\le 2.0\%$ trên tổng số active frames).
   - Trên 10 trials của Chữ R: Số frame bị nhận nhầm sang H ($R \rightarrow H$) phải bằng $0$ frame.

$\rightarrow$ Nếu thỏa mãn đồng thời cả 4 tiêu chí: Đệ trình Gate 5 phê duyệt **SKIP TASK 5**. Nếu bất kỳ tiêu chí nào bị vi phạm: Chuyển sang thực hiện Task 5 để tinh chỉnh Rule R.

### 6.3. Các giá trị Threshold và Guards đề xuất mang tính CANDIDATE — NOT APPROVED

> [!WARNING]
> Mọi giá trị ngưỡng và biểu thức logic dưới đây đều được gắn nhãn **`CANDIDATE — NOT APPROVED`**. Tuyệt đối **không mặc định** các giá trị này là phương án cuối cùng. Chúng chỉ là các điểm bắt đầu khảo sát trong quá trình mô phỏng offline tại Task tương ứng:
> - **Chữ Q:** Ngưỡng `0.70` là `CANDIDATE — NOT APPROVED`. Quá trình mô phỏng Task 1 phải quét qua cả dải $[0.85, 0.80, 0.75, 0.70]$ và đo đạc đối kháng với P/X.
> - **Chữ O:** Ngưỡng `O_D_TI_MAX = 0.28` và cơ chế Fallback O là `CANDIDATE — NOT APPROVED`. Cần kiểm thử hồi quy độc lập với A, S, C, X.
> - **Chữ N:** Ngưỡng `0.70` là `CANDIDATE — NOT APPROVED`. Phải đo đạc phân bố conf và nhầm lẫn M $\leftrightarrow$ N trước khi quyết định.
> - **Chữ R:** Ngưỡng `R_CROSS_IND_MAX = 0.0080` và gating `0.70` là `CANDIDATE — NOT APPROVED`. Mặc định ưu tiên SKIP nếu Task 3 đã giúp R ổn định.
> - **Chữ H:** Biểu thức V-guard ($|idx\_dx| > |idx\_dy|$ và $idx\_horiz \ge 0.50$) là `CANDIDATE — NOT APPROVED`. Cần đo trên video D0500N và video V thật.

### 6.4. Phân tích rủi ro hồi quy (Regression Risks & Invariants)

Bất kỳ thay đổi nào cũng phải đảm bảo các bất biến sau:

1. **Bảo vệ tuyệt đối Chữ P (Rule 15) và Chữ X (Rule 18) `[ZERO REGRESSION INVARIANT]`:**
   - P và X đã đạt chuẩn **`WEBCAM VERIFIED — PASS`** (100% trial PASS).
   - *Rủi ro với Q:* Cả Q và P đều có cử chỉ ngón tay chúc xuống đất. Khi hạ ngưỡng Q, phải kiểm tra đối kháng nghiêm ngặt để đảm bảo Q không nhận diện nhầm thành P và P không bị Q cướp nhãn. Ràng buộc: Ngón giữa của Q gập ($m\_curl \ge 0.35$), ngón giữa của P duỗi ($d_{MW} \ge 1.40$).
   - *Rủi ro với O:* Chữ X có tư thế ngón trỏ móc câu chúi trước. Phải đảm bảo vòng khép O không nhận nhầm ngón trỏ móc câu của X.
2. **Bảo vệ cặp đối kháng O vs A, S, C:**
   - Khi nới lỏng $d_{TI}$ cho O và kích hoạt Fallback O, phải bảo vệ tuyệt đối Rule A (ngón cái dựng đứng $d_{TW} \ge 1.20$), Rule S (ngón cái vắt ngang), và Rule C (vòm mở rộng $d_{TI} \ge 0.55$).
3. **Bảo vệ cặp đối kháng H vs V, U, K:**
   - Việc bổ sung V-guard cho H không được làm hỏng Rule V khi người dùng thực hiện chữ V thực sự (hai ngón duỗi dựng đứng $ang_{IM} \ge 12^\circ$ và vector Y chiếm ưu thế).
   - Bảo vệ Rule U (hai ngón khép sát dựng đứng song song $ang_{IM} \le 6.5^\circ$).

---

## 7. HỆ THỐNG 7 CỔNG PHÊ DUYỆT (7 APPROVAL GATES: GATE 0 ĐẾN GATE 6)

Mọi bước chuyển giao trong kế hoạch bắt buộc dừng lại tại cổng phê duyệt tương ứng. **Cấm tự ý chuyển task khi chưa có văn bản phê duyệt từ Người dùng.**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      HỆ THỐNG 7 CỔNG PHÊ DUYỆT TUẦN TỰ                      │
│                                                                             │
│  [GATE 0: PLAN & PROVENANCE] ──────► Cho phép bắt đầu TASK 0                │
│             │                                                               │
│             ▼                                                               │
│  [GATE 1: TASK 1 PROPOSAL (Q)] ───► Cho phép minimal diff Chữ Q             │
│             │                                                               │
│             ▼                                                               │
│  [GATE 2: TASK 2 PROPOSAL (O)] ───► Cho phép minimal diff Chữ O             │
│             │                                                               │
│             ▼                                                               │
│  [GATE 3: TASK 3 PROPOSAL (H)] ───► Cho phép minimal diff Chữ H             │
│             │                                                               │
│             ▼                                                               │
│  [GATE 4: TASK 4 PROPOSAL (M/N)] ─► Cho phép minimal diff Chữ M & N         │
│             │                                                               │
│             ▼                                                               │
│  [GATE 5: TASK 5 EVAL/SKIP (R)] ──► Quyết định SKIP hoặc minimal diff Chữ R │
│             │                                                               │
│             ▼                                                               │
│  [GATE 6: FINAL ACCEPTANCE] ──────► Nghiệm thu chính thức 22 chữ cái        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Chi tiết tiêu chí từng Cổng Phê Duyệt:

| Gate | Tên Cổng Phê Duyệt | Điều kiện vào (Entry Conditions) | Điều kiện PASS / Tiêu chí chấp thuận | Điều kiện FAIL / STOP & Rollback | Yêu cầu phê duyệt từ Người dùng |
| :---: | :---| :---| :---| :---| :---|
| **GATE 0** | **Phê duyệt Kế hoạch & Xác minh Hiện trạng** | File docs `VSL_6_LETTERS_REPAIR_PLAN.md` hoàn thiện, không sửa mã nguồn, working tree sạch. | 1. Thống nhất quy tắc ưu tiên nguồn QIPEDC.<br>2. Phân tách Legacy Suite (451/551) vs Canonical Suite (H miền Nam D0500N; O, Q, R, M, N toàn quốc).<br>3. Thống nhất Regression Contract: 198 tests cho 22 chữ tĩnh (198/198 PASS) và 8 tests FSM trong Backlog 29 chữ.<br>4. Xác nhận bản vá an ninh Static Head (PyTorch 2.2.2 compatibility + SHA-256 AND condition).<br>5. Chuẩn hóa 4 tiêu chí SKIP R và nhãn CANDIDATE cho thresholds. | Bất kỳ mâu thuẫn provenance nào chưa được làm rõ, hoặc có hành vi tự sửa code trước Gate 0. | **BẮT BUỘC:** Người dùng phê duyệt bằng văn bản trước khi cho phép chạy TASK 0. |
| **GATE 1** | **Phê duyệt Đề xuất Kỹ thuật Chữ Q** | Hoàn thành TASK 0; hoàn thành mô phỏng offline quét dải ngưỡng CANDIDATE $[0.85, 0.70]$ của Q; hoàn thành kiểm thử đối kháng P/X. | Recall của Q tăng có ý nghĩa trên D0510 và webcam; False Positive trên P và X bằng 0.0%; không hồi quy 22 chữ. | P hoặc X bị nhận nhầm thành Q, hoặc False Positive > 0.0% $\rightarrow$ STOP, không hạ ngưỡng. | **BẮT BUỘC:** Người dùng duyệt phương án threshold Q trước khi sửa `recognizer.py`. |
| **GATE 2** | **Phê duyệt Đề xuất Kỹ thuật Chữ O** | Hoàn thành TASK 1; khảo sát xong khoảng cách $d_{TI}$ trên D0506 và log webcam; thử nghiệm tách biệt 3 yếu tố CANDIDATE (Rule O, Fallback O, Guard O/A). | Triệt tiêu O $\rightarrow$ A trên dữ liệu replay và webcam; False Positive sang S/C/X bằng 0.0%; unit tests PASS. | Gộp chung 3 thay đổi mà không đo độc lập, hoặc phát sinh nhầm lẫn sang S/C $\rightarrow$ STOP, điều chỉnh lại. | **BẮT BUỘC:** Người dùng duyệt phương án Fallback O & Rule O trước khi sửa code. |
| **GATE 3** | **Phê duyệt Đề xuất Kỹ thuật Chữ H** | Hoàn thành TASK 2; hoàn thành khảo sát vector duỗi ngón ngang trên D0500N; hoàn thành thiết kế V-guard và H-guard chống R. | V-guard bảo vệ được H ngang; Rule V thật và Rule R thật không bị suy giảm độ chính xác; triệt tiêu H $\rightarrow$ R/V. | Rule V bị hỏng trên các mẫu V thật, hoặc False Positive H/R tăng $\rightarrow$ STOP, rollback minimal diff. | **BẮT BUỘC:** Người dùng duyệt V-guard và Rule H trước khi sửa code. |
| **GATE 4** | **Phê duyệt Đề xuất Kỹ thuật Chữ M & N** | Hoàn thành TASK 3; khảo sát xong phân bố confidence thô của M (D0504) và N (D0505); đánh giá mức độ khả thi của calibration gating vs geometric anchors. | Giải phóng frame đúng của M và N; kiểm soát được tỷ lệ nhầm lẫn chéo M $\leftrightarrow$ N; bảo vệ tuyệt đối A/S/T. | Heuristic anchor rule gây bất ổn định hoặc tăng False Positive sang A/S/T $\rightarrow$ STOP, chỉ giữ calibration gating an toàn. | **BẮT BUỘC:** Người dùng duyệt phương án M & N trước khi sửa code. |
| **GATE 5** | **Phê duyệt Đánh giá hoặc SKIP Chữ R** | Hoàn thành TASK 3 & 4; đo đạc lại hiệu năng của Rule R sau khi H đã được sửa trên webcam holdout mới (10 trials). | Nếu R thỏa mãn đồng thời 4 tiêu chí tại Mục 6.2 $\rightarrow$ **Duyệt SKIP TASK 5**; Nếu R chưa đạt $\rightarrow$ Duyệt minimal diff tinh chỉnh Rule R. | Tinh chỉnh R làm tăng nhầm lẫn H $\rightarrow$ R trở lại $\rightarrow$ STOP, rollback về Rule R ban đầu. | **BẮT BUỘC:** Người dùng duyệt quyết định SKIP hoặc minimal diff của R. |
| **GATE 6** | **Nghiệm thu Toàn diện (Final Acceptance)** | Hoàn thành toàn bộ task sửa chữ được duyệt; 198/198 tests 22 chữ cái tĩnh đạt 100% PASS; báo cáo minh bạch 8 tests FSM Backlog; hoàn thành phiên Controlled Webcam Pilot Holdout mới. | 6 chữ mục tiêu đạt Active Precision $\ge 70\%$, Trial Accuracy $\ge 7/10$; P & X duy trì 100% PASS; 16 chữ còn lại không hồi quy; 198 tests tĩnh PASS 100%. | Phát sinh hồi quy trên 16 chữ ổn định, hoặc trial pass < 7/10 $\rightarrow$ Revert commit của task gây lỗi, không merge. | **BẮT BUỘC:** Người dùng ký duyệt đóng đợt sửa chữa và cho phép tạo commit/merge. |

---

## 8. KẾ HOẠCH CHI TIẾT THEO CHECKPOINT (TASK 0 ĐẾN TASK 6)

### 8.1. Hợp đồng Kiểm thử Hồi quy Toàn diện (Regression Contract: 198 + 8 = 206 tests)

Để đảm bảo kỷ luật kỹ thuật cao nhất, không che giấu lỗi và phân định rạch ròi phạm vi nghiệm thu giữa **Bảng 22 chữ cái tĩnh không dấu** và **Chức năng dấu thanh động**:

1. **Tổng số test hiện có trong repository `[SOURCE VERIFIED]`:**
   - Khi chạy `pytest backend/tests --collect-only`, hệ thống thu thập được **627 tests** phân bổ trên 40 test files (bao gồm chữ cái, chữ số, diacritic, pilot HUD, telemetry, v.v.).
2. **Con số "147 unit tests" lịch sử:**
   - Con số 147 tests nhắc đến trong các báo cáo cũ là tập con hình học cơ bản (`test_alphabet_classifier.py` 137 tests + `test_mode_filtering.py` 10 tests). Tuyệt đối **không dùng con số 147 để đại diện cho toàn bộ hệ thống recognition**.
3. **Phân tách Hợp đồng Hồi quy Cốt lõi (206 tests):**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 HỢP ĐỒNG KIỂM THỬ HỒI QUY TOÀN DIỆN (206 TESTS)             │
│                                                                             │
│  [PHẠM VI 1: 22 CHỮ CÁI TĨNH KHÔNG DẤU] (198 tests) ──► 198/198 PASS (100%) │
│   ├─ test_alphabet_classifier.py                 : 137 tests (PASS)         │
│   ├─ test_alphabet_recognition_integration.py     : 9 tests   (PASS)         │
│   ├─ test_alphabet_static_head_integration.py     : 13 tests  (PASS)         │
│   ├─ test_mode_filtering.py                       : 10 tests  (PASS)         │
│   ├─ test_mode_separation.py                      : 16 tests  (PASS)         │
│   └─ test_static_head_fallback.py (nhóm tĩnh)     : 13 tests  (PASS)         │
│                                                                             │
│  [PHẠM VI 2: DIACRITIC FSM DẤU THANH ĐỘNG BACKLOG] (8 tests) ──► 0/8 (FAIL) │
│   └─ test_static_head_fallback.py (test_a4, test_d1..d7): 8 tests           │
│      (FSM đang chủ động tắt trong mode letters từ commit f5a26c2f)          │
└─────────────────────────────────────────────────────────────────────────────┘
```

   - **Phạm vi 1 — Static 22-Letter Regression Suite (198 tests):**
     - Bao gồm toàn bộ các bài test về quy tắc hình học, nạp và suy luận Static Head, lọc nhãn, cô lập 4 chế độ nhận diện, và fallback cho 3 chữ cái nền tĩnh A, D, E.
     - **Baseline đã kiểm chứng thực tế:** **198/198 PASSED (100% PASS)** tại TASK 0.C và TASK 0.E.
     - **Quy tắc bắt buộc:** Mọi checkpoint sửa code của 6 chữ H, O, Q, R, M, N bắt buộc phải chạy đầy đủ 198 tests này và đạt 198/198 PASS. Tuyệt đối **không được tuyên bố 198/198 PASS** nếu chưa thực sự chạy lệnh kiểm thử trên checkpoint tương ứng.
   - **Phạm vi 2 — Diacritic FSM Regression Backlog (8 tests):**
     - Gồm 8 tests chuyển trạng thái FSM: `test_a4_fallback_feeds_fsm_session`, `test_d1_fallback_a_achieves_base_stable`, `test_d2_fallback_e_achieves_base_stable`, `test_d3_fallback_d_achieves_base_stable`, `test_d4_fallback_a_to_a_breve`, `test_d5_fallback_d_to_d_stroke`, `test_d6_fallback_a_to_a_circumflex`, `test_d7_fallback_e_to_e_circumflex`.
     - **Hiện trạng:** 0/8 PASS (8 FAIL) do `DiacriticStateMachine` trong `mode="letters"` đang bị chủ động ngắt từ commit `f5a26c2f` (*"Chưa kích hoạt lại DiacriticStateMachine trong nhiệm vụ này"*).
     - **Kỷ luật:** Giữ nguyên 8 tests trong mã nguồn; tuyệt đối **không thêm `@unittest.skip`**, không xóa assertion, không sửa test code để làm đẹp báo cáo. Đưa toàn bộ 8 tests này vào Backlog chức năng dấu thanh động của giai đoạn mở rộng 29 chữ cái. Tuyệt đối không tự ý bật lại FSM trong đợt sửa 6 chữ tĩnh này.
   - **Nguyên tắc Minh bạch Toàn diện (Full Test Transparency):**
     - Mọi báo cáo sau mỗi checkpoint bắt buộc trình bày đầy đủ cả 2 chỉ số:
       - *Static 22-Letter Regression:* X/198 PASS.
       - *Diacritic FSM Backlog:* Y/8 PASS (hiện 0/8).
       - *Tổng hợp:* (X+Y)/206 tests.
     - Tuyệt đối **không bao giờ tuyên bố 206/206 PASS** khi 8 tests FSM chưa được giải quyết. Việc tách phạm vi chỉ phục vụ nghiệm thu độc lập cho 22 chữ cái tĩnh, không xóa bỏ nghĩa vụ khôi phục chức năng tạo dấu động 29 chữ cái trong tương lai.

---

### 8.2. Chi tiết từng Task thực thi

#### TASK 0 — Baseline, Provenance và Xác minh Môi trường (READ-ONLY, ĐÃ HOÀN THÀNH)
- **Mục tiêu:** Xác minh 100% tính toàn vẹn của mã nguồn, commit, dữ liệu provenance và thiết lập baseline kiểm thử hồi quy trước khi chạm vào bất kỳ dòng code nào.
- **Hiện trạng đã kiểm chứng và xử lý (TASK 0.A đến TASK 0.E):**
  1. *Git & Working Tree:* HEAD `ce9efdeea77d` trên nhánh `feat/controlled-pilot-29-letters`, working tree sạch.
  2. *Khắc phục Blocker nạp Model:* Đã bổ sung bản vá tương thích PyTorch 2.2.2 / NumPy 1.26.4 trong [`backend/models/alphabet_static_head.py`](file:///d:/SIGN_LANGUAGE/backend/models/alphabet_static_head.py); Static Head nạp thành công 100%, 22 classes, input_dim 72, SHA-256 `cab8123e...` giữ nguyên vẹn.
  3. *Khắc phục Blocker An ninh:* Đã chuyển điều kiện xác thực từ `or` sang `and` (`is_internal_default and file_sha256 == KNOWN_HEAD_CHECKPOINT_SHA256`). Đã kiểm thử từ chối thành công checkpoint giả mạo. Ghi nhận rủi ro TOCTOU cho tương lai.
  4. *Khắc phục Blocker Treo TestClient:* Đã xác minh nguyên nhân do Windows asyncio `socket.socketpair()` bị Sandbox cô lập loopback; đã thực thi thành công 16/16 tests `test_mode_separation.py` với quyền ngoài sandbox được cấp phép.
  5. *Baseline Dữ liệu:* Đã hoàn thành đo đạc Legacy Suite (451/551, 81.85%) và Canonical Suite cho 6 chữ (H: 64.9%, O: 65.0%, Q: 73.7%, R: 68.8%, M: 89.5%, N: 64.0%).
  6. *Hợp đồng Hồi quy:* Đã thiết lập Hợp đồng 198 + 8 tests (198/198 PASS trên phạm vi 22 chữ tĩnh).
- **Tiêu chí hoàn thành (Exit Criteria):**
  - [x] 198/198 tests 22 chữ cái tĩnh PASS.
  - [x] Báo cáo minh bạch 8 tests FSM trong backlog chức năng 29 chữ.
  - [x] Bộ baseline Legacy và Canonical được ghi nhận đầy đủ trong SSOT.
  - [x] Hệ thống sẵn sàng bước vào GATE 1 (Chữ Q).
- **STOP Condition:** Đã vượt qua toàn bộ blockers kỹ thuật và an ninh; sẵn sàng nghiệm thu TASK 0.

---

#### TASK 1 — Chữ Q: Rule Q, Ngưỡng Gating và Phòng chống Xung đột P/X
- **Mục tiêu:** Khắc phục 28/29 frame rơi vào `NEUTRAL` của chữ Q, triệt tiêu rào cản ngưỡng nhân tạo 0.85 mà không gây hồi quy sang P và X.
- **Các bước thực hiện:**
  1. *Source Audit & Data Analysis:* Khảo sát phân bố confidence thô của Q từ Static Head trên video toàn quốc `D0510.mp4` và log webcam.
  2. *Offline Candidate Simulation (Không sửa production):* Thử nghiệm quét ngưỡng ứng viên CANDIDATE $0.85 \rightarrow 0.80 \rightarrow 0.75 \rightarrow 0.70$ trên tập offline Q, P, X. Đo lường tỷ lệ True Positive (Q) và False Positive (P, X).
  3. *Lập đề xuất kỹ thuật tối thiểu (Minimal Proposal):* Nêu rõ dòng code cần sửa trong `recognizer.py`, rủi ro đối kháng và kết quả replay.
  4. **DỪNG TẠI GATE 1 — Chờ User Phê Duyệt.**
  5. *Implement Minimal Diff:* Chỉ sửa ngưỡng của Q trong `DEFAULT_SUPPLEMENTAL_THRESHOLDS["Q"]` theo phương án đã duyệt.
  6. *Unit Test & Replay Verification:* Chạy unit test riêng của Q, test đối kháng P/X và regression suite 22 chữ.
  7. *Webcam Holdout Gate:* Mời người dùng mở Controlled Pilot HUD thử nghiệm 10 trials cho chữ Q và các trial đối chứng P, X.

---

#### TASK 2 — Chữ O: Rule O, Fallback O/A và Bảo vệ Cặp Đối Kháng A/S/C
- **Mục tiêu:** Khắc phục 21 frame O bị nhầm sang A và hiện tượng rơi vào NEUTRAL khi vòng tròn lỏng tay; nâng Trial Accuracy của O từ 50% lên $\ge 70\%$.
- **Các bước thực hiện:**
  1. *Source & Data Audit:* Phân tích khoảng cách $d_{TI}$, $d_{TM}$, $d_{TW}$ trên video toàn quốc `D0506.mp4` và các frame FAIL của Rep 2 trong phiên `024454`.
  2. *Thử nghiệm tách biệt 3 yếu tố CANDIDATE (Không gộp chung):*
     - Yếu tố A: Nới lỏng nhẹ ngưỡng hình học `O_D_TI_MAX` ($0.23 \rightarrow 0.28$) trong `alphabet_classifier.py`.
     - Yếu tố B: Bổ sung `"O"` vào `ALLOWED_FALLBACK_BASE_LETTERS` trong `recognizer.py` để Static Head có quyền cứu O.
     - Yếu tố C: Xây dựng guard chặn: Nếu hình thái tay có vòm tròn rỗng thì cấm Static Head fallback nhận thành `A`.
  3. *Replay & Cross-letter Evaluation:* Đo lường độ chính xác của O và đo False Positive trên tập đối kháng A, S, C, X.
  4. **DỪNG TẠI GATE 2 — Chờ User Phê Duyệt.**
  5. *Implement Minimal Diff:* Sửa code chính xác theo phương án được duyệt.
  6. *Regression & Webcam Holdout:* Chạy full unit tests, replay 22 chữ và mời người dùng test holdout 10 trials cho O trên webcam.

---

#### TASK 3 — Chữ H: Xử lý V-guard, Triệt tiêu H $\rightarrow$ R/V, Bảo toàn H ngang
- **Mục tiêu:** Khắc phục 13 frame H $\rightarrow$ V và 5 frame H $\rightarrow$ R; khôi phục khả năng nhận diện H duỗi ngang trên webcam.
- **Các bước thực hiện:**
  1. *Source & Data Audit:* Sử dụng landmarks từ nguồn Miền Nam chuẩn dự phòng `D0500N` (Ưu tiên 2); loại bỏ hoàn toàn `D0500B` khỏi benchmark canonical. Phân tích vector duỗi của ngón trỏ và ngón giữa ($idx\_dx, idx\_dy, ang_{IM}$).
  2. *Thiết kế cơ chế phân lập CANDIDATE:*
     - *Phân lập H vs V:* Bổ sung điều kiện trong Rule V: Nếu hai ngón duỗi theo phương ngang ($|idx\_dx| > |idx\_dy|$ và $idx\_horiz \ge 0.50$), Rule V từ chối kích hoạt để nhường quyền cho Rule H.
     - *Phân lập H vs R:* Trong Rule R, nếu hướng ngón tay nằm ngang, yêu cầu chỉ số $cross\_indicator$ phải thật sự khắt khe mới cho phép nhận R.
  3. *Replay Verification:* Kiểm thử trên tập Canonical D0500N, tập V (`D0516`), tập R (`D0511`), tập U (`D0514`), tập K (`D0502`).
  4. **DỪNG TẠI GATE 3 — Chờ User Phê Duyệt.**
  5. *Implement Minimal Diff & Regression Test.*
  6. *Webcam Holdout:* Kiểm thử webcam 10 trials cho H và đối chứng V, R.

---

#### TASK 4 — Chữ M & Chữ N: Chẩn đoán Gating và Thử nghiệm Ràng buộc Hình học
- **Mục tiêu:** Giải phóng 97 frame đúng của M và 291 frame đúng của N đang bị bộ lọc gating giam cầm; triệt tiêu tình trạng 100% NEUTRAL (NO_HAND) trên webcam.
- **Các bước thực hiện:**
  1. *Source & Data Audit:* Đối chiếu video toàn quốc `D0504.mp4` (M) và `D0505.mp4` (N). Khảo sát phân bố confidence thực tế của MLP trên webcam.
  2. *Giai đoạn 1 — Tuning Gating Ngưỡng CANDIDATE:*
     - Thử nghiệm hạ ngưỡng của N trong `DEFAULT_SUPPLEMENTAL_THRESHOLDS["N"]` từ $0.75$ về $0.70$ trên dữ liệu replay.
     - Đánh giá mức độ giải phóng frame đúng và tỷ lệ nhầm lẫn chéo M $\leftrightarrow$ N.
  3. *Giai đoạn 2 — Nghiên cứu Geometric Anchor Rules (Nếu tuning ngưỡng chưa đủ):*
     - Khảo sát khả năng xây dựng luật hình học dựa trên 21 landmarks MediaPipe: kiểm tra tọa độ đầu ngón cái luồn dưới đốt ngón giữa/áp út.
     - Phân định M (3 ngón trùm) vs N (2 ngón trùm) qua độ co cụm của khớp ngón.
     - *Lưu ý an toàn:* Nếu dữ liệu landmark 2D không đủ độ tin cậy để phân tách M vs N một cách ổn định, **dừng lại ở phương án calibration gating**, không cố viết heuristic phức tạp gây hồi quy sang A/S/T.
  4. **DỪNG TẠI GATE 4 — Chờ User Phê Duyệt.**
  5. *Implement Minimal Diff & Full Regression.*
  6. *Webcam Holdout:* Kiểm thử webcam 10 trials cho M và 10 trials cho N.

---

#### TASK 5 — Chữ R: Đánh giá hoặc SKIP sau khi H đã Ổn định (TÙY CHỌN)
- **Mục tiêu:** Đánh giá lại hiệu năng của Rule R sau khi đã xử lý xong Task 3 (H) và Task 4 (M/N).
- **Quy trình thực hiện:**
  1. Chạy bài kiểm thử Webcam Holdout mới (10 trials của R, mỗi trial 5.0s).
  2. Đối chiếu với **4 tiêu chí định lượng SKIP R** tại Mục 6.2:
     - Trial Pass Rate $\ge 7/10$ ($70.0\%$).
     - Active Precision trong Stable Hold $\ge 75.0\%$.
     - Neutral Ratio trong Stable Hold $\le 30.0\%$.
     - Triệt tiêu nhầm lẫn chéo hai chiều: $H \rightarrow R \le 2.0\%$ và $R \rightarrow H = 0$ frames.
  3. **Nếu thỏa mãn 4 tiêu chí:** Đệ trình Gate 5 quyết định **CHÍNH THỨC SKIP TASK 5 (Không sửa code Rule R)**.
  4. **Nếu không thỏa mãn:** Trình Gate 5 đề xuất CANDIDATE tinh chỉnh tối thiểu $cross\_indicator$ ($0.0060 \rightarrow 0.0080$) và gating ($0.75 \rightarrow 0.70$).
- **DỪNG TẠI GATE 5 — Chờ User Phê Duyệt quyết định SKIP hoặc đề xuất sửa.**

---

#### TASK 6 — Nghiệm thu Toàn diện (Full Regression & Official Webcam Holdout)
- **Mục tiêu:** Kiểm tra và nghiệm thu dứt điểm toàn bộ hệ thống sau khi hoàn thành các task sửa chữ.
- **Các tiêu chuẩn nghiệm thu bắt buộc:**
  1. *Unit Test Suite:* 100% tests thuộc Phạm vi 22 Chữ cái Tĩnh (198/198 tests) PASS. Đồng thời báo cáo minh bạch 8 tests Diacritic FSM trong Backlog chức năng 29 chữ (không tính là PASS nhưng không xóa bỏ/bỏ qua).
  2. *Replay Benchmark 22 chữ:* Chạy replay trên cả Legacy Suite (551 mẫu) và Canonical Suite, ghi nhận độ chính xác tổng thể không bị suy giảm.
  3. *Bảo toàn P & X:* Chữ P và Chữ X duy trì 100% Trial PASS (Zero False Positive).
  4. *Webcam Holdout chính thức:*
     - Người dùng thực hiện kiểm thử trên Controlled Pilot HUD (10 trials/chữ, mỗi trial 5.0s).
     - Tiêu chí đạt cho mỗi chữ trong 6 chữ (H, O, Q, R, M, N):
       - Active Frame Precision trên pha Stable Hold $\ge 70.0\%$.
       - Trial Accuracy đạt tối thiểu **7/10 trials PASS ($\ge 70.0\%$)**.
       - Tỷ lệ rơi vào `NEUTRAL` trong pha Stable Hold giảm về mức an toàn.
  5. *Rollback Plan:* Nếu bất kỳ checkpoint nào gây hồi quy nghiêm trọng không thể khắc phục, chỉ revert duy nhất commit của task đó bằng patch sạch, không làm ảnh hưởng đến các task đã PASS trước đó.
- **DỪNG TẠI GATE 6 — Trình User quyết định đóng toàn bộ đợt sửa chữa 6 chữ cái.**

---

## 9. QUY TẮC NGHIỆM THU CHUNG CHO MỌI CHECKPOINT

Mọi tác vụ kỹ thuật trong kế hoạch này bắt buộc tuân thủ chu trình 8 bước nghiêm ngặt:

```
[BƯỚC 1: SOURCE & DATA EVIDENCE]
       │
       ▼
[BƯỚC 2: PROPOSED CANDIDATE FIX & REGRESSION RISKS]
       │
       ▼
[BƯỚC 3: USER APPROVAL GATE] ◄── (DỪNG LẠI CHỜ DUYỆT BẰNG VĂN BẢN)
       │
       ▼
[BƯỚC 4: IMPLEMENT MINIMAL DIFF]
       │
       ▼
[BƯỚC 5: UNIT TESTS & OFFLINE REPLAY]
       │
       ▼
[BƯỚC 6: FULL 22-LETTER REGRESSION]
       │
       ▼
[BƯỚC 7: CONTROLLED WEBCAM PILOT HOLDOUT]
       │
       ▼
[BƯỚC 8: ACCEPTANCE / ROLLBACK DECISION]
```

- **Kỷ luật Scoped Changes:** Mỗi task chỉ được sửa đúng file và đúng hàm đã được phê duyệt. Không tự ý "tiện tay" dọn dẹp, refactor hay đổi tên biến ngoài phạm vi.
- **Tính độc lập của dữ liệu kiểm chứng:** Tuyệt đối không dùng dữ liệu webcam đã thu thập để tune ngưỡng làm tập holdout kiểm chứng cuối cùng. Tập webcam holdout phải được người dùng thu mới hoàn toàn sau khi code đã được sửa.
- **Minh bạch số liệu:** Mọi báo cáo sau mỗi checkpoint bắt buộc trình bày đầy đủ: số mẫu, mẫu số tính toán, số frame PASS/FAIL, delta cải thiện, và kết quả kiểm thử hồi quy 16 chữ còn lại.

---

## 10. TEMPLATE BÁO CÁO MỖI CHECKPOINT

Mỗi khi hoàn thành một bước thử nghiệm hoặc chuẩn bị xin phê duyệt Gate, Antigravity phải xuất trình báo cáo chuẩn hóa theo mẫu:

```markdown
### BÁO CÁO CHECKPOINT: [TÊN TASK] — [TRẠNG THÁI: READY FOR APPROVAL / PASS / FAIL / BLOCKED / SKIPPED]
- **Gate liên quan:** [Gate 0 .. Gate 6]
- **Nhánh Git / Commit HEAD / Trạng thái Tree:** [Xác nhận clean tree]
- **Nguồn Provenance áp dụng:** [Toàn quốc không hậu tố? Miền Nam N dự phòng? Đã loại trừ B/T?]
- **Bộ đánh giá:** [Legacy Regression Suite hay Canonical Evaluation Suite?]
- **Chữ cái mục tiêu / Cặp đối kháng kiểm tra:** [Ví dụ: Q vs P, X, G]
- **Tập dữ liệu & Fixture sử dụng:** [Tên file video, số frame, session ID]
- **Số liệu định lượng trước và sau (Mẫu số rõ ràng):**
  - Overall Frame Accuracy: [Trước] -> [Sau]
  - Active Precision (Non-neutral): [Trước] -> [Sau]
  - Trial Pass Rate: [Trước] -> [Sau]
- **Phân tích nguyên nhân gốc / Bằng chứng kiểm chứng:** [Ghi rõ mã nguồn, điều kiện logic]
- **File và Hàm đề xuất sửa đổi (Minimal Diff - CANDIDATE):** [File path, dòng code, diff cụ thể]
- **Kết quả Unit Tests thực tế:** [Số test PASS/FAIL/SKIP cụ thể]
- **Kết quả Replay 22 chữ cái tĩnh:** [Đánh giá hồi quy]
- **Kết quả Webcam Holdout (nếu có):** [Session ID mới, số trial PASS/FAIL]
- **Tác động hồi quy (Regression Impact):** [Ảnh hưởng tới 16 chữ còn lại, đặc biệt P và X]
- **Bằng chứng còn thiếu / Điều kiện dừng (STOP Condition):** [Nếu có]
- **Quyết định đề xuất người dùng phê duyệt:** [Nội dung cần User gật đầu trước khi đi tiếp]
```

---

## 11. CHECKLIST PHÊ DUYỆT GATE 0 (GATE 0 APPROVAL CHECKLIST)

Các điều kiện cần thiết để người dùng phê duyệt Gate 0 và cho phép bước vào TASK 0:

- [ ] **1. Quy tắc nguồn QIPEDC:** Đã xác nhận Ưu tiên 1 Toàn quốc (không hậu tố) $\rightarrow$ Ưu tiên 2 Miền Nam (hậu tố N); cấm dùng hậu tố B và T làm canonical hoặc calibration.
- [ ] **2. Phân tách hai bộ đánh giá:** Đã xác nhận phân tách Legacy Suite (giữ nguyên dataset `static_v2.npz`, ghi chú D0500B là nguồn Bắc) và Canonical Suite (dùng `D0500N.json` cho H); cam kết không sửa dataset trong đợt này.
- [ ] **3. Thứ tự thực thi chuẩn:** Đã xác nhận thứ tự bắt buộc: `TASK 0 → Q → O → H → M/N → R (Optional) → Final Regression`.
- [ ] **4. Tiêu chuẩn SKIP R:** Đã xác nhận 4 tiêu chí định lượng (Trial Pass $\ge 7/10$, Active Precision $\ge 75\%$, Neutral Ratio $\le 30\%$, triệt tiêu $H \leftrightarrow R$) để quyết định SKIP TASK 5.
- [ ] **5. Kiến trúc & Smoothing Buffer:** Đã xác nhận kiến trúc 3 tầng, logic HUD gán `NO_HAND` khi activeFrames=0, lọc frame conf $\ge 0.70$, bầu chọn khi $\ge 5$ frames, và `cooldown_sec = 1.2s` từ mã nguồn thực tế [`recognizer.py#L128`](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L128).
- [ ] **6. Regression Contract (198 + 8 = 206):** Đã phân tách rõ ràng Hợp đồng Kiểm thử Hồi quy: Nhóm 22 chữ cái tĩnh đạt 198/198 PASS (100%); nhóm FSM Dấu thanh động 8 tests ghi nhận FAIL và đưa vào Backlog chức năng 29 chữ; cam kết không tuyên bố 206/206 PASS và luôn báo cáo đầy đủ cả hai nhóm.
- [ ] **7. Nhãn Candidate:** Đã gắn nhãn `CANDIDATE — NOT APPROVED` cho toàn bộ các đề xuất threshold Q, O, N, R và geometric guards.
- [ ] **8. Hệ thống 7 Gates:** Đã thiết lập hoàn chỉnh 7 Gates từ Gate 0 đến Gate 6 kèm điều kiện vào, tiêu chí PASS/FAIL và yêu cầu phê duyệt người dùng.
- [ ] **9. Bảo lưu SSOT lịch sử:** Đã xác nhận bảo lưu nguyên vẹn [`docs/VSL_8_LETTERS_FIX_PLAN.md`](file:///d:/SIGN_LANGUAGE/docs/VSL_8_LETTERS_FIX_PLAN.md).

---

> [!CAUTION]
> **TRẠNG THÁI HIỆN TẠI: `AWAITING GATE 0 APPROVAL`**
> Mọi hành vi sửa code, tinh chỉnh threshold, thay đổi model weights, sửa dataset hoặc triển khai TASK 0 đều bị **KHÓA TUYỆT ĐỐI** cho đến khi nhận được phê duyệt chính thức bằng văn bản từ Người dùng cho Gate 0.
