# TÀI LIỆU ĐIỀU PHỐI TRUNG TÂM: KẾ HOẠCH FIX 6 CHỮ CÁI VSL (O, Q, M, N, E, H → R)
*(Vietnamese Sign Language Alphabet Recognition Optimization Plan — SSOT)*

---

## 1. MỤC TIÊU TỔNG THỂ & PHẠM VI

Kế hoạch này kế thừa toàn bộ thành quả đã hoàn thành và nghiệm thu dứt điểm của **Chữ P** (Task 1 cũ) và **Chữ X** (Task 2 cũ) đạt chuẩn **`WEBCAM VERIFIED — PASS`**. Mã nguồn nhận diện của P (Rule 15) và X (Rule 18) trong [backend/vision/alphabet_classifier.py](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py) cùng toàn bộ các chữ cái đã ổn định khác được **khóa bất biến làm Baseline chống hồi quy (Zero Regression Baseline)**.

Tài liệu này đóng vai trò là **nguồn thông tin chính thức duy nhất (Single Source of Truth - SSOT)** điều phối việc xử lý **6 chữ cái mục tiêu còn lại**:
1. **TASK 1 — Chữ O**: Khởi động lại từ đầu (`NOT STARTED`) — khắc phục tình trạng nhận diện `NEUTRAL` và nhầm lẫn sang A / S.
2. **TASK 2 — Chữ Q**: Khắc phục tình trạng `NEUTRAL` và nhầm lẫn sang G.
3. **TASK 3 — Chữ M**: Khắc phục tình trạng bộ lọc gating từ chối các dự đoán đúng của MLP.
4. **TASK 4 — Chữ N**: Khắc phục tình trạng bộ lọc gating từ chối các dự đoán đúng của MLP.
5. **TASK 5 — Chữ E**: Khắc phục tình trạng `NEUTRAL` và nhầm lẫn sang C.
6. **TASK 6 — Chữ H → R**: Triệt tiêu nhầm lẫn tồn dư H → R mà không phá vỡ logic V-guard.

### Mục tiêu kỹ thuật then chốt:
- **Độ chính xác**: Nâng cao Frame Accuracy và Trial Accuracy trên cả hai môi trường: Replay (offline dataset/fixtures) và Webcam thực tế (Holdout độc lập) với tiêu chí $\ge 70\%$.
- **Triệt tiêu False Rejection**: Giảm thiểu tối đa tình trạng rơi vào `NEUTRAL` trong pha giữ ký hiệu ổn định (Stable Hold).
- **Kiểm soát False Positive**: Ngăn chặn nhầm lẫn chéo giữa các cặp đối kháng: O vs A/S, Q vs G, M vs N, E vs C, H vs R/V.
- **Bảo toàn tuyệt đối (Zero Regression)**: Không làm giảm độ chính xác của P, X và toàn bộ 29 chữ cái VSL đã được công nhận.

---

## 2. NGUYÊN TẮC TRIỂN KHAI CỐT LÕI

Mọi phiên làm việc bắt buộc tuân thủ nghiêm ngặt các nguyên tắc sau:

1. **Tuân thủ SSOT**: Mọi quyết định kỹ thuật, tiêu chuẩn dữ liệu và đánh giá nghiệm thu đều phải căn cứ vào tài liệu này. Không làm việc dựa trên suy đoán.
2. **Quy trình 7 giai đoạn nghiêm ngặt**: Mỗi TASK phải thực hiện tuần tự qua 7 bước:
   `Source Audit` → `Data Audit` → `Root Cause Analysis` → `Propose Solution (Chờ phê duyệt)` → `Implement` → `Replay Verification` → `Webcam Verification`.
   *Tuyệt đối không nhảy cóc, không sửa code khi chưa audit và chưa có phê duyệt từ User.*
3. **Phân biệt rạch ròi nhãn xác minh**:
   - `[SOURCE VERIFIED]`: Đã đối chiếu và kiểm chứng trực tiếp từ mã nguồn thực tế.
   - `[REPLAY VERIFIED]`: Đã kiểm chứng qua dữ liệu replay/pilot log thực tế.
   - `[WEBCAM VERIFIED]`: Đã kiểm chứng qua thử nghiệm webcam holdout thực tế.
   - `[NOT VERIFIED]`: Giả thuyết hoặc thông tin chưa được kiểm chứng đầy đủ.
4. **Quy ước ưu tiên dữ liệu VID (BẤT DI BẤT DỊCH CHO TOÀN BỘ DỰ ÁN)**:
   Khi tìm dữ liệu mẫu hoặc replay để kiểm tra thuật toán, đánh giá phân bố và xây dựng quy tắc hình học:
   
   | Loại file VID / Dữ liệu | Ý nghĩa vùng miền | Mức ưu tiên | Nguyên tắc áp dụng |
   |---|---|:---:|---|
   | **File không có hậu tố** (ví dụ: `D0501.mp4`, `D0510.mp4`, `D0506`) | **Dữ liệu Toàn quốc (Cả nước)** | **Ưu tiên 1 (Cao nhất)** | Luôn tìm và ưu tiên tuyệt đối số 1 trước tiên trong mọi tình huống. |
   | **File có hậu tố `N`** (ví dụ: `D0509N.mp4`, `D0503N.mp4`) | **Dữ liệu Miền Nam** | **Ưu tiên 2** | Chỉ sử dụng khi KHÔNG CÓ file toàn quốc; bắt buộc ghi rõ trong tài liệu lý do sử dụng. |
   | **File có hậu tố `B`** (ví dụ: `D0509B.mp4`) | **Dữ liệu Miền Bắc** | **Ưu tiên 3 / Đối chiếu** | Chỉ dùng để đối chiếu phương ngữ, không dùng làm đại diện toàn quốc duy nhất. |

   > [!IMPORTANT]
   > **NGUYÊN TẮC BẤT DI BẤT DỊCH (IMMUTABLE INVARIANT)**:
   > Dù trong bất kỳ tình huống nào và dù User/Tester có bất kỳ yêu cầu hay giả định nào, quy tắc thứ tự ưu tiên dữ liệu này là **bất di bất dịch**:
   > 1. Ưu tiên 1 tuyệt đối: Dữ liệu cả nước (không có hậu tố).
   > 2. Ưu tiên 2: Dữ liệu miền Nam (hậu tố `N`), chỉ dùng khi không tồn tại dữ liệu cả nước và phải giải trình minh bạch.
   > 3. Tuyệt đối không được đảo ngược thứ tự ưu tiên, không được bỏ qua file cả nước để chỉ lấy file miền Nam/Bắc làm căn cứ duy nhất.
5. **Kiểm soát phạm vi sửa đổi (Scoped Changes & Minimal Diff)**:
   - Chỉ sửa đổi trong phạm vi hẹp của TASK hiện tại. Không refactor diện rộng, không sửa threshold toàn cục khi chưa đánh giá tác động toàn diện.
6. **Bảo vệ thành quả P & X (Zero Regression Invariants)**:
   - Rule 15 (P-down + North upright) và Rule 18 (X-hook) không được bị ảnh hưởng.
   - Toàn bộ test suite liên quan phải luôn đạt 100% PASS sau mỗi thay đổi.
7. **Đánh giá trên 5 pha động học (Kinematic Phases)**:
   - Mọi phân tích Static Geometric Classifier chỉ được lấy mẫu và đánh giá trên **Pha 3: Stable Hold**. Tuyệt đối không đánh giá độ chính xác static rule trên Entry hoặc Exit.
8. **Nguyên tắc không đánh dấu thủ công (Zero Manual Marking)**:
   - Tester chỉ chọn Target Label; HUD tự động ghi nhận dữ liệu và đối chiếu khách quan với dự đoán thực tế. Không can thiệp gán nhãn thủ công trong lúc ghi nhận.

---

## 3. BASELINE VÀ TÌNH TRẠNG HIỆN TẠI

### 3.1. Baseline 6 Chữ Cái Mục Tiêu (Tổng hợp từ Pilot Baseline & Holdout)

| TASK | Chữ Cái | Nguồn Baseline | Tổng Frame | Đúng | Sai | Neutral | Frame Acc | Trial Acc | Nhầm lẫn chính & Telemetry |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **TASK 1** | **O** | Pilot 2141 & Holdout 004312 | 324 | 1 | 25 | 298 | **0.31%** | 10.0% (1/10) | 25 frame nhầm sang **A** (MLP fallback), 298 frame NEUTRAL. Baseline trước khi hoàn thành audit TASK 1.1–1.3. |
| **TASK 2** | **Q** | Pilot 2141 | 303 | 1 | 2 | 300 | **0.33%** | 10.0% (1/10) | 2 frame nhầm sang **G** (`geometric_rule`), 300 frame NEUTRAL. |
| **TASK 3** | **M** | Pilot 2141 | 323 | 2 | 0 | 321 | **0.62%** | 10.0% (1/10) | MLP raw đúng 99 frame nhưng 97 frame bị bộ lọc gating từ chối. |
| **TASK 4** | **N** | Pilot 2141 | 328 | 4 | 0 | 324 | **1.22%** | 20.0% (2/10) | MLP raw đúng 295 frame nhưng 291 frame bị bộ lọc gating từ chối. |
| **TASK 5** | **E** | Pilot 2141 | 329 | 10 | 11 | 308 | **3.04%** | 20.0% (2/10) | 11 frame nhầm sang **C**, 308 frame NEUTRAL. |
| **TASK 6** | **H → R** | Pilot 08/11 (Webcam) | 90 | 6 | 30 | 54 | **6.67%** | — | 28/30 frame sai bị nhầm thành **R** (93.3% số frame sai). |

### 3.2. Baseline Các Chữ Cái Đã Khóa Hoàn Thành (Zero Regression Reference)
- **Chữ P (Task 1 cũ)**: `WEBCAM VERIFIED — PASS`. Nhánh P-down Rule 15 đạt Frame Acc 99.63% trên Stable Hold, 100% Trial Acc (10/10 PASS), 0 False Positives.
- **Chữ X (Task 2 cũ)**: `WEBCAM VERIFIED — PASS`. Nhánh X-hook Rule 18 đạt Frame Acc 100.0%, 100% Trial Acc (10/10 PASS), 0 False Positives.

---

## 4. DANH SÁCH 6 TASK TRIỂN KHAI

### TASK 1 — FIX O (BẮT ĐẦU LẠI TỪ ĐẦU)
- **Trạng thái**: `NOT STARTED` (Chưa làm gì, bắt đầu bài bản từ Giai đoạn 1).
- **Mục tiêu**: Khắc phục tình trạng rơi vào `NEUTRAL` (false rejection) và triệt tiêu nhầm lẫn O → A / O → S.
- **Trọng tâm điều tra**:
  - Hình thái khép vòm của ngón cái và ngón trỏ (`d_TI`, góc mở ngón, độ cong các ngón `idx_curl`, `mid_curl`).
  - Sự khác biệt về feature hình học giữa O (vòng tròn hở/khép giữa ngón cái và ngón trỏ) với tư thế nắm tay (A, S) hoặc mở vòm chữ C.
  - Phân tích luồng thực thi: Geometric Rule vs MLP Static Head fallback và cơ chế temporal smoothing.
- **Kiểm tra hồi quy bắt buộc**: Chữ O, A, S, C, X.
- **Các bước triển khai cụ thể (Quy trình 7 bước)**:
  1. **TASK 1.1 — Source Audit (Giai đoạn 1) — DONE `[SOURCE VERIFIED]`**: Đã audit `evaluate_rule_o_diagnostics`, Rule 6, các ngưỡng/feature và luồng fallback sang MLP theo báo cáo đã cung cấp.
  2. **TASK 1.2 — Data Audit & Kinematic Partition (Giai đoạn 2) — DONE `[REPLAY VERIFIED]`**: Đã rà soát các fixture và pilot được nêu trong báo cáo; ghi nhận giới hạn pilot 004312 dùng preset 5 giây cũ nên không gán pha hồi tố và không dùng làm holdout Stable Hold. Với mọi lần chọn VID/replay tiếp theo, bắt buộc tìm và dùng file **không hậu tố (toàn quốc) trước**; chỉ khi không tồn tại file toàn quốc mới dùng file hậu tố **`N` (miền Nam)**, đồng thời ghi rõ lý do. Hậu tố `B` chỉ dùng đối chiếu, không làm căn cứ đại diện duy nhất.
  3. **TASK 1.3 — Root Cause Analysis (Giai đoạn 3) — DONE `[ROOT CAUSE IDENTIFIED]`**: Đã ghi nhận nguyên nhân chính: ngưỡng `idx_straight` và `d_TI` overfit fixture studio; fallback gating chặn O khi Rule 6 `NO_MATCH`.
  4. **TASK 1.4 — Propose Candidate Rule & Regression Test Plan (Giai đoạn 4) — NEXT / PENDING**: Đề xuất công thức và bộ ngưỡng Candidate O dựa trên dữ liệu phù hợp, lập kế hoạch hồi quy (bắt buộc O, A, S, C, X và kiểm tra baseline P/X). Tuân thủ ưu tiên VID toàn quốc trước, hậu tố `N` chỉ khi không có dữ liệu toàn quốc và phải giải trình. **Dừng tại đây để báo cáo và chờ User phê duyệt trước khi sửa code.**
  5. **TASK 1.5 — Implementation & Unit Testing (Giai đoạn 5) — PENDING**: Chỉ sau khi Candidate được duyệt; triển khai minimal diff trong `alphabet_classifier.py` và bổ sung unit tests.
  6. **TASK 1.6 — Replay & Full Regression Verification (Giai đoạn 6) — PENDING**: Chạy replay trên fixtures/dataset và regression suite, đo False Positive; chỉ dùng dữ liệu theo thứ tự ưu tiên toàn quốc → miền Nam `N` khi không có toàn quốc. Gán nhãn `[REPLAY VERIFIED]` sau khi có kết quả.
  7. **TASK 1.7 — Independent Webcam Holdout Verification (Giai đoạn 7) — PENDING**: Thu holdout webcam độc lập 10 trials theo protocol 6.0s/5 pha; nghiệm thu Stable Frame Acc ≥70%, Trial Acc ≥7/10 và False Positive 0.0%. Gán nhãn `[WEBCAM VERIFIED]` khi đạt.

### TASK 2 — FIX Q
- **Trạng thái**: `NOT STARTED`.
- **Mục tiêu**: Khắc phục 300 frame neutral và triệt tiêu 2 frame Q → G.
- **Trọng tâm điều tra**:
  - Hướng vector chỉ của ngón tay: ngón trỏ và ngón cái chúc xuống dưới (Q) vs hướng nằm ngang (G).
  - Điều kiện geometric rule của Q trong [alphabet_classifier.py](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py).
  - Phân tích nguyên nhân rơi vào NEUTRAL và cơ chế phân nhánh giữa Q và G/P.
- **Kiểm tra hồi quy bắt buộc**: Chữ Q, G, P, D.

### TASK 3 — FIX M
- **Trạng thái**: `NOT STARTED`.
- **Mục tiêu**: Giải phóng các dự đoán đúng của MLP (99 frame raw correct) đang bị bộ lọc gating từ chối (97 frame bị reject).
- **Trọng tâm điều tra**:
  - Cơ chế gating và lý do M bị ép trả về `UNKNOWN` trong rule classifier.
  - Ràng buộc hình học phân biệt tư thế 3 ngón phủ qua ngón cái (M) vs 2 ngón (N) vs ngón cái kẹp giữa ngón trỏ và giữa (T).
  - Tuyệt đối không hạ bừa threshold mà phải xây dựng ràng buộc hình học đặc trưng an toàn.
- **Kiểm tra hồi quy bắt buộc**: Chữ M, N, A, S, E, T.

### TASK 4 — FIX N
- **Trạng thái**: `NOT STARTED`.
- **Mục tiêu**: Giải phóng 295 frame raw correct của MLP đang bị bộ lọc gating từ chối (291 frame bị reject).
- **Trọng tâm điều tra**:
  - Cơ chế gating của N và mối liên hệ qua lại với TASK 3 (M).
  - Ràng buộc hình học phân biệt tư thế 2 ngón phủ qua ngón cái (N) so với 3 ngón (M).
- **Kiểm tra hồi quy bắt buộc**: Chữ N, M, A, S, T.

### TASK 5 — FIX E
- **Trạng thái**: `NOT STARTED`.
- **Mục tiêu**: Khắc phục 308 frame neutral và triệt tiêu 11 frame E → C.
- **Trọng tâm điều tra**:
  - Phân tích khoảng cách giữa các đầu ngón tay và ngón cái để phân biệt dứt khoát độ mở vòm của C so với các đầu ngón gập tì lên ngón cái của E.
  - Điều kiện geometric rule của E và luồng fallback MLP.
- **Kiểm tra hồi quy bắt buộc**: Chữ E, C, O, A, S.

### TASK 6 — FIX H → R
- **Trạng thái**: `NOT STARTED`.
- **Mục tiêu**: Khắc phục triệt để lỗi 28 frame H → R trên webcam mà không phá vỡ logic V-guard đã thiết lập.
- **Trọng tâm điều tra**:
  - Phép đo bắt chéo ngón tay (`finger_crossing` hoặc tọa độ tương đối giữa đầu ngón trỏ và ngón giữa).
  - Phân biệt ngón đặt song song phẳng (H) vs ngón bắt chéo thực sự (R) từ các góc quay webcam khác nhau.
  - Thứ tự ưu tiên (precedence) và mối quan hệ giữa Rule H và Rule R.
- **Kiểm tra hồi quy bắt buộc**: Chữ H, R, V, U, K.

---

## 5. QUY TRÌNH 7 GIAI ĐOẠN CHUẨN

Mỗi TASK bắt buộc đi qua đầy đủ 7 bước trước khi được nghiệm thu:

```
[GĐ 1: SOURCE AUDIT] ── Kiểm tra code hiện hành, ghi nhận logic và điều kiện hiện tại.
       │
       ▼
[GĐ 2: DATA AUDIT] ──── Khảo sát dữ liệu log, fixture, pilot; phân loại rõ 5 pha động học;
                       áp dụng quy ước ưu tiên VID bất biến (Toàn quốc > Miền Nam N).
       │
       ▼
[GĐ 3: ROOT CAUSE] ──── Xác định nguyên nhân gốc (đối chiếu code vs dữ liệu thực tế).
       │
       ▼
[GĐ 4: PROPOSE] ─────── Đề xuất phương án cụ thể (file, dòng, logic), dự báo rủi ro hồi quy.
       │                *(DỪNG LẠI CHỜ USER PHÊ DUYỆT TRƯỚC KHI IMPLEMENT)*
       ▼
[GĐ 5: IMPLEMENT] ───── Sửa đổi code chính xác theo phương án đã duyệt (minimal diff).
       │
       ▼
[GĐ 6: REPLAY] ──────── Chạy kiểm thử offline trên fixtures, dataset và regression suite.
       │
       ▼
[GĐ 7: WEBCAM] ──────── Kiểm thử holdout độc lập trên webcam (5 pha, trial >= 70%).
```

---

## 6. TIÊU CHÍ NGHIỆM THU (ACCEPTANCE CRITERIA)

Một TASK chỉ được đóng (`PASS`) khi thỏa mãn đồng thời 3 nhóm tiêu chí:

1. **Độ chính xác (Accuracy Criteria)**:
   - Frame Accuracy trên pha **Stable Hold** của webcam holdout đạt $\ge 70\%$.
   - Trial Accuracy đạt tối thiểu **7/10 trials ($\ge 70\%$)** xác nhận đúng ký hiệu qua temporal smoothing.
   - Tỷ lệ `NEUTRAL` trong vùng Stable Hold giảm mạnh về mức cho phép.
2. **Không hồi quy (Zero Regression Criteria)**:
   - **P và X**: Giữ nguyên 100% độ chính xác đã đạt được, không phát sinh False Positive sang P và X.
   - **Bảng chữ cái**: Toàn bộ unit test suite [backend/tests/test_alphabet_classifier.py](file:///d:/SIGN_LANGUAGE/backend/tests/test_alphabet_classifier.py) và các test suite phụ trợ phải đạt **100% PASS**.
   - **Cặp đối kháng**: Tỷ lệ False Positive giữa các cặp đối kháng (O vs A/S, Q vs G, M vs N, E vs C, H vs R/V) phải bằng $0.0\%$ hoặc được triệt tiêu qua temporal filter.
3. **Hiệu năng thời gian thực (Real-time Criteria)**:
   - Độ trễ xử lý classifier < 50ms/frame.
   - Ký hiệu giữ ổn định phải hiển thị liên tục, không bị chập chờn (flicker).

---

## 7. BẢNG TIẾN ĐỘ 6 TASK

*Các trạng thái hợp lệ: `NOT STARTED` | `IN PROGRESS` | `ROOT CAUSE IDENTIFIED` | `IMPLEMENTED` | `REPLAY VERIFIED` | `WEBCAM VERIFIED` | `PASS` | `FAIL` | `BLOCKED`.*

| TASK | Chữ Cái | GĐ 1: Audit | GĐ 2: Data | GĐ 3: Root Cause | GĐ 4: Propose | GĐ 5: Implement | GĐ 6: Replay | GĐ 7: Webcam | Trạng Thái Hiện Tại |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **TASK 1** | **O** | DONE `[SOURCE VERIFIED]` | DONE `[REPLAY VERIFIED]` | DONE `[ROOT CAUSE IDENTIFIED]` | PENDING | PENDING | PENDING | PENDING | `ROOT CAUSE IDENTIFIED` |
| **TASK 2** | **Q** | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | `NOT STARTED` |
| **TASK 3** | **M** | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | `NOT STARTED` |
| **TASK 4** | **N** | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | `NOT STARTED` |
| **TASK 5** | **E** | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | `NOT STARTED` |
| **TASK 6** | **H → R** | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | `NOT STARTED` |

---

## 8. HẠ TẦNG KIỂM THỬ VÀ GHI DỮ LIỆU CHUẨN HÓA

Hệ thống sử dụng **Controlled Webcam Pilot HUD (29 Chữ Cái)** tại [src/components/PilotTestingHUD.jsx](file:///d:/SIGN_LANGUAGE/src/components/PilotTestingHUD.jsx) làm công cụ chuẩn hóa:

### 8.1. Protocol Trial 6.0 giây và 5 Pha Động Học
Mỗi trial kéo dài chuẩn **6.0 giây** được phân chia cố định:
- **Pha 1: PRE_IDLE** $[0.0, 0.5)\text{s}$ (0.5 giây): Tay thả lỏng ở vị trí nghỉ.
- **Pha 2: ENTRY** $[0.5, 1.5)\text{s}$ (1.0 giây): Tay đưa lên tạo hình chữ cái.
- **Pha 3: STABLE_HOLD** $[1.5, 4.5)\text{s}$ (**3.0 giây**): Giữ ổn định tư thế. **Căn cứ duy nhất đánh giá Static Classifier**.
- **Pha 4: EXIT** $[4.5, 5.5)\text{s}$ (1.0 giây): Buông lỏng cử chỉ, hạ tay xuống.
- **Pha 5: POST_IDLE** $[5.5, 6.0]\text{s}$ (0.5 giây): Tay hạ hoàn toàn về vị trí nghỉ.

### 8.2. JSONL frame schema hiện hành: `2.1-kinematic-phase`

Runtime hiện hành (`backend/vision/recognizer.py:PILOT_SCHEMA_VERSION` và `/api/pilot/config`) ghi JSONL frame theo `2.1-kinematic-phase`. Schema 2.1 là phần mở rộng tương thích của `2.0-kinematic`: giữ nguyên các trường và ý nghĩa của 2.0, đồng thời thêm:

| Trường | Schema 2.0 | Schema 2.1 | Ý nghĩa |
|---|---|---|---|
| `phase` | Không có | Có | Pha trial gắn cho frame; null nếu caller không cung cấp. |
| `classifier_config` | Không có | Có | Snapshot ngưỡng runtime của classifier (hiện gồm `O_D_TI_MAX`, `O_IDX_STRAIGHT_MAX`). |

Các trường 2.0 được giữ nguyên gồm metadata frame/session (`timestamp`, `session_id`, `trial_id`, `signer_id`, `frame_idx`, `expected_sign`), tracking/camera (`tracking_status`, kích thước ảnh, handedness và confidence), phiên bản extractor/rule, `image_landmarks_21`, `normalized_landmarks_21`, kết quả nhận diện (`final_sign`, `final_confidence`, `prediction_source`, `confirmed`), `geometric`, `static_head`, gating/rejection và các khối `sync`, `temporal`. `expected_sign` tiếp tục biểu thị nhãn ground truth của trial; các trường hình học và dự đoán giữ ý nghĩa cũ.

Trong các file thực tế đối chiếu, toàn bộ 458 record schema 2.0 có đủ hai mảng landmark dạng 21×3 và cùng bộ khóa feature hình học; cả 529 record schema 2.1 có `phase` và `classifier_config`, giữ nguyên toàn bộ 29 khóa cấp cao của 2.0. Trong 529 record 2.1, 527 có landmark arrays 21×3 và 2 không có landmark hợp lệ (tracking không hợp lệ/mất tay), phù hợp với quy tắc lưu null khi không có dữ liệu hợp lệ; điều này không phải mất landmark hợp lệ. `geometric.features` có cùng bộ khóa trong các record có feature object.

Reader tương thích nên coi `phase` và `classifier_config` là optional khi đọc log 2.0, không tự sinh giá trị cho log cũ; JSONL 2.1 vẫn giữ các tên trường cần thiết cho reader 2.0 dạng permissive. Trong repository, `verify_pilot_smoke_test.py` yêu cầu các trường cốt lõi dùng chung và bỏ qua trường bổ sung, nên tương thích với cả hai schema. Các helper đọc JSONL trong test cũng parse record theo tên trường. Riêng `backend/tools/replay_diagnostic.py` hiện truy cập legacy keys `raw_landmarks`, `geometry`, `mlp`, `ensemble`, khác với cả schema 2.0 và 2.1 (`image_landmarks_21`, `geometric`, `static_head`, `final_sign`); công cụ này không tương thích với hai schema hiện hành và cần adapter riêng nếu muốn replay chúng. Do đó không khẳng định mọi công cụ đọc đều hỗ trợ replay 2.1. Dữ liệu JSONL 2.0 cũ tiếp tục parse nguyên trạng; không có migration hay chỉnh sửa file cũ.

`backend/vision/pilot_logger.py:PILOT_SCHEMA_VERSION = 2.0-kinematic` hiện áp dụng cho **session manifest**, không phải JSONL frame schema; đây là hợp đồng riêng và không được đồng nhất với `recognizer.py`.

### 8.3. Trial-scoped pilot logging và accounting

Logger duy trì hai phạm vi counter:

- **Session cumulative**: các trường lịch sử `queued_count`, `logged_count`, `dropped_queue_full`, `dropped_io_error`, `total_dropped`, `pending_in_queue` vẫn tính mọi record trong logger session. Nhánh `out_of_trial` tách counter cho record không có cặp `session_id` + `trial_id` hợp lệ; các counter legacy không bị thay nghĩa.
- **Trial scoped**: khóa `(session_id, trial_id)` tách `queued_count`, `logged_count`, hai loại dropped, `total_dropped`, `pending_count`, `record_count`, trạng thái `sealed`, `flush_barrier_passed` và `late_record_count`. Counters được cập nhật lúc enqueue/drop và khi worker hoàn tất write/drop; không suy ra từ frontend response.

API contract:

1. `POST /api/pilot/trial/open` đăng ký trial duy nhất trong session hiện hoạt; ID trùng bị từ chối.
2. `POST /api/pilot/trial/flush` seal trial, đặt FIFO flush barrier và trả accounting riêng của trial.
3. `GET /api/pilot/trial/accounting?session_id=...&trial_id=...` đọc lại trạng thái backend và phát hiện record đến muộn.

Một trial chỉ được báo `VERIFIED` khi trial đã tồn tại và seal, có ít nhất một record, barrier hoàn tất thành công, `pending_count=0`, `queued_count=logged_count`, không có dropped record và không có record đến sau seal. Request đã bắt đầu producer trước barrier được tuần tự hóa để enqueue trước barrier; record của trial khác và record ngoài trial không làm thay đổi trial counters. Record gắn trial đã seal vẫn được ghi để bảo toàn dữ liệu nhưng tăng `late_record_count`, khiến trial không còn verified. Session mismatch, trial ID sai, logger unavailable, queue pending, drop và flush failure có mã lỗi riêng; trường hợp thiếu bằng chứng giữ trạng thái UNKNOWN/INVALID.

Flush barrier chỉ xác nhận worker đã xử lý FIFO record đến barrier và không bảo đảm `fsync` hoặc độ bền vật lý trên thiết bị. Không thay đổi chính sách ghi record ngoài trial, không rewrite JSONL cũ; sidecar có trial ID tiếp tục dùng tệp riêng và từ chối overwrite.

---

## 9. NHẬT KÝ THAY ĐỔI (CHANGE LOG)

### [2026-11-08] Khóa Baseline Hoàn Thành Cho Chữ P và Chữ X
- **Chữ P (Task 1 cũ)**: Nghiệm thu thành công qua webcam holdout `pilot_20261108_183703` (Stable Frame Acc: 99.63%, Trial Acc: 10/10 PASS). Tích hợp nhánh P-down vào Rule 15 [backend/vision/alphabet_classifier.py](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py). Trạng thái: `WEBCAM VERIFIED — PASS`.
- **Chữ X (Task 2 cũ)**: Nghiệm thu thành công qua webcam holdout `pilot_20261108_193712` (Stable Frame Acc: 100.0%, Trial Acc: 10/10 PASS). Tích hợp nhánh X-hook vào Rule 18 [backend/vision/alphabet_classifier.py](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py). Trạng thái: `WEBCAM VERIFIED — PASS`.
- **Hạ tầng kiểm thử**: Nâng cấp HUD 29 chữ cái, protocol 6.0s 5 pha; JSONL frame bắt đầu từ schema `2.0-kinematic`.

### [2026-11-09] Mở rộng JSONL frame schema lên `2.1-kinematic-phase`
- Bổ sung `phase` và snapshot `classifier_config`; giữ nguyên các trường/ý nghĩa schema 2.0, gồm raw/normalized 21 landmarks, geometric/static-head diagnostics, sync và temporal.
- Schema version của session manifest trong `pilot_logger.py` vẫn là `2.0-kinematic` và là hợp đồng riêng.
- Không sửa đổi JSONL lịch sử; reader cần chấp nhận hai schema và chỉ đọc các trường mới khi chúng hiện diện.

### [2026-11-10] Thêm trial-scoped accounting cho pilot logger
- Thêm accounting theo `(session_id, trial_id)`, counters riêng cho record ngoài trial, API open/flush/read accounting và seal để phát hiện late record.
- Giữ nguyên counter session cumulative, JSONL lịch sử, schema frame, manifest và thuật toán nhận diện.

### [2026-11-09 03:10] Tái Cấu Trúc SSOT: Khởi Động Kế Hoạch 6 Chữ Cái (O, Q, M, N, E, H → R)


### [2026-11-09] Hoàn Thành TASK 1.1–1.3 — Chữ O
- **TASK 1.1 — Source Audit:** `DONE [SOURCE VERIFIED]` theo kết quả audit đã cung cấp; Rule O hiện hành dùng `idx_straight <= 0.48`, `d_TI <= 0.22`, Outer Gate có cờ `pinky_ext`, và fallback gating không cho nhãn `O` qua khi Rule 6 `NO_MATCH`.
- **TASK 1.2 — Data Audit:** `DONE [REPLAY VERIFIED]` theo báo cáo 1.3. Các căn cứ được nêu gồm fixture `O/A/S/C.json` (N lần lượt 20/22/15/20) và pilot `pilot_pilot_20261109_004312.jsonl` (349 active frames, 10 trials). Pilot 004312 dùng HUD preset 5 giây cũ, không có nhãn pha; vì vậy không gán pha hồi tố và không dùng làm holdout chuẩn.
- **TASK 1.3 — Root Cause Analysis:** `DONE [ROOT CAUSE IDENTIFIED]` theo báo cáo READ-ONLY. Trong 349 active frames: 53 MATCH (15.19%), 296 `NO_MATCH` (84.81%); 167 frame bị loại riêng bởi `idx_straight`, 93 bởi `d_TI`, 3 bởi Outer Gate; 9 frame đồng thời trượt hai điều kiện chính. Báo cáo quy nguyên nhân chính cho hai ngưỡng bị fit theo fixture studio D0506; chiếu 2D là yếu tố khuếch đại, landmark jitter là yếu tố thứ cấp, và `pinky_ext` gây một số false rejection. Khi Rule O không match, fallback gating từ chối nhãn O của MLP nên frame trượt thành `NEUTRAL`.
- **Rủi ro đối kháng:** Báo cáo kiểm tra 57 fixture frames A/S/C và ghi nhận 0 match Rule O; các ngưỡng khoảng cách đầu ngón tới cổ tay và khoảng cách `d_TI` tạo các rào cản độc lập. Đây là kết quả fixture, chưa phải xác minh holdout webcam.
- **Hạn chế/kết luận chưa đủ dữ liệu:** Chưa có ground truth 3D, protocol webcam 6 giây có phân pha, hoặc đa dạng người ký. Các số liệu pilot cũ không được xem là Stable Hold chính thức.
- **Bước tiếp theo:** TASK 1.4 — đề xuất Candidate Rule và kế hoạch kiểm thử hồi quy; dừng chờ User phê duyệt trước khi sửa code. Chưa có thay đổi implementation.
### [2026-11-09 03:56] Cập Nhật Nguyên Tắc Bất Biến: Ưu Tiên Dữ Liệu Cả Nước & Miền Nam
- Bổ sung bảng phân cấp dữ liệu VID và khẳng định nguyên tắc **bất di bất dịch**:
  1. Ưu tiên 1 tuyệt đối: File không có hậu tố (dữ liệu cả nước / toàn quốc).
  2. Ưu tiên 2: File có hậu tố `N` (dữ liệu miền Nam), chỉ dùng khi không có file toàn quốc và phải ghi rõ lý do.
  3. Áp dụng nhất quán cho mọi tình huống và mọi TASK, không thể bị ghi đè.
- Cập nhật quy trình Giai đoạn 2 (Data Audit) để tuân thủ quy ước ưu tiên VID này.



