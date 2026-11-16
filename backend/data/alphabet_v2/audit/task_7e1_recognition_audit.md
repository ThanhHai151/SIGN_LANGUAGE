# BÁO CÁO KIỂM TOÁN TASK 7E.1 — KIỂM TOÁN 29 CHỮ CÁI VÀ LẬP DANH SÁCH CHỮ CHƯA NHẬN DIỆN ĐƯỢC

**Dự án:** VSL Connect AI  
**Nhiệm vụ:** Task 7E.1 — Kiểm toán 29 chữ cái và lập danh sách chữ chưa nhận diện được  
**Thời gian thực hiện:** 31/10/2026  
**Chế độ thực thi:** **READ-ONLY AUDIT** (Tuyệt đối không sửa code, không đổi threshold, không sửa FSM, không huấn luyện model, không đổi checkpoint)  
**Tập tin số liệu máy đọc:** [task_7e1_recognition_audit.json](file:///d:/SIGN_LANGUAGE/backend/data/alphabet_v2/audit/task_7e1_recognition_audit.json)

---

## 1. TỔNG QUAN VÀ SỐ LIỆU ĐIỀU HÀNH (EXECUTIVE METRICS)

Sau khi tích hợp `DiacriticStateMachine` vào luồng camera (Task 7D), chúng tôi đã thực hiện kiểm toán toàn diện độc lập trên toàn bộ mã nguồn nhận diện ([recognizer.py](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py), [diacritic_state_machine.py](file:///d:/SIGN_LANGUAGE/backend/vision/diacritic_state_machine.py), [alphabet_classifier.py](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py), [alphabet_static_head.py](file:///d:/SIGN_LANGUAGE/backend/models/alphabet_static_head.py)), API backend `/api/recognize_landmarks`, frontend React ([CameraSignRecognizer.jsx](file:///d:/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx)), cùng với việc chạy kiểm thử thực nghiệm trên **551 mẫu tĩnh Dataset V2** và **7 chuỗi video động thực tế (QIPEDC)**.

### Tổng hợp phân loại trạng thái 29 chữ cái VSL:
- **Tổng số chữ cái:** 29 chữ.
- **PASS (Đã có bằng chứng nhận diện đúng, ổn định cao $\ge 75\%$):** **11 chữ** (37.93%)
- **UNSTABLE (Nhận diện lúc đúng lúc sai, accuracy thấp $26\% - 71\%$ hoặc tỷ lệ NEUTRAL cao):** **10 chữ** (34.48%)
- **FAIL (Hoàn toàn không nhận diện được, $0\%$ hoặc bị chặn ngưỡng/nhận nhầm nặng):** **8 chữ** (27.59%)
- **NOT_TESTED:** **0 chữ** (100% 29 chữ đều đã được đưa qua luồng chạy thực nghiệm).
- **BLOCKED:** **0 chữ**.

> [!CAUTION]
> **Kết luận trung thực:** Mặc dù hệ thống đã định nghĩa đủ 29 nhãn trong danh mục và vượt qua 74/74 unit/integration tests mô phỏng (mocked), **trên dữ liệu thực tế chỉ có 11/29 chữ (37.9%) hoạt động tin cậy**. Toàn bộ **7 chữ cái động có dấu ($Ă, Â, Ê, Ô, Ơ, Ư, Đ$) đều FAIL (0% thành công)** do các lỗi kiến trúc nghiêm trọng trong cơ chế reset của FSM và sự xung đột của các luật hình học Tầng 1.

---

## 2. BẢNG KẾT QUẢ CHI TIẾT ĐỦ 29 CHỮ CÁI VSL

*Quy ước trạng thái:*
- **PASS:** Tỷ lệ nhận diện đúng trên dữ liệu thực tế $\ge 75\%$, không bị nhầm lẫn nghiêm trọng.
- **UNSTABLE:** Nhận diện được nhưng không ổn định (tỷ lệ đúng $< 72\%$, tỷ lệ rơi về `NEUTRAL` cao từ $30\% - 72\%$, hoặc chập chờn theo góc quay).
- **FAIL:** Tỷ lệ nhận diện đúng xấp xỉ $0\%$, bị chặn bởi ngưỡng quá cao, hoặc luôn bị cướp nhãn sang ký tự khác.

| Chữ | Loại | Tầng nhận diện | Kết quả thực nghiệm | Nhãn nhận nhầm | Nguyên nhân sơ bộ & Dẫn chứng code | Trạng thái |
| :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **A** | Tĩnh / Nền | Tầng 1 (Geom Rule) | 17/64 đúng (26.6%) | `NEUTRAL` (71.9%), `O` (1.5%) | Luật A quá khắt khe: `d_TW>=0.50`, `d_TM>=0.45` ([alphabet_classifier.py#L303](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py#L303)). | **UNSTABLE** |
| **Ă** | Động có dấu | Tầng 3 (Diacritic FSM) | 0/89 đúng (0.0%) | `V` (Token chốt), `NEUTRAL` | Base A bị NEUTRAL 100%. Khi xòe ngón làm dấu, Rule chữ V cướp nhãn ([alphabet_classifier.py#L263](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py#L263)) $\rightarrow$ FSM reset. | **FAIL** |
| **Â** | Động có dấu | Tầng 3 (Diacritic FSM) | 0/108 đúng (0.0%) | `NEUTRAL` (75%), `O` (1 mẫu) | Base A bị 22/23 frames NEUTRAL. FSM không bao giờ vào `BASE_STABLE` ([diacritic_state_machine.py#L402](file:///d:/SIGN_LANGUAGE/backend/vision/diacritic_state_machine.py#L402)). | **FAIL** |
| **B** | Tĩnh thuần | Tầng 1 (Geom Rule) | 13/22 đúng (59.1%) | `NEUTRAL` (36.4%), `O` (1 mẫu) | Ngưỡng duỗi ngón `d_IW>=0.95`, `d_MW>=0.95` trượt khi ngón hơi cong ([alphabet_classifier.py#L243](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py#L243)). | **UNSTABLE** |
| **C** | Tĩnh thuần | Tầng 1 (Geom Rule) | 12/20 đúng (60.0%) | `NEUTRAL` (40.0%) | Ràng buộc đồng thời 5 cự ly $d_{TI}, d_{TM}, d_{IW}, d_{MW}, d_{PW}$ quá nhạy với góc camera ([alphabet_classifier.py#L295](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py#L295)). | **UNSTABLE** |
| **D** | Tĩnh / Nền | Tầng 1 (Geom Rule) | 15/46 đúng (32.6%) | `NEUTRAL` (67.4%) | Điều kiện tiếp xúc ngón cái-giữa $d_{TI}\le 0.40, d_{TM}\le 0.35$ quá hẹp trên 2D ([alphabet_classifier.py#L272](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py#L272)). | **UNSTABLE** |
| **Đ** | Động có dấu | Tầng 3 (Diacritic FSM) | 0/105 đúng (0.0%) | `NEUTRAL` (91.4%) | Base D bị 26/26 frames NEUTRAL trong pha BASE_POSE. FSM không có baseline snapshot. | **FAIL** |
| **E** | Tĩnh / Nền | Tầng 1 (Geom Rule) | 10/35 đúng (28.6%) | `NEUTRAL` (71.4%) | Điều kiện 4 ngón tì ngón cái $d_{TM}\le 0.35, d_{TW}\le 0.45$ trượt khi ngón cái nằm sâu ([alphabet_classifier.py#L318](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py#L318)). | **UNSTABLE** |
| **Ê** | Động có dấu | Tầng 3 (Diacritic FSM) | 0/76 đúng (0.0%) | `NEUTRAL` (73.7%) | Base E bị 20/20 frames NEUTRAL trong pha BASE_POSE. FSM không có baseline snapshot. | **FAIL** |
| **G** | Tĩnh bổ sung | Tầng 2 (Static Head) | 26/29 đúng (89.7%) | `NEUTRAL` (10.3%) | Hoạt động tốt qua MLP 72D ([alphabet_static_head_v1.pt](file:///d:/SIGN_LANGUAGE/backend/models/alphabet_static_head_v1.pt)). Ngưỡng 0.70 hợp lý. | **PASS** |
| **H** | Tĩnh bổ sung | Tầng 2 (Static Head) | 16/18 đúng (88.9%) | `NEUTRAL` (11.1%) | Hoạt động tốt qua MLP 72D ([alphabet_static_head_v1.pt](file:///d:/SIGN_LANGUAGE/backend/models/alphabet_static_head_v1.pt)). Ngưỡng 0.70 hợp lý. | **PASS** |
| **I** | Tĩnh thuần | Tầng 1 (Geom Rule) | 14/18 đúng (77.8%) | `NEUTRAL` (22.2%) | Nhận diện I tốt ([alphabet_classifier.py#L251](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py#L251)), nhưng cướp nhãn của Ư và Ơ khi vẩy ngón út. | **PASS** |
| **K** | Tĩnh bổ sung | Tầng 2 (Static Head) | 14/17 đúng (82.4%) | `NEUTRAL` (17.6%) | Hoạt động tốt qua MLP 72D với ngưỡng 0.75 ([recognizer.py#L114](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L114)). | **PASS** |
| **L** | Tĩnh thuần | Tầng 1 (Geom Rule) | 11/16 đúng (68.8%) | `NEUTRAL` (31.2%) | Ràng buộc góc vuông $ang_{TI}\ge 70^\circ$ trượt khi ngón cái không mở tối đa ([alphabet_classifier.py#L276](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py#L276)). | **UNSTABLE** |
| **M** | Tĩnh bổ sung | Tầng 2 (Static Head) | 17/19 đúng (89.5%) | `NEUTRAL` (10.5%) | Hoạt động tốt qua MLP 72D với ngưỡng 0.70. | **PASS** |
| **N** | Tĩnh bổ sung | Tầng 2 (Static Head) | 15/21 đúng (71.4%) | `NEUTRAL` (23.8%), `M` (4.8%) | Ranh giới phân lớp hẹp giữa 2 ngón (N) và 3 ngón (M) qua hình chiếu 2D. | **UNSTABLE** |
| **O** | Tĩnh / Nền | Tầng 1 (Geom Rule) | 37/48 đúng (77.1%) | `NEUTRAL` (18.8%), `E` (4.1%) | Nhận diện tĩnh O tốt ([alphabet_classifier.py#L288](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py#L288)). Dễ biến dạng khi di chuyển. | **PASS** |
| **Ô** | Động có dấu | Tầng 3 (Diacritic FSM) | 0/84 đúng (0.0%) | `O` (Token chốt), `NEUTRAL` | Base O nhận diện tốt, nhưng khi nâng tay vẽ nón thì frame 34 rơi về NEUTRAL $\rightarrow$ [recognizer.py#L858](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L858) reset FSM ngay lập tức! | **FAIL** |
| **Ơ** | Động có dấu | Tầng 3 (Diacritic FSM) | 0/89 đúng (0.0%) | `O` (Token chốt), `I`, `NEUTRAL` | Base O nhận diện tốt, nhưng khi vươn ngón út làm râu thì bị Rule chữ I cướp $\rightarrow$ [recognizer.py#L858](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L858) reset FSM! | **FAIL** |
| **P** | Tĩnh bổ sung | Tầng 2 (Static Head) | 16/20 đúng (80.0%) | `NEUTRAL` (20.0%) | Hoạt động tốt qua MLP 72D với ngưỡng 0.70. | **PASS** |
| **Q** | Tĩnh bổ sung | Tầng 2 (Static Head) | 5/19 đúng (26.3%) | `NEUTRAL` (73.7%) | Ngưỡng `req_threshold = 0.85` ([recognizer.py#L114](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L114)) quá cao, loại bỏ 73.7% mẫu Q thật. | **FAIL** |
| **R** | Tĩnh bổ sung | Tầng 2 (Static Head) | 11/14 đúng (78.6%) | `NEUTRAL` (21.4%) | Hoạt động tốt qua MLP 72D với ngưỡng 0.75. | **PASS** |
| **S** | Tĩnh thuần | Tầng 1 (Geom Rule) | 11/15 đúng (73.3%) | `NEUTRAL` (26.7%) | Nhận diện ổn định qua thế nắm tay ngón cái vắt ngang ([alphabet_classifier.py#L310](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py#L310)). | **PASS** |
| **T** | Tĩnh thuần | Tầng 1 (Geom Rule) | 11/20 đúng (55.0%) | `NEUTRAL` (45.0%) | Khó đo 2D ngón cái kẹp trỏ/giữa $d_{TI}\le 0.40, d_{TM}\ge 0.20$ ([alphabet_classifier.py#L282](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py#L282)). | **UNSTABLE** |
| **U** | Tĩnh / Nền | Tầng 2 (Static Head) | 27/32 đúng (84.4%) | `NEUTRAL` (15.6%) | Nhận diện U tĩnh tốt qua MLP 72D với ngưỡng 0.70. | **PASS** |
| **Ư** | Động có dấu | Tầng 3 (Diacritic FSM) | 0/118 đúng (0.0%) | `U` (Token chốt), `I`, `NEUTRAL` | Base U nhận diện tốt, nhưng khi bật ngón út làm râu thì bị Rule chữ I cướp $\rightarrow$ [recognizer.py#L858](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L858) reset FSM! | **FAIL** |
| **V** | Tĩnh thuần | Tầng 1 (Geom Rule) | 13/17 đúng (76.5%) | `NEUTRAL` (23.5%) | Nhận diện V tốt ([alphabet_classifier.py#L263](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py#L263)), nhưng bắt nhầm Ă khi người dùng xòe 2 ngón. | **PASS** |
| **X** | Tĩnh bổ sung | Tầng 2 (Static Head) | 17/25 đúng (68.0%) | `NEUTRAL` (28.0%), `T` (4.0%) | Ngón trỏ cong móc câu dễ nhầm với T và Q trên 2D. | **UNSTABLE** |
| **Y** | Tĩnh thuần | Tầng 1 (Geom Rule) | 11/16 đúng (68.8%) | `NEUTRAL` (31.2%) | Thế Shaka đòi hỏi cả ngón cái và út duỗi thẳng ([alphabet_classifier.py#L256](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py#L256)). | **UNSTABLE** |

---

## 3. KIỂM TRA ĐỘC LẬP CHUYÊN SÂU 7 CHỮ ĐỘNG CÓ DẤU ($Ă, Â, Ê, Ô, Ơ, Ư, Đ$)

Đối chiếu trực tiếp 8 câu hỏi kiểm toán bắt buộc trên 7 chuỗi video QIPEDC thực tế:

| Câu hỏi kiểm toán | Ă | Â | Ê | Ô | Ơ | Ư | Đ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Nhận diện được chữ nền trong BASE_POSE?** | **KHÔNG** (0/19) | **KHÔNG** (0/23) | **KHÔNG** (0/20) | **CÓ** (18/19) | **CÓ** (7/9) | **CÓ** (13/14) | **KHÔNG** (0/26) |
| **2. FSM chuyển sang BASE_STABLE?** | **KHÔNG** | **KHÔNG** | **KHÔNG** | **CÓ** (Fr 19) | **CÓ** (Fr 39) | **CÓ** (Fr 23) | **KHÔNG** |
| **3. FSM phát hiện đúng dấu động học?** | **KHÔNG** | **KHÔNG** | **KHÔNG** | **KHÔNG** | **KHÔNG** | **KHÔNG** | **KHÔNG** |
| **4. Có đạt DIACRITIC_CONFIRMED?** | **KHÔNG** | **KHÔNG** | **KHÔNG** | **KHÔNG** | **KHÔNG** | **KHÔNG** | **KHÔNG** |
| **5. Có bị ENTRY/EXIT Guard chặn nhầm?** | Không | Không | Không | Không | Không | Không | Không |
| **6. Bị mất landmarks giữa chuyển động?** | Không (Chất lượng MediaPipe trích xuất tốt) | Không | Không | Không | Không | Không | Không |
| **7. Bị Rule tĩnh Tầng 1 cướp nhãn?** | **CÓ (Bị V cướp)** | Không | Không | Không | **CÓ (Bị I cướp)** | **CÓ (Bị I cướp)** | Không |
| **8. Kết quả cuối cùng camera hiển thị** | **Token V** | **NEUTRAL** | **NEUTRAL** | **Token O** | **Token O** | **Token U** | **NEUTRAL** |
| **KẾT LUẬN CUỐI CÙNG** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** |

---

## 4. PHÂN TÍCH NGUYÊN NHÂN LỖI CỐT LÕI (ROOT CAUSE ANALYSIS)

Dựa trên việc truy vết mã nguồn và dữ liệu thực tế, các lỗi thất bại và không ổn định tập trung vào **4 nguyên nhân kỹ thuật mang tính hệ thống**:

### 4.1. Lỗi thiết kế luồng Reset FSM (Pipeline Reset Flaw — Nghiêm trọng nhất)
- **Tập tin:** [backend/vision/recognizer.py](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py) (dòng 815–860).
- **Đoạn mã gây lỗi:**
  ```python
  if base_sign in BASE_GROUPS and p_prim and len(p_prim) >= 21:
      fsm_res = fsm.update(...)
  ...
  else:
      # Chữ tĩnh ngoài 5 nhóm nền (B, C, G, H...) hoặc không có mốc hợp lệ: reset FSM
      fsm.reset()  # <--- CRITICAL BUG!
  ```
- **Cơ chế phát sinh lỗi:**
  - FSM được thiết kế để phân tích **chuỗi chuyển động vẩy dấu**.
  - Nhưng trong lúc người dùng thực hiện động tác vẩy dấu (nâng tay vẽ nón trong Ô, mở râu trong Ư/Ơ), bàn tay **bắt buộc phải thay đổi vị trí hoặc hình dạng**, khiến bộ phân loại tĩnh (Geometric Rules) **ngay lập tức mất dấu chữ nền** và trả về `NEUTRAL` (hoặc nhầm sang chữ khác).
  - Vì `base_sign` lúc này là `NEUTRAL` (không nằm trong `BASE_GROUPS`), nhánh `else` gọi `fsm.reset()`.
  - **Hậu quả:** Toàn bộ trạng thái `BASE_STABLE` và `baseline` snapshot bị xóa sạch **đúng vào frame đầu tiên khi người dùng bắt đầu vẽ dấu**! FSM không bao giờ có cơ hội so sánh frame hiện tại với snapshot nền.
  - *Bằng chứng thực nghiệm:* Với chữ **Ô** (Video D0507B), FSM đạt `BASE_STABLE` ở frame 19-33. Nhưng vừa bước sang frame 34 của pha `DIACRITIC_MOTION`, `base_sign` rơi về `NEUTRAL`, `fsm.reset()` lập tức bị gọi, khiến toàn bộ pha vẽ nón (frame 35–54) bị bỏ qua hoàn toàn.

### 4.2. Xung đột phân cấp nhận diện (Rule Hijacking của Tầng 1)
- **Tập tin:** [backend/vision/alphabet_classifier.py](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py) (Rule chữ I dòng 251, Rule chữ V dòng 263).
- **Cơ chế phát sinh lỗi:**
  - Tầng 1 (12 luật hình học) có độ ưu tiên cao nhất và luôn chạy trước.
  - Khi người dùng làm cử chỉ chữ **Ă** (từ dáng A xòe 2 ngón trỏ và giữa): Rule chữ V lập tức nhận diện là chữ `V` (confidence 0.95).
  - Khi người dùng làm cử chỉ chữ **Ư** hoặc **Ơ** (từ dáng U/O bật ngón út làm râu): Rule chữ I lập tức nhận diện ngón út duỗi là chữ `I` (confidence 0.95).
  - **Hậu quả:** Ký tự `base_sign` bị đổi thành `V` hoặc `I`. Vì `V` và `I` không nằm trong `BASE_GROUPS`, `fsm.reset()` lại bị kích hoạt, đồng thời camera chốt nhầm token sang `V` hoặc `I`!

### 4.3. Luật hình học chữ nền quá khắt khe (Over-constrained Geometric Rules)
- **Tập tin:** [backend/vision/alphabet_classifier.py](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py) (A dòng 303, D dòng 272, E dòng 318).
- **Cơ chế phát sinh lỗi:**
  - Các luật hình học tĩnh cho chữ nền $A, D, E$ đòi hỏi các ngưỡng tỷ lệ khoảng cách ngón tay quá hẹp trên không gian 2D, không tính đến sự đa dạng giải phẫu của bàn tay người thật.
  - Tỷ lệ rơi vào `NEUTRAL` trên dữ liệu thực tế: **A (71.9%), D (67.4%), E (71.4%)**.
  - **Hậu quả:** Trong các video thực tế của $Ă, Â, Ê, Đ$, trong suốt toàn bộ pha `BASE_POSE` (khi người ký đang giữ nguyên tư thế nền chuẩn), hệ thống **100% trả về `NEUTRAL`**.
  - Vì chữ nền không nhận diện được, FSM bị "đói dữ liệu", không bao giờ tích lũy đủ 4 frame ổn định để chuyển sang `BASE_STABLE`.

### 4.4. Ngưỡng Rejection Threshold chữ Q quá cao trong Static Head
- **Tập tin:** [backend/vision/recognizer.py](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py) dòng 114:
  ```python
  DEFAULT_SUPPLEMENTAL_THRESHOLDS = { ... "Q": 0.85 ... }
  ```
- **Cơ chế phát sinh lỗi:**
  - Trong Task 6C, để tránh việc ngón cái khép tạo ảo giác 2D bị nhầm sang chữ X, ngưỡng của Q đã bị nâng lên tới `0.85`.
  - Trên tập dữ liệu thực tế 19 mẫu Q, chỉ có 5 mẫu (26.3%) đạt confidence $\ge 0.85$. **73.7% số mẫu Q thật bị từ chối và gán thành `NEUTRAL`**.

---

## 5. THỨ TỰ ƯU TIÊN XỬ LÝ (RECOMMENDED FIX PRIORITY)

Để đưa hệ thống đạt khả năng nhận diện thực chất đủ 29 chữ cái qua camera, các nhiệm vụ kỹ thuật cần được xử lý theo lộ trình ưu tiên sau:

```
[Ưu tiên 1: Sửa Pipeline Reset Flaw]
              │
              ▼
[Ưu tiên 2: Khắc phục Rule Hijacking Tầng 1 (V & I)]
              │
              ▼
[Ưu tiên 3: Nới lỏng Geometric Rules & Mở Fallback cho Static Head (A, D, E)]
              │
              ▼
[Ưu tiên 4: Hiệu chuẩn ngưỡng tin cậy chữ Q]
              │
              ▼
[Ưu tiên 5: Thu thập dữ liệu kiểm thử camera đa người ký (Multi-Signer)]
```

1. **Ưu tiên 1 (Cực kỳ khẩn cấp): Sửa lỗi Pipeline Reset Flaw trong `recognizer.py`:**
   - Thay đổi logic reset: Khi FSM đã đạt `STATE_BASE_STABLE` hoặc `STATE_DIACRITIC_CANDIDATE`, nếu frame tiếp theo trả về `NEUTRAL`, **tuyệt đối KHÔNG gọi `fsm.reset()`**.
   - Cho phép FSM tiếp tục sống trong khoảng thời gian tìm kiếm dấu `max_diacritic_search_sec = 0.65s` để phân tích chuyển động trong `motion_buffer`. Chỉ reset khi hết thời gian search window hoặc mất mốc bàn tay quá 5 frames.
2. **Ưu tiên 2 (Khẩn cấp): Khắc phục hiện tượng Rule Hijacking của Tầng 1:**
   - Khi FSM đang ở trạng thái `STATE_BASE_STABLE` của nhóm A: tạm thời ức chế Rule chữ V.
   - Khi FSM đang ở trạng thái `STATE_BASE_STABLE` của nhóm U hoặc O: tạm thời ức chế Rule chữ I để cho phép ngón út đóng vai trò là nét vẽ dấu râu của Ư và Ơ.
3. **Ưu tiên 3: Nới lỏng luật hình học và mở Fallback Static Head cho A, D, E:**
   - Mở rộng tập `ALLOWED_SUPPLEMENTAL_LETTERS` trong [recognizer.py](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py) để cho phép Static Head cứu các trường hợp A, D, E bị Geometric Rule bỏ sót. (Mô hình `alphabet_static_head_v1.pt` vốn đã được huấn luyện sẵn cho 22 lớp bao gồm cả A, D, E).
4. **Ưu tiên 4: Hiệu chuẩn lại ngưỡng tin cậy của chữ Q:**
   - Hạ ngưỡng `req_threshold` của Q từ 0.85 xuống 0.70, kết hợp kiểm tra độ cong của ngón trỏ (`index_curl`) để chống nhầm sang X một cách tự nhiên bằng hình học.
5. **Ưu tiên 5: Triển khai kiểm thử camera đa người ký (Multi-Signer Validation):**
   - Áp dụng quy trình theo kế hoạch [multisigner_evaluation_plan.md](file:///d:/SIGN_LANGUAGE/backend/data/alphabet_v2/audit/multisigner_evaluation_plan.md) trên 3–5 người ký mới để tinh chỉnh adaptive thresholds cho góc camera và tốc độ vẩy tay thực tế.

---

## 6. KẾT LUẬN VÀ TRẢ LỜI CÁC CÂU HỎI TRỌNG TÂM

1. **Hiện tại có bao nhiêu chữ thực sự đã được xác nhận nhận diện đúng qua camera?**
   - **Chỉ có 11 chữ cái** (chiếm 37.9%) có bằng chứng hoạt động chính xác và tin cậy trên dữ liệu thực tế:
     - 4 chữ qua Geometric Rules: **I, O, S, V**.
     - 7 chữ qua Static Head MLP: **G, H, K, M, P, R, U**.
2. **Có bao nhiêu chữ chưa nhận diện được?**
   - **Có 18 chữ cái** (chiếm 62.1%) chưa đạt yêu cầu:
     - **8 chữ FAIL hoàn toàn (0% hoặc bị chặn/cướp nhãn):** **Q, Ă, Â, Ê, Ô, Ơ, Ư, Đ**.
     - **10 chữ UNSTABLE (chập chờn, tỷ lệ NEUTRAL cao):** **A, B, C, D, E, L, N, T, X, Y**.
3. **Những chữ nào cần sửa trước?**
   - Cần sửa trước **7 chữ cái động có dấu ($Ă, Â, Ê, Ô, Ơ, Ư, Đ$)** và **3 chữ nền ($A, D, E$)**, vì đây là cụm chữ cái có sự liên kết hữu cơ trực tiếp qua FSM. Sửa lỗi Reset Flaw và Rule Hijacking sẽ ngay lập tức giải phóng khả năng nhận diện cho cả 7 chữ động này.
4. **Lỗi tập trung ở tầng nào?**
   - Lỗi tập trung chủ yếu ở **Tầng 3 (Logic tích hợp FSM vào pipeline trong recognizer.py)** và **Tầng 1 (Luật hình học quá chặt chẽ và xung đột phân cấp cướp nhãn của Tầng 3)**. Checkpoint Static Head MLP ở Tầng 2 hoạt động tương đối tốt (7/10 chữ đạt PASS).
5. **Cần bổ sung dữ liệu kiểm thử gì?**
   - Cần bổ sung dữ liệu kiểm thử camera đa người ký (Multi-Signer Live Frames), bao gồm video nhiều người ký thực hiện các chữ động với tốc độ và biên độ vẩy tay tự nhiên khác nhau, nhằm chuẩn hóa các ngưỡng $vy, \Delta y$ và bán kính mở ngón của FSM.

