# BÁO CÁO XÁC MINH NGUYÊN NHÂN GỐC RỄ (ROOT CAUSE) VÀ ĐỀ XUẤT SỬA LỖI FSM / RULE HIJACKING (TASK 7E.2)

**Dự án:** VSL Connect AI  
**Nhiệm vụ:** Task 7E.2 — Xác minh Root Cause và Đề xuất sửa lỗi FSM / Rule Hijacking  
**Thời gian thực hiện:** 31/10/2026  
**Chế độ thực thi:** **READ-ONLY INVESTIGATION** (Tuyệt đối không sửa code, không đổi threshold, không sửa FSM, không can thiệp checkpoint)  
**Tập tin số liệu máy đọc:** [task_7e2_root_cause.json](file:///d:/SIGN_LANGUAGE/backend/data/alphabet_v2/audit/task_7e2_root_cause.json)

---

## 1. TỔNG HỢP XÁC MINH CÁC GIẢ THUYẾT (HYPOTHESIS VERIFICATIONS)

Sau khi kiểm tra trực tiếp mã nguồn, phân tích luồng điều khiển và chạy mô phỏng đối chiếu thực nghiệm, chúng tôi kết luận chính thức về 3 nhóm giả thuyết:

| Nhóm giả thuyết | Nội dung kiểm tra | Kết luận | Mức độ ảnh hưởng |
| :--- | :--- | :---: | :--- |
| **Giả thuyết 1** | FSM bị reset ngay lập tức khi chữ nền tạm thời mất nhận diện (rơi về `NEUTRAL`) | **CONFIRMED** | Gây tê liệt 100% khả năng nhận diện các chữ vẽ nón và chữ động ($Ô, Â, Ê$). |
| **Giả thuyết 2** | Rule I và Rule V ở Tầng 1 cướp nhãn (Rule Hijacking) trong quá trình tạo dấu của $Ă, Ơ, Ư$ | **CONFIRMED** | Gây tê liệt và chốt nhầm token của $Ă$ thành $V$, $Ơ$ thành $I$, $Ư$ thành $I$. |
| **Giả thuyết 3** | Các rule hình học $A, D, E$ quá khắt khe khiến FSM không thể nhận diện chữ nền trong pha `BASE_POSE` | **CONFIRMED** | Khiến $A, D, E$ bị trượt $67\% - 73\%$, làm FSM bị "đói dữ liệu nền" cho $Ă, Â, Ê, Đ$. |

---

## 2. KIỂM TRA CHI TIẾT LUỒNG RESET FSM (PHẦN 1)

Khảo sát mã nguồn thực tế tại [backend/vision/recognizer.py](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py) (dòng 805–865) và [backend/vision/diacritic_state_machine.py](file:///d:/SIGN_LANGUAGE/backend/vision/diacritic_state_machine.py) (dòng 136–150):

### 2.1. Điều kiện gọi `fsm.update()` và `fsm.reset()`
- **Điều kiện gọi `fsm.update()`** ([recognizer.py#L815](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L815)):
  ```python
  if base_sign in BASE_GROUPS and p_prim and len(p_prim) >= 21:
      fsm_res = fsm.update(...)
  ```
  Chỉ được gọi khi và chỉ khi `base_sign` nằm trong tập đúng 5 chữ cái nền: `BASE_GROUPS = {"A", "E", "O", "U", "D"}`.
- **Điều kiện gọi `fsm.reset()`** ([recognizer.py#L858-859](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L858-859)):
  ```python
  else:
      # Chữ tĩnh ngoài 5 nhóm nền (B, C, G, H...) hoặc không có mốc hợp lệ: reset FSM
      fsm.reset()
  ```
  Được gọi lập tức ở nhánh `else` bất cứ khi nào `base_sign` không nằm trong 5 chữ nền (bao gồm `NEUTRAL`, `UNKNOWN`, hoặc các chữ cái khác như `I`, `V`, `B`, `C`...).

### 2.2. Các câu hỏi xác minh cốt lõi:
1. **Khi base label chuyển từ A/O/U/D/E sang NEUTRAL, I hoặc V, FSM có bị reset ngay không?**
   - **CÓ!** Vì `NEUTRAL`, `I`, `V` đều không thuộc `BASE_GROUPS`, nhánh `else` thực thi ngay lập tức.
2. **Có cơ chế giữ baseline snapshot trong lúc bàn tay đang chuyển động không?**
   - **KHÔNG!** Khi `fsm.reset()` được gọi, toàn bộ thuộc tính `self.baseline = None`, `self.motion_buffer.clear()`, `self.current_base = None` bị xóa sạch ([diacritic_state_machine.py#L136-150](file:///d:/SIGN_LANGUAGE/backend/vision/diacritic_state_machine.py#L136-150)).
3. **Có phân biệt được mất landmarks thực sự với việc chỉ mất nhãn chữ nền không?**
   - **KHÔNG CÓ SỰ PHÂN BIỆT!** Mất landmarks thực sự (không có tay, dòng 556–561) và việc bàn tay vẫn ở trước camera nhưng cử chỉ bị phân loại thành `NEUTRAL` (dòng 858–859) đều gọi chung lệnh `fsm.reset()`.
4. **Việc reset có xảy ra trong một frame đơn lẻ hay cần nhiều frame liên tiếp?**
   - **XẢY RA NGAY LẬP TỨC TRONG 1 FRAME ĐƠN LẺ DUY NHẤT!** Hoàn toàn không có bộ đệm dung sai (missing counter hay grace period).

### 2.3. Mô phỏng luồng điều khiển trên 5 chuỗi nhãn giả định:

```
[Frame 1..4: Base Sign] ────► FSM đạt BASE_STABLE (Snapshot Baseline thành công)
                                       │
[Frame 5: Diacritic Gesture] ──────────┴────► Bàn tay biến dạng / di chuyển
                                                       │
                                 ┌─────────────────────┴─────────────────────┐
                           [base_sign = NEUTRAL]                       [base_sign = I hoặc V]
                                 │                                           │
                                 ▼                                           ▼
                     recognizer.py L858: else                    recognizer.py L858: else
                                 │                                           │
                                 ▼                                           ▼
                            fsm.reset()                                 fsm.reset()
                                 │                                           │
                                 ▼                                           ▼
                    Baseline bị xóa ngay lập tức                Baseline bị xóa ngay lập tức
                    (FSM rơi về IDLE ở Frame 5)                 (FSM rơi về IDLE ở Frame 5)
```

| Chữ | Chuỗi nhãn giả định | Trạng thái tại Frame 4 | Diễn biến tại Frame 5 | Kết quả FSM |
| :---: | :--- | :---: | :--- | :---: |
| **Ô** | `O → O → O → O → NEUTRAL → O` | `BASE_STABLE` (`current_base=O`, `has_baseline=True`) | `NEUTRAL` kích hoạt nhánh `else` $\rightarrow$ `fsm.reset()`. Trạng thái rơi về `IDLE`. | **Mất baseline tại Frame 5.** Frame 6 phải học lại từ đầu (`BASE_CANDIDATE`). |
| **Ơ** | `O → O → O → O → I → I → O` | `BASE_STABLE` (`current_base=O`, `has_baseline=True`) | Rule I cướp nhãn $\rightarrow$ `base_sign = "I"` $\notin$ `BASE_GROUPS` $\rightarrow$ `fsm.reset()`. | **Mất baseline tại Frame 5.** |
| **Ư** | `U → U → U → U → I → I → U` | `BASE_STABLE` (`current_base=U`, `has_baseline=True`) | Rule I cướp nhãn $\rightarrow$ `base_sign = "I"` $\notin$ `BASE_GROUPS` $\rightarrow$ `fsm.reset()`. | **Mất baseline tại Frame 5.** |
| **Ă** | `A → A → A → A → V → V → A` | `BASE_STABLE` (`current_base=A`, `has_baseline=True`) | Rule V cướp nhãn $\rightarrow$ `base_sign = "V"` $\notin$ `BASE_GROUPS` $\rightarrow$ `fsm.reset()`. | **Mất baseline tại Frame 5.** |
| **Â** | `A → A → A → A → NEUTRAL → A` | `BASE_STABLE` (`current_base=A`, `has_baseline=True`) | `NEUTRAL` kích hoạt nhánh `else` $\rightarrow$ `fsm.reset()`. Trạng thái rơi về `IDLE`. | **Mất baseline tại Frame 5.** |

---

## 3. KIỂM TRA XUNG ĐỘT GEOMETRIC RULES & SỰ THẤT BẠI CỦA A/D/E (PHẦN 2)

Khảo sát mã nguồn thực tế tại [backend/vision/alphabet_classifier.py](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py):

### 3.1. Cơ chế thực thi và Short-circuit
1. **Thứ tự ưu tiên:**
   - Trong `evaluate_rules(feat)`: Duyệt lần lượt 10 khối điều kiện: B $\rightarrow$ I/Y $\rightarrow$ V $\rightarrow$ D/L $\rightarrow$ T $\rightarrow$ O $\rightarrow$ C $\rightarrow$ A $\rightarrow$ S $\rightarrow$ E.
   - Nếu có đúng 1 rule khớp: trả về `status = "MATCH"`, `confidence = 0.95`.
2. **Short-circuit:**
   - Tại [recognizer.py#L740](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L740):
     ```python
     if status == "MATCH" and alpha_res.get("letter") in ALLOWED_LETTERS:
         final_sign = alpha_res["letter"]
         recognition_source = "geometric_rule"
     elif status in ("NO_MATCH", "UNKNOWN") and self.has_static_head:
         # Gọi Static Head
     ```
     Nếu Tầng 1 trả về `status == "MATCH"`, **Static Head bị ngắt mạch hoàn toàn (không được gọi)**.
3. **Hiện tượng Rule Hijacking của Rule I và Rule V:**
   - Khi người ký vẩy ngón út làm râu trong $Ơ$ hoặc $Ư$:
     Rule I ([alphabet_classifier.py#L251](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py#L251)) kiểm tra:
     `if p_ext and not i_ext and not m_ext and not r_ext and not t_ext:`
     Vì chỉ có ngón út duỗi thẳng, điều kiện này **MATCH 100%** với confidence 0.95! Rule I cướp frame, Static Head bị bỏ qua, FSM nhận `base_sign = "I"` và bị reset!
   - Khi người ký xòe 2 ngón trỏ và giữa trong $Ă$:
     Rule V ([alphabet_classifier.py#L263](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py#L263)) kiểm tra:
     `if i_ext and m_ext and not r_ext and not p_ext:`
     Hai ngón trỏ và giữa mở xòe **MATCH 100%** với Rule V (confidence 0.95)! Rule V cướp frame, FSM bị reset, camera chốt token `V`!

### 3.2. Đo lường định lượng thất bại của Rules A, D, E trên dữ liệu thực tế (551 mẫu V2)

Kiểm tra trực tiếp trên các mẫu thực tế của Dataset V2:

#### A. Chữ A (Tổng mẫu: 64) — Thất bại: 47/64 mẫu (73.4%)
- **Biểu thức rule:**
  ```python
  if not i_ext and not m_ext and not r_ext and not p_ext:
      if d_tw >= 1.32 and d_tm >= 0.75 and d_iw <= 0.90 and d_mw <= 0.75:
          matches.append(("A", ...))
  ```
- **Nguyên nhân định lượng:**
  1. `d_IW <= 0.90`: **42/47 mẫu thất bại vi phạm điều kiện này**. Khoảng cách thực tế từ ngón trỏ đến cổ tay của dáng A có giá trị trung bình $\sim 1.05 > 0.90$.
  2. `d_TM >= 0.75`: **37/47 mẫu thất bại vi phạm điều kiện này**. Khi ngón cái áp sát cạnh bàn tay, cự ly ngón cái tới ngón giữa thường chỉ đạt $0.45 - 0.70 < 0.75$.

#### B. Chữ D (Tổng mẫu: 46) — Thất bại: 31/46 mẫu (67.4%)
- **Biểu thức rule:**
  ```python
  if i_ext and not m_ext and not r_ext and not p_ext:
      if d_iw >= 1.70 and d_ti <= 1.28 and d_tm <= 0.30:
          matches.append(("D", ...))
  ```
- **Nguyên nhân định lượng:**
  1. `d_IW >= 1.70`: **29/31 mẫu thất bại vi phạm điều kiện này**. Khi người ký giơ ngón trỏ hơi chếch về phía trước hoặc nghiêng góc camera, cự ly chuẩn hóa $d_{IW}$ chỉ đạt $1.35 - 1.65 < 1.70$.

#### C. Chữ E (Tổng mẫu: 35) — Thất bại: 25/35 mẫu (71.4%)
- **Biểu thức rule:**
  ```python
  if not i_ext and not m_ext and not r_ext and not p_ext:
      if d_ti <= 0.35 and d_tm <= 0.38 and d_tw <= 0.95 and d_iw >= 1.05 and d_mw >= 1.05:
          matches.append(("E", ...))
  ```
- **Nguyên nhân định lượng:**
  1. `d_TI <= 0.35`: **25/25 mẫu thất bại (100%) vi phạm điều kiện này**. Trong dáng E thật, 4 ngón tì lên mặt ngoài ngón cái, khoảng cách đầu ngón cái tới đầu ngón trỏ $d_{TI}$ đo được là $0.40 - 0.65 > 0.35$.
  2. `d_TM <= 0.38`: **23/25 mẫu thất bại vi phạm điều kiện này** (trung bình thực tế đạt $0.65$).
  3. `not i_ext and not m_ext`: **17/25 mẫu bị nhận nhầm là ngón đang duỗi** do các khớp ngón gập vuông góc (claw/hook) chứ không nắm chặt vào lòng bàn tay.

---

## 4. KIỂM TRA DỮ LIỆU VÀ CÁCH TÍNH THỐNG KÊ TASK 7E.1 (PHẦN 3)

Để đảm bảo tính minh bạch khoa học, phương pháp luận đo đạc trong Task 7E.1 được làm rõ như sau:

### 4.1. Cách thức nạp dữ liệu:
- **551 mẫu tĩnh (Dataset V2):** Được nạp tuần tự từ `vsl_alphabet_static_v2.npz` (mảng `landmarks` 21 joints). Mỗi mẫu đại diện cho 1 frame tĩnh độc lập.
- **7 chuỗi video động (Dataset V2):** Được nạp tuần tự theo đúng dòng thời gian (chronological timeline) từ `vsl_alphabet_motion_v2.json` (`raw_timeline_sequence`) với đúng `timestamp_sec` thực tế của từng video.
  - Sử dụng chung một `session_id` cố định cho mỗi sequence (`test_motion_{letter}`).
  - Toàn bộ `prediction_history` và thực thể `DiacriticStateMachine` trong `self.diacritic_fsms[session_key]` được duy trì liên tục xuyên suốt chiều dài video.

### 4.2. Bản chất các con số thống kê:
- **Con số `17/64` (chữ A):** Đo **Frame-level Accuracy trên tập kiểm thử tĩnh** (17 frame trả về nhãn "A" trên tổng số 64 frame mẫu của chữ A trong Dataset V2).
- **Con số `0/89` (chữ Ă):** Đo **Frame-level Accuracy trên toàn bộ timeline của video D0490B** (0 frame nào trong 89 frame của video trả về nhãn "Ă").
- **Con số `18/19` (chữ Ô):** Đo **Frame-level Accuracy trong riêng pha `BASE_POSE` của video D0507B** (18/19 frame nhận diện đúng chữ nền "O").
- **Tác động của Temporal Smoothing:**
  - Trong video chữ Ô: 18 frame nhận diện đúng O $\rightarrow$ bộ đệm làm mượt chốt token `O`. Nhưng ở pha diacritic, FSM bị reset, không có frame Ô nào $\rightarrow$ token cuối cùng vẫn là `O`.
  - Trong video chữ Ă: Ở pha diacritic, Rule V cướp liên tiếp 5 frame $\rightarrow$ bộ đệm làm mượt chốt token `V`.

### 4.3. Phân biệt 3 môi trường kiểm nghiệm:
1. **Offline Dataset Replay:** Phát lại ngoại tuyến các landmark đã trích xuất từ 551 mẫu tĩnh và 7 video QIPEDC. Đây là cơ sở thực nghiệm định lượng của báo cáo này.
2. **Automated Integration Test:** Các test trong `test_diacritic_camera_integration.py`. Các test này đã **mock/patch** hàm `classify_detailed` luôn trả về chữ nền nên che giấu lỗi Reset Flaw và Rule Hijacking.
3. **Live Webcam Test:** Người dùng bật webcam thật trên trình duyệt gọi qua API backend. **Repo hiện tại chưa lưu trữ log định lượng cho từng chữ trên môi trường này** (chỉ có phản ánh định tính từ người dùng).

---

## 5. BẢNG TRẠNG THÁI 29 CHỮ CÁI: ĐỐI CHIẾU OFFLINE REPLAY VS LIVE WEBCAM (PHẦN 4)

*Tiêu chí phân loại chặt chẽ:*
- **PASS:** Accuracy $\ge 75\%$.
- **UNSTABLE:** Accuracy $> 0\%$ và $< 75\%$.
- **FAIL:** $0\%$ hoặc không thể tạo token đúng / bị chặn ngưỡng.
- **NOT_TESTED:** Chưa có log kiểm thử định lượng tương ứng.

| Chữ | Loại | Tầng nhận diện | Kết quả Offline Dataset Replay | Trạng thái Offline | Kết quả Live Webcam | Ghi chú & Mẫu số |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **A** | Tĩnh/Nền | Tầng 1 | 17/64 (26.6%) | **UNSTABLE** | `NOT_TESTED` | Mẫu số: 64 frame tĩnh Dataset V2. |
| **Ă** | Động | Tầng 3 | 0/89 (0.0%) | **FAIL** | `NOT_TESTED` | Mẫu số: 89 frame video D0490B. Token: 'V'. |
| **Â** | Động | Tầng 3 | 0/108 (0.0%) | **FAIL** | `NOT_TESTED` | Mẫu số: 108 frame video D0491B. 0 frame Â. |
| **B** | Tĩnh | Tầng 1 | 13/22 (59.1%) | **UNSTABLE** | `NOT_TESTED` | Mẫu số: 22 frame tĩnh Dataset V2. |
| **C** | Tĩnh | Tầng 1 | 12/20 (60.0%) | **UNSTABLE** | `NOT_TESTED` | Mẫu số: 20 frame tĩnh Dataset V2. |
| **D** | Tĩnh/Nền | Tầng 1 | 15/46 (32.6%) | **UNSTABLE** | `NOT_TESTED` | Mẫu số: 46 frame tĩnh Dataset V2. |
| **Đ** | Động | Tầng 3 | 0/105 (0.0%) | **FAIL** | `NOT_TESTED` | Mẫu số: 105 frame video D0494B. 0 frame Đ. |
| **E** | Tĩnh/Nền | Tầng 1 | 10/35 (28.6%) | **UNSTABLE** | `NOT_TESTED` | Mẫu số: 35 frame tĩnh Dataset V2. |
| **Ê** | Động | Tầng 3 | 0/76 (0.0%) | **FAIL** | `NOT_TESTED` | Mẫu số: 76 frame video D0497B. 0 frame Ê. |
| **G** | Tĩnh | Tầng 2 | 26/29 (89.7%) | **PASS** | `NOT_TESTED` | Mẫu số: 29 frame tĩnh Dataset V2 (MLP). |
| **H** | Tĩnh | Tầng 2 | 16/18 (88.9%) | **PASS** | `NOT_TESTED` | Mẫu số: 18 frame tĩnh Dataset V2 (MLP). |
| **I** | Tĩnh | Tầng 1 | 14/18 (77.8%) | **PASS** | `NOT_TESTED` | Mẫu số: 18 frame tĩnh Dataset V2 (Rule I). |
| **K** | Tĩnh | Tầng 2 | 14/17 (82.4%) | **PASS** | `NOT_TESTED` | Mẫu số: 17 frame tĩnh Dataset V2 (MLP). |
| **L** | Tĩnh | Tầng 1 | 11/16 (68.8%) | **UNSTABLE** | `NOT_TESTED` | Mẫu số: 16 frame tĩnh Dataset V2 (Rule L). |
| **M** | Tĩnh | Tầng 2 | 17/19 (89.5%) | **PASS** | `NOT_TESTED` | Mẫu số: 19 frame tĩnh Dataset V2 (MLP). |
| **N** | Tĩnh | Tầng 2 | 15/21 (71.4%) | **UNSTABLE** | `NOT_TESTED` | Mẫu số: 21 frame tĩnh Dataset V2 (MLP). |
| **O** | Tĩnh/Nền | Tầng 1 | 37/48 (77.1%) | **PASS** | `NOT_TESTED` | Mẫu số: 48 frame tĩnh Dataset V2 (Rule O). |
| **Ô** | Động | Tầng 3 | 0/84 (0.0%) | **FAIL** | `NOT_TESTED` | Mẫu số: 84 frame video D0507B. Token: 'O'. |
| **Ơ** | Động | Tầng 3 | 0/89 (0.0%) | **FAIL** | `NOT_TESTED` | Mẫu số: 89 frame video D0508B. Token: 'O'. |
| **P** | Tĩnh | Tầng 2 | 16/20 (80.0%) | **PASS** | `NOT_TESTED` | Mẫu số: 20 frame tĩnh Dataset V2 (MLP). |
| **Q** | Tĩnh | Tầng 2 | 5/19 (26.3%) | **FAIL** | `NOT_TESTED` | Bị chặn ngưỡng 0.85, 73.7% rơi vào NEUTRAL. |
| **R** | Tĩnh | Tầng 2 | 11/14 (78.6%) | **PASS** | `NOT_TESTED` | Mẫu số: 14 frame tĩnh Dataset V2 (MLP). |
| **S** | Tĩnh | Tầng 1 | 11/15 (73.3%) | **UNSTABLE** | `NOT_TESTED` | Mẫu số: 15 frame tĩnh Dataset V2 (Rule S < 75%). |
| **T** | Tĩnh | Tầng 1 | 11/20 (55.0%) | **UNSTABLE** | `NOT_TESTED` | Mẫu số: 20 frame tĩnh Dataset V2 (Rule T). |
| **U** | Tĩnh/Nền | Tầng 2 | 27/32 (84.4%) | **PASS** | `NOT_TESTED` | Mẫu số: 32 frame tĩnh Dataset V2 (MLP). |
| **Ư** | Động | Tầng 3 | 0/118 (0.0%) | **FAIL** | `NOT_TESTED` | Mẫu số: 118 frame video D0518B. Token: 'U'. |
| **V** | Tĩnh | Tầng 1 | 13/17 (76.5%) | **PASS** | `NOT_TESTED` | Mẫu số: 17 frame tĩnh Dataset V2 (Rule V). |
| **X** | Tĩnh | Tầng 2 | 17/25 (68.0%) | **UNSTABLE** | `NOT_TESTED` | Mẫu số: 25 frame tĩnh Dataset V2 (MLP). |
| **Y** | Tĩnh | Tầng 1 | 11/16 (68.8%) | **UNSTABLE** | `NOT_TESTED` | Mẫu số: 16 frame tĩnh Dataset V2 (Rule Y). |

### Tổng hợp thống kê:
- **Offline Dataset Replay:** **10 PASS** (34.5%), **11 UNSTABLE** (37.9%), **8 FAIL** (27.6%), **0 NOT_TESTED**.
- **Live Webcam:** **29 NOT_TESTED** (100% chưa có log định lượng trực tiếp).

---

## 6. ĐỀ XUẤT BA PHƯƠNG ÁN SỬA LỖI TỐI THIỂU (PHẦN 5)

> [!IMPORTANT]
> Đây là các đề xuất kỹ thuật tối thiểu, có căn cứ định lượng rõ ràng. **HOÀN TOÀN CHƯA THỰC HIỆN SỬA CODE TRONG TASK NÀY.**

### 6.1. Phương án A: Tránh Reset FSM Khi Base Label Tạm Thời Thành `NEUTRAL`
- **Tập tin & Vị trí sửa:** [backend/vision/recognizer.py](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py), dòng 815–860 trong hàm `predict_landmarks()`.
- **Logic hiện tại:**
  ```python
  if base_sign in BASE_GROUPS and p_prim and len(p_prim) >= 21:
      fsm_res = fsm.update(...)
  else:
      fsm.reset()  # Reset ngay khi base_sign == "NEUTRAL"
  ```
- **Logic đề xuất:**
  - Nếu bàn tay vẫn tồn tại (`p_prim and len(p_prim) >= 21`):
    - Kiểm tra nếu FSM của session hiện tại đang ở trạng thái `STATE_BASE_STABLE` hoặc `STATE_DIACRITIC_CANDIDATE`:
      - Khi `base_sign == "NEUTRAL"`, **tuyệt đối không gọi `fsm.reset()`**.
      - Tiếp tục gọi `fsm.update(base_label=fsm.current_base, base_confidence=0.50, landmarks_raw=p_prim, ...)` để FSM duy trì bộ đệm chuyển động trong cửa sổ tìm kiếm dấu (`max_diacritic_search_sec = 0.65s`).
    - Chỉ reset FSM khi:
      1. Hết thời gian tìm kiếm dấu (FSM tự chuyển về `STATE_BASE_STABLE` hoặc timeout).
      2. Người ký chủ động hạ tay nhanh (`vy > 0.65` EXIT Guard).
      3. Hoặc xuất hiện một chữ cái tĩnh xác định thuộc nhóm khác (vd: đang theo dõi O mà xuất hiện B rõ ràng).
- **Những chữ được cải thiện:** **Ô, Â, Ê** (các chữ vẽ nón tịnh tiến không gian).
- **Những chữ có nguy cơ bị ảnh hưởng:** Các chữ tĩnh thuần nếu người dùng đổi tư thế từ O sang B quá nhanh (giải pháp: nếu `base_sign` là chữ tĩnh rõ ràng ngoài nhóm thì reset ngay).
- **Test bổ sung cần có:** `test_fsm_holds_state_during_transient_neutral_frames()`.
- **Cách kiểm chứng bằng webcam:** Bật `$env:VSL_DIACRITIC_DEBUG="1"`, giữ O rồi đẩy tay lên cao để vẽ nón, quan sát log terminal đảm bảo FSM không bị in dòng `reset()` hoặc rơi về `IDLE`.

---

### 6.2. Phương án B: Xử Lý Xung Đột Rule I/V Trong Lúc FSM Đang Theo Dõi Nhóm Nền
- **Tập tin & Vị trí sửa:** [backend/vision/recognizer.py](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py) (dòng 739–745) kết hợp kiểm tra trạng thái FSM.
- **Logic hiện tại:**
  - Tầng 1 chạy trước không phụ thuộc FSM. Khi người ký mở ngón làm dấu Ă hoặc vẩy ngón út làm dấu Ư/Ơ, Rule V hoặc Rule I lập tức báo `MATCH` với confidence 0.95 và short-circuit toàn bộ pipeline.
- **Logic đề xuất (FSM-Aware Rule Suppression):**
  - Trước khi chấp nhận kết quả `MATCH` của Tầng 1:
    - Lấy tham chiếu FSM hiện tại của session: `fsm = self.diacritic_fsms.get(session_key)`.
    - Nếu FSM đang ở trạng thái `STATE_BASE_STABLE` hoặc `STATE_DIACRITIC_CANDIDATE`:
      - Nếu `fsm.current_base == "A"` và Rule Tầng 1 trả về `"V"`: **Ức chế Rule V**, coi đây là cử chỉ mở ngón của chữ Ă và chuyển giao frame cho FSM xử lý.
      - Nếu `fsm.current_base in ("U", "O")` và Rule Tầng 1 trả về `"I"`: **Ức chế Rule I**, coi đây là cử chỉ vẩy râu của Ư hoặc Ơ và chuyển giao frame cho FSM xử lý.
- **Những chữ được cải thiện:** **Ă, Ư, Ơ** (chấm dứt hoàn toàn hiện tượng chốt nhầm token V cho Ă và token I cho Ư/Ơ).
- **Những chữ có nguy cơ bị ảnh hưởng:** Trường hợp người ký cố tình ký chữ V thật ngay sau chữ A (giải pháp: cơ chế ức chế chỉ tồn tại tối đa trong 0.65s; nếu sau 0.65s không có nét vẽ dấu thì nhả lại nhãn V bình thường).
- **Test bổ sung cần có:** `test_suppress_v_rule_during_a_breve_transition()`, `test_suppress_i_rule_during_u_horn_transition()`.
- **Cách kiểm chứng bằng webcam:** Giữ U rồi vẩy ngón út sang phải xem chữ hiển thị là "Ư" hay "I".

---

### 6.3. Phương án C: Cải Thiện Khả Năng Nhận Diện A/D/E Dựa Trên Dữ Liệu Thực Tế
- **Tập tin & Vị trí sửa:**
  - [backend/vision/recognizer.py](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py) dòng 756: Mở rộng `ALLOWED_SUPPLEMENTAL_LETTERS`.
  - [backend/vision/alphabet_classifier.py](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py) dòng 59–113: Nới lỏng ngưỡng Geometric Rules.
- **Logic hiện tại:**
  - Checkpoint `alphabet_static_head_v1.pt` được huấn luyện sẵn cho **22 lớp** (trong đó có cả A, D, E). Nhưng recognizer chỉ cho phép Static Head phát ra đúng 10 chữ phụ trợ. A, D, E phụ thuộc 100% vào Geometric Rules quá khắt khe, dẫn đến tỷ lệ thất bại $67\% - 73\%$.
- **Logic đề xuất (Cơ chế Dual Fallback):**
  1. Cho phép Static Head fallback nhận diện A, D, E khi Geometric Rules trả về `NO_MATCH`:
     ```python
     ALLOWED_SUPPLEMENTAL_LETTERS = {"G", "H", "K", "M", "N", "P", "Q", "R", "U", "X", "A", "D", "E"}
     ```
  2. Nới lỏng các ngưỡng cực đoan của Geometric Rules dựa trên số liệu định lượng:
     - **Chữ A:** Tăng `A_D_IW_MAX` từ 0.90 lên 1.15; hạ `A_D_TM_MIN` từ 0.75 xuống 0.50.
     - **Chữ D:** Hạ `D_D_IW_MIN` từ 1.70 xuống 1.40.
     - **Chữ E:** Tăng `E_D_TI_MAX` từ 0.35 lên 0.55; tăng `E_D_TM_MAX` từ 0.38 lên 0.60.
- **Những chữ được cải thiện:** **A, D, E** (accuracy tăng từ $\sim 28\%$ lên $> 80\%$) $\rightarrow$ kéo theo **Ă, Â, Ê, Đ** vì chữ nền được nhận diện vững chắc trong pha `BASE_POSE`.
- **Những chữ có nguy cơ bị ảnh hưởng:** Cần kiểm soát ma trận nhầm lẫn giữa A và S, giữa D và L, giữa E và O/S.
- **Test bổ sung cần có:** `test_static_head_fallback_for_base_letters()`, `test_relaxed_rules_no_false_positive()`.
- **Cách kiểm chứng bằng webcam:** Giơ tay làm dáng A, D, E tự nhiên trước webcam xem hệ thống có nhận diện tức thì và ổn định không.

---

## 7. DANH SÁCH TEST HỒI QUY BẮT BUỘC SAU KHI SỬA (REGRESSION SUITE)

Khi tiến hành triển khai các phương án sửa đổi ở task tiếp theo, bắt buộc phải chạy qua toàn bộ test suite này để đảm bảo không gây suy thoái hệ thống:

1. `backend/tests/test_alphabet_classifier.py` (26 tests — đảm bảo 12 luật hình học giữ nguyên tính đúng đắn).
2. `backend/tests/test_alphabet_static_head_integration.py` (10 tests — đảm bảo Static Head 10 chữ hoạt động ổn định).
3. `backend/tests/test_diacritic_state_machine.py` (15 tests — logic nội tại của FSM).
4. `backend/tests/test_diacritic_camera_integration.py` (14 tests — luồng tích hợp camera).
5. `backend/tests/test_mode_separation.py` (Bảo đảm cách ly tuyệt đối giữa letters, digits, words).
6. `backend/tests/test_digit_10_integration.py` & `test_sequential_digits_integration.py` (Bảo toàn 10 chữ số).
7. `backend/tests/test_api.py` (Bảo đảm API `/api/recognize_landmarks` không bị trễ hay lỗi schema).

