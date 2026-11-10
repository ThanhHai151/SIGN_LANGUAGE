# BÁO CÁO TÍCH HỢP TASK 7D — TÍCH HỢP DIACRITIC STATE MACHINE VÀO LUỒNG CAMERA

**Dự án:** VSL Connect AI  
**Nhiệm vụ:** Task 7D — Tích hợp Diacritic State Machine vào luồng nhận diện camera VSLRecognizer  
**Thời gian hoàn thành:** 31/10/2026  
**Trạng thái:** HOÀN THÀNH — 74/74 TESTS PASS (100%)  

---

## 1. TỔNG QUAN VÀ MỤC TIÊU ĐẠT ĐƯỢC

Mục tiêu cốt lõi của **Task 7D** là hoàn thiện khả năng nhận diện đầy đủ **29 chữ cái tiếng Việt** ($A \dots Y$, bao gồm 7 chữ cái có dấu: $Ă, Â, Ê, Ô, Ơ, Ư, Đ$) trong luồng nhận diện chữ cái trực tiếp (`mode == "letters"`) của [recognizer.py](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py), thông qua việc tích hợp máy trạng thái động học [DiacriticStateMachine](file:///d:/SIGN_LANGUAGE/backend/vision/diacritic_state_machine.py).

### Các nguyên tắc kỹ thuật đã thực thi tuyệt đối:
1. **Phân cấp 3 tầng ưu tiên nghiêm ngặt (Strict 3-Tier Hierarchy):**
   - **Tầng 1 (Geometric Rules):** 12 luật hình học chạy trước nhất ($A, B, C, D, E, I, L, O, S, T, V, Y$).
   - **Tầng 2 (Static Head):** Bổ sung 10 chữ cái tĩnh ($G, H, K, M, N, P, Q, R, U, X$) thông qua checkpoint [alphabet_static_head_v1.pt](file:///d:/SIGN_LANGUAGE/backend/models/alphabet_static_head_v1.pt).
   - **Tầng 3 (Diacritic FSM):** Kích hoạt **chỉ khi** chữ cái tĩnh nhận diện được thuộc 5 nhóm chữ cái nền:
     - Nhóm $A \rightarrow \{A, Ă, Â\}$
     - Nhóm $E \rightarrow \{E, Ê\}$
     - Nhóm $O \rightarrow \{O, Ô, Ơ\}$
     - Nhóm $U \rightarrow \{U, Ư\}$
     - Nhóm $D \rightarrow \{D, Đ\}$
2. **Zero-Lag Passthrough cho 17 chữ cái thuần tĩnh:**
   - 17 chữ cái không có dấu ($\{B, C, G, H, I, K, L, M, N, P, Q, R, S, T, V, X, Y\}$) hoàn toàn **không đi qua FSM**, trả về kết quả ngay lập tức với độ trễ $< 5\text{ ms}$.
3. **Cách ly phiên độc lập (Session Isolation):**
   - Mỗi luồng camera/phiên kết nối sở hữu một FSM instance độc lập trong từ điển `self.diacritic_fsms[session_key]`, ngăn ngừa rò rỉ trạng thái động học giữa nhiều người ký.
4. **Không can thiệp checkpoint hoặc kiến trúc cũ:**
   - Không huấn luyện lại [weights.pt](file:///d:/SIGN_LANGUAGE/backend/models/weights.pt) (84 classes).
   - Không thay đổi [sign_sstcn_attention_model.pth](file:///d:/SIGN_LANGUAGE/backend/models/weights/sign_sstcn_attention_model.pth) (102 classes).
   - Không thay đổi [alphabet_static_head_v1.pt](file:///d:/SIGN_LANGUAGE/backend/models/alphabet_static_head_v1.pt).
   - Không can thiệp các chế độ `digits`, `words`, `sentence`, `letters_numbers`.

---

## 2. KIẾN TRÚC VÀ LUỒNG DỮ LIỆU TÍCH HỢP

```
                   MediaPipe Hand Landmarks (21 joints)
                                   │
                                   ▼
                       [ mode == "letters" ? ]
                                   │
                 ┌─────────────────┴─────────────────┐
               [YES]                                [NO]
                 │                                   │
                 ▼                                   ▼
       ┌──────────────────┐               [ Chế độ Words / Digits ]
       │ Tầng 1: 12 Luật  │               (Giữ nguyên 100% logic cũ)
       │ Hình học tĩnh    │
       └─────────┬────────┘
                 │
       [ Có kết quả không? ]
         ├── YES ──┐
         │         │
         └── NO    │
             │     │
             ▼     │
       ┌──────────────────┐
       │ Tầng 2: VSL      │
       │ Static Head (MLP)│
       └─────────┬────────┘
                 │
                 ├── [ Chữ cái tĩnh: base_sign, base_conf ]
                 │
                 ▼
       [ Thuộc 5 nhóm nền: A, E, O, U, D ? ]
                 │
         ├── NO ───► [ 17 chữ thuần tĩnh: B, C, G, H, I, K, L, M, N, P, Q, R, S, T, V, X, Y ]
         │           └── Trả về ngay lập tức (Zero-lag < 5ms)
         │
         └── YES ──► [ Tầng 3: DiacriticStateMachine(session_id) ]
                     │
                     ├── Frame Landmark Dispatcher (update baseline / track kinematics)
                     ├── Kinematic Threshold Evaluator (dy, dx, wrist-to-tip ratio)
                     │
                     ├── [ FSM State: IDLE / BASE_STABLE / COOLDOWN ]
                     │   └── Giữ nguyên nhãn chữ nền (A, E, O, U, D)
                     │
                     └── [ FSM State: DIACRITIC_CONFIRMED ]
                         ├── Phân giải ký tự: Ă, Â, Ê, Ô, Ơ, Ư, Đ
                         ├── Nội suy final_confidence bảo thủ (0.75 - 0.92)
                         └── Prime bộ đệm làm mượt prediction_history (tránh độ trễ nuốt chữ)
```

---

## 3. CÁC NỘI DUNG ĐÃ SỬA ĐỔI TRONG `recognizer.py`

### 3.1. Bổ sung bảng chữ cái tiếng Việt đầy đủ vào danh mục cho phép
Trước đây, danh mục `ALLOWED_LETTERS` chỉ lấy 27 lớp đầu của `sign_model.CLASSES` (tiếng Anh + Đ), khiến các ký tự có dấu $Ă, Â, Ê, Ô, Ơ, Ư$ bị chuyển thành `NEUTRAL`.
Đã bổ sung bộ hằng số đầy đủ:
```python
VIETNAMESE_ALPHABET_29 = [
    "A", "Ă", "Â", "B", "C", "D", "Đ", "E", "Ê", "G", "H", "I",
    "K", "L", "M", "N", "O", "Ô", "Ơ", "P", "Q", "R", "S", "T",
    "U", "Ư", "V", "X", "Y"
]
ALLOWED_LETTERS = set(CLASSES[:27]) | set(VIETNAMESE_ALPHABET_29)
```

### 3.2. Quản lý FSM theo phiên và cơ chế dọn dẹp (Session Isolation)
- Khởi tạo từ điển FSM trong `VSLRecognizer.__init__`:
  ```python
  self.diacritic_fsms = {}  # session_key -> DiacriticStateMachine
  self.debug_diacritic = os.environ.get("VSL_DIACRITIC_DEBUG", "0") == "1"
  ```
- Phương thức `reset_diacritic_fsm(session_id=None)` cho phép reset chính xác phiên làm việc khi người dùng đổi phiên hoặc reset camera.
- Khi người dùng đổi `mode` (rời khỏi `letters`), FSM của phiên tự động được reset về `STATE_IDLE`.
- Khi không phát hiện bàn tay (`no_hand`), FSM tự động được gọi `update(None, None)` để chuyển về `STATE_IDLE` và xóa baseline động học.

### 3.3. Tối ưu bộ đệm làm mượt (Smoothing Buffer Priming)
Khi FSM chuyển sang trạng thái `STATE_DIACRITIC_CONFIRMED`, nếu vẫn để nguyên lịch sử 10-15 frame của chữ cái nền cũ trong `prediction_history`, thuật toán đa số phiếu `collections.Counter` sẽ gây trễ từ 5–8 frame trước khi nhãn có dấu được hiển thị.
Giải pháp: Nạp trước `self.smoothing_window // 2` frame `(final_sign, final_conf)` vào đầu hàng đợi giúp token có dấu xuất hiện ngay tức thì mà không làm mất tính ổn định.

---

## 4. KẾT QUẢ KIỂM THỬ TỔNG THỂ (74/74 TESTS PASS)

Tất cả các bài kiểm tra từ cấp độ luật hình học, Static Head, đến FSM và tích hợp camera đều vượt qua 100%:

| Bộ kiểm thử (Test Suite) | Số lượng | Kết quả | Mô tả phạm vi |
| :--- | :---: | :---: | :--- |
| [test_alphabet_classifier.py](file:///d:/SIGN_LANGUAGE/backend/tests/test_alphabet_classifier.py) | 26 | **PASS (26/26)** | 12 luật hình học tĩnh (A, B, C, D, E, I, L, O, S, T, V, Y) |
| [test_alphabet_recognition_integration.py](file:///d:/SIGN_LANGUAGE/backend/tests/test_alphabet_recognition_integration.py) | 9 | **PASS (9/9)** | Tích hợp nhận diện chữ cái tĩnh cơ bản qua recognizer |
| [test_alphabet_static_head_integration.py](file:///d:/SIGN_LANGUAGE/backend/tests/test_alphabet_static_head_integration.py) | 10 | **PASS (10/10)** | Tích hợp 10 chữ cái tĩnh bổ sung (G, H, K, M, N, P, Q, R, U, X) |
| [test_diacritic_state_machine.py](file:///d:/SIGN_LANGUAGE/backend/tests/test_diacritic_state_machine.py) | 15 | **PASS (15/15)** | Unit test logic chuyển trạng thái FSM và động học 7 chữ |
| [test_diacritic_camera_integration.py](file:///d:/SIGN_LANGUAGE/backend/tests/test_diacritic_camera_integration.py) | 14 | **PASS (14/14)** | Tích hợp camera: 5 nhóm nền, 17 chữ passthrough, session isolation |
| **TỔNG CỘNG** | **74** | **PASS (74/74)** | **Thời gian chạy: 3.26 giây** |

### Chi tiết 14 test case tích hợp camera mới ([test_diacritic_camera_integration.py](file:///d:/SIGN_LANGUAGE/backend/tests/test_diacritic_camera_integration.py)):
1. `test_a_to_a_breve_transition`: A chuyển động vòng cung lên tạo chữ **Ă**.
2. `test_a_to_a_circumflex_transition`: A nhấc cao dứt khoát tạo chữ **Â**.
3. `test_e_to_e_circumflex_transition`: E nhấc cao theo trục Y tạo chữ **Ê**.
4. `test_o_to_o_circumflex_transition`: O nhấc cao theo trục Y tạo chữ **Ô**.
5. `test_o_to_o_horn_transition`: O giật móc ngang sang phải tạo chữ **Ơ**.
6. `test_u_to_u_horn_transition`: U giật móc ngang sang phải tạo chữ **Ư**.
7. `test_d_to_d_stroke_transition`: D gạch ngang sang trái tạo chữ **Đ**.
8. `test_pure_static_letters_passthrough_zero_lag`: 17 chữ thuần tĩnh bỏ qua FSM, không bị trễ.
9. `test_geometric_rules_preserved`: Đảm bảo 12 luật hình học giữ nguyên 100% độ chính xác.
10. `test_static_head_letters_preserved`: Đảm bảo Static Head 10 chữ tĩnh hoạt động chuẩn xác.
11. `test_lost_landmarks_safe_neutral`: Mất bàn tay chuyển an toàn về NEUTRAL, reset FSM.
12. `test_session_isolation_and_reset`: Kiểm tra tính độc lập tuyệt đối giữa 2 session đồng thời.
13. `test_other_modes_unaffected`: Kiểm tra chế độ `digits` và `words` không bị tác động.
14. `test_cross_group_transition_blocked`: Ngăn chặn tuyệt đối chuyển nhóm sai quy tắc (VD: $A \rightarrow Ô$).

---

## 5. XÁC MINH TOÀN VẸN CÁC CHECKPOINT (SHA-256)

Toàn bộ các tệp trọng số pre-trained trong hệ thống đều được kiểm tra tính toán băm SHA-256 và giữ nguyên trạng thái gốc:

| Tệp mô hình | SHA-256 Checksum | Trạng thái |
| :--- | :--- | :---: |
| [backend/models/weights.pt](file:///d:/SIGN_LANGUAGE/backend/models/weights.pt) | `6965b4ffd9eed45c10317f0cc4eff453470ea4552c081f18ea997d74365907be` | **NGUYÊN VẸN** |
| [backend/models/weights/sign_sstcn_attention_model.pth](file:///d:/SIGN_LANGUAGE/backend/models/weights/sign_sstcn_attention_model.pth) | `2a8ddcc4d06c4f23aa7910caf77371848034dcdf8f4749d134c5894764cf9697` | **NGUYÊN VẸN** |
| [backend/models/alphabet_static_head_v1.pt](file:///d:/SIGN_LANGUAGE/backend/models/alphabet_static_head_v1.pt) | `cab8123e23ed41a6088ba1c6e80a8aaaa9a382663418cc669eacff834795c254` | **NGUYÊN VẸN** |

---

## 6. HƯỚNG DẪN VẬN HÀNH VÀ KIỂM TRA THỰC TẾ QUA CAMERA

### 6.1. Bật chế độ Debug Diacritic FSM
Để theo dõi chi tiết từng bước chuyển trạng thái FSM (từ `IDLE` $\rightarrow$ `BASE_STABLE` $\rightarrow$ `DIACRITIC_CONFIRMED`) trên terminal:
```powershell
$env:VSL_DIACRITIC_DEBUG = "1"
python backend/app.py
```
Khi bật cờ này, mỗi khi người ký thực hiện động tác vẩy dấu, log sẽ in rõ:
```
[DIACRITIC_DEBUG] session=cam1 | base=A -> Ă (d_wrist_y=-0.082, arc_prom=0.045) | conf=0.88
```

### 6.2. Chạy toàn bộ test suite
```powershell
python -m pytest backend/tests/test_alphabet_classifier.py backend/tests/test_alphabet_recognition_integration.py backend/tests/test_alphabet_static_head_integration.py backend/tests/test_diacritic_state_machine.py backend/tests/test_diacritic_camera_integration.py
```

### 6.3. Khuyến nghị kiểm thử thực tế và giới hạn hiện tại
> [!NOTE]
> Hệ thống hiện đã hỗ trợ đầy đủ 29 chữ cái về mặt logic và tích hợp luồng camera. Tuy nhiên, các ngưỡng động học ($dy, dx$, arc prominence) được tham chiếu từ dữ liệu Dataset V2 (1 người ký mẫu). Trong môi trường thực tế với nhiều người ký có tốc độ vẩy tay hoặc góc camera khác nhau, cần tiếp tục thu thập dữ liệu đa người ký (Multi-Signer Evaluation theo kế hoạch Task 6B.2) để tinh chỉnh adaptive thresholds nếu cần thiết.

