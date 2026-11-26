# BÁO CÁO BÀN GIAO — TỐI ƯU ĐỘ TRỄ NHẬN DẠNG PHỤ ĐỀ VÀ PHÁT GIỌNG NÓI (TTS) CHO CHỮ CÁI & CHỮ SỐ

**Dự án:** VSL Connect AI (`D:\SIGN_LANGUAGE`)  
**Phạm vi:** Chế độ `letters` (29 chữ cái tiếng Việt) và `digits` (chữ số `0–10`) tại màn hình **Camera → Ký hiệu sang Văn bản**.

---

## 1. Nguyên nhân gốc rễ đã xác minh từ mã nguồn thực tế

1. **Treo TTS chữ cái nền (`A, D, E, O, U`) khi người dùng hạ tay ngay sau khi chốt ký hiệu:**
   - Tại `src/components/CameraSignRecognizer.jsx`, vòng lặp MediaPipe trước đây dừng sớm (`if (!hasHands) return;`) nên không gửi gói tin báo mất bàn tay (`landmarks: []`) về Backend.
   - Tại `backend/vision/recognizer.py`, `_pending_tts_token` của nhóm chữ nền chỉ được xả khi Backend nhận được frame tiếp theo (giữ tĩnh đủ thời gian hoặc nhận cử chỉ thả tay). Hệ quả là nếu người dùng ký xong rồi hạ tay ngay, `_pending_tts_token` bị treo trong RAM Backend cho tới lần đưa tay ký hiệu kế tiếp.
2. **Xung đột hai nguồn điều phối TTS giữa Frontend và Backend:**
   - Trước đây `CameraSignRecognizer.jsx` tự đặt `pendingTtsTimerRef` (`1200ms`) song song với bộ đếm chờ dấu `1200–1600ms` của Backend, đồng thời `speakVietnamese()` gọi `window.speechSynthesis.cancel()` vô điều kiện. Khi hai nguồn kích hoạt lệch nhau vài chục ms, lệnh sau hủy luồng âm thanh vừa bắt đầu của lệnh trước.
3. **Cửa sổ Temporal Smoothing `5/10` frame cộng hưởng với nhịp gửi tuần tự (`isRecognizingRef`):**
   - Yêu cầu tích lũy 5 phiếu hợp lệ trên cửa sổ 10 frame làm tăng thời gian chốt Phụ đề thêm 1 chu kỳ frame so với cấu hình `4/7`.
4. **Khiếm khuyết TTS và chống lặp ở chế độ chữ số (`digits`):**
   - Phản hồi xác nhận cử chỉ động Số `10` tại `recognizer.py` thiếu trường `tts_speak` (chỉ có `tts_text`) và ghi đè không điều kiện token `"1"` cuối câu ngay cả khi số `1` đó là cử chỉ độc lập đã thả tay trước đó.
   - Chế độ `digits` cho các số `0–9` chưa có cờ theo dõi thả tay (`_digit_seen_release`), dẫn tới việc giữ nguyên một chữ số quá `1.2s` (`cooldown_sec`) bị chốt lặp lại vào Phụ đề và đọc lặp mỗi `1.2s`.
5. **Closure cũ (`ttsEnabled`) và rò rỉ trạng thái khi chuyển chế độ / xóa phụ đề:**
   - Callback MediaPipe khởi tạo một lần khi bật camera đã giữ giá trị `ttsEnabled` cũ trong closure.
   - `clear_sentence()` chưa xóa `prediction_history`, có thể khiến các frame tích lũy sát thời điểm bấm xóa kích hoạt lại ký hiệu cũ ngay frame kế tiếp.

---

## 2. Danh sách file đã sửa và bổ sung

- **Đã sửa:**
  - `backend/vision/recognizer.py` — Cửa sổ làm mượt `4/7` riêng cho `letters` & `digits` (bảo toàn `5/10` cho chế độ Từ vựng và Pilot Task 3.10), mã định danh sự kiện `confirm_event_id` & `tts_event_id`, tối ưu thời gian chờ dấu tĩnh (`0.85s` cho `A, E, O, U`; `1.18s` cho `D`; bảo vệ chuyển động tạo dấu tới `1.60s`), xả ngay `_pending_tts_token` khi nhận `landmarks: []`, bổ sung `tts_speak` cho Số `10`, chống lặp giữ tay 5s+ cho `digits`, dọn sạch trạng thái khi đổi mode / xóa câu / reset.
  - `src/services/api.js` — Nạp trước và cache giọng đọc `vi-VN` (`initSpeechSynthesisVoices` + sự kiện `voiceschanged`), bỏ gọi `speechSynthesis.cancel()` vô điều kiện khi không cần ngắt, bổ sung hàm `cancelVietnameseSpeech()` và telemetry đo `onstartLatencyMs`.
  - `src/components/CameraSignRecognizer.jsx` — Tích hợp bộ điều phối TTS duy nhất (`createTtsOrchestrator`), bỏ `pendingTtsTimerRef` (`1200ms`) và nhánh fallback đọc trùng, thêm `dispatchInstantRelease` gửi `landmarks: []` sau 2 frame mất tay liên tiếp (`~60ms`), đồng bộ `ttsEnabledRef`, dọn timer/TTS khi tắt camera, đổi chế độ, xóa phụ đề, tắt loa hoặc rời trang.
- **Đã thêm mới:**
  - `src/utils/ttsOrchestrator.js` — Bộ điều phối phát âm tập trung phía Frontend, chống đọc trùng theo `eventId`.
  - `tests/ttsOrchestrator.test.js` — Unit test cho bộ điều phối TTS Frontend.
  - `backend/tests/test_latency_and_tts_optimization.py` — Bộ kiểm thử tích hợp toàn diện cho 29 chữ cái, số `0–10`, giữ tay 5s, hạ tay tức thì, chuyển chế độ và bảo toàn chế độ Từ vựng.
  - `docs/BAO_CAO_PHU_DE_VA_DO_TRE_TTS.md` — Báo cáo phân tích hiện trạng ban đầu.
  - `docs/BAO_CAO_BAN_GIAO_TOI_UU_DO_TRE_VA_TTS.md` — Báo cáo bàn giao tổng hợp.

*(Lưu ý: `backend/vision/digit_10_detector.py`, `alphabet_classifier.py`, `digit_classifier.py`, `diacritic_state_machine.py`, `digit_aggregator.py` và toàn bộ trọng số mô hình được giữ nguyên 100%).*

---

## 3. Cấu hình trước và sau tối ưu

| Hạng mục cấu hình | Trước khi sửa (Baseline) | Sau khi sửa (Optimized) |
| :--- | :--- | :--- |
| Temporal Smoothing (`letters`, `digits`) | Cửa sổ `10` frame, cần `5` phiếu (`5/10`) | Cửa sổ `7` frame, cần `4` phiếu (`4/7`, công thức `(w + 1) // 2`) |
| Temporal Smoothing (`words` / Từ vựng / Pilot 3.10) | Cửa sổ `10` frame, cần `5` phiếu (`5/10`) | **Giữ nguyên** `10` frame, `5` phiếu (`5/10`) |
| Chờ dấu tĩnh chữ nền `A, E, O, U` | `1200 ms` (`1600 ms` khi candidate) | **`850 ms`** (`1600 ms` khi FSM candidate / đang tạo chuyển động dấu) |
| Chờ dấu tĩnh chữ nền `D` (`D → Đ`) | `1200 ms` (`1600 ms` khi candidate) | **`1180 ms`** (bảo toàn video chuẩn QIPEDC `D0495` có pha giữ `D` tĩnh `1126 ms`) |
| Xử lý khi hạ tay ngay sau chữ nền | Frontend bỏ qua frame mất tay; treo `_pending_tts_token` hoặc đợi timer `1200 ms` | Frontend gửi `landmarks: []` sau 2 frame mất tay (`~67 ms` @ 30 FPS) → Backend xả `tts_speak` **ngay lập tức** |
| Chống lặp khi giữ yên chữ số `0–9` (5 giây) | Lặp lại mỗi `1.2s` (`cooldown_sec`) | Khóa bằng `_digit_seen_release` (chỉ nhận lại sau khi thả tay / `NEUTRAL >= 3` frame + `> 0.30s`) |
| Phản hồi TTS của Số `10` | Chỉ có `tts_text: "mười"`, thiếu `tts_speak` | Có đầy đủ `tts_speak: "mười"`, `tts_event_id`, bảo vệ không ghi đè số `1` độc lập trước đó |
| Điều phối TTS trên Frontend | 2 nơi gọi chồng chéo + `speechSynthesis.cancel()` vô điều kiện | Duy nhất `ttsOrchestrator.js`, chống trùng theo `tts_event_id` / `confirm_event_id`, preload giọng `vi-VN` |

---

## 4. Độ trễ trước và sau theo từng giai đoạn & nhóm ký hiệu

### 4.1. Độ trễ hạ tầng suy luận & API thực tế
- **Backend Inference (`predict_landmarks`):**
  - `letters` (`N = 551` mẫu thật): Trước median `0.878 ms` (P95 `2.247 ms`, max `7.046 ms`) → Sau median **`0.573 ms`** (P95 `1.473 ms`, max `4.644 ms`).
  - `digits` (`N = 22` mẫu thật): Trước median `0.364 ms` (P95 `0.868 ms`, max `4.836 ms`) → Sau median **`0.335 ms`** (P95 `0.377 ms`, max `3.054 ms`).
- **HTTP Round-Trip thực tế qua Vite Proxy (`http://127.0.0.1:5173/api/recognize_landmarks` → FastAPI `:8000`, `N = 30`):**
  - Min `5.420 ms` | Median **`9.912 ms`** | P95 **`29.679 ms`** | Max **`30.921 ms`**.

### 4.2. Độ trễ xác nhận Phụ đề & phát tín hiệu TTS theo nhóm ký hiệu

| Nhóm ký hiệu | Chỉ số đo | Trước khi sửa (`5/10`) | Sau khi sửa (`4/7`) |
| :--- | :--- | :--- | :--- |
| **17 chữ cái thường** (`B, C, G, H, I, K, L, M, N, P, Q, R, S, T, V, X, Y`) | Ổn định → Phụ đề (30 / 20 / 15 FPS)<br>Xác nhận → Phát lệnh TTS | `133.3 / 200.0 / 266.7 ms`<br>`0.0 ms` | **`100.0 / 150.0 / 200.0 ms`** (-25%)<br>**`0.0 ms`** (tức thì) |
| **5 chữ cái nền — Giữ nguyên tay tĩnh** (`A, E, O, U` và `D`) | Ổn định → Phụ đề (30 / 20 / 15 FPS)<br>Xác nhận → Phát lệnh TTS (30 FPS) | `133.3 / 200.0 / 266.7 ms`<br>Median `1233.3 ms`, Max `1233.3 ms` (+ xung đột cancel) | **`100.0 / 150.0 / 200.0 ms`**<br>Median **`866.7 ms`** (`A,E,O,U`), P95/Max **`1200.0 ms`** (`D`) |
| **5 chữ cái nền — Ký xong hạ tay ngay** (`A, D, E, O, U`) | Xác nhận → Phát lệnh TTS (30 / 20 / 15 FPS) | Backend bị treo vô hạn tới ký hiệu sau; timer phụ Frontend `1200.0 ms` | **`66.7 / 100.0 / 133.3 ms`** (xả ngay khi hạ tay, không cần ký hiệu kế tiếp) |
| **7 chữ cái có dấu** (`Ă, Â, Đ, Ê, Ô, Ơ, Ư` — replay QIPEDC) | Chữ nền → Chốt chữ có dấu<br>Chốt chữ có dấu → Phát lệnh TTS | Median `500.0 ms` (`334–1084 ms`)<br>`0.0 ms` | Median **`542.0 ms`** (`375–1126 ms`, chốt chữ nền sớm hơn 1 frame)<br>**`0.0 ms`** (chỉ đọc đúng 1 lần chữ có dấu) |
| **Chữ số tĩnh `0–9`** (`N = 10` chữ số) | Ổn định → Phụ đề (30 / 20 / 15 FPS)<br>Xác nhận → Phát lệnh TTS | `133.3 / 200.0 / 266.7 ms`<br>`0.0 ms` | **`100.0 / 150.0 / 200.0 ms`** (-25%)<br>**`0.0 ms`** (tức thì, không qua timer chữ nền) |
| **Chữ số động `10`** (`D0539`) | Thời lượng cử chỉ → Phụ đề<br>Xác nhận → Phát lệnh TTS | `930.0 ms`<br>**Lỗi thiếu `tts_speak`** (không đọc ở chế độ instant) | **`930.0 ms`** (bảo toàn detector)<br>**`0.0 ms`** (`tts_speak: "mười"` phát ngay lập tức) |

---

## 5. Chất lượng nhận dạng và tỷ lệ lỗi trước / sau

- **Tập 22 chữ cái tĩnh (`vsl_alphabet_static_v2.npz`, `N = 551`):**
  - Tỷ lệ nhận đúng (Accuracy): Trước `454/551` (**`82.40%`**) → Sau `454/551` (**`82.40%`**) — **Không suy giảm**.
  - Tỷ lệ nhận nhầm chéo (False Positive): **`0.36%`** (`2/551`) — Không đổi.
  - Tỷ lệ từ chối an toàn về `NEUTRAL` (Miss / Reject): **`17.24%`** (`95/551`) — Không đổi.
- **Tập 7 chữ cái có dấu động (`Ă, Â, Đ, Ê, Ô, Ơ, Ư` từ chuỗi QIPEDC):**
  - Nhận đúng `7/7` (**`100%`**), `0` trường hợp đọc nhầm chữ nền trước khi ra dấu, `0` trường hợp đọc kép.
- **Tập chữ số `0–9` (`N = 22` mẫu thật) & Số `10` (`D0539`):**
  - Nhận đúng `22/22` (**`100%`**) cho `0–9` và `100%` cho Số `10`, `0` lần chốt nhầm số `1` trong khi đang xoay cổ tay tạo số `10`, `0` lần lặp khi giữ nguyên tay 5 giây.

---

## 6. Kết quả Unit Tests và Regression Tests

- **Backend (`pytest`):** **`718 passed, 1 skipped`** (bao gồm toàn bộ kiểm thử chữ cái, FSM dấu, chữ số `0–9`, cử chỉ động Số `10`, `VSLSequentialDigitAggregator`, tách biệt chế độ, scope Từ vựng, SSTCN, Pilot logging Task 3.10 và bộ test mới `test_latency_and_tts_optimization.py`).
  - *Ghi chú minh bạch:* Có 2 test cũ trong `test_api.py` (`test_camera_classes_linked_to_lexicon` kỳ vọng 102 từ camera so với 66 từ đã thu gọn trước đó, và `test_lexicon_search`) thuộc phạm vi dữ liệu từ điển cũ không liên quan tới thay đổi lần này; theo đúng ràng buộc mục 12, không chỉnh sửa các test đó.
- **Frontend (`node --test` & `npm run build`):** **`25/25 passed`** (`ttsOrchestrator.test.js`, `wordScopeGuard.test.js`, `pilotProtocol.test.js`), `npm run lint` **`0 errors`**, `npm run build` thành công.

---

## 7. Các trường hợp có thể tiệm cận hoặc vượt 2 giây & Rủi ro còn tồn tại

1. **Độ trễ khởi tạo Web Speech API của trình duyệt (lần đọc đầu tiên hoặc giọng đọc đám mây):**
   - Mã nguồn đã gọi `initSpeechSynthesisVoices()` ngay khi mở màn hình và ưu tiên giọng `vi-VN` cài sẵn trên máy (`localService === true`). Tuy nhiên, nếu Windows chưa cài gói giọng đọc tiếng Việt offline (*Microsoft An / HoaiMy*) và trình duyệt Chrome/Edge dùng giọng đọc online lần đầu tiên, thời gian từ `speechSynthesis.speak()` tới `utterance.onstart` của riêng trình duyệt có thể mất thêm `300–900 ms` ở câu/từ đầu tiên.
2. **Người dùng giữ tĩnh chữ `D` cực lâu rồi mới gập ngón tạo `Đ`:**
   - Ngưỡng chờ tĩnh của `D` được đặt ở `1.18s` (cộng `~0.10–0.15s` xác nhận Phụ đề + HTTP RTT = tổng `~1.30–1.35s`, vẫn dưới mục tiêu `2.0s`). Nếu người dùng giữ `D` bất động quá `1.18s` mới bắt đầu gập ngón trỏ, hệ thống sẽ đọc `D` trước theo đúng thiết kế tránh treo âm thanh.
3. **Máy yếu / ánh sáng kém làm giảm FPS MediaPipe xuống dưới 10 FPS:**
   - Ở `< 10 FPS`, 4 frame xác nhận sẽ mất `> 400 ms` thay vì `100–150 ms`.

---

## 8. Trạng thái từng Checkpoint

| Checkpoint | Nội dung | Trạng thái |
| :--- | :--- | :--- |
| **CHECKPOINT 1** | Đo độ trễ hiện tại (Baseline) trên từng chặng pipeline | **PASS** |
| **CHECKPOINT 2** | Tối ưu xác nhận Phụ đề (`4/7` cho `letters` & `digits`, bảo toàn `NO_MATCH`/`CONFLICT`/`NEUTRAL` & Từ vựng) | **PASS** |
| **CHECKPOINT 3** | Tối ưu TTS chữ cái (tức thì cho 17 chữ thường, `850ms`/`1180ms` + xả ngay khi hạ tay cho chữ nền, bảo toàn 7 chữ có dấu) | **PASS** |
| **CHECKPOINT 4** | Tối ưu TTS chữ số (`0–9` tức thì, chống lặp giữ 5s, sửa `tts_speak` cho Số `10`, giữ nguyên `VSLSequentialDigitAggregator`) | **PASS** |
| **CHECKPOINT 5** | Thống nhất điều phối TTS duy nhất trên Frontend (`ttsOrchestrator.js`), chống trùng theo Event ID, dọn sạch lifecycle | **PASS** |
| **CHECKPOINT 6** | Kiểm thử tự động toàn diện & đo lại độ trễ thực tế (median, P95, max) | **PASS** |
| **CHECKPOINT 7** | Controlled Webcam Pilot (Server mới đã chạy & xác minh live; chờ người dùng kiểm thử tay trực tiếp trước webcam) | **PENDING** *(chờ kiểm thử webcam thực tế)* |
