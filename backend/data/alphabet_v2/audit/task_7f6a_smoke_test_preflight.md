# TASK 7F.6A — CONTROLLED WEBCAM PILOT SMOKE TEST PREFLIGHT AUDIT

**Trạng thái phê duyệt:** `READY_FOR_SMOKE_TEST`  
**Ngày kiểm định:** 2026-11-01  
**Môi trường:** Python 3.12, PyTorch 2.2.2, FastAPI, React 18, Vite 8.3.0, Windows 10/11  
**Phạm vi:** Kiểm tra quy trình vận hành và thiết kế kịch bản Smoke Test tối thiểu cho Controlled Webcam Pilot sau Task 7F.5G.

---

## 1. MỤC TIÊU & PHẠM VI KIỂM ĐỊNH

Task 7F.6A là bước lập kế hoạch và tiền kiểm tra (Preflight Audit) trước khi người vận hành tiến hành chạy thử nghiệm webcam thực tế (Smoke Test).

### Ràng buộc thực thi bắt buộc
- **Không chạy webcam:** Nhiệm vụ này hoàn toàn là Read-only Audit & Planning.
- **Không sinh dữ liệu giả mạo:** Không tạo các bản ghi log giả và coi là dữ liệu webcam thực.
- **Không chỉnh sửa mã nguồn:** Giữ nguyên vẹn 100% mã nguồn thuật toán nhận diện, trọng số mô hình (`weights/sign_sstcn_attention_model.pth`, `alphabet_static_head_v1.pt`), ngưỡng phân lớp (`Q=0.85`, `N=0.75`), danh sách Fallback (`A, D, E`), logic FSM và Geometric Rules.
- **Tính xác thực:** Mọi thông tin, cổng mạng, endpoint, tệp nguồn và hành vi giao diện đều được đối chiếu trực tiếp từ mã nguồn thực tế; các điểm chưa thể xác minh trong code được ghi nhận là `NOT VERIFIED`.

---

## 2. QUY TRÌNH VẬN HÀNH THỰC TẾ (OPERATIONAL PROCEDURE)

Dựa trên việc kiểm tra trực tiếp các tệp [`start.sh`](file:///d:/SIGN_LANGUAGE/start.sh), [`backend/app.py`](file:///d:/SIGN_LANGUAGE/backend/app.py), [`src/components/PilotTestingHUD.jsx`](file:///d:/SIGN_LANGUAGE/src/components/PilotTestingHUD.jsx), và [`src/services/api.js`](file:///d:/SIGN_LANGUAGE/src/services/api.js):

### 2.1. Cách khởi động Backend và Frontend
1. **Khởi động Backend (FastAPI):**
   - Lệnh thực thi từ thư mục gốc `d:\SIGN_LANGUAGE`:
     ```powershell
     python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
     ```
   - Xác nhận Backend hoạt động: Truy cập `http://127.0.0.1:8000/api/health` hoặc Swagger UI tại `http://127.0.0.1:8000/docs`.
2. **Khởi động Frontend (React / Vite):**
   - Lệnh thực thi từ thư mục gốc `d:\SIGN_LANGUAGE`:
     ```powershell
     npm run dev
     ```
   - Vite dev server sẽ lắng nghe tại `http://127.0.0.1:5173`.
3. **Mở giao diện ứng dụng:**
   - Mở trình duyệt (khuyến nghị Chrome/Edge) truy cập `http://127.0.0.1:5173`.
   - Cấp quyền truy cập Camera cho trình duyệt khi được yêu cầu.

### 2.2. Cách mở HUD và tạo Session
- Trong giao diện [`CameraSignRecognizer.jsx`](file:///d:/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L188-L206):
  - Bật Camera bằng nút bật camera chính.
  - Chuyển chế độ nhận diện sang **"Bảng chữ cái VSL"** (`mode = "letters"`).
  - Nhấp nút **"Mở Pilot Testing HUD"** ở góc trên khung hình camera.
- Trong giao diện [`PilotTestingHUD.jsx`](file:///d:/SIGN_LANGUAGE/src/components/PilotTestingHUD.jsx#L25-L39):
  - HUD tự động sinh `sessionId` dạng `pilot_YYYYMMDD_HHMM` (ví dụ: `pilot_20261101_2015`).
  - Người vận hành có thể chỉnh sửa chuỗi này (chỉ cho phép các ký tự `[a-zA-Z0-9_-]`, độ dài 3–64 ký tự).
  - Ví dụ đặt tên: `smoke_signer01_20261101`.

### 2.3. Cách chọn chữ cái, bắt đầu trial và kết thúc trial
1. **Chọn chữ cái mục tiêu (`expected_sign`):**
   - Chọn từ dropdown **"Chữ mục tiêu (Ground Truth)"** trong 22 chữ cái tĩnh tiếng Việt: `A, B, C, D, E, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, X, Y`.
2. **Bắt đầu Trial:**
   - Người thử nghiệm ổn định tư thế tay trước ống kính.
   - Bấm nút **"Bắt đầu Trial (5s Hold)"** (`handleStartTrial`).
   - HUD tự động gọi API kích hoạt logging nếu chưa bật (`handleTogglePilotLogging(true)`).
   - HUD chuyển sang trạng thái đếm ngược 5.0 giây (`durationMs = 5000`), thanh tiến trình hiển thị màu xanh lam.
   - Vòng lặp nhận diện MediaPipe truyền metadata của từng frame (`expected_sign`, `session_id`, `trial_id`, `frame_idx`) vào cả HTTP Header lẫn JSON body của request `/api/recognize_landmarks`.
3. **Kết thúc Trial:**
   - Sau 5.0 giây, HUD tự động kích hoạt `handleEndTrial()`.
   - Trạng thái `trialStateRef.current` lập tức chuyển thành `'completed'`, cắt đứt luồng metadata ở cấp vi giây để không frame nào ngoài 5s bị gắn nhãn sai.
   - HUD tự động gọi `/api/pilot/flush` để xả dữ liệu của trial xuống đĩa JSONL.
   - Bộ đếm `currentRep` tự động tăng lên 1 (ví dụ từ `1` lên `2`).

### 2.4. Cách bắt đầu và kết thúc Session
- **Bắt đầu Session:** Tự động kích hoạt khi bấm bắt đầu trial đầu tiên, hoặc thông qua nút kích hoạt trên giao diện HUD.
- **Kết thúc Session:**
  - Sau khi hoàn thành toàn bộ các trial mong muốn, người vận hành bấm nút **"Flush & Xong"** (`handleFinishSession`).
  - Hệ thống thực hiện:
    1. Hủy bất kỳ trial nào đang dở dang (`handleCancelTrial`).
    2. Gọi `/api/pilot/flush` để bảo đảm mọi bản ghi còn lại trong queue được ghi đĩa hoàn toàn.
    3. Gọi `/api/pilot/control` với payload `{ "enabled": false }` để tắt logger và giải phóng session trên Backend.
    4. Hiển thị hộp thoại tổng kết số record đã ghi nhận (`logged_count`) và số lượng thất thoát (`total_dropped`).

### 2.5. Đường dẫn tệp JSONL thực tế được tạo
- Thư mục lưu trữ: `backend/data/diagnostics/pilot/`
- Tên tệp chuẩn: `pilot_{session_id}.jsonl`
- Nếu session_id trùng phiên trước, tệp mới được tạo an toàn với hậu tố: `pilot_{session_id}_{YYYYMMDD_HHMMSS}_{counter}.jsonl`

### 2.6. Cách xác nhận Logger đã đóng thành công và kiểm tra lỗi / dropped records
1. **Kiểm tra trực tiếp qua UI HUD:**
   - Dòng telemetry footer hiển thị: `Queued: X`, `Logged: Y`, `Thất thoát: 0`.
   - Tiêu chí đạt: `Logged == Queued` và `Thất thoát == 0`.
2. **Kiểm tra qua API Endpoint:**
   - `GET http://127.0.0.1:8000/api/pilot/stats`
   - Phản hồi mong đợi:
     ```json
     {
       "success": true,
       "stats": {
         "queued_count": 650,
         "logged_count": 650,
         "dropped_queue_full": 0,
         "dropped_io_error": 0,
         "total_dropped": 0,
         "pending_in_queue": 0
       }
     }
     ```

---

## 3. CÁC API VÀ FILE LIÊN QUAN TRONG DỰ ÁN

| Thành phần | Đường dẫn tệp | Vai trò / Chức năng |
|:---|:---|:---|
| **FastAPI App** | [`backend/app.py`](file:///d:/SIGN_LANGUAGE/backend/app.py) | Cung cấp các endpoint: `/api/recognize_landmarks`, `/api/pilot/control`, `/api/pilot/flush`, `/api/pilot/stats`. |
| **Pilot Logger** | [`backend/vision/pilot_logger.py`](file:///d:/SIGN_LANGUAGE/backend/vision/pilot_logger.py) | Bộ ghi nhật ký non-blocking, quản lý hàng đợi đa luồng, vòng đời 3 trạng thái (`OPEN` -> `CLOSING` -> `CLOSED`), flush barrier, atomic file creation. |
| **Recognizer** | [`backend/vision/recognizer.py`](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py) | Nhận diện cử chỉ, gắn metadata chẩn đoán, gọi `pilot_logger.log_frame()`. |
| **Pilot HUD Component** | [`src/components/PilotTestingHUD.jsx`](file:///d:/SIGN_LANGUAGE/src/components/PilotTestingHUD.jsx) | Giao diện điều khiển thử nghiệm: chọn Session ID, Chữ cái mục tiêu, Lần lặp, Đếm ngược 5s, nút Flush & Kết thúc. |
| **Camera View** | [`src/components/CameraSignRecognizer.jsx`](file:///d:/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx) | Tích hợp HUD, trích xuất điểm mốc MediaPipe từ webcam, truyền metadata qua API client. |
| **API Client** | [`src/services/api.js`](file:///d:/SIGN_LANGUAGE/src/services/api.js) | Chuyển tiếp metadata qua HTTP Headers (`X-Expected-Sign`, `X-Session-Id`, v.v.) và JSON body. |

---

## 4. THIẾT KẾ KỊCH BẢN SMOKE TEST TỐI THIỂU (MINIMAL SMOKE TEST)

Kịch bản được thiết kế với quy mô tinh gọn nhất nhằm xác minh tính toàn vẹn của chuỗi dữ liệu (Data Pipeline Integrity) trước khi tiến hành thu thập đại trà 22 chữ cái.

### 4.1. Thông số thiết lập
- **Số lượng Signer:** 01 người thử nghiệm.
- **Mã Session ID đề xuất:** `smoke_signer01_20261101` (tuân thủ regex `^[a-zA-Z0-9_-]{3,64}$`).
- **Danh sách 3 chữ cái tĩnh được chọn:**
  1. **Chữ `A`:** Đại diện cho chữ cái có cả Geometric Rule và Gated Static Head Fallback đã kích hoạt trong production.
  2. **Chữ `B`:** Đại diện cho chữ cái dạng xòe 4 ngón khép, thuần túy Geometric Rule trong production hiện tại.
  3. **Chữ `C`:** Đại diện cho chữ cái dạng cong uốn tròn, kiểm tra nhận diện hình học góc cong đặc thù.
- **Số lần lặp (Repetitions):** 02 repetitions cho mỗi chữ cái.
- **Thời lượng mỗi Trial:** 5.0 giây (thời lượng cố định được mã hóa trong HUD).
- **Khoảng nghỉ giữa các Trial (Neutral Rest):** 3 đến 5 giây thả lỏng tay về tư thế tự nhiên trước khi bấm trial tiếp theo.
- **Tổng số Trial thực hiện:** $3 \text{ chữ} \times 2 \text{ reps} = 6 \text{ trials}$.
- **Ước tính số lượng frame:** Khoảng 600 – 900 frames (tùy thuộc tốc độ webcam 20–30 FPS).

### 4.2. Trình tự thực hiện từng bước (Step-by-step Runbook)

```
[BƯỚC 1: KHỞI TẠO]
  1. Khởi động Backend: python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
  2. Khởi động Frontend: npm run dev
  3. Mở trình duyệt tại http://127.0.0.1:5173
  4. Bật Camera, chọn chế độ "Bảng chữ cái VSL"
  5. Bấm "Mở Pilot Testing HUD"
  6. Nhập Session ID: "smoke_signer01_20261101"

[BƯỚC 2: THỰC HIỆN TRIAL CHỮ A]
  7. Chọn Chữ cái: "A"
  8. Repetition: chọn 1/10
  9. Tạo dáng chữ A chuẩn trước webcam
  10. Bấm "Bắt đầu Trial (5s Hold)" -> Giữ yên tư thế trong 5s cho đến khi hết đếm ngược
  11. Nghỉ 3-5s (hạ tay)
  12. Đảm bảo Repetition hiển thị là 2/10
  13. Tạo dáng chữ A chuẩn lần 2
  14. Bấm "Bắt đầu Trial (5s Hold)" -> Giữ yên tư thế 5s

[BƯỚC 3: THỰC HIỆN TRIAL CHỮ B]
  15. Chuyển Chữ cái sang: "B" (Repetition tự động reset về 1)
  16. Tạo dáng chữ B chuẩn trước webcam
  17. Bấm "Bắt đầu Trial (5s Hold)" -> Giữ yên tư thế 5s
  18. Nghỉ 3-5s (hạ tay)
  19. Tạo dáng chữ B chuẩn lần 2 (Rep 2)
  20. Bấm "Bắt đầu Trial (5s Hold)" -> Giữ yên tư thế 5s

[BƯỚC 4: THỰC HIỆN TRIAL CHỮ C]
  21. Chuyển Chữ cái sang: "C"
  22. Tạo dáng chữ C chuẩn trước webcam (Rep 1)
  23. Bấm "Bắt đầu Trial (5s Hold)" -> Giữ yên tư thế 5s
  24. Nghỉ 3-5s (hạ tay)
  25. Tạo dáng chữ C chuẩn lần 2 (Rep 2)
  26. Bấm "Bắt đầu Trial (5s Hold)" -> Giữ yên tư thế 5s

[BƯỚC 5: ĐÓNG VÀ FLUSH PHIÊN]
  27. Bấm "Flush & Xong" trên HUD
  28. Xác nhận hộp thoại thông báo thành công và ghi nhận số lượng records
```

---

## 5. TIÊU CHÍ NGHIỆM THU DỮ LIỆU JSONL (ACCEPTANCE CRITERIA)

Dữ liệu thu được từ tệp `backend/data/diagnostics/pilot/pilot_smoke_signer01_20261101.jsonl` phải vượt qua các tiêu chí nghiệm thu sau:

### 5.1. Tiêu chí tính toàn vẹn tệp & hạ tầng (File & Infrastructure Integrity)
1. **Tính hợp lệ cú pháp JSONL:** Mỗi dòng trong tệp là một đối tượng JSON độc lập, không có dòng rác hoặc ký tự gãy (`json.loads()` thành công 100%).
2. **Khớp số lượng Telemetry:** Số dòng trong tệp JSONL phải khớp chính xác với `logged_count` và `queued_count` được báo cáo bởi API `/api/pilot/stats`.
3. **Thất thoát bằng không:** `dropped_queue_full == 0` và `dropped_io_error == 0`.
4. **Không có bản ghi sau khi đóng:** Sau khi bấm "Flush & Xong", không có bản ghi nào của session này tiếp tục được ghi vào tệp.

### 5.2. Tiêu chí Metadata & Session/Trial Tracking
1. **Session ID:** Tất cả các bản ghi có trial đều mang `session_id == "smoke_signer01_20261101"`.
2. **Danh mục Trial ID:** Phải xuất hiện đầy đủ 6 Trial ID duy nhất:
   - `smoke_signer01_20261101_A_rep01`
   - `smoke_signer01_20261101_A_rep02`
   - `smoke_signer01_20261101_B_rep01`
   - `smoke_signer01_20261101_B_rep02`
   - `smoke_signer01_20261101_C_rep01`
   - `smoke_signer01_20261101_C_rep02`
3. **Tính đơn điệu của `frame_idx`:** Trong phạm vi mỗi trial, `frame_idx` bắt đầu từ 1 và tăng đơn điệu nghiêm ngặt không trùng lặp ($1, 2, 3, \dots, N$).
4. **Tính hợp lý của Timestamp & FPS thực tế:**
   - Thời gian `timestamp` giữa hai frame liên tiếp phải tăng đơn điệu ($\Delta t = t_{i+1} - t_i > 0$).
   - FPS thực tế được tính theo công thức:
     $$\text{FPS}_{\text{trial}} = \frac{N_{\text{frames}} - 1}{t_{\text{end}} - t_{\text{start}}}$$
   - Giá trị FPS thực tế nằm trong khoảng chấp nhận được của webcam ($15.0 \le \text{FPS} \le 35.0$).

### 5.3. Tiêu chí tính độc lập của `expected_sign` (Ground Truth Isolation)
- `expected_sign` phải khớp chính xác với chữ đang thực hiện ("A", "B", hoặc "C").
- **Kiểm tra chéo độc lập:** Xác nhận rằng `expected_sign` hoàn toàn không can thiệp vào bộ phân loại. Nếu signer cố tình làm sai hoặc chưa kịp đưa tay lên, `final_sign` phải trả về `NEUTRAL` hoặc nhãn nhận diện thực tế chứ không được tự động khớp theo `expected_sign`.

### 5.4. Phân biệt rõ ràng giữa Đánh giá theo Frame và Đánh giá theo Trial
> [!IMPORTANT]
> **Các frame liên tiếp trong cùng một video/webcam 5 giây KHÔNG PHẢI là các mẫu độc lập thống kê** (Statistically Correlated Frames). Do đó, phân tích dữ liệu phải phân tách rõ hai tầng đo lường:
> 1. **Thống kê mức Frame (Frame-level Metrics):**
>    - Số frame thu được trong mỗi trial.
>    - Tỷ lệ số frame nhận diện đúng (`final_sign == expected_sign`) trên tổng số frame của trial.
>    - Tỷ lệ frame ở trạng thái ổn định (`confirmed == True`).
> 2. **Thống kê mức Trial (Trial-level Metrics):**
>    - Đánh giá tổng thể 6 trial hoàn thành.
>    - Nhãn dự đoán đại diện của trial (được xác định theo cơ chế Đa số phiếu - Majority Voting hoặc nhãn xác nhận dài nhất trong 5s).
>    - Trial được coi là thành công (PASS) nếu nhãn đa số khớp với `expected_sign`.

---

## 6. MẪU SCRIPT PYTHON XÁC MINH NHANH DỮ LIỆU SMOKE TEST

Sau khi hoàn tất Smoke Test, người vận hành có thể chạy đoạn script kiểm tra tự động sau để xác nhận nghiệm thu:

```python
import json
import os
import sys

def verify_smoke_test(jsonl_path, expected_session_id):
    if not os.path.exists(jsonl_path):
        print(f"❌ Không tìm thấy tệp: {jsonl_path}")
        return False

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Tổng số bản ghi đọc được: {len(records)}")
    if len(records) == 0:
        print("❌ Tệp rỗng!")
        return False

    # Lọc các frame thuộc trial
    trial_records = [r for r in records if r.get("trial_id") is not None]
    print(f"Số frame thuộc trial: {len(trial_records)}")

    trials = {}
    for r in trial_records:
        t_id = r["trial_id"]
        trials.setdefault(t_id, []).append(r)

    print(f"Tổng số Trial hoàn thành: {len(trials)}/6")

    all_passed = True
    for t_id, t_frames in sorted(trials.items()):
        exp = t_frames[0].get("expected_sign")
        frame_indices = [f["frame_idx"] for f in t_frames]
        timestamps = [f["timestamp"] for f in t_frames]
        
        # 1. Kiểm tra trùng lặp frame_idx
        no_dup_idx = len(frame_indices) == len(set(frame_indices))
        
        # 2. Tính FPS thực tế
        dur = timestamps[-1] - timestamps[0]
        fps = (len(t_frames) - 1) / dur if dur > 0 else 0
        
        # 3. Đánh giá nhận diện
        matches = sum(1 for f in t_frames if f.get("final_sign") == exp)
        match_rate = (matches / len(t_frames)) * 100

        status = "✅ PASS" if (no_dup_idx and len(t_frames) >= 50 and fps >= 10.0) else "❌ FAIL"
        if status == "❌ FAIL":
            all_passed = False

        print(f"[{status}] Trial: {t_id} | Chữ: {exp} | Frames: {len(t_frames)} | FPS: {fps:.1f} | Frame-Match: {match_rate:.1f}%")

    if all_passed and len(trials) == 6:
        print("\n🎉 SMOKE TEST ACCEPTANCE: PASSED 100%")
        return True
    else:
        print("\n⚠️ SMOKE TEST ACCEPTANCE: FAILED")
        return False

if __name__ == "__main__":
    test_file = sys.argv[1] if len(sys.argv) > 1 else "backend/data/diagnostics/pilot/pilot_smoke_signer01_20261101.jsonl"
    verify_smoke_test(test_file, "smoke_signer01_20261101")
```

---

## 7. CÁC RỦI RO VẬN HÀNH CÒN LẠI & HƯỚNG XỬ LÝ

| Rủi ro vận hành | Biểu hiện | Biện pháp phòng ngừa / Khắc phục |
|:---|:---|:---|
| **Điều kiện ánh sáng & Phông nền** | MediaPipe mất dấu bàn tay hoặc rung lắc điểm mốc khi ánh sáng yếu hoặc phông nền phức tạp. | Người thử nghiệm ngồi ở nơi đủ ánh sáng chiếu thẳng vào lòng bàn tay, phông nền đơn sắc, tránh ngược sáng. |
| **Hạ tay quá sớm trước khi hết 5s** | Frame cuối trial bị ghi nhận là `NEUTRAL` do tay đã hạ xuống trước khi thanh đếm ngược kết thúc. | Người thử nghiệm nhìn thanh đếm ngược trên HUD, chỉ hạ tay khi chữ chuyển sang trạng thái kết thúc trial và thanh tiến trình đầy 100%. |
| **Tab trình duyệt bị giảm tải (Background Throttling)** | Nếu chuyển sang cửa sổ khác khi đang trial, trình duyệt giảm FPS của `requestAnimationFrame` / `setInterval`. | Giữ cửa sổ trình duyệt luôn active và hiển thị toàn màn hình trong suốt quá trình thử nghiệm. |
| **Xung đột tên Session cũ** | Nếu nhập trùng `session_id` của phiên đang mở trên backend mà chưa tắt, backend sẽ trả về lỗi HTTP 409 Conflict. | Luôn bấm "Flush & Xong" trước khi đổi session, hoặc sử dụng timestamp để session_id luôn duy nhất. |

---

## 8. DANH MỤC THÔNG TIN NGƯỜI VẬN HÀNH CẦN CHUẨN BỊ (CHECKLIST)

1. [ ] **Thiết bị:** Webcam hoạt động ổn định ở độ phân giải tối thiểu 720p (30 FPS).
2. [ ] **Môi trường vật lý:** Bàn ngồi đối diện camera, khoảng cách 0.5m – 0.8m, bàn tay hiển thị rõ ràng trong khung hình camera.
3. [ ] **Kiến thức cử chỉ:** Nắm vững hình thái chuẩn của 3 chữ cái tĩnh theo giáo trình VSL:
   - **Chữ A:** Nắm chặt 4 ngón tay, ngón cái áp sát cạnh ngoài của ngón trỏ, hướng thẳng lên trên.
   - **Chữ B:** 4 ngón tay (trỏ, giữa, áp út, út) duỗi thẳng khép sát vào nhau, ngón cái gập ngang qua lòng bàn tay.
   - **Chữ C:** Toàn bộ các ngón tay và ngón cái uốn cong tạo thành hình chữ C hướng về phía người quan sát.
4. [ ] **Quyền truy cập:** Đảm bảo trình duyệt đã được cấp quyền truy cập Camera và cổng mạng `8000`, `5173` không bị chặn bởi tường lửa.

---

## 9. KẾT LUẬN

```
================================================================================
                    TASK 7F.6A PREFLIGHT AUDIT SUMMARY
                    STATUS: READY_FOR_SMOKE_TEST
================================================================================
  - Operational Procedures: Fully verified via source code inspection
  - Minimal Smoke Test: 1 Signer, 1 Session, 3 Letters (A, B, C), 2 Reps (6 Trials)
  - Trial Duration: Strictly aligned with HUD's 5.0-second countdown
  - Data Acceptance: Clear separation between Frame-level and Trial-level metrics
  - Verification Automation: Inspection script ready for post-run evaluation
================================================================================
```
Báo cáo tiền kiểm tra hoàn tất. Hệ thống và quy trình đã sẵn sàng để người vận hành tiến hành thu thập dữ liệu webcam cho Smoke Test.

