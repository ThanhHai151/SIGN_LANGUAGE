# BÁO CÁO KIỂM TOÁN VÀ THIẾT LẬP MÔI TRƯỜNG LOGGING SẠCH

**PROMPT ID:** A01.1-P01  
**PROMPT NAME:** ANTIGRAVITY | PREPARE CLEAN WEBCAM TEST LOGS  
**TRẠNG THÁI:** COMPLETED  
**NGÀY THỰC HIỆN:** 11/11/2026  

---

## 1. Mục tiêu và Phạm vi
Chuẩn bị môi trường ghi nhận log sạch sẽ, cô lập tuyệt đối cho chiến dịch kiểm thử webcam 15 lượt **A01.1-T01** của mô hình SSTCN (Word Mode, 63 classes: 39–101), đồng thời bảo toàn 100% dữ liệu lịch sử của đợt kiểm thử VSL-P010 mà không làm ảnh hưởng đến pipeline nhận diện cốt lõi hay các chế độ khác (`letters`, `digits`, `sentence`).

---

## 2. Kết quả Rà soát Hệ thống Logging (Word Debug Logging Audit)

Đã rà soát toàn bộ các vị trí xử lý và ghi nhận Word Debug Logging trong repository:

| Thành phần | Vị trí file | Trách nhiệm & Cơ chế hoạt động |
| :--- | :--- | :--- |
| **Debug Engine Singleton** | [`backend/vision/word_debugger.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/vision/word_debugger.py) | Quản lý vòng đời phiên test (`start_session`, `stop_session`, `record_prediction`, `record_blocked_sequence`, `log_event`). Tự động ghi `session.json`, `events.jsonl`, `trial_summary.csv`, `sequences/sequence_*.npy` và append vào master `trials_summary.csv`. |
| **Pipeline Bridge** | [`backend/vision/recognizer.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/vision/recognizer.py) | Gửi chuỗi landmark (30, 75, 3) và kết quả dự đoán của SSTCN vào `word_debugger.record_prediction` khi có session active; chặn và ghi nhận chuỗi không hợp lệ qua `record_blocked_sequence`. |
| **REST API Server** | [`backend/app.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/app.py) | Cung cấp các endpoints: `/api/debug/word/targets`, `/api/debug/word/session`, `/api/debug/word/event`, `/api/debug/word/status`. Nhận tham số `session_prefix` để định danh chiến dịch. |
| **Frontend API Service** | [`src/services/wordDebugApi.js`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/services/wordDebugApi.js) | Giao tiếp với backend API debug, truyền cờ `sessionPrefix = 'A01_1_T01'` khi khởi tạo session. |
| **Frontend UI Panel** | [`src/components/WordDebugPanel.jsx`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/WordDebugPanel.jsx) | Giao diện điều khiển chọn từ (ground truth class 39–101), số thứ tự trial (1–15), nút Start/Stop Test, hiển thị badge chiến dịch `🏷️ Đợt test: A01_1_T01` và bảng tóm tắt kết quả sau mỗi trial. |
| **Camera Component** | [`src/components/CameraSignRecognizer.jsx`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx) | Truyền `debug_meta` (FPS, tracker_engine, latency, reset counters) kèm frame buffer lên backend. |

---

## 3. Quá trình Sao lưu và Cô lập Dữ liệu Lịch sử (Archival)

### 3.1. Thư mục Lưu trữ Lưu động (Archive Path)
Toàn bộ dữ liệu 14 phiên test lịch sử của đợt VSL-P010 đã được di chuyển và sao lưu vào:  
`E:\NAM4\SEMINAR\SIGN_LANGUAGE\debug\archive\vsl_p010_historical_reconstructed_2026-11-07\`

### 3.2. Bằng chứng Toàn vẹn Dữ liệu (Integrity Verification)
- **Số lượng file sao lưu:** **544 files**
- **Tổng dung lượng:** **14.710.739 bytes** (~14.03 MB)
- **Cấu trúc dữ liệu được bảo toàn:**
  - 14 thư mục phiên test độc lập (`Ban_trial_01_...`, `Buoi_toi_trial_01_...`, `Tam_biet_trial_...`, `The_nao_trial_...`).
  - 501 file ma trận landmarks `.npy` (`sequences/sequence_*.npy`).
  - 14 file nhật ký sự kiện frame-by-frame `events.jsonl`.
  - 14 file metadata phiên `session.json`.
  - 14 file tóm tắt kết quả theo từng phiên `trial_summary.csv`.
  - 1 file master CSV snapshot lịch sử `trials_summary.csv` (14 dòng tương ứng 14 trials của VSL-P010).
- **Kiểm định mã băm SHA-256:** 100% tệp tin được kiểm tra checksum SHA-256 khớp tuyệt đối giữa thư mục nguồn và thư mục lưu trữ.
- **Manifest chi tiết:** Đã tạo file manifest `debug/archive/vsl_p010_historical_reconstructed_2026-11-07/ARCHIVE_MANIFEST.json`.

---

## 4. Thiết lập Môi trường Logging Sạch cho Đợt Test A01.1-T01

1. **Khởi tạo thư mục làm việc sạch:**
   - Xóa toàn bộ session cũ khỏi `debug/words/` sau khi đã hoàn tất xác nhận sao lưu.
   - Khởi tạo mới file `debug/words/trials_summary.csv` chỉ với dòng tiêu đề chuẩn 23 cột.
   - Thêm tài liệu hướng dẫn cấu trúc thư mục tại [`debug/words/README.md`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/debug/words/README.md).

2. **Cơ chế gắn nhãn chiến dịch (Campaign Tagging):**
   - Hỗ trợ tham số `session_prefix` (mặc định: `"A01_1_T01"`).
   - Tên thư mục session được tạo tự động theo định dạng:  
     `A01_1_T01_<SanitizedWord>_trial_<TrialID>_<YYYY-MM-DD_HHMMSS>`  
     *(Ví dụ: `A01_1_T01_Buoi_toi_trial_01_2026-11-11_230000`)*.
   - File `session.json` lưu rõ trường `"campaign": "A01_1_T01"`.
   - Master CSV `trials_summary.csv` ghi nhận rõ cột `session_id` có tiền tố `A01_1_T01_`, giúp việc lọc và phân tích dữ liệu 15 trials hoàn toàn tách biệt với dữ liệu cũ.

3. **Chống trộn lẫn Checkpoint:**
   - Khi phiên test bắt đầu, hệ thống tự động đọc và khóa metadata checkpoint đang nạp từ backend (`checkpoint_type`, `checkpoint_filename`, `checkpoint_sha256`, `checkpoint_epoch`, `model_device`).
   - Nếu trong quá trình chạy mà checkpoint bị thay đổi hoặc backend restart không nhất quán, hệ thống ghi nhận event `CHECKPOINT_MISMATCH` và gán verdict `INVALID_CHECKPOINT_MISMATCH`.

4. **Bảo toàn 100% Logic Nhận diện:**
   - Mã nguồn mô hình [`backend/models/sstcn_model.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/models/sstcn_model.py) giữ nguyên kiến trúc và trọng số.
   - Bảng ánh xạ từ điển [`backend/models/vsl_102_dictionary.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/models/vsl_102_dictionary.py) giữ nguyên 102 lớp.
   - Không thay đổi bất kỳ logic inference nào của các chế độ khác (`letters`, `digits`, `sentence`).

---

## 5. Kết quả Kiểm thử Tự động (End-to-End Verification)

Kịch bản kiểm thử tự động [`scratch/verify_clean_logging.py`](file:///e:/NAM4/SEMINAR/SIGN_LANGUAGE/scratch/verify_clean_logging.py) được thực thi trong môi trường ảo của dự án (`.venv\Scripts\python.exe`):

```text
=== TEST START: VERIFY CLEAN WEBCAM LOGGING ENVIRONMENT ===
[WordDebug] START  target='Thời gian' (class 39) trial=1
            checkpoint='sign_sstcn_epoch_32.pth' (historical)
            -> E:\NAM4\SEMINAR\SIGN_LANGUAGE\debug\words\A01_1_T01_Thoi_gian_trial_01_2026-11-11_224537
[OK] Session started successfully: A01_1_T01_Thoi_gian_trial_01_2026-11-11_224537
[OK] session.json verified with campaign: A01_1_T01
[WordDebug] Window #001 target=Thời gian -> top1=Thời gian (0.95) [OK ] | ckpt=sign_sstcn_epoch_32.pth | zero L/R/pose=0/0/0 | infer=12.5ms
[OK] Sequence recorded. Status: {'active': True, 'session_id': 'A01_1_T01_Thoi_gian_trial_01_2026-11-11_224537', 'campaign': 'A01_1_T01', 'target_word': 'Thời gian', 'target_class': 39, 'trial': 1, 'seq_count': 1, 'reset_count': 0, 'blocked_fallback_count': 0, 'checkpoint': {'checkpoint_type': 'historical', 'checkpoint_filename': 'sign_sstcn_epoch_32.pth', 'checkpoint_sha256': 'abcdef1234567890', 'checkpoint_epoch': 32, 'model_device': 'cpu'}}
[WordDebug] STOP   A01_1_T01_Thoi_gian_trial_01_2026-11-11_224537 (1 sequences) -> Verdict: CORRECT
[OK] Session stopped with summary verdict: CORRECT
[OK] Master CSV verified: 1 row recorded with session_id=A01_1_T01_Thoi_gian_trial_01_2026-11-11_224537
[OK] Archive verification passed: 544 files intact.
[OK] Test session cleaned up. Clean environment ready for user's 15 trials!
=== ALL VERIFICATION CHECKS PASSED ===
```

Kiểm tra biên dịch Frontend bằng `npm run build`:
- **Kết quả:** `✓ built in 708ms` — PASS hoàn toàn không có lỗi cú pháp hoặc thiếu module.

---

## 6. Hướng dẫn Người dùng Thực hiện 15 Lượt Kiểm thử Webcam (A01.1-T01)

### Bước 1: Khởi động Backend
Mở Terminal PowerShell và chạy lệnh:
```powershell
.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```
*Lưu ý: Backend sẽ tự động load checkpoint SSTCN hiện tại và nạp MediaPipe Holistic.*

### Bước 2: Khởi động Frontend
Mở một cửa sổ Terminal khác và chạy:
```powershell
npm run dev
```
Truy cập trình duyệt tại địa chỉ:
`http://localhost:5173/?wordDebug=1` *(hoặc cổng Vite được thông báo, bắt buộc có tham số `?wordDebug=1` để bật bảng điều khiển Word Debug)*.

### Bước 3: Thao tác thực hiện từng Trial (15 lượt)
1. Trên giao diện web, chuyển sang chế độ **"Từ ngữ (Word Mode)"**.
2. Quan sát bảng màu hổ phách phía dưới: **🧪 WORD DEBUG TEST (ĐỘC LẬP TỪNG TRIAL)** có gắn nhãn `🏷️ Đợt test: A01_1_T01`.
3. Với mỗi lượt test trong số 15 trials:
   - **Target word:** Chọn từ cần test từ danh sách dropdown (ví dụ: `Buổi tối (class 40)`, `Chào (class 55)`, `Bạn (class 56)`...).
   - **Trial:** Nhập số thứ tự trial (ví dụ: `1`, `2`, `3`...).
   - Nhấn nút **"Start Test"**: Trạng thái chuyển sang `● Recording`.
   - Thực hiện cử chỉ trước webcam trong khoảng 3–5 giây.
   - Nhấn nút **"Stop Test"**: Phiên test kết thúc, bảng tóm tắt kết quả (Verdict: `CORRECT`/`PARTIAL`/`FAIL`, số cửa sổ đúng, landmark source...) sẽ hiển thị tức thì.
   - Hệ thống tự động lưu trữ dữ liệu vào thư mục `debug/words/A01_1_T01_<Word>_trial_<TrialID>_<Timestamp>/` và ghi dòng tóm tắt vào `debug/words/trials_summary.csv`.
   - Ô Trial tự động tăng lên để bạn sẵn sàng cho lượt test tiếp theo.

