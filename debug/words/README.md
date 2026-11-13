# VSL Word Mode Webcam Logging Directory

Thư mục này dành riêng cho việc ghi log phiên kiểm thử webcam thực tế của mô hình SSTCN_Attention (Word Mode, 63 classes: 39–101).

---

## 1. Lưu trữ và Cô lập Dữ liệu (Archival & Isolation)

Toàn bộ các phiên log lịch sử thuộc đợt kiểm thử VSL-P010 (ngày 07/11/2026, 14 sessions, 501 file npy, 14 CSVs, 14 JSONs) đã được **sao lưu toàn vẹn 100% và cô lập** tại:
`debug/archive/vsl_p010_historical_reconstructed_2026-11-07/`

- **Số file đã sao lưu:** 544 files (~14.03 MB)
- **Kiểm định toàn vẹn:** 100% khớp mã băm SHA-256 (xem chi tiết tại `debug/archive/vsl_p010_historical_reconstructed_2026-11-07/ARCHIVE_MANIFEST.json`).
- **Master CSV lịch sử:** Đã được lưu trữ kèm bản snapshot 14 dòng của VSL-P010.

---

## 2. Chiến dịch Kiểm thử Hiện tại: `A01_1_T01`

- **Tên chiến dịch:** `A01_1_T01`
- **Mục tiêu:** Kiểm thử 15 lượt webcam (15 trials) với MediaPipe Holistic thực tế trên từ vựng Word Mode.
- **Quy tắc đặt tên session thư mục:**
  `A01_1_T01_<SanitizedWord>_trial_<XX>_<YYYY-MM-DD_HHMMSS>`
  *(Ví dụ: `A01_1_T01_Buoi_toi_trial_01_2026-11-11_223000`)*
- **Master Summary CSV:** `debug/words/trials_summary.csv`
  - Đã được khởi tạo sạch (clean header, 23 trường).
  - Tự động ghi nhận thông tin từng trial khi phiên kết thúc (`Stop Test`).
  - Phân tách rõ ràng định danh `session_id` có tiền tố `A01_1_T01_`.

---

## 3. Cấu trúc mỗi Session Folder

Mỗi session test độc lập gồm có:
```text
debug/words/A01_1_T01_<Word>_trial_<XX>_<Timestamp>/
├── session.json          # Metadata phiên test: campaign, target_word, trial, checkpoint, thời gian, summary
├── events.jsonl          # Nhật ký sự kiện chi tiết theo từng frame (SESSION_START, PREDICTION, SEQUENCE_RESET, SESSION_END...)
├── trial_summary.csv     # Tóm tắt kết quả (1 dòng) của riêng trial này
└── sequences/            # Chứa các file sequence landmark đầu vào (30, 75, 3) dạng float32
    ├── sequence_001.npy
    ├── sequence_002.npy
    └── ...
```

---

## 4. Bảo vệ Tính Toàn Vẹn Hệ Thống
- Không sửa đổi mã nguồn mô hình nhận diện (`backend/models/sstcn_model.py`).
- Không sửa đổi mapping từ điển (`backend/models/vsl_102_dictionary.py`).
- Không ảnh hưởng đến các mode khác (`letters`, `digits`, `sentence`).
- Tự động đóng phiên và ghi log khi chuyển phiên mới hoặc dừng test.

