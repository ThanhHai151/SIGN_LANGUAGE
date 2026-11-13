# VSL-P010 — Hướng Dẫn Chạy Lại (Run Instructions)

**Phạm vi:** Chỉ chạy công cụ phân tích; không tác động hệ thống runtime, không sửa weights, không sửa pipeline nhận diện.

---

## Yêu cầu

- Python environment: `.venv` trong thư mục dự án
- Dữ liệu: `debug/words/` phải có đủ 14 session với `events.jsonl`, `session.json`, `sequences/*.npy`
- Checkpoints phải tồn tại tại:
  - `backend/models/weights/sign_sstcn_attention_model.pth` (SHA: `2a8ddcc4...`)
  - `backend/models/weights/sign_sstcn_reconstructed_model.pth` (SHA: `370c5edc...`)
  - `backend/models/weights/sign_sstcn_reconstructed_last.pth` (SHA: `301dc6bc...`)

---

## Chạy từng bước

**Bước 0: Survey tổng quan sessions**
```powershell
Set-Location "E:\NAM4\SEMINAR\SIGN_LANGUAGE"
.venv\Scripts\python.exe scratch\vsl_p010_survey.py
```

**Bước 1: Kiểm tra tái hiện log**
```powershell
.venv\Scripts\python.exe scratch\vsl_p010_step1_reproducibility.py
```
Output: `docs/vsl/word_ab_fixed_input/_reproducibility_results.json`

**Bước 2: A/B Replay trên tất cả NPY**
```powershell
.venv\Scripts\python.exe scratch\vsl_p010_step2_ab_replay.py
```
Output:
- `docs/vsl/word_ab_fixed_input/replay_per_window.csv`
- `docs/vsl/word_ab_fixed_input/replay_by_source_and_word.csv`

**Bước 3: Pose & Temporal Audit**
```powershell
.venv\Scripts\python.exe scratch\vsl_p010_step3_pose_audit.py
```
Output:
- `docs/vsl/word_ab_fixed_input/_pose_temporal_audit_raw.csv`
- `docs/vsl/word_ab_fixed_input/_pose_audit_summary.json`

---

## Lưu ý

- **Không cần khởi động backend hay frontend.** Các script chạy độc lập hoàn toàn.
- **Không sửa bất kỳ file runtime nào** (`backend/app.py`, `backend/vision/recognizer.py`, `src/`, v.v.).
- **Không gọi** `model.train()`, `optimizer.step()` hay bất kỳ bước cập nhật weights nào.
- Nếu thêm session mới vào `debug/words/`, chạy lại từ Bước 0 để cập nhật toàn bộ.
- Thời gian chạy ước tính: Bước 1 ~15s, Bước 2 ~30s, Bước 3 ~10s (CPU, không GPU).

---

## Cấu trúc Output

```
docs/vsl/word_ab_fixed_input/
├── reproducibility_check.md        ← Báo cáo tái hiện log
├── replay_per_window.csv           ← Kết quả A/B từng NPY (501 rows)
├── replay_by_source_and_word.csv   ← Thống kê theo nhóm
├── pose_temporal_audit.md          ← Phân tích Pose & temporal
├── conclusion.md                   ← Kết luận A/B
├── run_instructions.md             ← File này
├── _reproducibility_results.json   ← Raw data Step 1
├── _pose_temporal_audit_raw.csv    ← Raw data Step 3
└── _pose_audit_summary.json        ← Summary Step 3
```

