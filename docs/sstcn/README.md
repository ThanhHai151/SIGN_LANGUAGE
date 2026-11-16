# Pipeline SSTCN 102 lớp

Pipeline này được xây dựng lại từ kiến trúc đang chạy và dữ liệu khôi phục được.
Đây là quy trình mới có thể tái lập; không phải bản phục hồi nguyên vẹn của script train lịch sử.
Kết quả đo thực tế và phần chưa xác minh nằm trong [STATUS.md](STATUS.md).

## Dữ liệu và quy ước

- Mapping cố định: [`backend/data/sstcn/classes.json`](../../backend/data/sstcn/classes.json).
  Có 102 đầu ra `0..101`; `UNKNOWN` với mã 102 là trạng thái từ chối dự đoán, không phải lớp train.
- Input: `float32 [N,3,30,75]`; 75 khớp theo thứ tự **33 pose, 21 tay trái, 21 tay phải**.
- Dữ liệu `.npy` cũ chứa `{"keypoints": array, "label": int}`. Bộ đọc chỉ cho phép
  các kiểu NumPy cần thiết; không gọi `np.load(..., allow_pickle=True)` trên dữ liệu tải về.
- Skeleton đã đủ 30 frame được giữ nguyên tọa độ; không chuẩn hóa lại theo cổ tay/vai.
- Với video mới: BGR → RGB → MediaPipe Holistic, thiếu khớp thì điền 0,
  lấy mỗi 2 frame rồi nội suy tuyến tính về 30 frame. Đây là lựa chọn được ghi nhận
  của pipeline mới; cách lấy mẫu video trong lần train cũ chưa được xác minh.
- Video phải được cắt thành một cử chỉ. Trích toàn bộ một video dài có nhiều cử chỉ
  thành một sequence sẽ làm mất ý nghĩa của nhãn.

## 1. Tải và kiểm tra dữ liệu thật

Chạy các lệnh từ thư mục gốc repo:

```bash
python -m backend.training.fetch_sstcn_data --extract
```

Script mặc định ưu tiên [GitHub Releases — SSTCN dataset v1](https://github.com/ThanhHai151/SIGN_LANGUAGE/releases/tag/sstcn-dataset-v1),
rồi thử nguồn Drive gốc nếu GitHub không truy cập được. Release chứa đủ 6 ZIP gốc
(tổng 1.793.877.824 byte, khoảng 1,67 GiB), kèm `SHA256SUMS.txt` và `sources.json`.
Chỉ tải từ GitHub: thêm `--source github`; chỉ tải từ Drive: thêm `--source drive`.

Nguồn khôi phục ban đầu: [vsl_skeleton_data trên Google Drive](https://drive.google.com/drive/folders/132sA_E7PMwnuC_WZpFi50I--Ncbo9LwX).
Link được tìm trong báo cáo cũ [test_set_result_report_nhom3.docx](legacy/test_set_result_report_nhom3.docx).
Danh sách ID và checksum nằm trong [`sources.json`](../../backend/data/sstcn/sources.json).
Script kiểm tra checksum, chặn đường dẫn ZIP không hợp lệ và ghi `download_receipt.json`.
Dataset đã khôi phục và split cụ thể của lần train mới được thống kê tại
[`dataset_summary.json`](../../backend/data/sstcn/dataset_summary.json) và
[`splits.csv`](../../backend/data/sstcn/splits.csv).

Chỉ khôi phục bộ test:

```bash
python -m backend.training.fetch_sstcn_data --only test_set.zip --extract
```

Archive và dữ liệu giải nén ở `backend/data/sstcn/archives/` và `raw/`, được bỏ qua bởi git.
Git lưu code, mapping, split, checkpoint và báo cáo; các ZIP lớn nằm trong Releases.
Metadata giữ nguyên nguồn gốc và thông tin giấy phép chưa được nguồn gốc chỉ định.
Không cần credentials khi tải các asset công khai.

## 2. Chuẩn bị train/validation/test

```bash
python -m pip install -r backend/training/requirements-sstcn.txt
python -m backend.training.prepare_sstcn --seed 42 --val-fraction 0.15
```

Kết quả ở `backend/data/sstcn/prepared/`:

| File | Nội dung |
|---|---|
| `sequences.npy` | Array có thể mmap, `[N,3,30,75]` |
| `labels.npy` | Nhãn số nguyên, `[N]` |
| `classes.json` | Mapping 102 lớp |
| `manifest.json` | Từng mẫu, nhãn, đường dẫn nguồn, group, split, hash, bản trùng/mẫu lỗi, thống kê lớp |

`test_set` của nguồn được giữ làm test. Các file `sign_XXX_vid_YY_ZZ.npy`
có cùng `XXX/YY` được gom vào một nhóm trước khi chia train/validation;
`ZZ` là biến thể augmentation. Việc nhóm áp dụng xuyên archive một cách bảo thủ.
Validation/test chỉ giữ bản `_00`; các bản augmentation của nhóm validation bị loại.
Việc xem `_00` là bản gốc được suy ra từ quy ước filename; script tạo các archive
lịch sử chưa được khôi phục để đối chiếu hoàn toàn.
Chia train/validation theo lớp, có seed, không dùng test để chọn checkpoint.

Các sequence trùng chính xác được loại trước khi chia. Video trùng dưới tên khác
được gom cùng group. Nếu phát hiện bản trùng giữa train và test, bản test được giữ
và toàn bộ group train tương ứng, gồm các bản augmentation, bị loại.
Mẫu trùng có nhãn khác nhau gây lỗi để tránh train dữ liệu mâu thuẫn.
Không có ID người thực hiện, nên chưa thể chứng minh split độc lập theo người.
Seed này tạo split mới, không khôi phục được split train/validation lịch sử.

Đầu ra đã tồn tại chỉ được thay khi có `--overwrite`, và phải là thư mục dataset
do pipeline tạo. Để thử nghiệm một phần archive, dùng thư mục đầu ra riêng:

```bash
python -m backend.training.prepare_sstcn \
  --include-archives data_01 test_set \
  --output backend/data/sstcn/prepared_baseline
```

## 3. Train và tiếp tục train

```bash
python -m backend.training.train_sstcn \
  --data backend/data/sstcn/prepared \
  --output backend/runs/sstcn_full \
  --epochs 40 --batch-size 8 --seed 42 --amp
```

Model giữ nguyên kiến trúc `SSTCN_Attention` 102 lớp để tương thích ứng dụng.
Trainer dùng AdamW, giảm learning rate theo validation loss, gradient clipping,
early stopping và chọn checkpoint theo validation accuracy rồi validation loss.
`--amp` dùng mixed precision khi có CUDA. CPU chạy được khi bỏ cờ này.
Dữ liệu nguồn đã có augmentation; `--augment` chỉ dùng khi muốn tăng cường thêm
trực tuyến, và chỉ áp dụng trên train.

Tiếp tục từ `last.pth` vào cùng thư mục run; `--epochs` là tổng số epoch:

```bash
python -m backend.training.train_sstcn \
  --data backend/data/sstcn/prepared \
  --output backend/runs/sstcn_full \
  --resume backend/runs/sstcn_full/last.pth \
  --epochs 60 --batch-size 8 --seed 42 --amp
```

Khởi tạo từ trọng số cũ, tạo optimizer/run mới:

```bash
python -m backend.training.train_sstcn \
  --output backend/runs/sstcn_finetune \
  --init-from backend/models/weights/sign_sstcn_attention_model.pth \
  --lr 0.0001 --epochs 15 --amp
```

Fine-tune từ checkpoint lịch sử không chứng minh được đánh giá độc lập: chưa biết
chính xác checkpoint đó từng thấy những video nào. Train từ đầu có provenance mới rõ ràng hơn.

Mỗi run lưu:

- `best.pth`, `last.pth`: model, optimizer, scheduler, scaler, RNG, epoch, history,
  mapping, preprocessing và hash manifest. Không ghi đè checkpoint của ứng dụng.
- `run_config.json`, `dataset_manifest.json`, `classes.json`: cấu hình, phiên bản,
  seed, commit, từng mẫu và nguồn khởi tạo.
- `history.csv`, `learning_curves.png`: loss/accuracy theo epoch.
- `validation/`, `test/`: báo cáo cuối cho checkpoint được chọn bằng validation.

## 4. Đánh giá và replay

```bash
python -m backend.training.evaluate_sstcn \
  --checkpoint backend/models/weights/sign_sstcn_attention_model.pth \
  --data backend/data/sstcn/prepared \
  --output backend/runs/sstcn_original_evaluation --split test
```

Xuất `metrics.json` (accuracy, top-5, macro/weighted precision/recall/F1),
`per_class.csv`, `predictions.csv`, `confusion_matrix.csv/.npy/.png`.
Hàng của confusion matrix là nhãn thật; cột là dự đoán; ảnh chuẩn hóa theo hàng.

```bash
python -m backend.training.predict_sstcn \
  backend/data/sstcn/raw/test_set/test_set/sign_000/sign_000_vid_01_00.npy
```

Replay trả top-5 và áp dụng confidence threshold mặc định 0.7.
Accuracy của evaluator dùng argmax, không loại mẫu confidence thấp.

Để dùng checkpoint mới trong ứng dụng sau khi xem báo cáo:

```bash
VSL_SSTCN_WEIGHTS=backend/runs/sstcn_full/best.pth \
  python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Nếu đường dẫn checkpoint được chỉ định bị lỗi, loader báo lỗi thay vì tự suy luận
với trọng số ngẫu nhiên hoặc lặng lẽ chuyển sang checkpoint khác.

Checkpoint mới đã được train 12 epoch và xuất tại
`backend/models/weights/sign_sstcn_reconstructed_model.pth` (test accuracy 88,66%).
Lệnh dùng ngay và lệnh resume đúng cấu hình của run này nằm trong [STATUS.md](STATUS.md).

## 5. Preprocessing từ video mới

Các dependency được ghim của MediaPipe Holistic cũ cần môi trường Python 3.11–3.12
riêng vì phụ thuộc NumPy 1.x:

```bash
python3 -m venv .venv-sstcn
.venv-sstcn/bin/python -m pip install -r backend/training/requirements-preprocessing.txt
```

Trên Windows dùng `.venv-sstcn\Scripts\python.exe` thay cho đường dẫn `bin/python`.

Tổ chức video:

```text
videos/
  sign_000/clip_001.mp4
  sign_000/clip_002.mp4
  sign_001/clip_001.mp4
  ...
  test_set/sign_000/held_out_clip.mp4
```

Mỗi lớp cần ít nhất hai video nguồn để tạo train/validation. Những clip cắt từ
cùng một video nguồn cần được đổi tên theo cùng group `sign_XXX_vid_YY_ZZ`, hoặc
được gom trước; script không thể tự biết hai filename khác nhau thuộc cùng video/người.

```bash
.venv-sstcn/bin/python -m backend.training.prepare_sstcn \
  --input /path/to/videos --output backend/data/sstcn/prepared_videos
```

Preprocessing chỉ cần NumPy, OpenCV và MediaPipe. Replay video còn cần PyTorch:

```bash
.venv-sstcn/bin/python -m pip install -r backend/training/requirements-sstcn.txt
.venv-sstcn/bin/python -m backend.training.predict_sstcn /path/to/gesture.mp4
```

Các GIF demo cũ được giữ trong [legacy/](legacy/). Có thể chuyển thành MP4 để kiểm tra luồng:

```bash
ffmpeg -i docs/sstcn/legacy/chao.gif -c:v libx264 -pix_fmt yuv420p /tmp/chao.mp4
```

## 6. Kiểm thử

```bash
python -m unittest backend.tests.test_sstcn_pipeline -v
```

Kiểm tra định dạng/nội suy, bộ đọc dữ liệu cũ, ZIP, chia group có seed,
loại trùng xuyên test, toàn vẹn dữ liệu, augmentation giữ khớp bị thiếu,
metric confusion matrix, lỗi checkpoint và kết quả resume bằng train liên tục trên CPU.

Lần kiểm chứng khi khôi phục: **58 kiểm thử đạt**, gồm 12 kiểm thử pipeline và các
kiểm thử API/chế độ nhận diện liên quan; xem [validation.json](validation.json).
