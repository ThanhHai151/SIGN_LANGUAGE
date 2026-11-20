# Hướng dẫn Pipeline Đa nguồn & Nhận diện Nâng cao (`VSL Connect AI` v2)

Tài liệu tóm tắt cách sử dụng các nâng cấp đã tích hợp từ 3 dự án tham khảo (`KVSL`, `SignLang`, `Vietnamese-Sign-Language-Translation`) và 3 nguồn dataset VSL (Hugging Face, Kaggle/QIPEDC, Zenodo VSL400).

Để thử một từ QIPEDC ngay, chọn **Luyện bằng camera** trong tab **Từ Điển & Luyện Tập VSL**. Mỗi lượt ghi được so với video của mục từ đang chọn. Mẫu trích xuất lưu ở `backend/data/media_cache/practice/` và không được đưa vào dữ liệu train SSTCN.

## 1. Tính năng mới dùng được ngay khi chạy `./start.sh`
Trong tab **Camera → Văn bản**, chuyển sang chế độ **Từ**:
- **Lọc theo Chủ đề (`topic_id` — học từ KVSL):** Chọn chủ đề cụ thể để thu hẹp không gian dự đoán của SSTCN, tăng độ chính xác thực tế.
- **Chế độ Tự do (Adaptive Time Resampling):** Gửi kèm timestamp khung hình để backend nội suy đều về 30 frame độc lập với FPS webcam.
- **Chế độ Ghi theo nhịp 3s (Record-and-Check — học từ KVSL):** Đếm ngược 2s chuẩn bị → 3s ghi hình → chấm kết quả kèm Top-K dự đoán.
- **Hiệu chỉnh động tác của tôi (Personal Calibration — học từ LNT):** Nhập tên từ (từ có sẵn hay bị nhầm hoặc từ hoàn toàn mới), bấm **Ghi mẫu của tôi (3s)** 5–10 lần → lưu tại `backend/data/personal_calibration/`.

## 2. Fine-tune cá nhân ngay trên CPU (1 lệnh)
Sau khi ghi 5–10 mẫu cho các động tác của bạn:

```bash
python -m backend.training.finetune_personal \
  --base-checkpoint backend/models/weights/sign_sstcn_reconstructed_model.pth \
  --personal-dir backend/data/personal_calibration \
  --replay-data backend/data/sstcn/prepared \
  --output backend/models/weights/sign_sstcn_personal.pth \
  --epochs 15
```

Chạy ứng dụng với trọng số vừa tinh chỉnh:
```bash
VSL_SSTCN_WEIGHTS=backend/models/weights/sign_sstcn_personal.pth ./start.sh
```

## 3. Hợp nhất 3 nguồn Dataset & Train mở rộng nhiều từ vựng (Body-Centric v2 + KVSL Augmentation)

### Bước 1 — Nạp dữ liệu từ từng nguồn
```bash
# Nguồn 1a: Hugging Face Kateht/VOYA_VSL (161 file .npz theo lớp, mỗi file chứa nhiều mẫu)
python -m backend.training.ingest_voya_vsl \
  --output backend/data/vsl_unified/sources/voya_vsl

# Nguồn 1b: Hugging Face namwu/ViSignLanguage-Video (MP4)
python -m backend.training.ingest_videos \
  --source visign \
  --input-dir /path/to/ViSignLanguage-Video \
  --output backend/data/vsl_unified/sources/visign

# Nguồn 2: Kaggle/QIPEDC (tự động cắt đoạn active theo 3.285 file backend/data/motions/*.json)
python -m backend.training.ingest_videos \
  --source qipedc \
  --input-dir /path/to/kaggle_vsl_qipedc \
  --output backend/data/vsl_unified/sources/qipedc

# Nguồn 3: Zenodo VSL400 đa góc nhìn (front/left/right)
python -m backend.training.ingest_videos \
  --source vsl400 \
  --input-dir /path/to/VSL400 \
  --output backend/data/vsl_unified/sources/vsl400
```

### Bước 2 — Gộp, chuẩn hóa nhãn & Body-Centric Normalization v2
```bash
python -m backend.training.prepare_unified_dataset \
  --output backend/data/vsl_unified/prepared \
  --include-sstcn-words \
  --include-personal \
  --source-dir backend/data/vsl_unified/sources/voya_vsl \
  --source-dir backend/data/vsl_unified/sources/visign \
  --source-dir backend/data/vsl_unified/sources/qipedc \
  --source-dir backend/data/vsl_unified/sources/vsl400 \
  --preprocessing-version 2 \
  --overwrite
```

`manifest.json` ghi `split_group` và `signer_id` nếu nguồn có ID người ký hiệu. Các mẫu có chung người ký hiệu trong cùng một nguồn được giữ ở cùng split. Khi nguồn thiếu ID, pipeline chỉ có thể nhóm bằng mã bản ghi. Kiểm tra số mẫu từng lớp trong validation/test trước khi train để biết lớp nào có đủ dữ liệu đánh giá.

### Bước 3 — Đóng gói & Train trên Kaggle / Google Colab GPU
```bash
python -m backend.training.colab_train_sstcn \
  --data backend/data/vsl_unified/prepared \
  --bundle backend/data/vsl_unified/vsl_unified_gpu_bundle.zip
```
Làm theo lệnh in ra để train trên Colab/Kaggle GPU (tự bật `--augment --augment-profile kvsl --class-balanced --label-smoothing 0.1 --scheduler cosine --amp`), tải `best.pth` về đặt tại `backend/models/weights/sign_sstcn_unified_v2.pth` rồi khởi chạy:

```bash
VSL_SSTCN_WEIGHTS=backend/models/weights/sign_sstcn_unified_v2.pth ./start.sh
```
