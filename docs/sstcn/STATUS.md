# Trạng thái khôi phục SSTCN

Ngày thực hiện: 2026-10-30. Hướng dẫn chạy: [README.md](README.md).

## Đã bổ sung và kiểm chứng

| Phần | Trạng thái |
|---|---|
| Dataset | Khôi phục đủ 6 archive từ Drive trong báo cáo cũ, tổng 47.643 file skeleton, 102 lớp |
| Train SSTCN | Có trainer đúng kiến trúc đang triển khai, tách biệt với trainer MLP 84 lớp cũ |
| Video preprocessing | Chạy thực tế MediaPipe 0.10.21 trên MP4 chuyển từ GIF `chao`; đầu ra float32 `[3,30,75]`, không có NaN/Inf |
| Split | Chia mới có seed theo group video, loại trùng và augmentation của validation; manifest ghi từng mẫu |
| Mapping | Đủ `0..101`, khớp từ điển cũ; `UNKNOWN=102` không phải đầu ra model |
| Evaluation | Accuracy/top-5/precision/recall/F1, dự đoán từng mẫu, confusion matrix CSV/NPY/PNG |
| Checkpoint | Có best/last mới, trạng thái optimizer/scheduler/scaler/RNG và kiểm tra resume |
| Lịch sử | Đã giữ báo cáo DOCX, ảnh log training và ba GIF demo từ project cũ trong `legacy/` |

## Dataset đầy đủ đã chuẩn bị

| Split | Số mẫu |
|---|---:|
| Train | 38.818 |
| Validation | 1.407 |
| Test | 1.420 |

Không có mẫu lỗi. Loại 129 sequence trùng và 5.869 bản augmentation thuộc nhóm
validation; không tìm thấy sequence trùng chính xác giữa train và test.
Các con số này không chứng minh độc lập theo người thực hiện, vì nguồn không có ID người.

Từng mẫu và split: [`splits.csv`](../../backend/data/sstcn/splits.csv).
Thống kê/hash dữ liệu: [`dataset_summary.json`](../../backend/data/sstcn/dataset_summary.json).
Checksum của đủ 6 archive: [`sources.json`](../../backend/data/sstcn/sources.json).
Các archive được phân phối qua [GitHub Release `sstcn-dataset-v1`](https://github.com/ThanhHai151/SIGN_LANGUAGE/releases/tag/sstcn-dataset-v1),
không đưa thư mục dữ liệu giải nén 5,3 GB vào lịch sử Git.

## Run train đầy đủ đã hoàn tất

Run `backend/runs/sstcn_reconstructed_full` train từ khởi tạo ngẫu nhiên với toàn bộ
split trên: **12 epoch**, batch 32, seed 42, AdamW learning rate 0.001,
weight decay 0.0001, dropout 0.3, CUDA AMP. Không khởi tạo từ checkpoint lịch sử.
Checkpoint được chọn theo validation ở **epoch 11** (chỉ số trong file là 10).

Đánh giá cuối dùng float32, không loại mẫu confidence thấp:

| Metric | Checkpoint mới |
|---|---:|
| Validation accuracy (1.407 mẫu) | 85,7854% |
| Test accuracy (1.420 mẫu) | **88,6620%** |
| Test top-5 accuracy | 99,5070% |
| Test weighted F1 | 0,884439 |

Test accuracy cao hơn checkpoint lịch sử đo lại **2,0423 điểm phần trăm**.
Đây là kết quả trên split hiện có, chưa phải kiểm chứng trên người dùng/người ký hiệu mới.

Checkpoint mới được xuất sẵn:
[`sign_sstcn_reconstructed_model.pth`](../../backend/models/weights/sign_sstcn_reconstructed_model.pth),
kèm [metadata](../../backend/models/weights/sign_sstcn_reconstructed_model.json).
Checkpoint của epoch cuối cũng được giữ ở
[`sign_sstcn_reconstructed_last.pth`](../../backend/models/weights/sign_sstcn_reconstructed_last.pth).
Manifest chính xác của run được nén và lưu trong repo:
[`dataset_manifest.json.gz`](evaluation/sstcn_reconstructed_full/dataset_manifest.json.gz).
Model cũ vẫn là mặc định của ứng dụng; chọn bản mới bằng:

```bash
VSL_SSTCN_WEIGHTS=backend/models/weights/sign_sstcn_reconstructed_model.pth ./start.sh
```

Artifacts đã đưa vào repo:
[history](evaluation/sstcn_reconstructed_full/history.csv),
[log train](evaluation/sstcn_reconstructed_full/training_log.txt),
[learning curves](evaluation/sstcn_reconstructed_full/learning_curves.png),
[test metrics](evaluation/sstcn_reconstructed_full/test/metrics.json),
[confusion matrix](evaluation/sstcn_reconstructed_full/test/confusion_matrix.png),
[dự đoán từng mẫu](evaluation/sstcn_reconstructed_full/test/predictions.csv),
[model card](evaluation/sstcn_reconstructed_full/model_card.json).

`best.pth`, `last.pth` và manifest đầy đủ còn ở thư mục run local. Muốn tiếp tục
run này lên tổng 40 epoch:

```bash
python -m backend.training.train_sstcn \
  --data backend/data/sstcn/prepared \
  --output backend/runs/sstcn_reconstructed_full \
  --resume backend/runs/sstcn_reconstructed_full/last.pth \
  --epochs 40 --batch-size 32 --patience 5 --amp
```

Loss/accuracy trong log train có thể lệch nhẹ với báo cáo cuối do log dùng CUDA AMP,
còn evaluator dùng float32; checkpoint được chọn theo validation của trainer.
Sau khi xuất checkpoint mới, bộ 58 kiểm thử pipeline/API/chế độ nhận diện đã được
chạy lại thành công với chính checkpoint này qua `VSL_SSTCN_WEIGHTS`.

## Checkpoint đang được ứng dụng dùng

File: `backend/models/weights/sign_sstcn_attention_model.pth`.
SHA-256: `2a8ddcc4d06c4f23aa7910caf77371848034dcdf8f4749d134c5894764cf9697`.
File này giống hệt bản trong project SignLang cũ; nội dung còn `epoch=32`,
`loss=1.0744346380233765`, `model_state_dict`, `optimizer_state_dict`.
Các trường optimizer và kích thước tensor còn đọc được được lưu trong
[checkpoint_metadata.json](legacy/checkpoint_metadata.json).

Đánh giá argmax trực tiếp trên 1.420 skeleton trong `test_set.zip`, không loại mẫu
confidence thấp:

| Metric | Kết quả đo lại |
|---|---:|
| Accuracy | 86,6197% |
| Top-5 accuracy | 98,2394% |
| Weighted F1 | 0,859082 |

Chi tiết: [metrics.json](evaluation/sstcn_historical_test/metrics.json),
[confusion matrix](evaluation/sstcn_historical_test/confusion_matrix.png),
[từng dự đoán](evaluation/sstcn_historical_test/predictions.csv).

Đây là phép đo trên dữ liệu khôi phục, không chứng minh được checkpoint cũ chưa
từng thấy bộ test này trong quá trình tạo nó.

## Đợt train thử để kiểm chứng pipeline

Run `backend/runs/sstcn_reconstructed_baseline`, khởi tạo ngẫu nhiên, dùng riêng
`data_01` và `test_set`, seed 42, 3 epoch, AdamW 0.001, batch 8, CUDA AMP.
Có 7.283 mẫu train, 156 validation, 1.420 test; 21 bản trùng trong train được loại.
Checkpoint được chọn bằng validation, test accuracy sau run thử là **25,1408%**.
Kết quả này xác nhận luồng train/evaluate hoạt động; chưa đủ để thay model đang chạy.

Xem [history.csv](evaluation/sstcn_reconstructed_baseline/history.csv),
[learning curves](evaluation/sstcn_reconstructed_baseline/learning_curves.png),
[test metrics](evaluation/sstcn_reconstructed_baseline/test/metrics.json).

## Phân biệt với số liệu lịch sử

- README cũ ghi test accuracy **93,87%**.
- Báo cáo DOCX cũ ghi test accuracy **85,77%**, weighted F1 **0,8487**, trên 1.420 video.
- Ảnh log đính kèm DOCX ghi validation accuracy cuối **91,02%**, epoch 32.
- Phép đo mới trên checkpoint hiện tại cho **86,62% test accuracy**.

Các số liệu này không đồng nhất. Không gán số 93,87% cho checkpoint hiện tại.
Log cũ chỉ còn dạng ảnh; không tạo lại lịch sử train giả để lấp chỗ trống.

## Phần lịch sử chưa thể phục hồi chính xác

- Video gốc tạo nên các skeleton; các ZIP đã mở chứa skeleton, không phải video gốc.
  Link MP4 tham chiếu đầu tiên trong từ điển cũ không phân giải được DNS từ môi trường
  kiểm tra; chi tiết tại [source_access.json](source_access.json).
- Notebook/script và toàn bộ hyperparameter của lần tạo checkpoint cũ.
- Split train/validation lịch sử, ID người thực hiện, cách cắt video/lấy mẫu frame,
  normalization và augmentation chính xác trước khi xuất skeleton.
- Quan hệ chắc chắn giữa từng archive trên Drive và lần train checkpoint cũ.

Pipeline mới cung cấp quy trình thay thế, dữ liệu có nguồn, split có thể tái lập,
log và checkpoint có provenance cho các lần train từ thời điểm này.
