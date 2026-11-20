# VSL Connect AI trên cloud

## Ứng dụng

Docker build React với Node 24 và phục vụ giao diện cùng API FastAPI trên một cổng. Trang dùng HTTPS của hosting để truy cập webcam. Trình duyệt chạy MediaPipe cho webcam, còn FastAPI xử lý chuỗi điểm khớp và suy luận. Dữ liệu đi kèm gồm 3.363 mục từ, 3.285 tệp chuyển động 3D, 11 mẫu so khớp QIPEDC (gồm `mấy giờ?` và `năn nỉ`), nhân vật Avatar và các trọng số hiện có. Checkpoint mặc định có 102 lớp, trong đó 63 lớp được dùng trong chế độ Từ vựng. Video QIPEDC được lấy theo yêu cầu.

```bash
docker build -t vsl-connect:cloud .
docker run --rm -p 127.0.0.1:7860:7860 vsl-connect:cloud
```

Truy cập <http://127.0.0.1:7860> để thử cả hai chiều dịch. Dùng `GET /healthz` để kiểm tra checkpoint đã nạp.

Mở liên kết triển khai, đăng nhập Render bằng GitHub và bấm **Deploy Blueprint**. Render đọc `render.yaml` từ nhánh `cloud-deploy`, tạo dịch vụ **Free** ở Singapore, kiểm tra `/healthz` và tự sinh mật khẩu `VSL_PREVIEW_PASSWORD`. Các commit được đẩy lên nhánh `cloud-deploy` sẽ tự kích hoạt một lượt build và deploy. [Triển khai trên Render Free](https://render.com/deploy?repo=https://github.com/ThanhHai151/SIGN_LANGUAGE/tree/cloud-deploy)

Sau lần deploy đầu, mở URL `.onrender.com` trong Dashboard. Đăng nhập bằng tên `vsl` và lấy mật khẩu từ mục **Environment** của dịch vụ. Bản thử nghiệm phục vụ một người sử dụng tại một thời điểm vì backend còn giữ trạng thái phiên trên tiến trình. Dữ liệu hiệu chỉnh và bộ đệm phát sinh trên Render Free mất khi dịch vụ thay container. Sau một thời gian rảnh, yêu cầu đầu tiên có thể chờ dịch vụ khởi động. [Giới hạn Render Free](https://render.com/docs/free)

## Dữ liệu huấn luyện

Chạy phần dưới trong môi trường Python 3.11 hoặc 3.12 trên Kaggle, Colab hoặc một runner có đủ dung lượng. Dùng thư mục làm việc có thể lưu qua các phiên. Xử lý từng nguồn để có thể tiếp tục sau khi hết thời gian phiên. VOYA tải từng tệp lớp rồi xóa tệp nén sau khi trích xuất. Với nguồn video lớn, có thể gắn sẵn thư mục dữ liệu để giảm dung lượng đĩa ghi. Kiểm tra số file bị bỏ qua trong `summary.json` và `errors.json` của từng nguồn.

| Nguồn | Cách lấy dữ liệu |
| --- | --- |
| SSTCN giáo trình | Sáu gói lưu sẵn trên [GitHub Releases](https://github.com/ThanhHai151/SIGN_LANGUAGE/releases/tag/sstcn-dataset-v1), có SHA-256 trong dự án. |
| VOYA | Kho công khai khoảng 56 GB. Script tải từng tệp lớp và trích về chuẩn 75 khớp. [Kateht/VOYA_VSL](https://huggingface.co/datasets/Kateht/VOYA_VSL) |
| ViSignLanguage | Script tải bản video ZIP và CSV từ [namwu/ViSignLanguage-Video](https://huggingface.co/datasets/namwu/ViSignLanguage-Video). |
| Kaggle/QIPEDC | Gắn hai bộ [VSL-Vietnamese Sign Languages](https://www.kaggle.com/datasets/aresusayhi/vsl-vietnamese-sign-languages) và [VSL Cropped](https://www.kaggle.com/datasets/nguyenanfms/vsl-vietnamese-sign-language-v2) trong notebook. Với bộ Cropped, truyền thư mục video theo nhãn. |
| VSL400 | 74.259 clip và 400 ký hiệu. Xin quyền tải theo Data Usage Agreement rồi gắn dữ liệu vào runner riêng tư. Tạo CSV từ metadata Zenodo gồm `path,label,signer_id,view,group`. Dùng chung `group` cho các góc quay của một lần ký hiệu. [VSL400 trên Zenodo](https://zenodo.org/records/17943574) |

```bash
python -m pip install -r backend/training/requirements-preprocessing.txt huggingface_hub

VSL_WORK=/path/to/vsl-work
python -m backend.training.fetch_sstcn_data --source github --extract --output "$VSL_WORK/sstcn"
python -m backend.training.cloud_dataset --work-dir "$VSL_WORK" ingest voya
python -m backend.training.cloud_dataset --work-dir "$VSL_WORK" ingest visign
python -m backend.training.cloud_dataset --work-dir "$VSL_WORK" ingest qipedc --input-dir /path/to/qipedc
python -m backend.training.cloud_dataset --work-dir "$VSL_WORK" ingest qipedc-cropped --input-dir /path/to/qipedc-cropped
python -m backend.training.cloud_dataset --work-dir "$VSL_WORK" ingest vsl400 \
  --input-dir /path/to/vsl400 --metadata /path/to/vsl400.csv
python -m backend.training.cloud_dataset --work-dir "$VSL_WORK" prepare --sstcn-raw "$VSL_WORK/sstcn/raw"
```

ViSignLanguage cũng nhận `--input-dir` và `--metadata` khi đã giải nén dữ liệu ở nơi khác. Bước `prepare` yêu cầu đủ nguồn và xuất `manifest.json` cùng `harmonization_report.json` với số mẫu, lớp giữ lại và số mẫu đã khử trùng. Bộ SSTCN cũ đóng góp 63 lớp từ cho mô hình từ vựng mới. Chữ cái và chữ số dùng các bộ nhận diện riêng.

### Train và đánh giá

Trên phiên có GPU, dùng môi trường PyTorch theo bản CUDA của runner. Cài thêm `numpy` và `matplotlib` theo `backend/training/requirements-sstcn.txt` nếu cần.

```bash
python -m backend.training.train_sstcn --data "$VSL_WORK/prepared" --output "$VSL_WORK/run" \
  --device cuda --amp --augment --augment-profile kvsl --class-balanced \
  --label-smoothing 0.1 --scheduler cosine \
  --init-from backend/models/weights/sign_sstcn_attention_model.pth
python -m backend.training.evaluate_sstcn --data "$VSL_WORK/prepared" \
  --checkpoint "$VSL_WORK/run/best.pth" --output "$VSL_WORK/evaluation" --device cuda
```

Kiểm tra số lớp có đủ train/validation/test, các lớp dễ nhầm và hiệu quả trên người ký hiệu chưa có trong dữ liệu huấn luyện trước khi đổi model. Nếu tập dữ liệu có lớp thiếu trong validation, kiểm tra lại nhóm người ký hiệu và bổ sung mẫu độc lập cho lớp đó. Để dùng checkpoint đã đạt kiểm thử, đặt file tại `backend/models/weights/sign_sstcn_unified_v2.pth` trong nguồn Docker và cấu hình `VSL_SSTCN_WEIGHTS=/app/backend/models/weights/sign_sstcn_unified_v2.pth` trên Render. Build lại image rồi đối chiếu số lớp và phiên bản preprocessing tại `/api/health`.
