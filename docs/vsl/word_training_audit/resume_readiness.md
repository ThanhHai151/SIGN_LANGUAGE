# XÁC MINH KHẢ NĂNG TIẾP TỤC HUẤN LUYỆN (RESUME READINESS AUDIT)

**Dự án:** Nhận diện Ngôn ngữ Ký hiệu Việt Nam (VSL)  
**Mã nhiệm vụ:** VSL-P011 — Công việc 4  
**Phạm vi:** Kiểm tra tính toàn vẹn trạng thái của `sign_sstcn_reconstructed_last.pth`, `sign_sstcn_attention_model.pth` và đánh giá các kịch bản tiếp tục huấn luyện (Resume).  
**Ngày thực hiện:** 07/11/2026  

---

## 1. Kiểm tra trạng thái bên trong của Reconstructed Last

Đã thực hiện kiểm tra cấu trúc nhị phân của tệp [`backend/models/weights/sign_sstcn_reconstructed_last.pth`](file:///E:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/models/weights/sign_sstcn_reconstructed_last.pth):

| Trường trạng thái (State Key) | Hiện diện | Chi tiết kiểm chứng |
| :--- | :---: | :--- |
| `schema_version` | ✅ Có | Giá trị `1` (chuẩn cấu trúc mới có định danh). |
| `epoch` | ✅ Có | Giá trị `11` (chỉ số 0-based; đã hoàn thành 12 epoch từ 0 đến 11). |
| `model_state_dict` | ✅ Có | Đủ 100% trọng số của `SSTCN_Attention` (102 lớp, 75 khớp). |
| `optimizer_state_dict` | ✅ Có | AdamW optimizer state: chứa `state` momentum $m_t, v_t$ của toàn bộ tham số, `lr = 0.001`, `weight_decay = 0.0001`. |
| `scheduler_state_dict` | ✅ Có | `ReduceLROnPlateau`: `best = 0.8571`, `cooldown_counter = 0`, `num_bad_epochs = 1`, `patience = 3`, `factor = 0.5`. |
| `scaler_state_dict` | ✅ Có | `GradScaler` cho Mixed Precision (CUDA AMP): chứa scale factor hiện tại và thông số unscale. |
| `torch_rng_state` | ✅ Có | Kiểu `torch.ByteTensor` (kích thước chuẩn 5.056 bytes) lưu hạt giống sinh số ngẫu nhiên CPU. |
| `cuda_rng_state` | ✅ Có | Mảng trạng thái sinh số ngẫu nhiên GPU CUDA (1 GPU). |
| `history` | ✅ Có | Mảng danh sách 12 từ điển ghi lại đầy đủ thông số của cả 12 epoch trước đó. |
| `best_score` | ✅ Có | `[0.8571428571428571, -0.4710657465368954]` (thu được tại epoch 10/11). |
| `bad_epochs` | ✅ Có | Giá trị `1`. |
| `dataset_manifest_sha256` | ✅ Có | Chuỗi băm SHA-256: `e3340c0d55b27dce703a1575d721c339c73609dc19ad730d616f027099fb3920`. |
| `class_names` | ✅ Có | Đủ 102 tên lớp từ `"A"` đến `"Bộ đội"`. |
| `run_config` | ✅ Có | Đủ toàn bộ tham số chạy: seed 42, batch size 32, dropout 0.3, lr 0.001, amp True, counts (train 38818, val 1407, test 1420). |

---

## 2. Kiểm tra tính toàn vẹn của Dataset Manifest & Cấu hình Split

1. **Dataset Manifest Check:**
   - Trong checkpoint: Yêu cầu manifest có mã băm SHA-256 là `e3340c0d55b27dce703a1575d721c339c73609dc19ad730d616f027099fb3920`.
   - Trong repo: Tệp [`docs/sstcn/evaluation/sstcn_reconstructed_full/dataset_manifest.json.gz`](file:///E:/NAM4/SEMINAR/SIGN_LANGUAGE/docs/sstcn/evaluation/sstcn_reconstructed_full/dataset_manifest.json.gz) đã được giải nén kiểm tra bằng mã băm SHA-256:
     ```python
     # Kết quả chạy thực tế:
     Uncompressed manifest SHA256: e3340c0d55b27dce703a1575d721c339c73609dc19ad730d616f027099fb3920
     Matches: True
     ```
   - **Đánh giá:** File manifest lưu trữ trong tài liệu khớp chính xác 100% với yêu cầu của checkpoint `last.pth`.
2. **Cấu hình Phân chia Split:**
   - Tập dữ liệu được phân chia có seed cố định (`seed: 42`, `val_fraction: 0.15`).
   - Phương pháp phân chia: `stratified_source_video_groups` (nhóm các biến thể augmentation cùng video gốc vào cùng một split để chống rò rỉ dữ liệu).
   - Tệp phân chia đầy đủ từng dòng: [`backend/data/sstcn/splits.csv`](file:///E:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/data/sstcn/splits.csv) (7 MB, 41.645 dòng).

---

## 3. Đánh giá 3 kịch bản tiếp tục huấn luyện (Resume Scenarios)

### Kịch bản 1: Tái lập chính xác tuyệt đối (Exact Resume)
*Tiếp tục huấn luyện từ Epoch 12 lên Epoch 40 với cùng chuỗi số ngẫu nhiên, optimizer momentum và LR scheduler.*

- **Khả thi về mặt thuật toán & mã nguồn:**
  - Hàm `train()` trong [`backend/training/train_sstcn.py`](file:///E:/NAM4/SEMINAR/SIGN_LANGUAGE/backend/training/train_sstcn.py) (dòng 143 – 163) được thiết kế chuyên biệt cho việc này:
    ```python
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        if ckpt.get("dataset_manifest_sha256") != digest or ckpt.get("class_names") != CLASS_NAMES:
            raise ValueError("Resume requires the same prepared dataset and mapping")
        # Phục hồi model, optimizer, scheduler, scaler, RNG...
        torch.set_rng_state(ckpt["torch_rng_state"])
        if device.type == "cuda" and ckpt.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
    ```
- **RÀO CẢN DUY NHẤT HIỆN TẠI TRÊN MÁY TÍNH:**
  - Thư mục dữ liệu chuẩn bị `backend/data/sstcn/prepared/` (chứa `sequences.npy` kích thước ~1,1 GB) **chưa được tạo trên máy cục bộ**.
  - **Điều kiện cần để chạy được:**
    1. Tải 6 archive ZIP từ GitHub Release (~1,67 GB) qua `python -m backend.training.fetch_sstcn_data --extract`.
    2. Chạy `python -m backend.training.prepare_sstcn --seed 42 --val-fraction 0.15` để sinh thư mục `backend/data/sstcn/prepared/`.
    3. Tạo thư mục output `backend/runs/sstcn_reconstructed_full/`, đặt `sign_sstcn_reconstructed_last.pth` thành `last.pth` và `sign_sstcn_reconstructed_model.pth` thành `best.pth`.
    4. Chạy lệnh:
       ```bash
       python -m backend.training.train_sstcn \
         --data backend/data/sstcn/prepared \
         --output backend/runs/sstcn_reconstructed_full \
         --resume backend/runs/sstcn_reconstructed_full/last.pth \
         --epochs 40 --batch-size 32 --patience 5 --amp
       ```

### Kịch bản 2: Tiếp tục tương đối (Approximate Resume)
*Tiếp tục huấn luyện nhưng thay đổi một vài siêu tham số như batch size, learning rate hoặc tắt AMP.*

- **Khả thi:** **KHÔNG KHUYẾN KHÍCH VÀ BỊ CHẶN BỞI CODE.**
- **Lý do:** Mã nguồn `train_sstcn.py` (dòng 147 – 149) có điều kiện kiểm tra nghiêm ngặt:
  ```python
  for key in ("seed", "batch_size", "dropout", "lr", "weight_decay", "amp", "augment"):
      if ckpt["run_config"][key] != config[key]:
          raise ValueError(f"Resume configuration differs: {key}")
  ```
  Nếu cố tình đổi batch size hay LR khi dùng cờ `--resume`, mã nguồn sẽ chủ động văng lỗi `ValueError` để bảo toàn tính toàn vẹn của thực nghiệm khoa học.

### Kịch bản 3: Khởi động ấm từ mô hình lịch sử (Warm-start Historical)
*Nạp trọng số của `sign_sstcn_attention_model.pth` (epoch 32) làm điểm khởi đầu, rồi huấn luyện tiếp trên tập dữ liệu mới.*

- **Khả thi về mặt mã nguồn:**
  - Code `train_sstcn.py` hỗ trợ thông qua cờ `--init-from`:
    ```python
    if args.init_from:
        initial = torch.load(args.init_from, map_location="cpu", weights_only=True)
        model.load_state_dict(initial.get("model_state_dict", initial), strict=True)
    ```
- **Đặc điểm kỹ thuật:**
  - Trọng số nơ-ron được kế thừa từ Historical (epoch 32).
  - Tuy nhiên, Optimizer, Scheduler, Scaler và bộ đếm Epoch sẽ **bắt đầu lại từ Epoch 0**.
- **Rủi ro khoa học:**
  - Chúng ta không có thông tin chính xác về các video đã dùng để huấn luyện Historical ban đầu. Rất có thể một số video trong tập test chuẩn hiện tại đã từng nằm trong tập train của Historical trong quá khứ. Việc warm-start có thể làm mất đi tính minh bạch về ranh giới kiểm thử (train/test leakage).

---

## 4. Bảng tổng kết mức độ sẵn sàng Resume

| Tiêu chí đánh giá | Trạng thái | Đánh giá kỹ thuật |
| :--- | :---: | :--- |
| **Tính đầy đủ của Checkpoint Last** | ⭐ 100% | Có đủ Model, Optimizer, Scheduler, Scaler, RNG state, History, Manifest Hash. |
| **Tính sẵn sàng của Mã nguồn Trainer** | ⭐ 100% | `train_sstcn.py` có sẵn logic resume chuẩn xác, deterministic. |
| **Dữ liệu huấn luyện cục bộ (1,67 GB)** | ❌ 0% | Thư mục `prepared/` và `raw/` chưa có trên máy; cần tải qua internet nếu muốn chạy. |
| **Ý nghĩa kỹ thuật đối với độ chính xác Webcam** | ⚠️ Rất thấp | Train tiếp từ epoch 12 lên 40 chỉ làm tăng điểm benchmark offline; **không thể khắc phục được lỗi domain gap giữa Video tĩnh và Webcam sliding window**. |

