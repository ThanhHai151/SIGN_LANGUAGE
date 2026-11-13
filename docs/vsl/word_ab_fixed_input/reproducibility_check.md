# VSL-P010 — Kiểm Tra Tái Hiện Log (Reproducibility Check)

**Dự án:** Nhận diện Ngôn ngữ Ký hiệu Việt Nam (VSL)  
**Nhiệm vụ:** VSL-P010 — Đối chứng A/B SSTCN, Fixed-Input Replay  
**Phạm vi:** Chỉ đọc log và NPY; không sửa runtime code  
**Script:** `scratch/vsl_p010_step1_reproducibility.py`  
**Ngày thực hiện:** 07/11/2026

---

## 1. Mục tiêu

Xác minh rằng khi chạy lại inference trên đúng NPY tensor đã lưu, với đúng checkpoint gốc của từng session, kết quả Top-1 / Top-5 / confidence thu được **khớp hoàn toàn** với các PREDICTION event đã ghi trong `events.jsonl`. Nếu không khớp, phải dừng A/B và báo cáo bất nhất.

---

## 2. Phương pháp Replay

### 2.1 Pipeline đúng được dùng (theo `recognizer.py`)

```python
# Bước 1: Load NPY (đã được debugger lưu TRƯỚC khi preprocess thêm)
arr = np.load("sequences/sequence_XXX.npy")  # shape (30, 75, 3), float32

# Bước 2: Transpose sang định dạng SSTCN (giống recognizer.py:119-120)
tensor_seq = arr.transpose(2, 0, 1)      # (3, 30, 75)
tensor_seq = tensor_seq[np.newaxis, ...]  # (1, 3, 30, 75)
tensor_input = torch.tensor(tensor_seq, dtype=torch.float32)

# Bước 3: Words masking (recognizer.py:127-131)
masked_logits[:, :39] = -float('inf')    # Ẩn classes 0-38 (chữ cái + số)
probs = torch.softmax(masked_logits, dim=1)

# Bước 4: Top-5
top_k_probs, top_k_indices = torch.topk(probs, k=5)
```

### 2.2 Điểm quan trọng: NPY lưu ở giai đoạn nào?

Theo `word_debugger.record_prediction()`, NPY được lưu từ tham số `arr_input`, chính là `arr` đã được `predict_sequence()` nhận vào **sau bước reshape (30,75,3) nhưng TRƯỚC khi transpose**. Đây là tensor thô nguyên bản từ frontend, chưa qua bất kỳ normalization nào thêm. Replay lại từ NPY với đúng bước transpose tái hiện 100% input của model.

### 2.3 Checkpoint mapping

| Session checkpoint_type | SHA (8 ký tự đầu) | File |
|---|---|---|
| `historical` | `2a8ddcc4` | `sign_sstcn_attention_model.pth` |
| `reconstructed_best` | `370c5edc` | `sign_sstcn_reconstructed_model.pth` |

---

## 3. Kết quả Tái Hiện

### Tóm tắt tổng thể

| Chỉ số | Giá trị |
|---|---|
| Tổng sessions | 14 |
| Tổng NPY replay | **501** |
| NPY không tìm thấy | 0 |
| Shape sai (không phải 30×75×3) | 0 |
| Non-finite values (NaN/Inf) | 0 |
| **Top-1 class khớp** | **501/501 (100.0%)** |
| **Top-5 order khớp** | **501/501 (100.0%)** |
| Confidence diff ≤ 0.001 | 501/501 |
| Confidence diff ≤ 0.01 | 501/501 |
| Sai khác Top-1 | **0** |

> Log tái hiện hoàn toàn. 501/501 NPY replay cho Top-1 và Top-5 khớp chính xác 100% với PREDICTION events trong `events.jsonl`. Confidence difference = 0 (floating-point identical). Điều kiện "đủ điều kiện tiến hành A/B" được thỏa mãn.

---

## 4. Phân tích NPY — Shape, dtype, Finite

| Kiểm tra | Kết quả |
|---|---|
| Tất cả NPY đều có shape `(30, 75, 3)` | ✅ 501/501 |
| dtype = float32 | ✅ 501/501 |
| Tất cả giá trị finite (không NaN, không Inf) | ✅ 501/501 |
| Joint order: Pose (0-32), LH (33-53), RH (54-74) | ✅ Xác nhận theo `sstcn_model.py` L14-16 |

---

## 5. Xác minh Checkpoint Identity

Theo cơ chế VSL-P008 (hash trong `sstcn_model.py:_inspect_checkpoint_metadata`):

| Checkpoint | Full SHA-256 | Epoch | Cấu trúc dict |
|---|---|---|---|
| `historical` | `2a8ddcc4...` | 32 | `epoch, model_state_dict, optimizer_state_dict, loss` |
| `reconstructed_best` | `370c5edc...` | 10 | Schema v1: `model_state_dict, class_names, preprocessing, model_config, run_config, ...` |
| `reconstructed_last` | `301dc6bc...` | 11 | Schema v1 (tương tự reconstructed_best) |

Không có sự kiện `CHECKPOINT_MISMATCH` nào trong 14 sessions đủ điều kiện phân tích.

---

## 6. Kết luận

**Log tái hiện được 100%.** Việc tiến hành đối chứng A/B trên fixed-input là hợp lệ — cùng NPY tensor, cùng preprocess logic, hai model khác nhau.

*Script: `scratch/vsl_p010_step1_reproducibility.py`*  
*Dữ liệu: `_reproducibility_results.json`*

