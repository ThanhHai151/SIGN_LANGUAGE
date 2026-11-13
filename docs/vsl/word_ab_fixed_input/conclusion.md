# VSL-P010 — Kết Luận Đối Chứng A/B Fixed-Input

**Dự án:** Nhận diện Ngôn ngữ Ký hiệu Việt Nam (VSL)  
**Nhiệm vụ:** VSL-P010 — Đối chứng A/B SSTCN, Fixed-Input Replay  
**Script:** `scratch/vsl_p010_step2_ab_replay.py`  
**Ngày:** 07/11/2026

---

## 1. Điều kiện tiên quyết — Đã thỏa mãn

| Điều kiện | Trạng thái |
|---|---|
| Log tái hiện được 100% | ✅ 501/501 |
| NPY có shape (30,75,3), finite | ✅ 501/501 |
| Không có CHECKPOINT_MISMATCH trong sessions | ✅ |
| A/B dùng 2 instance model tách biệt, `model.eval()` | ✅ |
| Không gọi `model.train()` hay `optimizer.step()` | ✅ |

---

## 2. Tổng quan Dữ liệu

### 2.1 Phân bổ sessions và NPY

| Checkpoint | Sessions | NPY Windows | Landmark Source |
|---|---|---|---|
| **historical** (SHA `2a8ddcc4`, epoch 32) | 7 | 171 | 171 holistic_raw |
| **reconstructed_best** (SHA `370c5edc`, epoch 10) | 7 | 330 | 62 holistic_raw + 268 fallback |
| **Tổng (không trùng lặp)** | 14 | 501 | 233 holistic + 268 fallback |

### 2.2 Từ mục tiêu và phân bổ windows

| Từ | Total windows | Class |
|---|---|---|
| Bạn | 204 | 60 |
| Thế nào | 120 | 63 |
| Tên là gì | 177 | 61 |

---

## 3. Kết Quả A/B — Toàn Bộ 501 Tensor

> **Lưu ý đọc số liệu:** 501 windows từ sliding window có bước nhảy 6 frame; các windows chồng lấn, không độc lập. Các con số phản ánh **window-level accuracy** (chỉ báo mô tả), không phải trial accuracy chuẩn.

### 3.1 Top-1 Accuracy trên 501 tensor

| Checkpoint | Top-1 Correct | Accuracy |
|---|---|---|
| **Historical** | **43 / 501** | **8.6%** |
| Reconstructed Best | 29 / 501 | 5.8% |

### 3.2 Target in Top-5 trên 501 tensor

| Checkpoint | In Top-5 | Rate |
|---|---|---|
| **Historical** | **189 / 501** | **37.7%** |
| Reconstructed Best | 149 / 501 | 29.7% |

### 3.3 Phân tích Paired (cùng tensor)

| So sánh | Số windows |
|---|---|
| Historical đúng, Recon sai (hist_better) | **22** |
| Recon đúng, Hist sai (recon_better) | **8** |
| Cả hai cùng đúng | 21 |
| Cả hai cùng sai | **450** |
| Hai model đồng ý Top-1 | 189 / 501 (37.7%) |

---

## 4. Kết Quả theo Landmark Source

| Landmark Source | n | Historical Top-1% | Recon Top-1% |
|---|---|---|---|
| `holistic_raw` | 233 | **15.0%** | **10.3%** |
| `hands_fallback_pose` | 268 | **3.0%** | **1.9%** |

**Nhận xét:** Cả hai model đều giảm accuracy rõ rệt trên tensor `hands_fallback_pose` (3.0% vs 15.0% cho Historical; 1.9% vs 10.3% cho Recon). Điều này nhất quán với sự sai lệch tọa độ Pose (29/33 joints bất động, confirmed bằng NPY).

---

## 5. Kết Quả theo Từ Mục Tiêu

| Từ | n | Historical Top-1% | Recon Top-1% | Recon win? |
|---|---|---|---|---|
| Bạn | 204 | 6.9% | 1.5% | ❌ Hist tốt hơn |
| Thế nào | 120 | 14.2% | **19.2%** | ✅ Recon tốt hơn |
| Tên là gì | 177 | 6.8% | 1.7% | ❌ Hist tốt hơn |

---

## 6. Phân tích Trường hợp Đặc biệt: "Tên là gì" trên hands_fallback_pose

Đây là trường hợp được nêu trong yêu cầu: "0/95 Top-5 đối với từ 'Tên là gì' trên các session Reconstructed đầu vào hands_fallback_pose".

| Chỉ số | Giá trị |
|---|---|
| Tensor `hands_fallback_pose` cho "Tên là gì" | 95 windows |
| Historical: target (class 61) in Top-5 | **0 / 95** |
| Reconstructed: target (class 61) in Top-5 | **0 / 95** |
| Historical Top-1 distribution | Bạn(60): 78, Xin lỗi(53): 11, Tháng(42): 4, ... |
| Reconstructed Top-1 distribution | Bạn(60): 49, Tháng(42): 27, Xin lỗi(53): 16, ... |

**Phân tích:**

Cả hai model đều **không nhận ra "Tên là gì" (class 61) trong Top-5** trên 95 tensor `hands_fallback_pose`. Model dự đoán class 60 "Bạn" chiếm đa số. Điều này không phải lỗi class mapping vì:

1. Mapping `SIGN_DICT_102` đã xác minh: class 61 = "Tên là gì" ✅
2. Words masking hoạt động đúng (loại bỏ classes 0–38) ✅
3. Joint order đúng (Pose 0-32, LH 33-53, RH 54-74) ✅

**Giả thuyết (cần kiểm tra thêm):** Tensor `hands_fallback_pose` của "Tên là gì" có phân phối Pose khác biệt hoàn toàn so với training data của class 61. Khi 29/33 Pose joints là hằng số và tọa độ cổ tay/khuỷu tay được tính từ vai hardcoded, feature space không còn phân biệt được "Tên là gì" với "Bạn". Cần thực nghiệm holistic_raw thuần túy để xác nhận.

---

## 7. High-Confidence Wrong Predictions

| Checkpoint | Conf ≥ 0.8, nhưng sai | Tỷ lệ |
|---|---|---|
| Historical | 213 / 501 | 42.5% |
| Reconstructed Best | 224 / 501 | 44.7% |

Cả hai model đều tự tin cao nhưng sai trên nhiều window. Đây là hiện tượng phổ biến với softmax + masking khi tensor chứa nhiều missing landmarks: model buộc phải chọn một class, và thường chọn với confidence cao.

---

## 8. Trả lời 3 Câu Hỏi Cốt Lõi

### (1) Model nào tốt hơn trên cùng dữ liệu?

**Historical (epoch 32) tốt hơn trên 501 tensor tổng thể:**
- Top-1: 8.6% vs 5.8%
- Top-5: 37.7% vs 29.7%
- Paired: Historical better 22 windows, Recon better 8 windows
- Ngoại lệ: Reconstructed_best tốt hơn trên từ "Thế nào" (19.2% vs 14.2%)

**Giới hạn của kết luận này:** Dữ liệu không cân bằng. Historical chỉ chạy trên holistic_raw (171 windows), trong khi Reconstructed chạy trên cả holistic_raw (62) + hands_fallback_pose (268). Kết quả A/B công bằng nhất là trên **subset holistic_raw chung** (233 windows):
- Historical on holistic_raw: 35/233 = **15.0%**
- Reconstructed on holistic_raw: 24/233 = **10.3%**

Historical vẫn tốt hơn trên holistic_raw, nhưng khoảng cách nhỏ hơn (5%) và không có phân tầng session-level đầy đủ.

### (2) Lỗi nào có bằng chứng?

| Lỗi | Bằng chứng | Mức độ chắc chắn |
|---|---|---|
| **Lỗi preprocessing/landmark (hands_fallback_pose):** 29/33 Pose joints bất động, tọa độ vai hardcoded | Xác nhận 100% từ NPY gốc | **CÓ BẰNG CHỨNG** |
| **Lỗi temporal segmentation:** holistic_raw 2518ms vs fallback 1199ms, cùng 30 frames nhưng thời lượng khác nhau | Xác nhận từ log events | **CÓ BẰNG CHỨNG** |
| **Model dự đoán sai class "Tên là gì" khi dùng hands_fallback_pose:** 0/95 target in Top-5 ở cả hai model | Xác nhận từ A/B replay | **CÓ BẰNG CHỨNG** |
| **Phân phối sai lệch training vs runtime (distribution mismatch):** | Không có metadata training FPS/source | **GIẢ THUYẾT CHƯA XÁC NHẬN** |

### (3) Còn thiếu gì trước khi resume training?

| Thiếu | Mô tả |
|---|---|
| **Dataset training metadata** | FPS video gốc, số frame, tracking engine (Holistic hay Hands), normalization đã áp dụng |
| **Thực nghiệm holistic_raw thuần túy cho Reconstructed** | Cần chạy session với Reconstructed checkpoint + MediaPipe Holistic để có cặp so sánh công bằng hơn |
| **Ground truth mốc cử chỉ** | 501 windows chồng lấn từ 14 sessions 20–180 giây. Không thể dùng verdict CORRECT/PARTIAL/FAIL của session làm trial accuracy nếu không có mốc cử chỉ độc lập |
| **Validation set performance** | Chỉ có live webcam test; cần chạy evaluation trên validation/test set chuẩn để so sánh hai checkpoint |
| **Đánh giá Reconstructed_last** | Checkpoint epoch 11 chưa được đưa vào A/B replay |

---

## 9. Tóm Tắt Điều Hành

| Chỉ số | Giá trị |
|---|---|
| Total sessions phân tích | 14 |
| Total NPY replay | 501 |
| % tái hiện log gốc | **100%** |
| Checkpoint mismatch events | 0 |
| A/B: Historical Top-1 (toàn bộ 501) | **8.6%** |
| A/B: Reconstructed Top-1 (toàn bộ 501) | **5.8%** |
| A/B: Historical Top-1 (holistic_raw 233) | **15.0%** |
| A/B: Reconstructed Top-1 (holistic_raw 233) | **10.3%** |
| Chưa resume training | ✅ |
| Chưa sửa runtime pipeline | ✅ |

*Script: `scratch/vsl_p010_step2_ab_replay.py`*  
*Dữ liệu: `replay_per_window.csv`, `replay_by_source_and_word.csv`*

