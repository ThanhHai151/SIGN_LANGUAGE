# VSL-P010 — Kiểm Tra Pose & Temporal (Pose Temporal Audit)

**Script:** `scratch/vsl_p010_step3_pose_audit.py`  
**Ngày:** 07/11/2026

---

## 1. Phân nhóm Landmark Source

| Landmark Source | Số windows |
|---|---|
| `holistic_raw` | 233 |
| `hands_fallback_pose` | 268 |
| **Tổng** | **501** |

---

## 2. Window Duration — Chênh lệch Thời gian cửa sổ 30 frame

| Nhóm | Mean (ms) | Median (ms) | Std (ms) | Min | Max |
|---|---|---|---|---|---|
| `holistic_raw` | **2518** | 2449 | 296 | 2029 | 3755 |
| `hands_fallback_pose` | **1199** | 1189 | 115 | 980 | 1620 |

**Nhận xét:** Hai nhóm có thời lượng window khác biệt rõ rệt — holistic_raw ~2.5 lần dài hơn fallback. Lý do trực tiếp: MediaPipe Holistic xử lý tốc độ thấp (~13.6 FPS), còn MediaPipe Hands xử lý tốc độ cao (~26.1 FPS).

---

## 3. Client FPS Thực tế

| Nhóm | Mean FPS | Median | Std | Min | Max |
|---|---|---|---|---|---|
| `holistic_raw` | **13.6** | 14.0 | 1.1 | 11 | 16 |
| `hands_fallback_pose` | **26.1** | 26.0 | 1.9 | 22 | 30 |

---

## 4. Phân tích Tọa độ Vai Cố định (hands_fallback_pose)

### 4.1 Kết quả kiểm tra từ NPY gốc

| Chỉ số | Giá trị |
|---|---|
| Tensor gần đúng DEFAULT_POSE hardcoded | **268/268 (100%)** |
| Log flagged `is_static_shoulders=True` | 268/268 |
| Avg `shoulder_left_std` (σ theo 30 frames) | **0.000272** |
| Avg `shoulder_right_std` | **0.000193** |
| Avg số joint Pose tĩnh (σ < 1e-4) | **29.0 / 33** |

> Tọa độ DEFAULT_POSE hardcoded trong `extract75KeypointsFromHands()` tại `CameraSignRecognizer.jsx`:
> - Vai trái (joint 11): `[0.65, 0.45, 0]`  
> - Vai phải (joint 12): `[0.35, 0.45, 0]`
>
> 100% tensor `hands_fallback_pose` đều có tọa độ vai nằm trong bán kính 0.02 từ điểm hardcoded. Xác nhận bằng bằng chứng NPY.

### 4.2 So sánh với holistic_raw

| Chỉ số | holistic_raw | hands_fallback_pose |
|---|---|---|
| Avg shoulder_left_std | **0.000572** | 0.000272 |
| Avg static pose joints | **0.0 / 33** | **29.0 / 33** |

**Lưu ý quan trọng về đọc std:** std của holistic_raw (~0.000572) **cao hơn** so với fallback (~0.000272) — tuy nhiên cả hai đều rất nhỏ theo thang tọa độ MediaPipe (normalized 0-1). Sự khác biệt tuyệt đối ở mức 1-2 pixel màn hình 1920×1080. Điều này có nghĩa holistic_raw có biến thiên nhỏ nhưng thực sự (người dùng di chuyển nhẹ), còn fallback_pose hoàn toàn bất động.

Chỉ số `static pose joints (std < 1e-4)` phân biệt rõ hơn: holistic_raw = **0.0 joints tĩnh**, fallback = **29.0/33 joints tĩnh**.

---

## 5. Các khớp biến thiên trong hands_fallback_pose

Theo code `extract75KeypointsFromHands()` trong frontend:

| Joint | Nguồn | Biến thiên |
|---|---|---|
| 0–10 (đầu, mặt) | Hardcoded | Bất động |
| 11 (vai trái), 12 (vai phải) | Hardcoded `[0.65, 0.45, 0]` / `[0.35, 0.45, 0]` | **Bất động** |
| 13 (khuỷu trái), 14 (khuỷu phải) | Nội suy giữa vai giả lập và cổ tay | Biến thiên theo tay |
| 15 (cổ tay trái), 16 (cổ tay phải) | Lấy từ hand landmark | Biến thiên |
| 17–32 (thân dưới, chân) | Hardcoded | Bất động |
| 33–74 (21 LH + 21 RH) | Từ MediaPipe Hands | Biến thiên |

---

## 6. Câu hỏi về FPS/Sampling/Normalization và Dữ liệu Huấn luyện

Dữ liệu `preprocessing` trong checkpoint `reconstructed_best` (schema_version=1):

```json
{
  "shape": [3, 30, 75],
  "coordinates": "mediapipe_image_normalized_xyz",
  "missing_landmarks": "zero",
  "normalization": "none",
  "temporal_sampling": "linear_resample_to_30_after_frame_stride"
}
```

| Câu hỏi | Trạng thái |
|---|---|
| Training dùng MediaPipe Holistic hay Hands? | **Không có metadata rõ ràng** trong checkpoint. Tên `holistic` không xuất hiện. Field `temporal_sampling: linear_resample_to_30_after_frame_stride` gợi ý dữ liệu train có thể có số frame khác 30 (video gốc). |
| FPS training có khớp với runtime? | **Không rõ.** Training dùng video clip (không phải webcam realtime). Window duration runtime (1.2s–2.5s) không có metadata tương ứng trong train config. |
| Normalization | Cả training và runtime: `"none"` — không normalize tọa độ. Tọa độ gốc MediaPipe [0,1] được đưa thẳng vào model. |
| `historical` checkpoint preprocessing | **Không có metadata** — checkpoint epoch 32 chỉ chứa `model_state_dict`. Không thể xác nhận preprocessing settings của quá trình huấn luyện gốc. |

> **Giới hạn kết luận:** Không có metadata training FPS / video source / số frame gốc trong cả hai checkpoint. Không thể khẳng định hay bác bỏ distribution mismatch về temporal sampling.

---

## 7. Phân tích Missing Landmarks trong hands_fallback_pose

Từ log events (đã xác minh khớp 100% với NPY):

| Chỉ số | hands_fallback_pose | holistic_raw |
|---|---|---|
| Avg `right_hand_missing_ratio` | Cao hơn holistic | Thấp hơn |
| `both_hands_zero_frames` | 0 (MediaPipe Hands thường phát hiện ≥ 1 tay nếu trigger) | Có thể xảy ra |

---

## 8. Kết luận Pose Audit

**Observed facts:**

1. 100% tensor `hands_fallback_pose` có 29/33 joints Pose hoàn toàn cố định, xác nhận bằng NPY gốc.
2. Hai nhóm có window duration khác nhau gần gấp đôi (2518ms vs 1199ms) do FPS chênh lệch (13.6 vs 26.1 FPS).
3. `holistic_raw` không có joint Pose tĩnh nào (0/33), dữ liệu tọa độ cơ thể thực sự biến thiên theo cử chỉ.

**Hypotheses (không khẳng định nhân quả):**

- Tensor `hands_fallback_pose` có phân phối tọa độ Pose rất khác với dữ liệu huấn luyện từ video người thật (nơi cơ thể cử động). Model SSTCN có thể học được pattern cử chỉ từ Pose + Hand cùng nhau; khi Pose là hằng số, attention mechanism có thể bị ảnh hưởng.
- Window duration 1199ms (Hands) vs 2518ms (Holistic) có thể dẫn đến temporal sub-sampling khác nhau nếu training dùng video 25-30 FPS.

**Tests needed:**

- Cần metadata FPS và video source của tập huấn luyện gốc để xác nhận distribution mismatch temporal.
- Cần thực nghiệm với holistic_raw cho cả hai model (đủ sessions), loại bỏ hoàn toàn fallback_pose.

*Script: `scratch/vsl_p010_step3_pose_audit.py`*  
*Dữ liệu: `_pose_temporal_audit_raw.csv`, `_pose_audit_summary.json`*

