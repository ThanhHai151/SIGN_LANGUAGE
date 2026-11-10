import json
import os

# Load stats
detailed_stats = json.load(open('d:/SIGN_LANGUAGE/scratch/detailed_distribution_stats.json', encoding='utf-8'))
h_stats = detailed_stats['H_to_V']
v_stats = detailed_stats['V_to_V']

sim_data = json.load(open('d:/SIGN_LANGUAGE/scratch/counterfactual_simulation_results.json', encoding='utf-8'))
cand_sim = sim_data['candidate_simulation']
grid_sim = sim_data['sensitivity_grid']
matrix_sim = sim_data['interaction_matrix']
trial_sim = sim_data['per_trial']

# Features list
features_to_audit = [
    ('ang_IM_dir', 'Góc mở ngón trỏ - ngón giữa (độ)'),
    ('mid_horiz_ratio', 'Tỷ lệ phương ngang ngón giữa'),
    ('idx_horiz_ratio', 'Tỷ lệ phương ngang ngón trỏ'),
    ('d_IW', 'Khoảng cách cổ tay - đầu ngón trỏ'),
    ('d_MW', 'Khoảng cách cổ tay - đầu ngón giữa'),
    ('d_IM', 'Khoảng cách đầu ngón trỏ - giữa'),
    ('diff_x_IM', 'Độ lệch trục X đầu ngón trỏ - giữa'),
    ('idx_tip_y', 'Tọa độ chuẩn hóa Y đầu ngón trỏ'),
    ('mcp_width_2d', 'Độ rộng khớp bàn tay 2D (MCP)'),
    ('thumb_ring_dist', 'Khoảng cách ngón cái - áp út')
]

# Build Section 2 table lines
sec2_table_lines = []
sec2_table_lines.append("| Đặc trưng (Feature) | Tập | N | Min | P5 | P10 | P25 | Median | P75 | P90 | P95 | Max | Mean ± Std |")
sec2_table_lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

for feat_key, feat_desc in features_to_audit:
    hs = h_stats[feat_key]
    vs = v_stats[feat_key]
    
    # H row
    sec2_table_lines.append(
        f"| **`{feat_key}`**<br>*{feat_desc}* | **H → V** | {hs['count_valid']} | "
        f"{hs['min']:.3f} | {hs['p5']:.3f} | {hs['p10']:.3f} | {hs['p25']:.3f} | "
        f"**{hs['median']:.3f}** | {hs['p75']:.3f} | {hs['p90']:.3f} | {hs['p95']:.3f} | "
        f"{hs['max']:.3f} | {hs['mean']:.3f} ± {hs['std']:.3f} |"
    )
    # V row
    sec2_table_lines.append(
        f"| ↳ *(đối chứng chuẩn)* | **V → V** | {vs['count_valid']} | "
        f"{vs['min']:.3f} | {vs['p5']:.3f} | {vs['p10']:.3f} | {vs['p25']:.3f} | "
        f"**{vs['median']:.3f}** | {vs['p75']:.3f} | {vs['p90']:.3f} | {vs['p95']:.3f} | "
        f"{vs['max']:.3f} | {vs['mean']:.3f} ± {vs['std']:.3f} |"
    )

sec2_table_md = "\n".join(sec2_table_lines)

# Build Sensitivity Grid Table
grid_table_lines = []
grid_table_lines.append("| Ngưỡng mid_horiz_ratio | Ngưỡng ang_IM_dir | Loại trừ H → V (140 frames) | Tỷ lệ loại trừ H (%) | Bảo lưu V → V (362 frames) | Tỷ lệ bảo lưu V (%) | Loại trừ K → V (1 frame) |")
grid_table_lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
for g in grid_sim:
    grid_table_lines.append(
        f"| <= {g['mid_thresh']:.2f} | <= {g['ang_thresh']:.1f}° | "
        f"{g['h_elim']} / 140 | **{g['h_elim_pct']:.1f}%** | "
        f"{g['v_ret']} / 362 | **{g['v_ret_pct']:.1f}%** | "
        f"{g['k_elim']} / 1 |"
    )
grid_table_md = "\n".join(grid_table_lines)

# Build Per-Trial Table
trial_table_lines = []
trial_table_lines.append("| Mã Lượt Thử (Trial ID) | Nhãn Ký Tự | Tổng Frame | Rule V Khớp Ban Đầu | Loại trừ bởi A (mid<=0.50) | Loại trừ bởi B (ang<=38) | Loại trừ bởi A & B | Số Frame V Còn Lại | Tỷ Lệ Bảo Lưu V (%) |")
trial_table_lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

h_trials = [t for t in sorted(trial_sim.keys()) if trial_sim[t]['expected'] == 'H']
v_trials = [t for t in sorted(trial_sim.keys()) if trial_sim[t]['expected'] == 'V']

for t in h_trials:
    st = trial_sim[t]
    elim_a = st['rule_v_match'] - st['pass_A']
    elim_b = st['rule_v_match'] - st['pass_B']
    elim_ab = st['rule_v_match'] - st['pass_A_and_B']
    v_rem = st['pass_A_and_B']
    trial_table_lines.append(
        f"| `{t}` | **H** | {st['total_frames']} | {st['rule_v_match']} | "
        f"{elim_a} / {st['rule_v_match']} | {elim_b} / {st['rule_v_match']} | "
        f"**{elim_ab} / {st['rule_v_match']}** | {v_rem} | N/A (H FP) |"
    )

trial_table_lines.append("| **TỔNG CỘNG 10 TRIALS H** | **H** | **455** | **140** | **140 / 140 (100%)** | **140 / 140 (100%)** | **140 / 140 (100%)** | **0** | **100% Loại Trừ FP** |")

for t in v_trials:
    st = trial_sim[t]
    elim_a = st['rule_v_match'] - st['pass_A']
    elim_b = st['rule_v_match'] - st['pass_B']
    elim_ab = st['rule_v_match'] - st['pass_A_and_B']
    v_rem = st['pass_A_and_B']
    ret_pct = (v_rem / st['rule_v_match'] * 100) if st['rule_v_match'] > 0 else 100.0
    trial_table_lines.append(
        f"| `{t}` | **V** | {st['total_frames']} | {st['rule_v_match']} | "
        f"{elim_a} | {elim_b} | {elim_ab} | "
        f"**{v_rem} / {st['rule_v_match']}** | **{ret_pct:.1f}%** |"
    )

trial_table_lines.append("| **TỔNG CỘNG 10 TRIALS V** | **V** | **397** | **362** | **0 / 362 (0%)** | **0 / 362 (0%)** | **0 / 362 (0%)** | **362 / 362** | **100.0% Bảo Lưu TP** |")

trial_table_md = "\n".join(trial_table_lines)

# Write template
template = """# BÁO CÁO KIỂM TOÁN AN TOÀN PHẢN THỰC NGHIỆM QUY TẮC V (TASK 12.1.3)

> **Mã nhiệm vụ:** TASK 12.1.3 — Rule V Counterfactual Safety Audit  
> **Tệp dữ liệu:** `backend/data/diagnostics/pilot_pilot_20261107_2141.jsonl`  
> **Mã băm SHA-256 xác thực:** `fb4f725757bcbc73b96748896c4149c21927461314c72e68a49310b7cd76429f` (Khớp 100% với phiên thu thập chuẩn 220 trials)  
> **Nguyên tắc thực thi:** **STRICTLY READ-ONLY / COUNTERFACTUAL SIMULATION** — Không sửa mã nguồn, không nạp lại checkpoint, không thay đổi tệp telemetry thực tế.

---

## 1. XÁC ĐỊNH ĐẦY ĐỦ TẬP FRAME ĐANG KÍCH HOẠT QUY TẮC V (ACTIVE RULE V FRAMES)

Trong toàn bộ 2,058 active frames của đợt thử nghiệm chính thức 22 ký tự tĩnh × 10 lượt, điều kiện trích xuất toàn diện bao gồm tất cả các frame thỏa mãn:
- `geometric.status == "MATCH"`
- `final_sign == "V"`
- `prediction_source == "geometric_rule"`

### Bảng phân nhóm toàn diện theo nhãn mục tiêu thực tế (Ground-Truth Target Letter)

| Nhóm Ký Tự Mục Tiêu (Target Letter) | Số Lượng Frame Kích Hoạt Rule V | Tỷ Lệ Trong Tổng Tập Rule V | Bản Chất Phân Loại | Diễn Giải Thực Tế |
|:---|:---:|:---:|:---:|:---|
| **V (V → V)** | **362** | **71.97%** | **True Positive (TP)** | Cử chỉ chữ V thực tế được nhận diện chính xác qua luật hình học Rule V. |
| **H (H → V)** | **140** | **27.83%** | **False Positive (FP)** | Cử chỉ chữ H bị nhận nhầm sang V do Rule V thiếu ràng buộc góc mở và hướng ngón giữa. |
| **K (K → V)** | **1** | **0.20%** | **False Positive (FP)** | Frame 5 của trial `pilot_20261107_2141_K_rep01_att01` kích hoạt nhầm Rule V trong khoảnh khắc chuyển tư thế. |
| **Các chữ cái khác (19 chữ còn lại)** | **0** | **0.00%** | **True Negative (TN)** | Không có bất kỳ frame nào của 19 chữ cái tĩnh còn lại bị nhận nhầm thành V qua Rule V. |
| **TỔNG CỘNG** | **503** | **100.00%** | - | Toàn bộ quần thể kích hoạt Rule V trong dataset 2141. |

> [!NOTE]
> Báo cáo ghi nhận thêm **1 frame K → V duy nhất** (Frame 5 của `pilot_20261107_2141_K_rep01_att01`). Đây là bằng chứng quan trọng chứng minh Rule V không chỉ gây lỗi tràn nhãn với riêng H mà còn có nguy cơ rò rỉ sang các tư thế mở ngón trỏ - giữa khác nếu không có điều kiện bảo vệ chặt chẽ.

---

## 2. PHÂN BỐ ĐẶC TRƯNG CHI TIẾT (FEATURE DISTRIBUTIONS WITH PERCENTILES)

Thống kê phân vị hoàn chỉnh (P5, P10, P25, Median, P75, P90, P95) cùng Min, Max, Mean ± Std của 10 đặc trưng hình học cốt lõi trên 140 frames **H → V** và 362 frames **V → V**:

{{SEC2_TABLE}}

### Nhận xét chuyên sâu về khoảng cách phân ly (Separation Margins)

1. **Khoảng cách góc mở trỏ - giữa (`ang_IM_dir`):**
   - Phân vị 95 của **V → V** là $24.564^\\circ$, giá trị lớn nhất tuyệt đối là **$28.223^\\circ$**.
   - Phân vị 5 của **H → V** là $46.393^\\circ$, giá trị nhỏ nhất tuyệt đối là **$41.960^\\circ$**.
   - Khoảng trống phân ly tuyệt đối (Absolute Separation Gap) giữa Max(V) và Min(H) là:
     $$\\Delta = 41.960^\\circ - 28.223^\\circ = 13.737^\\circ$$
   - Ngưỡng đề xuất **$38.0^\\circ$** nằm hoàn toàn trong khoảng an toàn này (cách Max(V) $+9.78^\\circ$ và cách Min(H) $-3.96^\\circ$).

2. **Khoảng cách tỷ lệ ngang của ngón giữa (`mid_horiz_ratio`):**
   - Phân vị 95 của **V → V** là $0.323$, giá trị lớn nhất tuyệt đối là **$0.339$**.
   - Phân vị 5 của **H → V** là $0.781$, giá trị nhỏ nhất tuyệt đối là **$0.764$**.
   - Khoảng trống phân ly tuyệt đối giữa Max(V) và Min(H) là:
     $$\\Delta = 0.764 - 0.339 = 0.425$$
   - Ngưỡng đề xuất **$0.50$** nằm chính giữa khoảng an toàn lý tưởng (cách Max(V) $+0.161$ và cách Min(H) $-0.264$).

3. **Frame dị biệt K → V (Frame 5):**
   - `mid_horiz_ratio = 0.9916` (rất cao, vượt xa ngưỡng 0.50).
   - `ang_IM_dir = 76.32^\\circ` (rất rộng, vượt xa ngưỡng 38.0°).

---

## 3. MÔ PHỎNG PHẢN THỰC NGHIỆM VÀ PHÂN TÍCH ĐỘ NHẠY (COUNTERFACTUAL SIMULATION & SENSITIVITY)

### A. Kết quả mô phỏng các điều kiện ứng viên (Candidate Guard Conditions)

Mô phỏng phản thực nghiệm độc lập giả định bổ sung điều kiện bảo vệ vào Rule V:
- **Điều kiện A:** `mid_horiz_ratio <= 0.50`
- **Điều kiện B:** `ang_IM_dir <= 38.0`
- **Điều kiện A AND B:** `mid_horiz_ratio <= 0.50` **VÀ** `ang_IM_dir <= 38.0`

| Phương Án Điều Kiện Bảo Vệ | H → V Loại Trừ (Tỷ lệ %) | H → V Lọt Lưới (Frames) | V → V Bảo Lưu (Tỷ lệ %) | V → V Bị Hủy Nhầm (Frames) | K → V Loại Trừ (1 frame) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Điều kiện A** (`mid <= 0.50`) | **140 / 140 (100.0%)** | 0 | **362 / 362 (100.0%)** | 0 | **1 / 1 (100.0%)** |
| **Điều kiện B** (`ang <= 38.0°`) | **140 / 140 (100.0%)** | 0 | **362 / 362 (100.0%)** | 0 | **1 / 1 (100.0%)** |
| **Điều kiện A AND B** (Kết hợp cả hai) | **140 / 140 (100.0%)** | 0 | **362 / 362 (100.0%)** | 0 | **1 / 1 (100.0%)** |

> [!IMPORTANT]
> Cả 3 phương án đều đạt **100.0% loại trừ lỗi nhầm lẫn H → V (140/140 frames)** và **100.0% bảo lưu số frame V đúng (362/362 frames)**, đồng thời triệt tiêu hoàn toàn **1 frame lỗi K → V**.

---

### B. Ma trận quét độ nhạy đa chiều (Multi-Dimensional Sensitivity Grid)

Quét toàn diện không gian ngưỡng lân cận:
- `mid_horiz_ratio`: [0.40, 0.45, 0.50, 0.55, 0.60]
- `ang_IM_dir`: [28.0°, 32.0°, 36.0°, 38.0°, 40.0°, 44.0°]

{{GRID_TABLE}}

### Phân tích biên độ an toàn từ ma trận độ nhạy:
1. **Độ ổn định của `mid_horiz_ratio`:**
   - Khi quét đơn lẻ từ $0.40$ đến $0.60$, tỷ lệ loại trừ H luôn giữ nguyên $100.0\\%$ và tỷ lệ bảo lưu V luôn giữ nguyên $100.0\\%$.
   - Điều này khẳng định vùng đệm của `mid_horiz_ratio` là cực kỳ rộng ($0.34 \\to 0.76$, biên an toàn $\\pm 0.16$).
2. **Độ ổn định của `ang_IM_dir`:**
   - Tại ngưỡng quá chặt ($28.0^\\circ$): Bị mất $1$ frame V đúng (frame có góc $28.22^\\circ$), tỷ lệ bảo lưu giảm xuống $99.7\\%$.
   - Tại dải an toàn ($32.0^\\circ \\to 40.0^\\circ$): Đạt trạng thái tối ưu tuyệt đối ($100.0\\%$ bảo lưu V, $100.0\\%$ loại trừ H).
   - Tại ngưỡng nới lỏng ($44.0^\\circ$): Bắt đầu bị rò rỉ $3$ frame H lọt lưới (tỷ lệ loại trừ giảm còn $97.9\\%$) do có các frame H đạt góc $41.96^\\circ - 43.8^\\circ$.
   - Ngưỡng **$38.0^\\circ$** nằm hoàn hảo tại tâm của khoảng ổn định $32.0^\\circ - 40.0^\\circ$.

---

## 4. BẢNG TƯƠNG TÁC 2×2 (2×2 INTERACTION MATRIX)

Phân bổ chéo 4 góc phần tư giữa hai điều kiện:
- Chiều 1: `mid_horiz_ratio <= 0.50` vs `mid_horiz_ratio > 0.50`
- Chiều 2: `ang_IM_dir <= 38.0°` vs `ang_IM_dir > 38.0°`

### A. Ma trận đối với 140 frames nhầm lẫn H → V

| `ang_IM_dir` \\ `mid_horiz_ratio` | `mid_horiz_ratio <= 0.50` (Hướng đứng) | `mid_horiz_ratio > 0.50` (Hướng ngang) | TỔNG |
|:---|:---:|:---:|:---:|
| **`ang_IM_dir <= 38.0°` (Góc hẹp)** | **0 frame (0.0%)** | 0 frame (0.0%) | **0 frame** |
| **`ang_IM_dir > 38.0°` (Góc rộng)** | 0 frame (0.0%) | **140 frames (100.0%)** | **140 frames** |
| **TỔNG** | **0 frame** | **140 frames** | **140 frames** |

### B. Ma trận đối với 362 frames chữ V thực sự (V → V)

| `ang_IM_dir` \\ `mid_horiz_ratio` | `mid_horiz_ratio <= 0.50` (Hướng đứng) | `mid_horiz_ratio > 0.50` (Hướng ngang) | TỔNG |
|:---|:---:|:---:|:---:|
| **`ang_IM_dir <= 38.0°` (Góc hẹp)** | **362 frames (100.0%)** | 0 frame (0.0%) | **362 frames** |
| **`ang_IM_dir > 38.0°` (Góc rộng)** | 0 frame (0.0%) | 0 frame (0.0%) | **0 frame** |
| **TỔNG** | **362 frames** | **0 frame** | **362 frames** |

### C. Ma trận đối với 1 frame nhầm lẫn K → V

| `ang_IM_dir` \\ `mid_horiz_ratio` | `mid_horiz_ratio <= 0.50` (Hướng đứng) | `mid_horiz_ratio > 0.50` (Hướng ngang) | TỔNG |
|:---|:---:|:---:|:---:|
| **`ang_IM_dir <= 38.0°` (Góc hẹp)** | 0 frame (0.0%) | 0 frame (0.0%) | **0 frame** |
| **`ang_IM_dir > 38.0°` (Góc rộng)** | 0 frame (0.0%) | **1 frame (100.0%)** | **1 frame** |
| **TỔNG** | **0 frame** | **1 frame** | **1 frame** |

> [!NOTE]
> Phân bố 2×2 cho thấy **sự phân ly nhị phân hoàn hảo (Perfect Orthogonal Bipartition)**:
> - 100% frame V → V tập trung duy nhất ở góc phần tư **(Đứng + Hẹp)**.
> - 100% frame H → V và K → V tập trung duy nhất ở góc phần tư **(Ngang + Rộng)**.
> - Hai góc phần tư hỗn hợp chéo **(Đứng + Rộng)** và **(Ngang + Hẹp)** hoàn toàn trống (0 frame).

---

## 5. CHI TIẾT THEO TỪNG TRIAL (PER-TRIAL SIMULATION BREAKDOWN)

Bảng tổng hợp chi tiết toàn bộ 20 lượt thử nghiệm thực tế (10 lượt ký tự H và 10 lượt ký tự V):

{{TRIAL_TABLE}}

---

## 6. SỐ PHẬN CỦA CÁC FRAME ĐƯỢC LOẠI TRỪ TRONG PIPELINE HỆ THỐNG

Một câu hỏi cốt lõi của kiểm toán phản thực nghiệm: **Khi Rule V không khớp, 140 frame của H và 1 frame của K sẽ chuyển thành nhãn gì?**

Dựa trên cấu trúc pipeline thực tế trong [`backend/vision/recognizer.py`](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L1124-L1225) và kiểm thử mô phỏng độc lập:

1. **Ở Tầng 1 (Luật hình học):**
   - Đã kiểm tra đối chiếu toàn bộ các quy tắc hình học còn lại (B, I, Y, D, L, T, O, C, A, S, E, G, H, K, U, P, Q, R, X).
   - Kết quả: **0 / 140 frame H** và **0 / 1 frame K** thỏa mãn bất kỳ quy tắc nào khác.
   - Trạng thái hình học chuyển thành `geometric.status = "NO_MATCH"`.

2. **Ở Tầng 2 (Static MLP Gating):**
   - Khi `geometric.status == "NO_MATCH"`, hệ thống kích hoạt Static MLP (`static_alphabet_head.predict_feature`).
   - Kiểm tra thực tế trên 291 frame `H → NEUTRAL` trong cùng dataset cho thấy: Static MLP dự đoán thô nhãn `V` (với độ tin cậy trung bình $0.869$).
   - Tuy nhiên, quy tắc kiểm duyệt (`recognizer.py` dòng 1190–1225) quy định:
     - `ALLOWED_FALLBACK_BASE_LETTERS = {"A", "D", "E"}`
     - `ALLOWED_SUPPLEMENTAL_LETTERS = {"G", "H", "K", "M", "N", "P", "Q", "R", "U", "X"}`
     - Chữ `V` **HOÀN TOÀN KHÔNG NẰM TRONG CẢ HAI DANH SÁCH NÀY**.
   - Do đó, nếu Static MLP tiếp tục dự đoán `V`, tầng Gating sẽ lập tức **TỪ CHỐI** (`gating_status = "rejected"`).

3. **Kết luận về nhãn cuối cùng (Final Output):**
   - Toàn bộ **140 frame H → V** sẽ chuyển thành trạng thái **`NEUTRAL` (Trầm lặng / Đang theo dõi cử chỉ...)**, hoàn toàn chấm dứt tình trạng sinh nhãn sai hiển thị trên màn hình người dùng.
   - Tương tự, **1 frame K → V** cũng chuyển thành **`NEUTRAL`**.

---

## 7. GIỚI HẠN DỮ LIỆU, ĐIỀU ĐÃ XÁC NHẬN VS GIẢ ĐỊNH VS ẨN SỐ

### A. Sự kiện đã xác nhận thực tế (Confirmed Facts)
1. Trong 2,058 frames pilot 2141, Rule V đã kích hoạt tổng cộng 503 frames: 362 frame V thực sự (71.97%), 140 frame H nhầm lẫn (27.83%), và 1 frame K nhầm lẫn (0.20%).
2. Không có bất kỳ frame nào của 19 ký tự tĩnh khác bị kích hoạt Rule V.
3. Trong toàn bộ 140 frame H → V, ngón giữa đều có phương ngang rõ rệt (`mid_horiz_ratio >= 0.764`) và góc mở lớn (`ang_IM_dir >= 41.960°`).
4. Trong toàn bộ 362 frame V → V, ngón giữa luôn giữ phương đứng (`mid_horiz_ratio <= 0.339`) và góc mở hẹp/vừa (`ang_IM_dir <= 28.223°`).

### B. Giả định phản thực nghiệm (Counterfactual Assumptions)
1. Giả định hành vi thu nhận landmark của MediaPipe và tư thế của người dùng trong phiên pilot 2141 là đại diện chuẩn cho quy trình kiểm thử.
2. Giả định không có hiệu ứng phụ lan truyền đến các luồng xử lý ngoài `alphabet_classifier.py` và `recognizer.py`.

### C. Ẩn số chưa được kiểm chứng (Unverified Risks / Unknowns)
1. **Biến thiên sinh học của người dùng khác (User Intersubject Variability):** Dữ liệu pilot 2141 được thực hiện bởi một người dùng. Những người dùng có bàn tay linh hoạt mở ngón rất rộng (ví dụ V xòe $> 35^\\circ$) hoặc người có thói quen nghiêng cổ tay khi làm chữ V có thể tiến gần đến biên $38.0^\\circ$ hoặc $0.50$.
2. **Hiện tượng chữ H chưa thể nhận đúng (Root cause of H remains unsolved):** Việc loại trừ H → V chỉ giúp H không bị nhận sai thành V (chuyển sang `NEUTRAL`), nhưng **không làm cho chữ H nhận đúng thành H**. Như đã kết luận tại TASK 12.1.2, chữ H bị trượt do Rule H yêu cầu $d_{MW} \\ge 1.70$ trong khi thực tế cử chỉ H co ngón giữa có $d_{MW} \\approx 1.615 - 1.740$.
3. **Các góc máy quay nghiêng (Roll/Pitch angle):** Nếu người dùng đặt webcam ở góc nghiêng làm bàn tay xoay nghiêng quá $45^\\circ$, `mid_horiz_ratio` có thể tăng lên đối với chữ V chuẩn.

---

## 8. ĐÁNH GIÁ CUỐI CÙNG & KHUYẾN NGHỊ PHƯƠNG ÁN TỐI ƯU

### A. So sánh 3 phương án ứng viên

| Tiêu Chí Đánh Giá | Phương Án A (`mid <= 0.50`) | Phương Án B (`ang <= 38.0°`) | Phương Án A AND B (`mid <= 0.50` & `ang <= 38.0°`) |
|:---|:---:|:---:|:---:|
| **Hiệu quả loại trừ FP H → V (140)** | 100.0% (140/140) | 100.0% (140/140) | **100.0% (140/140)** |
| **Hiệu quả bảo lưu TP V → V (362)** | 100.0% (362/362) | 100.0% (362/362) | **100.0% (362/362)** |
| **Hiệu quả loại trừ FP K → V (1)** | 100.0% (1/1) | 100.0% (1/1) | **100.0% (1/1)** |
| **Độ rộng biên an toàn (Safety Margin)** | Rất rộng ($0.425$) | Rộng ($13.74^\\circ$) | **Đa chiều (Cực kỳ an toàn)** |
| **Bất biến trước xoay cổ tay 2D** | Không (phụ thuộc góc nghiêng đứng) | **Có (bất biến với phép xoay)** | **Kết hợp bù trừ lẫn nhau** |
| **Ngăn chặn tư thế ngón mở bất thường** | Yếu nếu ngón giữa đứng | **Rất mạnh (chặn góc mở)** | **Toàn diện 2 bậc tự do** |

### B. Trả lời trực tiếp câu hỏi kiểm toán

> **Câu hỏi:** *Phương án nào (A, B, hay A AND B) có bằng chứng dữ liệu tốt nhất để cân nhắc, và rủi ro nào vẫn chưa được giải quyết?*

1. **Phương án có bằng chứng dữ liệu tốt nhất:**
   **PHƯƠNG ÁN A AND B (`mid_horiz_ratio <= 0.50 and ang_IM_dir <= 38.0`)**.
   - *Căn cứ dữ liệu:* Trên toàn bộ 503 frames kích hoạt Rule V, phương án này tạo ra một ranh giới quyết định đa chiều (convex bounding region) bọc kín 100% cử chỉ V chuẩn và loại trừ sạch 100% cử chỉ H nhầm lẫn và K nhầm lẫn.
   - *Tính bù trừ sinh cơ học:* `mid_horiz_ratio` kiểm soát bậc tự do **hướng không gian của ngón giữa** (loại bỏ tư thế ngón giữa chỉ ngang của chữ H), trong khi `ang_IM_dir` kiểm soát bậc tự do **độ xòe tương đối giữa hai ngón** (bất biến trước góc nghiêng cổ tay). Hai điều kiện này trực giao về mặt hình học, tạo nên lớp phòng thủ kép ("defense-in-depth") vững chắc nhất.

2. **Rủi ro vẫn chưa được giải quyết (Unmitigated Risks):**
   - **Tỷ lệ nhận diện đúng của chữ H vẫn là 0%**: Việc sửa Rule V chỉ biến frame sai thành frame `NEUTRAL`. Muốn chữ H được nhận diện đúng (`H → H`), dự án cần một task riêng để tinh chỉnh điều kiện $d_{MW}$ của Rule H (hoặc cho phép Static MLP nhận H).
   - **Tư thế V xòe quá rộng hoặc nghiêng quá mức**: Cần ghi nhận trong hướng dẫn cử chỉ rằng chữ V trong VSL chuẩn mở góc vừa phải ($15^\\circ - 30^\\circ$) và hướng thẳng đứng lên trên.
"""

final_report = template.replace('{{SEC2_TABLE}}', sec2_table_md)
final_report = final_report.replace('{{GRID_TABLE}}', grid_table_md)
final_report = final_report.replace('{{TRIAL_TABLE}}', trial_table_md)

# Verify no placeholders
assert '{{' not in final_report, "Placeholder found!"

artifact_path = r"C:\Users\PC\.gemini\antigravity\brain\e7c77c34-c401-4b3d-8757-568dc34b3e88\vsl_task12_1_3_rule_v_counterfactual_report.md"

with open(artifact_path, 'w', encoding='utf-8') as f:
    f.write(final_report)

print(f"Report written successfully to {artifact_path} ({len(final_report)} bytes)")
