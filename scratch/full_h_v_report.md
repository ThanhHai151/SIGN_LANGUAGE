# BÁO CÁO GIÁM ĐỊNH PHÁP Y VỀ HIỆN TƯỢNG NHẦM LẪN H → V (TASK 12.1.2)

> **Mã nhiệm vụ:** TASK 12.1.2 — H → V Confusion Forensic Audit  
> **Tệp dữ liệu:** `backend/data/diagnostics/pilot_pilot_20261107_2141.jsonl` (SHA-256: `fb4f725757bcbc73b96748896c4149c21927461314c72e68a49310b7cd76429f`)  
> **Phạm vi đối tượng:** 455 Active Frames của ký tự mục tiêu H (0 frame đúng, 291 frame NEUTRAL, 164 frame sai, trong đó H → V chiếm 140 frame).  
> **Nguyên tắc thực hiện:** Chỉ đọc (STRICTLY READ-ONLY) — Bằng chứng thực tế từ mã nguồn và telemetry JSONL, không suy diễn, không sửa code.

---

## 1. PHÂN RÃ TOÀN BỘ 164 FRAME NHẦM LẪN CỦA CHỮ H THEO PIPELINE SOURCE

Kiểm toán trực tiếp từng bản ghi trong 455 active frames của nhãn mục tiêu H cho thấy:
- **Frame đúng (H → H):** `0 / 455` (0.00%)
- **Frame trầm lặng (H → NEUTRAL):** `291 / 455` (63.96%)
- **Frame sai (H → Khác):** `164 / 455` (36.04%)

### Bảng phân loại nguồn gốc dự đoán (Prediction Source Breakdown)

| Nhãn nhận diện cuối cùng (Final Sign) | Số lượng Frame | Tỷ lệ trong 164 frame sai | Nguồn nhận diện (Prediction Source) | Trạng thái Hình học (Geometric Status) | Gating Status | Trạng thái gọi MLP (Static Head) |
|:---:|:---:|:---:|:---|:---:|:---:|:---|
| **V** | **140** | **85.37%** | `geometric_rule` | `MATCH` | `passed_geometric` | **Không gọi (raw_pred = None)** |
| **K** | **22** | **13.41%** | `geometric_rule` | `MATCH` | `passed_geometric` | **Không gọi (raw_pred = None)** |
| **R** | **2** | **1.22%** | `geometric_rule` | `MATCH` | `passed_geometric` | **Không gọi (raw_pred = None)** |
| **Khác / MLP** | **0** | **0.00%** | - | - | - | - |
| **TỔNG SAI** | **164** | **100.00%** | `geometric_rule` (164/164) | `MATCH` (164/164) | `passed_geometric` (164/164) | **100% Bypassed MLP** |

> [!IMPORTANT]
> **Phát hiện pháp y 1:** Toàn bộ **140 frame H → V (100.0%)** đều bắt nguồn độc quyền từ **Tầng 1: Luật Hình học (Geometric Rule)** với độ ưu tiên tuyệt đối (`MATCH`). Static MLP hoàn toàn **KHÔNG THAM GIA** vào việc sinh ra nhãn V cho chữ H (MLP bị bypass 100% trong 140 frame này).

---

## 2. THỐNG KÊ CHI TIẾT 140 FRAME H → V THEO LƯỢT THỬ NGHIỆM (TRIAL BREAKDOWN)

### A. Phân bố nhầm lẫn theo từng Lượt thử nghiệm (Rep 01 → 10)

| Mã Lượt thử nghiệm (Trial ID) | Rep | Tổng Frame | H → V (Frames & %) | H → K (Frames & %) | H → R (Frames & %) | H → NEUTRAL (Frames & %) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `pilot_20261107_2141_H_rep01_att01` | **rep01** | 30 | **12** (40.0%) | 1 (3.3%) | 0 (0.0%) | 17 (56.7%) |
| `pilot_20261107_2141_H_rep02_att01` | **rep02** | 36 | **12** (33.3%) | 0 (0.0%) | 0 (0.0%) | 24 (66.7%) |
| `pilot_20261107_2141_H_rep03_att01` | **rep03** | 44 | **1** (2.3%) | 4 (9.1%) | 0 (0.0%) | 39 (88.6%) |
| `pilot_20261107_2141_H_rep04_att01` | **rep04** | 58 | **0** (0.0%) | 5 (8.6%) | 0 (0.0%) | 53 (91.4%) |
| `pilot_20261107_2141_H_rep05_att01` | **rep05** | 38 | **8** (21.1%) | 0 (0.0%) | 2 (5.3%) | 28 (73.7%) |
| `pilot_20261107_2141_H_rep06_att01` | **rep06** | 51 | **6** (11.8%) | 4 (7.8%) | 0 (0.0%) | 41 (80.4%) |
| `pilot_20261107_2141_H_rep07_att01` | **rep07** | 53 | **17** (32.1%) | 7 (13.2%) | 0 (0.0%) | 29 (54.7%) |
| `pilot_20261107_2141_H_rep08_att01` | **rep08** | 59 | **18** (30.5%) | 1 (1.7%) | 0 (0.0%) | 40 (67.8%) |
| `pilot_20261107_2141_H_rep09_att01` | **rep09** | 49 | **46** (93.9%) | 0 (0.0%) | 0 (0.0%) | 3 (6.1%) |
| `pilot_20261107_2141_H_rep10_att01` | **rep10** | 37 | **20** (54.1%) | 0 (0.0%) | 0 (0.0%) | 17 (45.9%) |
| **TỔNG CỘNG 10 REPS** | - | **455** | **140 (30.8%)** | **22 (4.8%)** | **2 (0.4%)** | **291 (64.0%)** |

### B. Đặc điểm phân bố không gian và thời gian
1. **Lượt có tỷ lệ H → V cao nhất:**
   - **`rep09` đạt tỷ lệ H → V kỷ lục: 93.9% (46 / 49 frames)**. Trong suốt lượt thử này, người dùng hầu như chỉ nhận được phản hồi là chữ V liên tục.
   - **`rep10` đạt tỷ lệ H → V: 54.1% (20 / 37 frames)**.
   - **`rep01` và `rep02` dao động từ 33.3% đến 40.0%**.
2. **Lượt không bị nhầm H → V:**
   - **`rep04` có 0 frame V (0.0%)**, nhưng lại có 5 frame K và 53 frame NEUTRAL (91.4%).
   - **`rep03` chỉ có 1 frame V (2.3%)**, còn lại là 4 frame K và 39 frame NEUTRAL (88.6%).
3. **Tính liên tục theo chuỗi thời gian (Temporal Clustering):**
   - Lỗi H → V xuất hiện theo các dải frame liên tục kéo dài:
     - `rep09`: Chuỗi `V:5` $\rightarrow$ `NEUTRAL:1` $\rightarrow$ `V:17` $\rightarrow$ `NEUTRAL:2` $\rightarrow$ `V:24`.
     - `rep08`: Đầu lượt là NEUTRAL (21 frames), sau đó duy trì chuỗi `V:11` và `V:4`.
     - `rep07`: Xuất hiện các xung `V:5` $\rightarrow$ `V:2` $\rightarrow$ `V:5` trước khi tay co lại thành `K:4` và `K:3`.

---

## 3. BẢNG SO SÁNH ĐẶC TRƯNG HÌNH HỌC THỰC TẾ: H → V vs H → NEUTRAL vs V → V

Bảng dữ liệu trích xuất từ 793 frames thực tế (140 frames H → V, 291 frames H → NEUTRAL, và 362 frames V → V):

| Tên Đặc trưng Hình học (Features) | H → V (140 frames)<br>Mean ± Std [Median] | H → NEUTRAL (291 frames)<br>Mean ± Std [Median] | V → V (362 frames)<br>Mean ± Std [Median] | Ngưỡng quy tắc V hiện hành (`alphabet_classifier.py`) |
|:---|:---:|:---:|:---:|:---|
| **`ang_IM_dir`** (°)<br>*Góc mở trỏ-giữa* | 52.961 ± 3.528 [52.660] | 55.381 ± 4.049 [55.300] | 20.574 ± 3.348 [21.865] |
| **`mid_horiz_ratio`**<br>*Tỷ lệ phương ngang ngón giữa* | 0.885 ± 0.043 [0.899] | 0.864 ± 0.041 [0.871] | 0.262 ± 0.065 [0.275] |
| **`idx_horiz_ratio`**<br>*Tỷ lệ phương ngang ngón trỏ* | 0.168 ± 0.072 [0.201] | 0.087 ± 0.062 [0.079] | 0.101 ± 0.050 [0.121] |
| **`d_MW`**<br>*Khoảng cách cổ tay - đầu ngón giữa* | 1.740 ± 0.033 [1.733] | 1.615 ± 0.060 [1.624] | 1.923 ± 0.025 [1.917] |
| **`d_IW`**<br>*Khoảng cách cổ tay - đầu ngón trỏ* | 2.068 ± 0.044 [2.074] | 1.966 ± 0.055 [1.959] | 1.862 ± 0.034 [1.848] |
| **`d_IM`**<br>*Khoảng cách đầu ngón trỏ - giữa* | 0.888 ± 0.067 [0.894] | 0.811 ± 0.065 [0.807] | 0.503 ± 0.053 [0.524] |
| **`diff_x_IM`**<br>*Độ chênh lệch trục X trỏ-giữa* | 0.649 ± 0.043 [0.644] | 0.605 ± 0.055 [0.595] | 0.494 ± 0.054 [0.517] |
| **`mcp_width_2d`**<br>*Độ rộng khớp bàn tay 2D (MCP)* | 0.562 ± 0.030 [0.567] | 0.514 ± 0.032 [0.509] | 0.514 ± 0.027 [0.518] |
| **`thumb_ring_dist`**<br>*Khoảng cách ngón cái - áp út* | 0.430 ± 0.044 [0.428] | 0.420 ± 0.044 [0.421] | 0.479 ± 0.072 [0.493] |
| **`thumb_mid_dist`**<br>*Khoảng cách ngón cái - giữa* | 0.229 ± 0.031 [0.230] | 0.183 ± 0.034 [0.178] | 0.610 ± 0.109 [0.634] |
| **`idx_tip_y`**<br>*Tọa độ chuẩn hóa Y đầu ngón trỏ* | -2.011 ± 0.034 [-2.014] | -1.935 ± 0.045 [-1.933] | -1.830 ± 0.033 [-1.821] |
| **`mid_tip_y`**<br>*Tọa độ chuẩn hóa Y đầu ngón giữa* | -1.415 ± 0.063 [-1.406] | -1.407 ± 0.065 [-1.404] | -1.875 ± 0.029 [-1.880] |
| **`ring_curl`**<br>*Độ gập ngón áp út (ring curl)* | 0.380 ± 0.046 [0.388] | 0.328 ± 0.043 [0.320] | 0.538 ± 0.145 [0.512] |
| **`pinky_curl`**<br>*Độ gập ngón út (pinky curl)* | 0.356 ± 0.054 [0.368] | 0.300 ± 0.052 [0.289] | 0.482 ± 0.128 [0.462] |

### Phân tích pháp y về các chỉ số đo lường thực tế

1. **Góc mở giữa ngón trỏ và ngón giữa (`ang_IM_dir`):**
   - Cử chỉ V thực sự (**V → V**): Góc mở tự nhiên trung bình là **$20.57°$** (Median: $21.87°$).
   - Cử chỉ H bị nhận thành V (**H → V**): Góc mở trung bình lên tới **$52.96°$** (Median: $52.66°$, Min: $43.8°$).
   - *Nguyên nhân quy tắc bị lọt lưới:* Quy tắc V chỉ đặt điều kiện chặn dưới:
     $$\text{ang\_IM\_dir} \ge \text{V\_ANG\_IM\_MIN} = 9.0°$$
     Hệ thống **hoàn toàn không có ngưỡng chặn trên** cho góc mở của chữ V! Do đó góc mở rộng $53°$ của chữ H hoàn toàn thỏa mãn điều kiện $\ge 9.0°$.

2. **Tỷ lệ định hướng ngang của ngón giữa (`mid_horiz_ratio`):**
   - Cử chỉ V thực sự (**V → V**): Ngón giữa dựng đứng theo phương dọc, `mid_horiz_ratio` chỉ đạt **$0.262$** (Median: $0.275$).
   - Cử chỉ H bị nhận thành V (**H → V**): Ngón giữa chỉ ngang sang bên cạnh, `mid_horiz_ratio` lên tới **$0.885$** (Median: $0.899$, độ nghiêng ngang $> 60°$ so với trục đứng).
   - *Nguyên nhân quy tắc bị lọt lưới:* Quy tắc V (dòng 568–579) **chỉ kiểm tra duy nhất ngón trỏ** (`idx_horiz_ratio <= 0.50`), **HOÀN TOÀN BỎ QUÊN ngón giữa**! Trong khi đó, người dùng thực hiện H với ngón trỏ dựng đứng nhẹ (`idx_horiz_ratio = 0.168 <= 0.50`), khiến Rule V lầm tưởng bàn tay đang dựng đứng.

3. **Cơ chế phân tách giữa H → V và H → NEUTRAL:**
   - Khoảng cách từ cổ tay đến ngón giữa (`d_MW`):
     - Ở **H → V**: `d_MW` đạt trung bình **$1.740$** (Median: $1.733 \ge 1.70$).
     - Ở **H → NEUTRAL**: `d_MW` chỉ đạt **$1.615$** (Median: $1.624 < 1.70$).
   - *Kết luận pháp y:* Khi người dùng đưa ngón giữa ra xa hơn một chút khiến $d_{MW} \ge 1.70$, Rule V lập tức kích hoạt và cướp nhãn thành V. Khi ngón giữa co lại một chút ($d_{MW} < 1.70$), Rule V bị trượt điều kiện `d_MW_ge_min` (chiếm 288/291 frame NEUTRAL, tương đương 99.0%). Khi Rule V trượt, hệ thống gọi MLP; MLP đoán thô thành V và bị Gating chặn lại, biến frame thành `NEUTRAL`.

---

## 4. ĐIỀU KIỆN MÃ NGUỒN CHÍNH XÁC DẪN ĐẾN LỖI (CODE PATH FORENSICS)

### A. Vị trí mã nguồn kích hoạt Rule V
Tệp tin: [`backend/vision/alphabet_classifier.py`](file:///d:/SIGN_LANGUAGE/backend/vision/alphabet_classifier.py)  
Hàm: `evaluate_rules(self, feat: Dict[str, Any])` (dòng 556–585)

```python
568: if i_ext and m_ext and not r_ext and not p_ext:
569:     if (
570:         d_iw >= V_D_IW_MIN              # d_iw = 2.068 >= 1.70 (THỎA MÃN)
571:         and d_mw >= V_D_MW_MIN          # d_mw = 1.740 >= 1.70 (THỎA MÃN)
572:         and d_im >= V_D_IM_MIN          # d_im = 0.888 >= 0.28 (THỎA MÃN)
573:         and diff_x_im >= V_DIFF_X_IM_MIN# diff_x = 0.649 >= 0.18 (THỎA MÃN)
574:         and ang_im_dir >= V_ANG_IM_MIN  # ang = 52.96° >= 9.0° (THỎA MÃN DO KHÔNG CÓ CHẶN TRÊN)
575:         and idx_horiz_ratio <= V_IDX_HORIZ_MAX  # idx_horiz = 0.168 <= 0.50 (THỎA MÃN VÌ CHỈ CHECK TRỎ)
576:         and idx_tip_y <= V_IDX_TIP_Y_MAX        # idx_tip_y = -2.011 <= -0.80 (THỎA MÃN)
577:         and v_width_ok                  # mcp_width = 0.562 >= 0.48 (THỎA MÃN)
578:         and v_thumb_ok                  # thumb_ring = 0.430 >= 0.21 (THỎA MÃN)
579:     ):
580:         matches.append((
581:             "V",
582:             f"Trỏ & Giữa xòe chữ V: d_IM={d_im:.2f}>={V_D_IM_MIN}, ang_IM={ang_im_dir:.1f}°>={V_ANG_IM_MIN}°, MCP_width={mcp_width_2d:.2f}>={V_MCP_WIDTH_MIN}",
583:             0.95
584:         ))
```

### B. Vị trí mã nguồn ưu tiên Hình học cướp quyền MLP
Tệp tin: [`backend/vision/recognizer.py`](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py)  
Hàm: `recognize_landmarks` (dòng 1124–1129)

```python
1124: if status == "MATCH" and alpha_res.get("letter") in ALLOWED_LETTERS:
1125:     final_sign = alpha_res["letter"]           # Gán final_sign = "V"
1126:     final_conf = alpha_res["confidence"]       # Gán final_conf = 0.95
1127:     recognition_source = "geometric_rule"      # Nguồn: geometric_rule
1128:     alpha_res["source"] = "geometric_rule"
1129:     gating_status = "passed_geometric"
```

### C. Tại sao cơ chế khử xung đột V trong Rule H không kích hoạt?
Trong `alphabet_classifier.py` (dòng 774–778), bộ lọc khử xung đột được định nghĩa:
```python
774: if is_h_horiz or is_h_north or is_h_south:
775:     if is_h_horiz or is_h_south:
776:         # Loại bỏ các phán đoán nhầm V, D, L, U, R khi cử chỉ thỏa mãn H ngang hoặc H Miền Nam
777:         matches[:] = [m for m in matches if m[0] not in ("V", "D", "L", "U", "R")]
```
Tuy nhiên:
- `is_h_horiz` **trượt 100% (455/455 frames)** do yêu cầu cả 2 ngón cùng nằm ngang (`idx_horiz_ratio >= 0.50`, thực tế chỉ $0.168$) và góc kẹp hẹp (`ang_im_dir <= 28.0°`, thực tế lên tới $53°$).
- `is_h_south` **trượt 100% (455/455 frames)** do yêu cầu lòng bàn tay quay nghiêng hướng camera (`mcp_width_2d <= 0.40`, thực tế $0.562$).
$\implies$ Cả 3 biến thể của Rule H đều trả về False, khiến khối mã dòng 777 **hoàn toàn không được thực thi**, để lại nhãn V trong danh sách `matches`.

---

## 5. PHÂN ĐỊNH RÕ RÀNG: BẰNG CHỨNG XÁC THỰC vs GIẢ THUYẾT vs ĐIỂM CHƯA THỂ KẾT LUẬN

### A. Những điều ĐÃ ĐƯỢC XÁC NHẬN CHẮC CHẮN (Confirmed Facts)
1. 100% trong số 140 frame H → V được sinh ra bởi quy tắc hình học `Trỏ & Giữa xòe chữ V` trong `alphabet_classifier.py` (dòng 556–585).
2. MLP hoàn toàn không tham gia vào việc sinh ra nhãn V cho chữ H (0 frame).
3. Rule V thiếu hai ràng buộc hình học quan trọng:
   - Thiếu chặn trên cho `ang_im_dir` (cho phép góc mở rộng $53°$ của H đi qua vì chỉ check $\ge 9.0°$).
   - Thiếu kiểm tra hướng thẳng đứng của ngón giữa (`mid_horiz_ratio`).
4. Toàn bộ 455 frame của chữ H trong đợt thử nghiệm này đều thất bại trước 3 biến thể hình học hiện tại của Rule H (H ngang, H Miền Bắc, H Miền Nam).
5. 291 frame H → NEUTRAL là các frame mà ngón giữa co lại một chút làm $d_{MW} < 1.70$, khiến Rule V trượt và MLP bị Gating chặn.

### B. Giả thuyết kỹ thuật cần kiểm chứng trong nhiệm vụ sửa lỗi (Hypotheses)
1. **Giả thuyết G1 (Ràng buộc hướng ngón giữa cho V):** Nếu thêm điều kiện `mid_horiz_ratio <= 0.50` và `ang_im_dir <= 38.0°` vào Rule V, toàn bộ 140 frame H → V sẽ bị loại bỏ ngay lập tức mà không làm ảnh hưởng đến nhận diện của V chuẩn (vì V chuẩn có `mid_horiz_ratio = 0.262` và `ang_im_dir = 20.57°`).
2. **Giả thuyết G2 (Định nghĩa biến thể H chữ L / H mở góc):** Khi người dùng làm cử chỉ H trước webcam, một ngón có xu hướng hơi chếch lên và một ngón chĩa ngang (góc mở $\approx 50°$). Việc thiết kế một biến thể hình học H phản ánh đúng dáng tay này sẽ kích hoạt cơ chế xóa nhầm V tại dòng 777.

### C. Những điều KHÔNG THỂ KẾT LUẬN từ dữ liệu hiện tại
1. Không thể kết luận người dùng đã làm sai hay đúng theo quy ước động tác chuẩn VSL chỉ từ landmark 2D, vì không có video RGB gốc được ghi lại đồng thời.
2. Không thể kết luận liệu mạng MLP có nhận diện được chữ H hay không nếu bỏ qua tầng hình học, vì hiện tại chữ H không nằm trong danh sách `ALLOWED_SUPPLEMENTAL_LETTERS` hoặc `ALLOWED_FALLBACK_BASE_LETTERS` của cơ chế Gating.

---

## 6. KẾT LUẬN CUỐI CÙNG & DANH SÁCH BẰNG CHỨNG CÒN THIẾU

### A. Kết luận pháp y
Hiện tượng nhầm lẫn 140 frame **H → V nằm 100% ở Tầng 1: Luật Hình học (Geometric Rule)**. Lỗi phát sinh do sự kết hợp của hai khiếm khuyết trong `backend/vision/alphabet_classifier.py`:
1. **Rule V quá lỏng lẻo:** Cho phép ngón giữa nằm ngang (`mid_horiz_ratio = 0.885`) và góc mở quá lớn (`ang_im_dir = 53°`) kích hoạt chữ V.
2. **Rule H quá khắt khe:** Đòi hỏi cả 2 ngón cùng nằm ngang (`idx_horiz_ratio >= 0.50`) và góc kẹp cực hẹp ($\le 28.0°$), khiến không một frame H thực tế nào kích hoạt được Rule H để triệt tiêu Rule V.

### B. Danh sách bằng chứng còn thiếu (Missing Evidence Checklist)
- Hiện tại hệ thống không lưu ảnh RGB của phiên webcam, do đó mọi phân tích đều dựa trên tọa độ 21 điểm landmarks 3D MediaPipe. (Tuy nhiên, dữ liệu telemetry landmarks và 18 trường diagnostic là **đầy đủ 100%** để xác định dứt khoát nguyên nhân mã nguồn).

---
*Báo cáo pháp y hoàn thành độc lập và đối chiếu toàn vẹn với tệp dữ liệu chẩn đoán webcam `pilot_pilot_20261107_2141.jsonl`.*
