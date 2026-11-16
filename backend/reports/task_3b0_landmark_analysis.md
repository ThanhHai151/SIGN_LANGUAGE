# BÁO CÁO TASK 3B.0 — PHÂN TÍCH LANDMARK THẬT TRƯỚC KHI THIẾT KẾ THUẬT TOÁN

**Dự án:** VSL Connect AI — Hệ thống giao tiếp hai chiều bằng Ngôn ngữ ký hiệu Việt Nam  
**Giai đoạn:** Task 3B.0 — Nghiên cứu và phân tích đặc trưng hình học từ dữ liệu landmark thật  
**Ngày thực hiện:** 30/10/2026  
**Người thực hiện:** Senior Computer Vision Engineer  
**Tập dữ liệu phân tích:** [`backend/tests/fixtures/real_alphabet_landmarks.json`](file:///d:/SIGN_LANGUAGE/backend/tests/fixtures/real_alphabet_landmarks.json) (203 bản ghi quét từ 29 video QIPEDC, gồm 122 mẫu `ok` và 81 mẫu `no_hand_detected`)  

---

## I. Xác minh chất lượng dữ liệu (Phần A)

### 1.1. Hiện trạng số lượng mẫu và tính toàn vẹn
- **Tổng số bản ghi:** **203 mẫu** phân bố đồng đều 7 frame cho mỗi video (tại các mốc tỷ lệ 10%, 25%, 40%, 50%, 60%, 75%, 90% thời lượng).
- **Mẫu thành công (`quality_status: "ok"`):** **122 mẫu** (đạt 60.1%). Mỗi chữ cái có từ 3 đến 6 mẫu thành công.
- **Mẫu không phát hiện tay (`quality_status: "no_hand_detected"`):** **81 mẫu** (chiếm 39.9%).
  - 100% mẫu này có `landmarks: null`, `confidence: 0`, không chứa dữ liệu giả.
  - Phân bố tập trung ở 2 biên: mốc 10% (`phase: "entry"`, tay chưa đưa lên) và mốc 90% (`phase: "exit"`, tay đã hạ xuống).
- **Lỗi định dạng dữ liệu (Data Format Errors):** **0 lỗi**. Cả 122 mẫu `ok` đều vượt qua kiểm định nghiêm ngặt của `validate_landmark_format()`, tọa độ hữu hạn, chuẩn hóa trong phạm vi quan sát.
- **Tính truy nguyên (Traceability):** 100% mẫu đều liên kết chính xác với file video thật trên đĩa [`backend/data/media_cache/videos/`](file:///d:/SIGN_LANGUAGE/backend/data/media_cache/videos/), có `frame_idx` và `timestamp_sec` trùng khớp hoàn toàn với luồng frame OpenCV.

### 1.2. Phân loại và nghi vấn về nhãn / pha (Label & Phase Ambiguities)

Hệ thống phân biệt rõ ràng 3 nhóm trạng thái mẫu:

1. **Nhóm mẫu chuẩn tư thế tĩnh (Static Steady Poses):**  
   Bao gồm các mẫu ở khoảng giữa ($40\% - 60\%$) của 22 chữ cái tĩnh (*A, B, C, D, E, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, X, Y*). Các mẫu này thể hiện trọn vẹn và ổn định hình thái cử chỉ của chữ cái.
2. **Nhóm mẫu ở giai đoạn chuyển tiếp (Transition Poses):**  
   Các mẫu ở mốc 25% hoặc 75% của một số chữ có chuyển động đưa tay nhanh (ví dụ chữ *X* `D0517B`, chữ *G* `D0498B`, chữ *M* `D0504`). Ở các frame này, bàn tay đang xoay hướng hoặc bắt đầu gập ngón, hình thái chưa đạt đỉnh điểm ổn định.
3. **Nghi vấn về pha đối với 7 chữ cái có dấu (Compound Diacritics):**  
   Bao gồm: **Ă** (`D0490B`), **Â** (`D0491B`), **Ê** (`D0497B`), **Ô** (`D0507B`), **Ơ** (`D0508N`), **Ư** (`D0515N`), **Đ** (`D0495`).
   - *Căn cứ:* Theo mô tả `how_to` trong từ điển QIPEDC, các chữ này gồm 2 pha vận động:
     - Pha 1 (Base pose): Tạo hình chữ cái gốc (A, E, O, U, D).
     - Pha 2 (Diacritic stroke): Thực hiện cử động tạo dấu (lật ngửa tay tạo dấu á, úp tay tạo nón, vẩy ngón út tạo râu, gập/vạch ngón tạo chữ đ).
   - *Kết luận:* Trong fixture hiện tại, các frame ở dải giữa của 7 chữ này được gắn nhãn minh bạch là `phase: "unknown"` vì chưa thể khẳng định một frame đơn lẻ là đang ở pha chữ gốc hay pha nét dấu nếu không theo dõi chuỗi thời gian (temporal sequence).

---

## II. Phân tích định lượng các cặp chữ cái dễ nhầm (Phần B)

Toàn bộ đặc trưng được tính toán tự động bằng công cụ phân tích [`backend/tools/analyze_real_landmarks.py`](file:///d:/SIGN_LANGUAGE/backend/tools/analyze_real_landmarks.py) trên 122 mẫu `ok`. Dữ liệu chi tiết được lưu trữ tại [`backend/reports/task_3b0_feature_analysis.json`](file:///d:/SIGN_LANGUAGE/backend/reports/task_3b0_feature_analysis.json).

Các khoảng cách và sai lệch tọa độ đều được chuẩn hóa theo kích thước lòng bàn tay $palm\_scale = \|\vec{P}_{\text{wrist}} - \vec{P}_{\text{middle\_mcp}}\|$.

---

### 2.1. Cặp R vs U (Bắt chéo ngón vs Khép ngón song song)
- **Mô tả cử chỉ:**
  - **U:** Hai ngón trỏ và giữa duỗi thẳng, khép sát song song nhau.
  - **R:** Ngón giữa bắt chéo ra phía sau ngón trỏ.
- **Bảng so sánh phân bố đặc trưng:**

| Đặc trưng hình học | Chữ R ($N=3$) (Mean ± Std [Min, Max]) | Chữ U ($N=4$) (Mean ± Std [Min, Max]) | Khoảng chồng lấn | Đánh giá phân tách |
| :--- | :--- | :--- | :---: | :---: |
| `curl_index` (Độ co ngón trỏ) | $0.076 \pm 0.053$ $[0.00, 0.12]$ | $0.005 \pm 0.009$ $[0.00, 0.02]$ | $[0.00, 0.02]$ | Chồng lấn (cả 2 đều duỗi) |
| `curl_middle` (Độ co ngón giữa) | $0.042 \pm 0.008$ $[0.03, 0.05]$ | $0.002 \pm 0.002$ $[0.00, 0.01]$ | **Tách biệt hoàn toàn** | U duỗi phẳng tuyệt đối |
| `d_IM` (Khoảng cách đầu Trỏ - Giữa) | $0.479 \pm 0.335$ $[0.23, 0.95]$ | $0.295 \pm 0.271$ $[0.13, 0.76]$ | $[0.23, 0.76]$ | **Chồng lấn mạnh trong 2D** |
| `diff_x_IM` ($x_{\text{index}} - x_{\text{middle}}$) | $0.025 \pm 0.100$ $[-0.10, 0.14]$ | $0.139 \pm 0.039$ $[0.11, 0.21]$ | $[0.11, 0.14]$ | Chồng lấn ở góc quay nghiêng |
| `diff_z_IM` ($z_{\text{index}} - z_{\text{middle}}$) | $-0.008 \pm 0.023$ $[-0.04, 0.01]$ | $-0.050 \pm 0.085$ $[-0.20, 0.01]$ | $[-0.04, 0.01]$ | Chồng lấn trục sâu z |

> [!WARNING]
> **Nhận xét then chốt cho R vs U:**
> Nếu chỉ dùng khoảng cách Euclid 2D/3D `d_IM`, **hoàn toàn không thể phân biệt R và U** vì khi bắt chéo ngón, khoảng cách giữa 2 đầu ngón tay cũng rất nhỏ ($0.23$). Để phân biệt tin cậy, thuật toán cần kiểm tra sự hoán đổi vị trí tương đối trên trục x (`diff_x_IM` đảo dấu khi ngón giữa bắt chéo qua ngón trỏ) kết hợp với vector chỉ phương của 2 ngón tay.

---

### 2.2. Cặp M vs N (3 ngón úp vs 2 ngón úp qua ngón cái)
- **Mô tả cử chỉ:**
  - **M:** 3 ngón (trỏ, giữa, áp út) úp xuống qua ngón cái; ngón út gập.
  - **N:** 2 ngón (trỏ, giữa) úp xuống qua ngón cái; ngón áp út và ngón út gập.
- **Bảng so sánh phân bố đặc trưng:**

| Đặc trưng hình học | Chữ M ($N=3$) (Mean ± Std [Min, Max]) | Chữ N ($N=5$) (Mean ± Std [Min, Max]) | Khoảng chồng lấn | Đánh giá phân tách |
| :--- | :--- | :--- | :---: | :---: |
| `curl_ring` (Độ co ngón áp út) | $0.013 \pm 0.012$ $[0.00, 0.03]$ | $0.180 \pm 0.285$ $[0.01, 0.75]$ | $[0.01, 0.03]$ | N có ngón áp út gập sâu hơn |
| `d_TI` (Khoảng cách Cái - Trỏ) | $1.021 \pm 0.272$ $[0.74, 1.39]$ | $0.887 \pm 0.523$ $[0.17, 1.60]$ | $[0.74, 1.39]$ | Chồng lấn lớn |
| `d_TM` (Khoảng cách Cái - Giữa) | $1.131 \pm 0.351$ $[0.83, 1.62]$ | $1.173 \pm 0.620$ $[0.20, 1.88]$ | $[0.83, 1.62]$ | Chồng lấn lớn |
| `d_IW` (Đầu ngón trỏ đến Cổ tay) | $3.463 \pm 0.706$ $[2.46, 3.98]$ | $3.236 \pm 1.273$ $[0.98, 4.65]$ | $[2.46, 3.98]$ | Cùng hướng trúc xuống |

> [!IMPORTANT]
> **Nhận xét then chốt cho M vs N:**
> Trong góc nhìn camera 2D chính diện, ngón cái bị các ngón tay khác che khuất một phần (occlusion). Điểm phân biệt duy nhất mang tính bản chất giữa M và N là **trạng thái của ngón áp út (Ring finger)**:
> - Ở chữ M: ngón áp út duỗi trúc xuống cùng ngón trỏ và ngón giữa.
> - Ở chữ N: ngón áp út gập thu vào lòng bàn tay.
> Bất kỳ luật hình học nào dựa vào khoảng cách ngón cái đều sẽ thất bại do độ bất định che khuất của MediaPipe.

---

### 2.3. Cặp S vs T (Nắm đấm ngón cái vắt ngoài vs Ngón cái kẹp giữa trỏ và giữa)
- **Mô tả cử chỉ:**
  - **S:** 4 ngón nắm chặt thành nắm đấm tròn, ngón cái vắt ngang bên ngoài mặt ngón trỏ/giữa.
  - **T:** Ngón trỏ bắt chéo qua ngón cái, ngón cái chen kẹp vào khe giữa ngón trỏ và ngón giữa.
- **Bảng so sánh phân bố đặc trưng:**

| Đặc trưng hình học | Chữ S ($N=4$) (Mean ± Std [Min, Max]) | Chữ T ($N=3$) (Mean ± Std [Min, Max]) | Khoảng chồng lấn | Đánh giá phân tách |
| :--- | :--- | :--- | :---: | :---: |
| `d_TI` (Khoảng cách Cái - Trỏ) | $0.389 \pm 0.100$ $[0.30, 0.56]$ | $0.116 \pm 0.015$ $[0.09, 0.13]$ | **Tách biệt hoàn toàn** | T có ngón cái kẹp sát trỏ |
| `d_IM` (Khoảng cách Trỏ - Giữa) | $0.321 \pm 0.094$ $[0.24, 0.48]$ | $0.926 \pm 0.070$ $[0.83, 0.98]$ | **Tách biệt hoàn toàn** | S nắm chụm, T mở khe |
| `curl_middle` (Độ co ngón giữa) | $0.138 \pm 0.094$ $[0.00, 0.23]$ | $0.000 \pm 0.000$ $[0.00, 0.00]$ | **Tách biệt hoàn toàn** | T duỗi/thẳng hơn |

> [!TIP]
> **Nhận xét then chốt cho S vs T:**
> Dữ liệu landmark thật cho thấy cặp S và T **có khả năng tách biệt hình học rất cao**. Khoảng cách `d_TI` của T ($0.09 - 0.13$) nhỏ hơn rõ rệt so với S ($0.30 - 0.56$). Đồng thời khoảng cách giữa đầu ngón trỏ và ngón giữa `d_IM` ở T mở rộng ($0.83 - 0.98$) do ngón cái chèn vào giữa, trong khi ở S các ngón xếp khít nhau ($0.24 - 0.48$).

---

### 2.4. Cặp C vs O (Khum mở rỗng vs Chụm tròn kín)
- **Mô tả cử chỉ:**
  - **C:** Các ngón cong hình vòng cung tròn, đầu ngón cái và ngón trỏ tách rời tạo khe hở mở.
  - **O:** Các ngón cong, đầu ngón trỏ (và các ngón khác) chạm sát tiếp xúc vào đầu ngón cái tạo thành vòng khuyên khép kín.
- **Phân bố thực tế:**
  - `d_TI` của Chữ C ($N=4$): Trung bình $0.782 \pm 0.115$ $[0.65, 0.94]$.
  - `d_TI` của Chữ O ($N=5$): Trung bình $0.165 \pm 0.048$ $[0.10, 0.22]$.
  - **Độ chồng lấn:** **Tách biệt hoàn toàn** (ngưỡng phân định tự nhiên nằm trong khoảng $[0.30, 0.50]$).

---

### 2.5. Cặp U vs V (2 ngón khép sát vs 2 ngón xòe chữ V)
- **Mô tả cử chỉ:** Cả hai chữ đều duỗi ngón trỏ và ngón giữa, gập 2 ngón còn lại.
- **Phân bố thực tế:**
  - Độ mở ngang `diff_x_IM` ($x_{\text{index}} - x_{\text{middle}}$):
    - Chữ U ($N=4$): $0.139 \pm 0.039$ $[0.11, 0.21]$ (khép sát).
    - Chữ V ($N=5$): $0.326 \pm 0.063$ $[0.21, 0.39]$ (xòe rộng).
  - Khoảng cách 3D `d_IM`: Chữ U đạt $0.295$, chữ V đạt $0.356$.
  - **Độ chồng lấn:** Biên trên của U ($0.21$) chạm biên dưới của V ($0.21$). Cần kết hợp thêm góc mở giữa 2 ngón tay $\angle(\text{Index\_TIP}, \text{MCP}, \text{Middle\_TIP})$ để bảo đảm tính phân tách khi góc nghiêng camera thay đổi.

---

### 2.6. Cặp D vs L (1 ngón trỏ duỗi vs Trỏ và Cái vuông góc)
- **Phân bố thực tế:**
  - `d_TI` (Khoảng cách đầu ngón cái đến ngón trỏ):
    - Chữ D ($N=4$): $1.075 \pm 0.201$ $[0.73, 1.24]$ (ngón cái gập vào ngón giữa).
    - Chữ L ($N=5$): $1.654 \pm 0.274$ $[1.31, 2.15]$ (ngón cái mở vuông góc sang ngang).
  - Góc giữa ngón cái và ngón trỏ: Chữ L đạt góc $\approx 75^\circ - 90^\circ$, chữ D đạt góc $< 35^\circ$.
  - **Độ chồng lấn:** Tách biệt tốt trên đặc trưng khoảng cách và góc mở ngón cái.

---

### 2.7. Cặp A vs E (Ngón cái duỗi đứng vs 4 ngón đè lên đầu ngón cái)
- **Phân bố thực tế:**
  - `d_TM` (Khoảng cách đầu ngón cái đến ngón giữa):
    - Chữ A ($N=4$): $0.841 \pm 0.040$ $[0.80, 0.90]$.
    - Chữ E ($N=4$): $0.427 \pm 0.190$ $[0.30, 0.76]$.
  - Chiều cao đầu ngón cái `d_TW` (so với cổ tay): Chữ A ngón cái dựng thẳng ($1.43 \pm 0.04$), chữ E ngón cái gập ngang dưới các đầu ngón tay ($1.01 \pm 0.27$).
  - **Độ chồng lấn:** Tách biệt rõ nét về hình thái ngón cái.

---

## III. Kiểm tra khả năng đánh giá độc lập (Phần C)

### 3.1. Thống kê nguồn video và người ký (Signer Diversity)
- **Tổng số video:** 29 video.
- **Số lượng video trên mỗi chữ cái:** **Đúng 1 video duy nhất cho mỗi chữ cái**.
- **Số người ký độc lập:** **1 người ký duy nhất** (cùng một diễn viên thực hiện toàn bộ 29 video chữ cái trong kho ngữ liệu QIPEDC).

### 3.2. Cảnh báo nghiêm ngặt về phương pháp luận đánh giá
1. **Không thể chia ngẫu nhiên khung hình (No Random Frame Splitting):**  
   Các frame trong cùng một video được ghi hình trong cùng một lần diễn (cùng điều kiện ánh sáng, cự ly camera, góc quay và đặc điểm bàn tay). Việc chia ngẫu nhiên các frame của cùng một video thành tập Train và Test sẽ gây hiện tượng **rò rỉ dữ liệu nghiêm trọng (Data Leakage)**, khiến mô hình chỉ "học vẹt" bàn tay của diễn viên đó.
2. **Giới hạn của tập Dev/Eval trong Fixture:**  
   Việc phân tách mẫu `dev` ($t \approx 40\%$) và `eval` ($t \approx 60\%$) trong fixture hiện tại chỉ có giá trị **kiểm định tính khớp hình học nội tại (Geometric Consistency Check)** của thuật toán trên dữ liệu video gốc, **tuyệt đối không được xem là bằng chứng về khả năng nhận diện tổng quát (Cross-Signer Generalization)** khi người dùng thực tế sử dụng webcam.

---

## IV. Danh mục các chữ cái cần xử lý theo thời gian (Temporal Processing)

Dựa trên phân tích vận động cử chỉ từ video QIPEDC, bảng chữ cái VSL được phân thành 2 nhóm kỹ thuật rõ rệt:

### 4.1. Nhóm 22 chữ cái tĩnh (Single-Frame Static Gestures)
- **Danh sách:** `A, B, C, D, E, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, X, Y`.
- **Cơ chế xử lý khuyến nghị:** Hoàn toàn có thể áp dụng bộ phân loại hình học tức thì (Rule-based Geometric Classifier, độ trễ $< 5\text{ms}$) trên từng frame độc lập, kết hợp bộ đệm thời gian chống rung (Temporal Smoothing Deque) sẵn có trong `recognizer.py`.

### 4.2. Nhóm 7 chữ cái nhiều pha / ghép dấu (Temporal Compound Gestures)
- **Danh sách:** `Ă, Â, Ê, Ô, Ơ, Ư, Đ`.
- **Đặc trưng:**
  - **Ă, Â:** Đều xuất phát từ tư thế chữ A, sau đó có động tác lật ngửa tay hoặc úp tay vẽ nón.
  - **Ê:** Xuất phát từ chữ E, sau đó úp tay vẽ nón.
  - **Ô, Ơ:** Xuất phát từ chữ O, sau đó vẽ nón hoặc vẩy ngón út vẽ râu.
  - **Ư:** Xuất phát từ chữ U, sau đó vẩy ngón út vẽ râu.
  - **Đ:** Xuất phát từ chữ D, sau đó có cử động gập ngón trỏ hoặc vạch ngang.
- **Rủi ro nếu xử lý tĩnh:** Nếu chỉ nhận diện trên 1 frame tĩnh, hệ thống sẽ **luôn luôn nhận nhầm thành chữ cái gốc** (Ă/Â thành A, Ê thành E, Ô/Ơ thành O, Ư thành U, Đ thành D).
- **Cơ chế xử lý bắt buộc:** Cần áp dụng **máy trạng thái hữu hạn theo thời gian (Temporal State Machine / Dynamic Transition Engine)**:
  - Trạng thái 1: Nhận diện và giữ tư thế chữ cái gốc ($T_{\text{hold}} \ge 200\text{ms}$).
  - Trạng thái 2: Theo dõi quỹ đạo chuyển động đầu ngón tay trong cửa sổ tiếp theo ($300\text{ms} - 600\text{ms}$) để kích hoạt chuyển đổi sang ký tự có dấu tương ứng.

---

## V. Đề xuất ưu tiên kỹ thuật cho Task 3B

1. **Ưu tiên 1 (Thiết kế Bộ phân loại Hình học Bảng chữ cái cho 22 chữ tĩnh):**
   - Xây dựng module `AlphabetClassifier` tương tự như `VSLDigitClassifier` đã thành công ở Stage 2C.
   - Định nghĩa các luật hình học phân định dựa trên các đặc trưng không chồng lấn đã được chứng minh trong báo cáo này (độ mở `diff_x_IM` cho U/V, `d_TI` cho S/T và C/O, `curl_ring` cho M/N).
2. **Ưu tiên 2 (Cơ chế phân tách riêng biệt cho cặp khó M/N và R/U):**
   - Đối với R/U: Bắt buộc dùng hoán vị tọa độ trục x và góc chéo vector, không dùng khoảng cách 2D đơn lẻ.
   - Đối với M/N: Tập trung vào phát hiện ngón áp út (Ring finger) thay vì tìm ngón cái bị che khuất.
3. **Ưu tiên 3 (Thiết kế State Machine cho 7 chữ cái có dấu):**
   - Tách biệt nhận diện 7 chữ có dấu thành cơ chế chuyển đổi trạng thái động trên nền tảng chữ cái gốc, tránh làm phức tạp hóa bộ phân loại tĩnh.
4. **Ưu tiên 4 (Mở rộng thu thập dữ liệu Cross-Signer):**
   - Thu thập thêm mẫu webcam thực tế của người dùng để bổ sung vào tập EVAL độc lập trước khi chốt độ chính xác nghiệm thu cuối cùng.

---

## VI. Trạng thái mã nguồn và kiểm thử

- **Mã nguồn production (`backend/vision/`, `src/components/`, `weights.pt`):** **TUYỆT ĐỐI NGUYÊN TRẠNG, KHÔNG SỬA ĐỔI.**
- **Tổng số kiểm thử hệ thống:** **216 / 216 PASSED** (Bảo toàn 100% tính tương thích và ổn định của dự án).
- **Dữ liệu phân tích chi tiết:** [`backend/reports/task_3b0_landmark_analysis.json`](file:///d:/SIGN_LANGUAGE/backend/reports/task_3b0_feature_analysis.json).

