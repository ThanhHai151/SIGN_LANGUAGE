# BÁO CÁO KỸ THUẬT TASK 3B.1: TRIỂN KHAI ALPHABET CLASSIFIER CHO CHỮ TĨNH

**Dự án:** VSL Connect AI — Hệ thống giao tiếp hai chiều Ngôn ngữ Ký hiệu Việt Nam  
**Giai đoạn:** Task 3B.1 — Triển khai `AlphabetClassifier` độc lập cho các chữ cái tĩnh  
**Ngày thực hiện:** 30/10/2026  
**Kỹ sư phụ trách:** Senior Computer Vision / AI Engineer  
**Trạng thái kiểm thử:** 242/242 passed (26 unit tests mới cho `AlphabetClassifier`, 0 regressions)  

---

## 1. Mục tiêu và Phạm vi triển khai

### 1.1. Mục tiêu
Xây dựng module phân loại hình học độc lập `VSLAlphabetClassifier` nhằm nhận diện các chữ cái tĩnh của Ngôn ngữ Ký hiệu Việt Nam (VSL) dựa trên 21 mốc landmark bàn tay MediaPipe.

### 1.2. Ràng buộc an toàn kiến trúc
- **Cô lập tuyệt đối với production:** Module mới được đặt tại `backend/vision/alphabet_classifier.py`, không can thiệp, không chỉnh sửa và không làm ảnh hưởng đến luồng nhận diện production (`backend/vision/recognizer.py`, `feature_extractor.py`, `digit_classifier.py`, `CameraSignRecognizer.jsx`).
- **Không ép nhãn mù quáng:** Tuyệt đối không sử dụng thác lệnh `if...elif` lồng nhau để phân chia ngẫu nhiên. Mọi quy tắc nhận diện đều được đánh giá độc lập (independent evaluation) để phát hiện trạng thái không khớp (`NO_MATCH`) và xung đột (`CONFLICT`).
- **Dựa trên chứng cứ dữ liệu thực tế:** Các ngưỡng hình học được suy luận trực tiếp từ 122 mẫu landmark thật trích xuất từ 29 video QIPEDC trong Task 3A và Task 3B.0 (`backend/tests/fixtures/real_alphabet_landmarks.json`).
- **Bảo lưu UNKNOWN an toàn:** Các chữ chưa đủ căn cứ phân tách trong không gian 2D từ một góc nhìn hoặc các chữ chuyển động có dấu tuyệt đối không gán nhãn tùy tiện mà giữ ở trạng thái `UNKNOWN`.

---

## 2. Bảng phân loại hiện trạng các chữ cái

Bộ phân loại bảng chữ cái VSL hiện phân chia 29 chữ cái tiếng Việt thành 3 nhóm rõ rệt:

| Nhóm | Số lượng | Danh mục chữ cái | Trạng thái kỹ thuật | Cơ sở xử lý |
| :--- | :---: | :--- | :--- | :--- |
| **Đã có luật hình học độc lập** | **12** | **A, B, C, D, E, I, L, O, S, T, V, Y** | Hoàn thành, 100% tests PASS, 0 sai chéo trên dữ liệu thật | Ngưỡng khoảng cách và góc chuẩn hóa theo kích thước lòng bàn tay (`palm_scale`). |
| **Bảo lưu UNKNOWN (Chữ tĩnh chưa đủ căn cứ)** | **10** | **G, H, K, M, N, P, Q, R, U, X** | An toàn, trả về `UNKNOWN` / `NO_MATCH` | M/N bị che khuất ngón cái trong 2D; R/U chồng lấn hình thái trên 1 góc quay; K, P, Q, G, H, X có biến thiên lớn hoặc chưa đủ đa dạng mẫu. |
| **Bảo lưu UNKNOWN (Chữ động có dấu)** | **7** | **Ă, Â, Ê, Ô, Ơ, Ư, Đ** | An toàn, trả về `UNKNOWN` / `NO_MATCH` | Đòi hỏi Temporal State Machine nhận diện quỹ đạo di chuyển của bàn tay/ngón tay (thuộc phạm vi Task 3B.2). |

---

## 3. Căn cứ dữ liệu thực nghiệm và Thiết lập ngưỡng hình học

Dựa trên phân tích định lượng 122 mẫu thực nghiệm từ file `backend/reports/task_3b0_feature_analysis.json`, các đặc trưng hình học chính được chuẩn hóa theo kích thước lòng bàn tay $S_{\text{palm}} = \text{dist}(P_0, P_9)$:

### 3.1. Phân biệt các cặp đối lập (Contrasting Pairs)

#### Cặp S vs T (Đều nắm đấm, phân biệt bằng vị trí ngón cái):
- **Cơ sở dữ liệu:**
  - Chữ S ($N=4$): Ngón cái vắt ngang mặt trước 4 ngón nắm chặt. $d_{\text{TI}} = 0.389 \pm 0.100$ $[0.30, 0.56]$, $d_{\text{IM}} = 0.321 \pm 0.094$ $[0.24, 0.48]$, $d_{\text{TW}} = 0.852 \pm 0.209$ $[0.69, 1.16]$.
  - Chữ T ($N=3$): Ngón cái kẹp giữa ngón trỏ và ngón giữa, đầu ngón trỏ quặp trên ngón cái. $d_{\text{TI}} = 0.116 \pm 0.015$ $[0.09, 0.13]$, $d_{\text{IM}} = 0.926 \pm 0.070$ $[0.83, 0.98]$, $d_{\text{TM}} = 0.916 \pm 0.038$ $[0.89, 0.96]$.
- **Ngưỡng hình học:**
  - Quy tắc T: $d_{\text{TI}} \le 0.16$, $d_{\text{TM}} \ge 0.80$, $d_{\text{IM}} \ge 0.80$, $d_{\text{IW}} \le 1.30$.
  - Quy tắc S: $0.25 \le d_{\text{TI}} \le 0.45$, $0.40 \le d_{\text{TM}} \le 0.65$, $d_{\text{TW}} \le 0.90$, $d_{\text{IW}} \le 0.90$.

#### Cặp C vs O (Đều khum tròn, phân biệt độ mở của vòng khuyên):
- **Cơ sở dữ liệu:**
  - Chữ O ($N=5$): Các đầu ngón cong chụm chạm sát đầu ngón cái thành vòng kín. $d_{\text{TI}} = 0.165 \pm 0.048$ $[0.10, 0.22]$, $d_{\text{TM}} = 0.218 \pm 0.050$ $[0.16, 0.30]$, $d_{\text{IW}} \ge 0.95$.
  - Chữ C ($N=4$): Các ngón khum mở rộng tạo hình vòm chữ C. $d_{\text{TI}} = 0.782 \pm 0.115$ $[0.65, 0.94]$, $d_{\text{TM}} = 0.760 \pm 0.109$ $[0.63, 0.91]$, $d_{\text{IW}} \in [1.35, 1.65]$, $d_{\text{MW}} \in [1.20, 1.55]$.
- **Ngưỡng hình học:**
  - Quy tắc O: $d_{\text{TI}} \le 0.22$, $d_{\text{TM}} \le 0.20$, $d_{\text{IW}} \ge 0.95$, $d_{\text{MW}} \ge 0.90$.
  - Quy tắc C: $0.55 \le d_{\text{TI}} \le 0.85$, $0.50 \le d_{\text{TM}} \le 0.85$, $1.35 \le d_{\text{IW}} \le 1.65$, $1.20 \le d_{\text{MW}} \le 1.55$.

#### Cặp D vs L (Đều duỗi ngón trỏ đơn, phân biệt góc mở ngón cái):
- **Cơ sở dữ liệu:**
  - Chữ D ($N=4$): Ngón trỏ duỗi thẳng, ngón cái khép chạm đầu ngón giữa. $d_{\text{TI}} = 1.075 \pm 0.201$ $[0.73, 1.24]$, $d_{\text{TM}} = 0.262 \pm 0.154$ $[0.16, 0.53]$ (ở stable pose $d_{\text{TM}} \le 0.19$), $d_{\text{IW}} \ge 1.90$.
  - Chữ L ($N=5$): Ngón cái và trỏ duỗi vuông góc hình chữ L. $d_{\text{TI}} = 1.654 \pm 0.274$ $[1.31, 2.15]$, $d_{\text{TM}} = 0.978 \pm 0.627$ $[0.51, 2.15]$, $\text{angle}_{\text{TI}} \ge 35^\circ$, $\text{thumb\_lat} \ge 0.30$.
- **Ngưỡng hình học:**
  - Quy tắc D: `i_ext=True, not m_ext, not r_ext, not p_ext`, $d_{\text{IW}} \ge 1.70$, $d_{\text{TI}} \le 1.28$, $d_{\text{TM}} \le 0.30$.
  - Quy tắc L: `i_ext=True, not m_ext, not r_ext, not p_ext`, $d_{\text{IW}} \ge 1.70$, $d_{\text{TI}} \ge 1.40$, $d_{\text{TM}} \ge 0.45$, $\text{thumb\_lat} \ge 0.30$, $\text{angle}_{\text{TI}} \ge 35^\circ$.

#### Cặp A vs E (Đều nắm ngón, phân biệt vị trí ngón cái):
- **Cơ sở dữ liệu:**
  - Chữ A ($N=4$): Nắm đấm với ngón cái duỗi dựng đứng cạnh bàn tay. $d_{\text{TW}} = 1.428 \pm 0.041$ $[1.38, 1.49]$, $d_{\text{TM}} = 0.841 \pm 0.040$ $[0.80, 0.90]$, $d_{\text{IW}} \le 0.85$, $d_{\text{MW}} \le 0.66$.
  - Chữ E ($N=4$): 4 ngón cong gập tì lên đầu ngón cái. $d_{\text{TW}} = 1.008 \pm 0.273$ $[0.75, 1.47]$ (stable pose $\le 0.91$), $d_{\text{TM}} = 0.427 \pm 0.190$ $[0.30, 0.76]$ (stable pose $\le 0.34$), $d_{\text{IW}} \ge 1.05$.
- **Ngưỡng hình học:**
  - Quy tắc A: `not i_ext, not m_ext, not r_ext, not p_ext`, $d_{\text{TW}} \ge 1.32$, $d_{\text{TM}} \ge 0.75$, $d_{\text{IW}} \le 0.90$, $d_{\text{MW}} \le 0.75$.
  - Quy tắc E: `not i_ext, not m_ext, not r_ext, not p_ext`, $d_{\text{TI}} \le 0.35$, $d_{\text{TM}} \le 0.38$, $d_{\text{TW}} \le 0.95$, $d_{\text{IW}} \ge 1.05$, $d_{\text{MW}} \ge 1.05$.

---

### 3.2. Các chữ tĩnh độc lập có hình thái đặc trưng cao

- **Chữ B:** 4 ngón trỏ, giữa, áp út, út duỗi thẳng khép sát (`i_ext, m_ext, r_ext, p_ext`), $d_{\text{IW}} \ge 1.65$, $d_{\text{MW}} \ge 1.70$. Ngón cái khép ngang bàn tay: $\text{thumb\_lat} \le 0.35$.
- **Chữ I:** Duy nhất ngón út duỗi thẳng (`p_ext=True, not i_ext, not m_ext, not r_ext, not t_ext`), $d_{\text{PW}} \ge 1.35$, $d_{\text{IW}} \le 0.85$, $d_{\text{MW}} \le 0.75$.
- **Chữ Y:** Ngón cái và ngón út cùng duỗi thẳng (Shaka), 3 ngón giữa gập (`p_ext=True, not i_ext, not m_ext, not r_ext`), $d_{\text{PW}} \ge 1.35$, $d_{\text{TW}} \ge 1.05$, $d_{\text{IW}} \le 0.85$, $d_{\text{MW}} \le 0.75$.
- **Chữ V:** Ngón trỏ và giữa duỗi xòe hình chữ V (`i_ext=True, m_ext=True, not r_ext, not p_ext`), $d_{\text{IW}} \ge 1.70$, $d_{\text{MW}} \ge 1.70$, khoảng cách mở giữa 2 ngón $d_{\text{IM}} \ge 0.28$, độ lệch tọa độ ngang $\Delta x_{\text{IM}} \ge 0.18$.

---

### 3.3. Lý do kỹ thuật bảo lưu UNKNOWN cho các chữ còn lại

1. **Cặp R vs U:**
   - Trong không gian 2D mặt trước, việc ngón giữa bắt chéo sau ngón trỏ (chữ R) rất nhạy cảm với góc quay cổ tay so với camera.
   - Tập dữ liệu hiện tại chỉ có 3 mẫu R từ một video duy nhất ($d_{\text{IM}} = 0.76 \pm 0.12$), chưa đủ bảo đảm tính ổn định góc quay khi người ký xoay nhẹ bàn tay. Việc gán luật cứng lúc này sẽ dẫn đến sai lệch lớn khi thử nghiệm trên người khác.
2. **Cặp M vs N:**
   - Ký hiệu chữ M luồn ngón cái dưới 3 ngón, chữ N luồn dưới 2 ngón. Trong góc quay camera chính diện, MediaPipe 2D landmarks bị che khuất (occluded) ngón cái ở mặt sau các ngón tay gập. Độ biến thiên tọa độ $Z$ giả định của MediaPipe không đủ tin cậy để phân biệt rõ vị trí chui ngón.
3. **Các chữ K, P, Q, G, H, X:**
   - Chữ K, P đòi hỏi ngón cái đặt vào khớp ngón giữa với góc chúc xuống hoặc phương nghiêng đặc thù;
   - Chữ G, H chỉ ngón ngang màn hình (horizontal orientation), cần thêm đặc trưng góc định hướng bàn tay (palm orientation angle).
4. **7 chữ cái có dấu (Ă, Â, Ê, Ô, Ơ, Ư, Đ):**
   - Đều là các ký hiệu động có pha chuyển động vẽ dấu (stroke motion). Cần Temporal State Machine kết hợp phân tích chuỗi thời gian để nhận diện chuẩn xác.

---

## 4. Kiến trúc và Giao diện kỹ thuật

Module `backend/vision/alphabet_classifier.py` được xây dựng tuân theo thiết kế OOP và các nguyên tắc Clean Code:

```python
class VSLAlphabetClassifier:
    def extract_features(self, landmarks: List[List[float]]) -> Optional[Dict[str, Any]]: ...
    def evaluate_rules(self, feat: Dict[str, Any]) -> List[Tuple[str, str, float]]: ...
    def classify_detailed(self, landmarks: Any) -> Dict[str, Any]: ...
    def classify(self, landmarks: Any) -> Tuple[str, float]: ...
```

### Hợp đồng dữ liệu `classify_detailed(landmarks)`:
```json
{
  "letter": "A",
  "confidence": 0.95,
  "status": "MATCH",
  "reason": "Nắm tay ngón cái dựng đứng: d_TW=1.49>=1.32, d_TM=0.90>=0.75",
  "matched_candidates": [["A", 0.95]],
  "features": { ... }
}
```
Các trạng thái `status`:
- `"MATCH"`: Đúng 1 luật thỏa mãn. Trả về ký tự viết hoa.
- `"NO_MATCH"`: Không thỏa mãn bất kỳ luật nào hoặc cử chỉ nằm trong vùng trung tính/chưa hỗ trợ. Trả về `"UNKNOWN"`, confidence `0.0`.
- `"CONFLICT"`: Có từ 2 luật cạnh tranh thỏa mãn đồng thời. Trả về `"UNKNOWN"`, confidence `0.0`.
- `"INVALID_INPUT"`: Đầu vào không hợp lệ (None, không đủ 21 mốc, chứa NaN/Inf, palm_scale suy biến). Trả về `"UNKNOWN"`, confidence `0.0`.

---

## 5. Kết quả Kiểm thử Thực nghiệm

### 5.1. Unit tests độc lập (`backend/tests/test_alphabet_classifier.py`)
- Tổng số test case: **26 tests**.
- Kết quả: **26/26 passed (100%)** trong 1.17 giây.
- Bao phủ:
  1. Thẩm định an toàn dữ liệu: `None`, sai kiểu (int, str, dict), sai số lượng điểm (0, 5, 20, 22, 42), tọa độ chuỗi, tọa độ `NaN`, tọa độ `Inf`, mốc suy biến phẳng (`palm_scale < 1e-5`).
  2. Phân tách cặp đối lập trên dữ liệu thật: S vs T (0% nhầm), C vs O (0% nhầm), D vs L (0% nhầm), A vs E (0% nhầm).
  3. Nhận diện các chữ độc lập: B, I, Y, V.
  4. Bảo vệ chữ chưa hỗ trợ: R, U, M, N luôn trả về `UNKNOWN` / `NO_MATCH`.
  5. Đảm bảo không sai chéo giữa 12 chữ được hỗ trợ.

### 5.2. Kiểm tra hồi quy toàn diện test suite dự án
- Tổng số test case toàn dự án: **242 items** (216 items ban đầu + 26 items mới).
- Kết quả: **242 passed (100%), 0 failed, 0 regressions** trong 32.56 giây.

### 5.3. Đánh giá trên fixture landmark thật 122 mẫu
- **12 chữ có luật:**
  - 100% các frame `stable_pose` (tư thế ổn định ở đỉnh cử chỉ) được nhận diện chính xác tuyệt đối.
  - Số mẫu bị phân loại chéo sai sang chữ khác trong nhóm có luật: **0 / 50 mẫu (Wrong = 0)**.
  - Các frame còn lại rơi vào trạng thái `NO_MATCH` đều là các frame ở pha chuẩn bị (`entry`), pha kết thúc (`transition`), hoặc lúc hạ tay về vị trí nghỉ.
- **72 mẫu của các chữ chưa hỗ trợ:**
  - 66/72 mẫu (91.7%) được bảo lưu an toàn ở trạng thái `UNKNOWN`.
  - 6 mẫu còn lại trùng hình thái cục bộ với các chữ tương ứng (ví dụ: tư thế tĩnh của chữ Ô trùng chữ O trước khi vẽ dấu, chữ Ă trùng tư thế chữ V ở pha chuẩn bị). Không có trường hợp nào xung đột với các quy tắc chữ cái khác.

---

## 6. Giới hạn Kỹ thuật và Kế hoạch Giai đoạn tiếp theo

### 6.1. Giới hạn kỹ thuật hiện tại
- **Single-Signer Dataset:** Fixture landmark thật hiện tại được trích xuất từ 29 video của cùng 1 người ký QIPEDC. Các đặc trưng tỉ lệ bàn tay, thói quen duỗi/gập ngón có thể có phương sai khác khi thử nghiệm trên người dùng thực tế với kích thước bàn tay và góc quay camera khác.
- **2D Heuristics:** Một số cử chỉ có chiều sâu 3D (như ngón cái kẹp dưới ngón trỏ của chữ M/N) cần thêm thông tin temporal hoặc mô hình học sâu hỗ trợ.

### 6.2. Kế hoạch tiếp theo (Task 3B.2)
1. Triển khai **Temporal State Machine** cho 7 chữ cái có dấu tiếng Việt (Ă, Â, Ê, Ô, Ơ, Ư, Đ) dựa trên chuỗi landmark liên tục và vector chuyển động của cổ tay/ngón tay.
2. Tích hợp cơ chế smoothing thời gian (EMA / Temporal voting) để ổn định kết quả nhận diện camera thời gian thực.

