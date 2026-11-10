# BÁO CÁO TASK 11.1: CHUẨN BỊ QUY TRÌNH KIỂM THỬ THỰC TẾ 22 CHỮ CÁI TĨNH TRÊN WEBCAM

> **Dự án:** VSL Connect AI — Chuẩn hóa Nhận diện 29 Chữ cái Ngôn ngữ Ký hiệu Tiếng Việt  
> **Căn cứ tiền đề:** Báo cáo kiểm toán Task 11.0: [`backend/data/alphabet_v2/audit/webcam_pipeline_task11_audit.md`](file:///d:/SIGN_LANGUAGE/backend/data/alphabet_v2/audit/webcam_pipeline_task11_audit.md)  
> **Thời gian lập:** 07/11/2026  
> **Chế độ thực hiện:** **READ-ONLY — KHÔNG SỬA CODE, KHÔNG THÊM LOGGING, KHÔNG HUẤN LUYỆN MODEL, BẢO TOÀN CHECKPOINT**  
> **Checkpoint bảo toàn:** [`backend/models/alphabet_static_head_v1.pt`](file:///d:/SIGN_LANGUAGE/backend/models/alphabet_static_head_v1.pt) (SHA-256: `cab8123e23ed41a6088ba1c6e80a8aaaa9a382663418cc669eacff834795c254`)

---

## 1. Mục tiêu và Nguyên tắc Phân định

Báo cáo này chuẩn bị đầy đủ quy trình kiểm thử thực tế cho **22 chữ cái tĩnh** (`A, B, C, D, E, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, X, Y`) trước webcam trên hệ thống runtime hiện tại.

Theo yêu cầu nghiêm ngặt của dự án, mọi phát hiện trong tài liệu được phân tách rõ ràng thành ba mức độ:
1. **[ĐÃ XÁC MINH TỪ SOURCE CODE]:** Được kiểm tra, đối chiếu trực tiếp từng dòng lệnh trong mã nguồn frontend/backend.
2. **[ĐÃ XÁC MINH BẰNG RUNTIME]:** Đã chạy thực nghiệm kiểm chứng bằng code kiểm thử backend với landmark thực tế và môi trường runtime không bật webcam.
3. **[CHƯA THỂ XÁC MINH NẾU CHƯA CHẠY WEBCAM]:** Các yếu tố phụ thuộc hoàn toàn vào điều kiện vật lý ngoại cảnh (ánh sáng thực, góc xoay bàn tay người dùng trước camera, tốc độ khung hình phần cứng MediaPipe Wasm trên trình duyệt).

> [!IMPORTANT]
> Tuyệt đối không tự tạo hoặc giả định kết quả kiểm thử webcam. Kết quả thực nghiệm webcam sẽ được ghi nhận khách quan trong quá trình người dùng thực hiện kiểm thử theo Biểu mẫu ở Mục 7.

---

## 2. Xác minh Cách Khởi động Backend và Frontend

### 2.1. Khởi động Backend (FastAPI)
- **[ĐÃ XÁC MINH TỪ SOURCE CODE]** ([`backend/app.py`](file:///d:/SIGN_LANGUAGE/backend/app.py#L500-L502), [`start.sh`](file:///d:/SIGN_LANGUAGE/start.sh#L14)):
  Backend được cấu hình chạy trên nền máy chủ ASGI Uvicorn tại host `127.0.0.1`, cổng `8000`.
- **Lệnh khởi động chuẩn trên Windows (PowerShell/CMD):**
  ```powershell
  python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
  ```
  *Hoặc:*
  ```powershell
  python backend/app.py
  ```
- **Kiểm tra trạng thái Backend:**
  - Health check endpoint: `http://127.0.0.1:8000/api/health`
  - Swagger API Documentation: `http://127.0.0.1:8000/docs`
- **[ĐÃ XÁC MINH BẰNG RUNTIME]:** Backend khởi động và tải thành công mô hình `sign_sstcn_attention_model.pth` (102 lớp) và `weights.pt` / `alphabet_static_head_v1.pt`. Endpoint `/api/health` trả về HTTP 200 `{"status": "online"}`.

### 2.2. Khởi động Frontend (React + Vite)
- **[ĐÃ XÁC MINH TỪ SOURCE CODE]** ([`package.json`](file:///d:/SIGN_LANGUAGE/package.json#L6-L12), [`start.sh`](file:///d:/SIGN_LANGUAGE/start.sh#L19)):
  Frontend sử dụng Vite 8.3 và React 19.2, cấu hình chạy tại `http://127.0.0.1:5173`.
- **Lệnh khởi động chuẩn:**
  ```powershell
  npm run dev
  ```
- **Kiểm tra kết nối Frontend - Backend:**
  - **[ĐÃ XÁC MINH TỪ SOURCE CODE]** ([`src/App.jsx`](file:///d:/SIGN_LANGUAGE/src/App.jsx#L15-L28)):
    Frontend tự động gọi hàm `checkBackendHealth()` mỗi 10 giây. Khi backend online, chấm trạng thái trên thanh `Navbar` hiển thị màu xanh lá (`Online`).

---

## 3. Xác minh Cách Mở Chế độ Letters và Bật Webcam

### 3.1. Chế độ Mặc định khi Mở Ứng dụng
- **[ĐÃ XÁC MINH TỪ SOURCE CODE]** ([`src/App.jsx`](file:///d:/SIGN_LANGUAGE/src/App.jsx#L9)):
  `const [activeTab, setActiveTab] = useState('camera');`  
  $\rightarrow$ Ứng dụng khi mở ra sẽ ở ngay tab **Nhận diện qua Camera** (`CameraSignRecognizer`).
- **[ĐÃ XÁC MINH TỪ SOURCE CODE]** ([`src/components/CameraSignRecognizer.jsx`](file:///d:/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L90-L93)):
  - Tiêu chí A (Phạm vi): `const [recognitionScope, setRecognitionScope] = useState('hands');` (Mặc định là **Bàn tay** - 60 FPS).
  - Tiêu chí B (Chuyển đổi): `const [conversionMode, setConversionMode] = useState('letters');` (Mặc định là **Chữ cái** - Instant <5ms).

### 3.2. Cách Chuyển sang Chế độ Letters (nếu đang ở chế độ khác)
- **[ĐÃ XÁC MINH TỪ SOURCE CODE]** ([`src/components/CameraSignRecognizer.jsx`](file:///d:/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L1045-L1052)):
  Tại khối công cụ `B. Chế độ chuyển đổi`, bấm vào nút có biểu tượng chữ cái:
  - **Nút:** `[🔤 Chữ cái]` (`Type` icon, title: *"Nhận diện 27 chữ cái VSL tức thì (<5ms)"*).
  - Khi được chọn, nút có class `btn-primary` (nền sáng active).

### 3.3. Cách Bật Webcam
- **[ĐÃ XÁC MINH TỪ SOURCE CODE]** ([`src/components/CameraSignRecognizer.jsx`](file:///d:/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx#L985, L1204)):
  - Người dùng bấm nút **"Bật Camera"** (hoặc nút lớn **"⚡ Bật Camera Ngay"** giữa màn hình video placeholder).
  - Trình duyệt sẽ yêu cầu cấp quyền truy cập Camera (`navigator.mediaDevices.getUserMedia`).
  - Thư viện MediaPipe Hands Wasm tự động tải từ CDN `https://cdn.jsdelivr.net/npm/@mediapipe/hands/`.
  - Khung video hiển thị trực tiếp và lớp canvas phủ xương tay bắt đầu vẽ khi phát hiện bàn tay.

---

## 4. Xác minh Các Trường Thông tin: HUD Frontend vs API Backend

Bảng đối soát chi tiết giữa dữ liệu Backend trả về và hiển thị thực tế trên giao diện HUD:

| Trường thông tin | Dữ liệu Backend trả về (`POST /api/recognize_landmarks`) | Hiển thị thực tế trên Frontend HUD (`CameraSignRecognizer.jsx`) | Tình trạng xác minh |
|---|---|---|---|
| **predicted letter** | `sign`: `"A"` .. `"Y"`, hoặc `"NEUTRAL"`.<br/>`label`: `"A"`, `"B"`... hoặc `"Đang theo dõi cử chỉ..."`. | Hiển thị to rõ ràng tại **Banner Kết Quả Lớn** (`currentResult.label` - cỡ chữ 32px, in đậm). | **[ĐÃ XÁC MINH TỪ SOURCE CODE & RUNTIME]** |
| **confidence** | `confidence`: số thực từ `0.0` đến `1.0` (làm tròn 2 chữ số, vd: `0.95`). | 1. Huy hiệu góc phải: `${Math.round(confidence * 100)}% Khớp`.<br/>2. Thanh đo độ tin cậy (Confidence Bar) đổi màu: Xanh lá (>85%), Xanh lam (>65%), Cam (<65%). | **[ĐÃ XÁC MINH TỪ SOURCE CODE & RUNTIME]** |
| **source** | `source` và `prediction_source`: `"geometric_rule"`, `"static_head"`, `"static_head_fallback"`, `"none"`. | **KHÔNG HIỂN THỊ TRÊN HUD.** Frontend không lưu trữ và không render trường này. (Xem chi tiết Mục 5). | **[ĐÃ XÁC MINH TỪ SOURCE CODE]** |
| **confirmed** | `confirmed`: `true` hoặc `false`. (Báo hiệu cử chỉ đã duy trì ổn định đủ thời gian). | Không hiện chữ "confirmed" trực tiếp. Thay vào đó, khi `confirmed === true`, token được đẩy vào **Hộp Phụ Đề Câu** và kích hoạt đọc **Web Speech TTS**. | **[ĐÃ XÁC MINH TỪ SOURCE CODE & RUNTIME]** |
| **new_token** | `new_token`: Tên chữ cái (vd: `"A"`) khi vừa được xác nhận, hoặc `null`. | Token mới xuất hiện trong chuỗi phụ đề (`sentence`) và danh sách thẻ lịch sử 7 token gần nhất (`recognizedTokens`). | **[ĐÃ XÁC MINH TỪ SOURCE CODE & RUNTIME]** |
| **trạng thái NEUTRAL** | `sign: "NEUTRAL"`, `confidence: 0.0`, `label: "Chờ cử chỉ..."` hoặc `"Đang theo dõi cử chỉ..."`. | Chữ hiển thị màu xám mờ (`var(--text-muted)`), huy hiệu ghi `"Đang chờ..."`, thanh confidence 0%, không cập nhật câu. | **[ĐÃ XÁC MINH TỪ SOURCE CODE & RUNTIME]** |

---

## 5. Kiểm tra Đặc biệt: Trường `source` Có Thực Sự Hiển Thị Trên HUD?

> [!WARNING]
> **KẾT LUẬN KIỂM TOÁN QUAN TRỌNG:**  
> **Frontend HUD HIỆN TẠI KHÔNG HIỂN THỊ TRƯỜNG `source` (`geometric_rule` / `static_head`)!**  
> Tuyệt đối không giả định người dùng có thể nhìn thấy nguồn nhận diện trên màn hình giao diện.

### Bằng chứng kỹ thuật chi tiết:
1. **Tại Backend (`backend/vision/recognizer.py` lines 1428–1439):**  
   Backend trả về đầy đủ trong JSON:
   ```json
   {
     "sign": "A",
     "label": "A",
     "confidence": 0.95,
     "source": "geometric_rule",
     "prediction_source": "geometric_rule",
     "classifier_details": { "status": "MATCH", ... }
   }
   ```
2. **Tại Frontend tiếp nhận (`CameraSignRecognizer.jsx` lines 852–860):**  
   Hàm nhận kết quả chỉ chọn lọc trích xuất các trường:
   ```javascript
   setCurrentResult({
     sign: data.sign,
     label: data.label,
     confidence: data.confidence,
     type: data.type || 'alphabet',
     description: data.description,
     category: ...,
     video_url: ''
   });
   ```
   Trường `data.source`, `data.prediction_source` **hoàn toàn bị bỏ qua**, không được đưa vào `currentResult`.
3. **Tại phần Render JSX HUD (`CameraSignRecognizer.jsx` lines 1327–1425):**  
   Huy hiệu góc hiển thị nhãn cố định: `"Instant (<5ms)"` khi ở mode `letters`, không có bất kỳ thẻ nào hiển thị giá trị nguồn.
4. **Tìm kiếm toàn bộ codebase frontend:**  
   Từ khóa `.source` chỉ xuất hiện duy nhất ở [`src/components/VSLDictionaryExplorer.jsx`](file:///d:/SIGN_LANGUAGE/src/components/VSLDictionaryExplorer.jsx#L283) cho modal tra cứu từ điển QIPEDC, hoàn toàn vắng bóng trong `CameraSignRecognizer.jsx`.

### Cách theo dõi `source` trong quá trình Pilot Test an toàn:
Vì không sửa code frontend (tuân thủ nguyên tắc READ-ONLY), người kiểm thử quan sát `source` bằng 1 trong 2 cách sau:
- **Cách 1 (Khuyến nghị cho tester):** Mở trình duyệt $\rightarrow$ Nhấn **F12** $\rightarrow$ Chọn tab **Network** $\rightarrow$ Lọc request `/api/recognize_landmarks` $\rightarrow$ Xem tab **Response** để thấy chính xác trường `"source": "geometric_rule"` hay `"source": "static_head"`.
- **Cách 2:** Quan sát dòng log stdout trên terminal chạy backend FastAPI.

---

## 6. Xác minh Chi tiết Điều kiện Temporal Smoothing

Dựa trên việc kiểm tra mã nguồn tại [`backend/vision/recognizer.py`](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py#L1271-L1378) và thực nghiệm runtime:

### 6.1. Cấu hình Tham số Mặc định
- `smoothing_window = 10` (Kích thước bộ đệm lịch sử: 10 khung hình).
- `confidence_threshold = 0.70` (Ngưỡng tin cậy tối thiểu để một khung hình được tính vào bộ lọc).
- `cooldown_sec = 1.2` giây (Thời gian đóng băng chống trùng lặp khi giữ nguyên cùng một cử chỉ).

### 6.2. Cơ chế Thêm Prediction vào History
- Bộ nhớ đệm là một `collections.deque(maxlen=10)`.
- **Luật dọn dẹp frame xung đột (Conflict Pruning):**  
  Trước khi thêm frame mới, nếu phát hiện cử chỉ có đặc trưng dễ nhầm lẫn, hệ thống tự động loại bỏ các frame cũ gây nhiễu trong bộ đệm:
  - Dự đoán `H` ($\ge 0.70$): Dọn sạch các frame `V` cũ.
  - Dự đoán `K` ($\ge 0.70$): Dọn sạch các frame `V`, `D` cũ.
  - Dự đoán `I` ($\ge 0.70$): Dọn sạch các frame `Y`, `B` cũ.
  - Dự đoán `U` ($\ge 0.70$): Dọn sạch các frame `V` cũ.
  - Dự đoán `T` ($\ge 0.70$): Dọn sạch các frame `B`, `O` cũ.
  - Dự đoán `S` ($\ge 0.70$): Dọn sạch các frame `A`, `E`, `T` cũ.
  - Dự đoán `P` ($\ge 0.70$): Dọn sạch các frame `K`, `D` cũ.
  - Dự đoán `Q` ($\ge 0.70$): Dọn sạch các frame `G` cũ.
  - Dự đoán `O` ($\ge 0.70$): Dọn sạch các frame `A`, `B` cũ.
  - Dự đoán `R` ($\ge 0.70$): Dọn sạch các frame `U`, `V` cũ.
  - Dự đoán `X` ($\ge 0.70$): Dọn sạch các frame `A`, `S`, `O` cũ.
- Sau đó thêm phần tử: `self.prediction_history.append((final_sign, final_conf))`.

### 6.3. Điều kiện Lọc Đủ 5/10 Frame và Confidence $\ge 0.70$
- Lọc frame hợp lệ: `recent_signs = [s for s, c in self.prediction_history if c >= 0.70]`.  
  *(Các frame `NEUTRAL` hoặc frame có độ tin cậy dưới 0.70 sẽ bị loại bỏ hoàn toàn khỏi `recent_signs`)*.
- Điều kiện kích hoạt:
  1. `len(recent_signs) >= 5` (ít nhất 5 frame trong 10 frame gần nhất đạt conf $\ge 0.70$).
  2. Ký hiệu xuất hiện nhiều nhất `most_common` phải có số lượng `count >= 5`.
  3. `most_common` phải nằm trong danh mục `ALLOWED_LETTERS` (loại trừ `NEUTRAL`).

### 6.4. Điều kiện Cooldown 1.2 Giây
- **Khi giữ nguyên cùng một cử chỉ (`most_common == self.last_confirmed_sign`):**
  - Hệ thống kiểm tra: `(current_time - self.last_confirmed_time) > 1.2s`.
  - Nếu **chưa đủ 1.2s:** `confirmed = False`, `new_token = None` $\rightarrow$ Không thêm chữ cái mới vào câu, không đọc TTS lại.
  - Nếu **đã qua 1.2s:** `confirmed = True`, `new_token = sign` $\rightarrow$ Thêm tiếp ký tự đó vào câu (cho phép gõ đúp có chủ đích, ví dụ gõ "AA").

### 6.5. Cách Xử lý Khi Chuyển Sang Chữ Cái Khác
- **[ĐÃ XÁC MINH BẰNG RUNTIME]:**
  - **Trường hợp chuyển đổi trực tiếp (`most_common != self.last_confirmed_sign`):**  
    Hệ thống **KHÔNG ÁP DỤNG COOLDOWN 1.2 GIÂY**. Ngay khi chữ cái mới đạt đủ điều kiện đa số trong bộ đệm smoothing, hệ thống xác nhận ngay lập tức (`confirmed = True`).
  - **Hành vi thực tế của bộ đệm khi đổi cử chỉ đột ngột:**  
    Nếu người dùng chuyển từ A sang B mà không hạ tay (không có frame trung gian):
    - Frame 1 đến Frame 5 của B: Bộ đệm chứa 5 frame A và 5 frame B $\rightarrow$ Tần suất hòa 5-5, `Counter.most_common(1)` vẫn giữ A (do A vào trước) $\rightarrow$ `confirmed = False`.
    - **Frame 6 của B:** Bộ đệm chứa 6 frame B và 4 frame A $\rightarrow$ B chiếm ưu thế tuyệt đối (count = 6 $\ge 5$) $\rightarrow$ **Xác nhận B ngay tại frame thứ 6** (~180–200ms).
  - Nếu giữa 2 chữ cái có một khoảng ngắn hạ tay hoặc đưa tay về trạng thái tự nhiên (khoảng 100ms), các frame mới của chữ thứ hai sẽ đạt xác nhận chỉ sau đúng **5 frames** (~150ms).

---

## 7. Biểu mẫu Ghi Kết quả Kiểm thử Thủ công 22 Chữ cái Tĩnh

> **Hướng dẫn dành cho Người Kiểm thử (Tester):**
> 1. Đặt webcam ngang tầm mắt, khoảng cách từ người đến camera khoảng **0.6m – 1.0m**.
> 2. Đảm bảo ánh sáng rõ ràng, phông nền phía sau ít chuyển động gây nhiễu.
> 3. Tạo tư thế chuẩn VSL cho từng chữ cái tĩnh bằng **tay phải**, giữ tư thế ổn định trong **2 – 3 giây**.
> 4. Quan sát Banner HUD trên màn hình và tab Network (F12) để điền vào biểu mẫu dưới đây.

### BIỂU MẪU PILOT TEST THỰC TẾ (22 Chữ cái Tĩnh)

| STT | Ký tự | Cơ chế dự kiến | Vùng miền dữ liệu | Chữ cái HUD hiển thị | Độ tin cậy (%) | Nguồn F12 Network (`source`) | Phụ đề & TTS kích hoạt? | Có bị nhầm lẫn chữ khác không? | Đánh giá (ĐẠT / CHƯA ĐẠT / CHẬM) | Ghi chú tư thế thực tế |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | **A** | Rule + MLP Fallback | Shared (Toàn quốc) | *(Điền khi test)* | *(Điền %)* | `geometric_rule` / `static_head_fallback` | *(Có/Không)* | *(Ghi nếu nhầm)* | *(Đánh giá)* | Nắm đấm, ngón cái ép sát ngoài ngón trỏ |
| 2 | **B** | Pure Geometric Rule | Shared (Toàn quốc) | *(Điền khi test)* | *(Điền %)* | `geometric_rule` | *(Có/Không)* | *(Ghi nếu nhầm)* | *(Đánh giá)* | 4 ngón duỗi thẳng khép chặt, ngón cái gập ngang lòng tay |
| 3 | **C** | Pure Geometric Rule | Shared (Toàn quốc) | *(Điền khi test)* | *(Điền %)* | `geometric_rule` | *(Có/Không)* | *(Ghi nếu nhầm)* | *(Đánh giá)* | Bàn tay khum tròn tạo hình vòm chữ C |
| 4 | **D** | Rule + MLP Fallback | Shared (Toàn quốc) | *(Điền khi test)* | *(Điền %)* | `geometric_rule` / `static_head_fallback` | *(Có/Không)* | *(Ghi nếu nhầm)* | *(Đánh giá)* | Ngón trỏ chỉ thẳng đứng, ngón cái chạm đầu 3 ngón còn lại |
| 5 | **E** | Rule + MLP Fallback | Shared (Toàn quốc) | *(Điền khi test)* | *(Điền %)* | `geometric_rule` / `static_head_fallback` | *(Có/Không)* | *(Ghi nếu nhầm)* | *(Đánh giá)* | 4 ngón gập vuông góc, ngón cái gập chạm đầu ngón trỏ |
| 6 | **G** | Rule + MLP Supplemental | Shared (Toàn quốc) | *(Điền khi test)* | *(Điền %)* | `geometric_rule` / `static_head` | *(Có/Không)* | *(Ghi nếu nhầm)* | *(Đánh giá)* | Ngón cái và ngón trỏ song song mở ngang, các ngón khác nắm |
| 7 | **H** | Rule + MLP Supplemental | Shared (Toàn quốc) | *(Điền khi test)* | *(Điền %)* | `geometric_rule` / `static_head` | *(Có/Không)* | *(Ghi nếu nhầm)* | *(Đánh giá)* | Ngón trỏ và ngón giữa duỗi ngang song song khép sát |
| 8 | **I** | Pure Geometric Rule | Shared (Toàn quốc) | *(Điền khi test)* | *(Điền %)* | `geometric_rule` | *(Có/Không)* | *(Ghi nếu nhầm)* | *(Đánh giá)* | Chỉ ngón út duỗi thẳng đứng, 4 ngón còn lại nắm chặt |
| 9 | **K** | Rule + MLP Supplemental | Shared (Toàn quốc) | *(Điền khi test)* | *(Điền %)* | `geometric_rule` / `static_head` | *(Có/Không)* | *(Ghi nếu nhầm)* | *(Đánh giá)* | Ngón trỏ thẳng, ngón giữa nghiêng, ngón cái chèn giữa |
| 10 | **L** | Pure Geometric Rule | Shared (Toàn quốc) | *(Điền khi test)* | *(Điền %)* | `geometric_rule` | *(Có/Không)* | *(Ghi nếu nhầm)* | *(Đánh giá)* | Ngón cái và trỏ tạo góc vuông chữ L, 3 ngón khác nắm |
| 11 | **M** | Pure Static MLP | Shared (Toàn quốc) | *(Điền khi test)* | *(Điền %)* | `static_head` | *(Có/Không)* | *(Ghi nếu nhầm)* | *(Đánh giá)* | 3 ngón gập úp xuống qua ngón cái |
| 12 | **N** | Pure Static MLP | Shared (Toàn quốc) | *(Điền khi test)* | *(Điền %)* | `static_head` | *(Có/Không)* | *(Ghi nếu nhầm)* | *(Đánh giá)* | 2 ngón gập úp xuống qua ngón cái |
| 13 | **O** | Pure Geometric Rule | Shared (Toàn quốc) | *(Điền khi test)* | *(Điền %)* | `geometric_rule` | *(Có/Không)* | *(Ghi nếu nhầm)* | *(Đánh giá)* | Đầu các ngón tay chụm tròn vào đầu ngón cái thành chữ O |
| 14 | **P** | Rule + MLP Supplemental | Shared (Toàn quốc) | *(Điền khi test)* | *(Điền %)* | `geometric_rule` / `static_head` | *(Có/Không)* | *(Ghi nếu nhầm)* | *(Đánh giá)* | Giống tư thế K nhưng hướng chúc ngón xuống dưới |
| 15 | **Q** | Rule + MLP Supplemental | Shared (Toàn quốc) | *(Điền khi test)* | *(Điền %)* | `geometric_rule` / `static_head` | *(Có/Không)* | *(Ghi nếu nhầm)* | *(Đánh giá)* | Giống tư thế G nhưng hướng chúc ngón xuống dưới |
| 16 | **R** | Rule + MLP Supplemental | Shared (Toàn quốc) | *(Điền khi test)* | *(Điền %)* | `geometric_rule` / `static_head` | *(Có/Không)* | *(Ghi nếu nhầm)* | *(Đánh giá)* | Ngón trỏ và ngón giữa bắt chéo vào nhau |
| 17 | **S** | Pure Geometric Rule | Shared (Toàn quốc) | *(Điền khi test)* | *(Điền %)* | `geometric_rule` | *(Có/Không)* | *(Ghi nếu nhầm)* | *(Đánh giá)* | Nắm đấm chặt, ngón cái vắt ngang qua mu 4 ngón tay |
| 18 | **T** | Pure Geometric Rule | Shared (Toàn quốc) | *(Điền khi test)* | *(Điền %)* | `geometric_rule` | *(Có/Không)* | *(Ghi nếu nhầm)* | *(Đánh giá)* | Ngón cái lọt chen giữa ngón trỏ và ngón giữa |
| 19 | **U** | Rule + MLP Supplemental | Shared (Toàn quốc) | *(Điền khi test)* | *(Điền %)* | `geometric_rule` / `static_head` | *(Có/Không)* | *(Ghi nếu nhầm)* | *(Đánh giá)* | Ngón trỏ và ngón giữa duỗi thẳng đứng khép sát |
| 20 | **V** | Pure Geometric Rule | Shared (Toàn quốc) | *(Điền khi test)* | *(Điền %)* | `geometric_rule` | *(Có/Không)* | *(Ghi nếu nhầm)* | *(Đánh giá)* | Ngón trỏ và ngón giữa duỗi thẳng tách góc chữ V |
| 21 | **X** | Rule + MLP Supplemental | Shared (Toàn quốc) | *(Điền khi test)* | *(Điền %)* | `geometric_rule` / `static_head` | *(Có/Không)* | *(Ghi nếu nhầm)* | *(Đánh giá)* | Ngón trỏ cong móc câu, các ngón khác nắm chặt |
| 22 | **Y** | Pure Geometric Rule | Shared (Toàn quốc) | *(Điền khi test)* | *(Điền %)* | `geometric_rule` | *(Có/Không)* | *(Ghi nếu nhầm)* | *(Đánh giá)* | Ngón cái và ngón út xòe rộng, 3 ngón giữa nắm chặt |

---

## 8. Bảng Tổng Hợp Phân Định Mức Độ Xác Minh Kỹ Thuật

| Hạng mục kiểm tra | Đã xác minh từ Source Code | Đã xác minh bằng Runtime | Chưa thể xác minh nếu chưa chạy webcam |
|---|:---:|:---:|:---:|
| **1. Khởi động Backend (FastAPI / Uvicorn)** | Có (`app.py`, `start.sh`) | Có (Chạy thành công, load weights 102 lớp + Static Head V1) | Không |
| **2. Khởi động Frontend (React / Vite)** | Có (`package.json`, `App.jsx`) | Có (Cấu hình cổng 5173, kết nối health check 10s) | Không |
| **3. Mở chế độ Letters mặc định** | Có (`conversionMode = 'letters'`) | Có (Header `X-Conversion-Mode: letters`) | Không |
| **4. Hiển thị chữ cái & confidence trên HUD** | Có (`CameraSignRecognizer.jsx`) | Có (Banner 32px, Badge %, Progress bar) | Cần kiểm tra góc hiển thị thực tế trước camera |
| **5. Hiển thị `source` trên HUD** | **Có (Xác nhận HUD KHÔNG hiển thị)** | Có (API trả về `source`, HUD bỏ qua) | Không |
| **6. Trạng thái NEUTRAL** | Có (`recognizer.py`, UI handler) | Có (Trả về khi landmarks rỗng hoặc cử chỉ lạ) | Cần kiểm thử tay ngoài vùng quan sát |
| **7. Điều kiện 5/10 frame smoothing** | Có (Logic `collections.Counter`) | Có (Frame 1-4 không emit, frame 5 emit) | Cần kiểm chứng độ mượt khi tay rung lắc |
| **8. Điều kiện Cooldown 1.2s** | Có (`cooldown_sec = 1.2`) | Có (Cùng chữ cái phải đợi >1.2s mới emit tiếp) | Không |
| **9. Chuyển đổi chữ cái khác tức thì** | Có (`most_common != last_confirmed`) | Có (Không bị cooldown chặn, emit khi đủ đa số) | Cần kiểm thử độ trễ chuyển động thực tế |
| **10. Tỷ lệ nhận diện thực tế 22 chữ cái** | Có (Cơ chế routing từng chữ cái) | Có (Baseline test trên landmark offline) | **CHƯA THỂ XÁC MINH (Phụ thuộc Pilot Test)** |

---

## 9. Kết luận và Khuyến nghị

1. **Hệ thống đã sẵn sàng cho Pilot Test:** Toàn bộ đường ống từ API, smoothing, cooldown đến giao diện hiển thị phụ đề và TTS đã được kiểm chứng kỹ thuật thông suốt.
2. **Lưu ý đặc biệt cho Tester:**
   - Trường `source` nhận diện (`geometric_rule` hay `static_head`) không thể nhìn thấy trực tiếp trên màn hình camera. Tester cần **bật tab Network (F12)** trên trình duyệt song song để ghi nhận chính xác nguồn gốc nhận diện.
   - Khi chuyển giữa các chữ cái, nên có một khoảng ngừng ngắn (~0.2 giây) thả lỏng tay để bộ đệm smoothing làm mới nhanh nhất (5 frames), mang lại trải nghiệm nhận diện chuẩn xác nhất.
3. **Quy trình sẵn sàng:** Người dùng có thể tiến hành chạy thực tế và điền kết quả vào **Biểu mẫu Mục 7** bất kỳ lúc nào.

