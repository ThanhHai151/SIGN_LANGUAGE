# BÁO CÁO TASK 11.2: GHI NHẬN KẾT QUẢ KIỂM THỬ THỰC TẾ 22 CHỮ CÁI TĨNH TRÊN WEBCAM (PILOT TEST RESULTS)

> **Dự án:** VSL Connect AI — Chuẩn hóa Nhận diện 29 Chữ cái Ngôn ngữ Ký hiệu Tiếng Việt  
> **Căn cứ quy trình:** Báo cáo chuẩn bị kiểm thử Task 11.1: [`backend/data/alphabet_v2/audit/webcam_pilot_task11_1.md`](file:///d:/SIGN_LANGUAGE/backend/data/alphabet_v2/audit/webcam_pilot_task11_1.md)  
> **Thời gian tạo biểu mẫu:** 07/11/2026  
> **Chế độ kiểm toán:** **READ-ONLY — KHÔNG TỰ TẠO KẾT QUẢ, KHÔNG SUY DIỄN ACCURACY, BẢO TOÀN CHECKPOINT**  
> **Checkpoint bảo toàn:** [`backend/models/alphabet_static_head_v1.pt`](file:///d:/SIGN_LANGUAGE/backend/models/alphabet_static_head_v1.pt) (SHA-256: `cab8123e23ed41a6088ba1c6e80a8aaaa9a382663418cc669eacff834795c254`)

---

## 1. Mục tiêu và Phạm vi Kỹ thuật

Báo cáo này cung cấp biểu mẫu chuẩn hóa toàn diện để **ghi nhận kết quả kiểm thử thực tế trước webcam** cho 22 chữ cái tĩnh tiếng Việt VSL, phục vụ đánh giá năng lực nhận diện của checkpoint Static MLP V1 kết hợp Geometric Rule Layer.

### Phạm vi cấu hình hệ thống:
- **Recognition Scope:** `hands` (Bàn tay — 60 FPS trích xuất qua MediaPipe Hands Wasm).
- **Conversion Mode:** `letters` (Chế độ nhận diện chuyên biệt chữ cái VSL — Instant <5ms).
- **Mô hình & Checkpoint:**
  - Ưu tiên 1: `Geometric Rule Layer` (20 luật hình học xác định).
  - Ưu tiên 2: `VSLStaticAlphabetHead` (Static MLP V1 qua checkpoint `alphabet_static_head_v1.pt`).
- **Danh mục 22 chữ cái kiểm thử:**  
  `A, B, C, D, E, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, X, Y`.
- **Cơ chế lọc mượt (Temporal Smoothing):** Window = 10 frames, Ngưỡng tin cậy = 0.70, Ngưỡng xác nhận = 5/10 frames, Cooldown = 1.2s.

> [!IMPORTANT]
> **NGUYÊN TẮC BẮT BUỘC:**  
> - Toàn bộ các trường kết quả kiểm thử thực tế được khởi tạo ở trạng thái **`NOT TESTED`**.
> - Tuyệt đối không tự điền kết quả giả định, không suy diễn accuracy từ dữ liệu offline.
> - Kết quả sẽ được người kiểm thử điền trực tiếp sau khi thực hiện thao tác trước webcam vật lý.

---

## 2. Thông tin Môi trường Kiểm thử Thực tế

*(Người kiểm thử điền thông tin chi tiết vào phần này trước khi bắt đầu phiên kiểm thử)*

| Tham số môi trường | Mô tả chi tiết | Trạng thái ghi nhận |
|---|---|:---:|
| **Người thực hiện kiểm thử** | Họ tên / Mã định danh Tester (người thực hiện thao tác ký hiệu) | `NOT TESTED` *(Chờ điền)* |
| **Ngày & giờ kiểm thử** | Thời điểm bắt đầu và kết thúc phiên thử nghiệm (YYYY-MM-DD HH:MM) | `NOT TESTED` *(Chờ điền)* |
| **Thiết bị Webcam / Camera** | Hãng sản xuất, model (vd: Integrated HD Webcam, Logitech C920, OBS Virtual Cam) | `NOT TESTED` *(Chờ điền)* |
| **Độ phân giải & FPS thực tế** | Độ phân giải luồng video (vd: 1280x720 @ 30 FPS hoặc 1920x1080 @ 60 FPS) | `NOT TESTED` *(Chờ điền)* |
| **Điều kiện ánh sáng** | Mô tả ánh sáng phòng (vd: Đèn LED trắng trần, ánh sáng tự nhiên qua cửa sổ, đèn rọi trước mặt; có bị ngược sáng hay chói lóa không) | `NOT TESTED` *(Chờ điền)* |
| **Khoảng cách đến Camera** | Khoảng cách từ ngực/bàn tay người thử đến ống kính webcam (khuyến nghị: 0.6m – 1.0m) | `NOT TESTED` *(Chờ điền)* |
| **Phông nền phía sau** | Mô tả hậu cảnh (vd: Tường trắng trơn tĩnh, phòng làm việc có người đi lại, v.v.) | `NOT TESTED` *(Chờ điền)* |
| **Tay thực hiện cử chỉ** | Tay thuận thực hiện (Tay phải / Tay trái; mặc định cấu hình là Right) | `NOT TESTED` *(Chờ điền)* |
| **Hệ điều hành & Trình duyệt** | Phiên bản OS (Windows 10/11) và Trình duyệt (Chrome/Edge phiên bản...) | `NOT TESTED` *(Chờ điền)* |

---

## 3. Biểu Mẫu Tổng Hợp Kết Quả Thực Tế 22 Chữ Cái Tĩnh

> **Quy ước điền kết quả cho Tester:**
> - **Số lần thử:** Khuyến nghị thực hiện **5 đến 10 lần thử** cho mỗi chữ cái (mỗi lần giữ tư thế 2 – 3 giây).
> - **Đúng:** Chữ cái hiển thị trên Banner HUD đúng với chữ cái đang ký hiệu, và token được thêm vào câu phụ đề.
> - **Sai:** Hệ thống nhận diện nhầm sang một chữ cái khác (vd: làm ký hiệu M nhưng nhận diện thành N hoặc S).
> - **NEUTRAL:** Giữ đúng tư thế nhưng HUD vẫn hiển thị "Chờ cử chỉ..." hoặc confidence < 0.70.
> - **Không phát sinh token:** Giữ tư thế đúng trên HUD nhưng phụ đề không ghi nhận chữ cái mới vào câu.
> - **Phát sinh token trùng:** Làm 1 lần nhưng hệ thống tự động emit 2 hoặc nhiều lần liên tiếp (do cooldown hoặc rung tay).
> - **Nguồn thực tế (F12):** Xem tab Network trên trình duyệt (`geometric_rule` hoặc `static_head`).

| STT | Ký tự | Cơ chế hệ thống | Vùng miền dữ liệu | Số lần thử | Đúng | Sai | NEUTRAL | Không token | Token trùng | Nguồn F12 (`source`) | Độ tin cậy quan sát | Độ trễ quan sát | Phụ đề & TTS | Nhầm lẫn phổ biến (nếu có) | Đánh giá tổng thể |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **A** | Rule + MLP Fallback | Shared (Toàn quốc) | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` |
| 2 | **B** | Pure Geometric Rule | Shared (Toàn quốc) | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` |
| 3 | **C** | Pure Geometric Rule | Shared (Toàn quốc) | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` |
| 4 | **D** | Rule + MLP Fallback | Shared (Toàn quốc) | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` |
| 5 | **E** | Rule + MLP Fallback | Shared (Toàn quốc) | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` |
| 6 | **G** | Rule + MLP Supplemental | Shared (Toàn quốc) | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` |
| 7 | **H** | Rule + MLP Supplemental | Shared (Toàn quốc) | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` |
| 8 | **I** | Pure Geometric Rule | Shared (Toàn quốc) | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` |
| 9 | **K** | Rule + MLP Supplemental | Shared (Toàn quốc) | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` |
| 10 | **L** | Pure Geometric Rule | Shared (Toàn quốc) | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` |
| 11 | **M** | Pure Static MLP | Shared (Toàn quốc) | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` |
| 12 | **N** | Pure Static MLP | Shared (Toàn quốc) | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` |
| 13 | **O** | Pure Geometric Rule | Shared (Toàn quốc) | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` |
| 14 | **P** | Rule + MLP Supplemental | Shared (Toàn quốc) | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` |
| 15 | **Q** | Rule + MLP Supplemental | Shared (Toàn quốc) | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` |
| 16 | **R** | Rule + MLP Supplemental | Shared (Toàn quốc) | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` |
| 17 | **S** | Pure Geometric Rule | Shared (Toàn quốc) | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` |
| 18 | **T** | Pure Geometric Rule | Shared (Toàn quốc) | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` |
| 19 | **U** | Rule + MLP Supplemental | Shared (Toàn quốc) | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` |
| 20 | **V** | Pure Geometric Rule | Shared (Toàn quốc) | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` |
| 21 | **X** | Rule + MLP Supplemental | Shared (Toàn quốc) | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` |
| 22 | **Y** | Pure Geometric Rule | Shared (Toàn quốc) | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` | `NOT TESTED` |

---

## 4. Bảng Nhật Ký Từng Lần Thử Nghiệm (Per-Trial Log Sheet)

Tester có thể sử dụng bảng mẫu dưới đây để ghi chép chi tiết từng lần thử nghiệm (Trial 1 $\rightarrow$ Trial 5) cho các ký tự trọng điểm (đặc biệt là nhóm thuần MLP `M, N`, nhóm phụ trợ `G, H, P, Q, R, X` và nhóm đối kháng `S, A, E, T`):

```markdown
### Nhật ký chi tiết: Chữ [KÝ TỰ CẦN TEST]
- Trial 1: [Đúng/Sai/Neutral] | Conf: ___% | Source: ___________ | Latency: ___ms | Ghi chú: _________________
- Trial 2: [Đúng/Sai/Neutral] | Conf: ___% | Source: ___________ | Latency: ___ms | Ghi chú: _________________
- Trial 3: [Đúng/Sai/Neutral] | Conf: ___% | Source: ___________ | Latency: ___ms | Ghi chú: _________________
- Trial 4: [Đúng/Sai/Neutral] | Conf: ___% | Source: ___________ | Latency: ___ms | Ghi chú: _________________
- Trial 5: [Đúng/Sai/Neutral] | Conf: ___% | Source: ___________ | Latency: ___ms | Ghi chú: _________________
```

---

## 5. Tổng Kết Định Lượng Hệ Thống Thực Tế

*(Phần này sẽ được tính toán tự động hoặc tổng hợp sau khi toàn bộ 22 chữ cái đã được kiểm thử)*

| Chỉ số thống kê hệ thống | Giá trị thực tế | Diễn giải & Ghi chú |
|---|:---:|---|
| **Tổng số lượt thử nghiệm (Total Trials)** | `NOT TESTED` | Tổng số lần thực hiện cử chỉ trước camera |
| **Tổng số lượt nhận diện đúng (Total Correct)** | `NOT TESTED` | Số lần HUD hiển thị đúng và phụ đề ghi nhận đúng |
| **Độ chính xác thực tế toàn diện (Live Accuracy %)** | `NOT TESTED` | $\frac{\text{Tổng số đúng}}{\text{Tổng số lần thử}} \times 100\%$ *(Tuyệt đối không suy diễn)* |
| **Tỷ lệ nhận diện sai (Error / Confusion %)** | `NOT TESTED` | Tỷ lệ nhầm sang ký hiệu khác |
| **Tỷ lệ không nhận diện được (NEUTRAL Rate %)** | `NOT TESTED` | Tỷ lệ không vượt qua ngưỡng conf $\ge 0.70$ hoặc không khớp luật |
| **Tỷ lệ kẹt / không phát sinh token (No-Token Rate %)** | `NOT TESTED` | Tỷ lệ HUD nhận đúng nhưng bộ đệm smoothing không emit token |
| **Tỷ lệ phát sinh token trùng lặp (Duplicate Rate %)** | `NOT TESTED` | Tỷ lệ bị gõ đúp do cooldown hoặc tay rung lắc |
| **Độ trễ trung bình từ khi tạo dáng đến khi emit token** | `NOT TESTED` | Thời gian quan sát được (thông thường ~150ms – 250ms) |

---

## 6. Bảng Phân Định Mức Độ Xác Minh Kỹ Thuật

| Hạng mục kiểm tra | Đã xác minh từ Source Code | Đã xác minh bằng Runtime | Chưa thể xác minh nếu chưa chạy webcam |
|---|:---:|:---:|:---:|
| **1. Cấu hình luồng nhận diện (`mode='letters'`, `scope='hands'`)** | Có ([`CameraSignRecognizer.jsx`](file:///d:/SIGN_LANGUAGE/src/components/CameraSignRecognizer.jsx)) | Có (Header HTTP, backend router) | Không |
| **2. Khởi tạo và phản hồi của API `/api/recognize_landmarks`** | Có ([`app.py`](file:///d:/SIGN_LANGUAGE/backend/app.py)) | Có (Response JSON đầy đủ 11 trường) | Không |
| **3. Logic Temporal Smoothing & Cooldown 1.2s** | Có ([`recognizer.py`](file:///d:/SIGN_LANGUAGE/backend/vision/recognizer.py)) | Có (Xác minh qua script test 5 frame / 1.3s) | Không |
| **4. Cơ chế dọn dẹp frame xung đột (Conflict Pruning)** | Có (11 luật dọn dẹp: H, K, I, U, T, S, P, Q, O, R, X) | Có (Xóa bỏ ngay các frame xung đột cũ) | Không |
| **5. Trường `source` hiển thị trên màn hình HUD** | **Có (HUD không hiển thị trường này)** | Có (API trả về nhưng UI bỏ qua) | Không |
| **6. Năng lực nhận diện 22 chữ cái tĩnh thực tế trước webcam** | Có (Cơ chế phân giải từng ký tự) | Có (Baseline offline 87.70% clean / 91.65% orig) | **`NOT TESTED` (Phụ thuộc hoàn toàn vào Pilot Test thực tế)** |
| **7. Ảnh hưởng của góc xoay bàn tay & ánh sáng môi trường** | Không | Không | **`NOT TESTED` (Phụ thuộc hoàn toàn vào Pilot Test thực tế)** |
| **8. Tần suất nhầm lẫn giữa M và N khi thực hiện trực tiếp** | Có (MLP Head gating conf $\ge 0.70$) | Có (Offline baseline phân biệt tốt) | **`NOT TESTED` (Phụ thuộc hoàn toàn vào Pilot Test thực tế)** |

---

## 7. Hướng Dẫn Quy Trình Thao Tác Cho Tester

Để đảm bảo kết quả ghi nhận có giá trị khoa học và đồng nhất:

1. **Chuẩn bị hệ thống:**
   - Mở Terminal 1: Chạy backend: `python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000`.
   - Mở Terminal 2: Chạy frontend: `npm run dev`.
   - Mở trình duyệt tại: `http://127.0.0.1:5173`.
2. **Chuẩn bị công cụ giám sát:**
   - Nhấn phím **F12** trên trình duyệt để mở Developer Tools.
   - Chuyển sang tab **Network**, nhập `/api/recognize_landmarks` vào ô Filter.
   - Bấm vào một request để sẵn sàng xem trường `"source"` trong tab **Response**.
3. **Bật Camera và thiết lập:**
   - Bấm nút **"Bật Camera"**.
   - Kiểm tra đảm bảo tiêu chí A là **Bàn tay** và tiêu chí B là **Chữ cái**.
4. **Thực hiện từng chữ cái:**
   - Đưa tay phải lên trước ngực, lòng bàn tay hướng về webcam ở cự ly khoảng 0.7m.
   - Tạo dáng chuẩn xác cho chữ cái cần kiểm tra.
   - Giữ nguyên tư thế trong 2 đến 3 giây:
     - Quan sát xem Banner lớn có hiện đúng chữ cái không.
     - Quan sát xem ô phụ đề câu có nhảy thêm token không.
     - Quan sát Web Speech TTS có phát âm đúng không.
     - Nhìn tab Network F12 xem `"source"` là `"geometric_rule"` hay `"static_head"`.
   - Hạ tay xuống khoảng 0.5 giây để bộ đệm reset về `NEUTRAL`.
   - Lặp lại 5 lần cho chữ cái đó, ghi số lần Đúng / Sai / NEUTRAL vào Biểu mẫu Mục 3.
5. **Hoàn tất và lưu kết quả:**
   - Sau khi hoàn thành đủ 22 chữ cái, điền thông tin môi trường tại Mục 2 và cập nhật bảng tổng kết tại Mục 5.

