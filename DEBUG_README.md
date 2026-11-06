# 🧪 TÀI LIỆU HƯỚNG DẪN GIAO DIỆN DEBUG (WORD DEBUG)

> **Lưu ý quan trọng:** File này được viết độc lập với `README.md` gốc của dự án. Khi dự án hoàn thiện, bạn có thể xóa file này và thư mục log mà không ảnh hưởng đến tài liệu hay tính năng chính của website.

---

## 1. Giới thiệu chức năng Word Debug

Giao diện **Word Debug (🧪 WORD DEBUG TEST)** là công cụ hỗ trợ kiểm thử và thu thập log thực nghiệm cho **Chế độ Từ vựng (Word Mode - SSTCN 63 từ, class 39–101)**. 

### Đặc điểm thiết kế cô lập:
- **Không can thiệp logic nhận diện:** Mô hình AI và luồng camera vẫn chạy nguyên bản, module debug chỉ đọc luồng frame/kết quả để ghi log.
- **Có Ground Truth:** Mỗi phiên test gắn liền với một từ mục tiêu (`target_word`), nhãn lớp (`target_class`) và lượt thử (`trial`).
- **Tự động lưu trữ:** Tự lưu file ma trận tọa độ `sequence_*.npy`, nhật ký `events.jsonl` và siêu dữ liệu `session.json`.
- **Dễ dàng ẩn/xóa:** Mặc định giao diện debug **hoàn toàn ẩn** đối với người dùng thông thường, chỉ kích hoạt khi có cờ cấu hình.

---

## 2. Cách khởi động hệ thống và bật giao diện Debug

### Bước 1: Khởi động Backend & Frontend

Chạy **y hệt như hướng dẫn trong [README.md](README.md)** (mục "🔄 Các lần chạy sau"), không có gì khác:

```bash
# Terminal 1 — Backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.app:app --reload

# Terminal 2 — Frontend
npm run dev
```

---

### Bước 2: Kích hoạt giao diện Debug

Điểm khác duy nhất là thêm `?wordDebug=1` vào đường link:

```text
http://localhost:5173/?wordDebug=1
```

(Dùng đúng địa chỉ Local mà Vite in ra trên Terminal, chỉ thêm `?wordDebug=1` vào cuối.)

---

## 3. Quy trình thực hiện Test trên giao diện

1. Truy cập tab **Nhận diện Ký hiệu qua Camera** (mặc định mở đầu).
2. Tại thanh điều khiển chế độ nhận diện, chọn **"Từ vựng"** (Word Mode).
3. Khi đó, khung màu vàng cam nét đứt **🧪 WORD DEBUG TEST** sẽ xuất hiện ngay phía dưới các nút chọn chế độ:
   - **Target word:** Bấm vào danh sách thả xuống và chọn từ bạn chuẩn bị thực hiện cử chỉ (ví dụ: `Hỏi giờ (class 55)`, `Thời gian (class 89)`, `Con gái (class 46)`, v.v.).
   - **Trial:** Nhập số thứ tự lần thử (mặc định là `1`).
4. Bật Camera và đứng vào khung hình sao cho MediaPipe bắt rõ nét khung xương.
5. Bấm nút **"Start Test"**:
   - Trạng thái chuyển sang **`● Recording`** màu đỏ.
   - Hệ thống bắt đầu ghi nhận các sequence 30 frames và log top-1 prediction kèm timestamp thời gian thực.
6. Thực hiện cử chỉ ngôn ngữ ký hiệu của từ đã chọn.
7. Bấm nút **"Stop Test"**:
   - Phiên ghi log kết thúc và được lưu vào thư mục `debug/words/`.
   - Số `Trial` sẽ tự động tăng lên 1 đơn vị để sẵn sàng cho lần thử tiếp theo.

---

## 4. Cấu trúc thư mục log được tạo ra

Mỗi phiên test sẽ tạo ra một thư mục riêng trong `debug/words/` theo định dạng:
```text
debug/words/<Ten_tu_khong_dau>_trial_<So_lan_thu>_<YYYY-MM-DD_HHMMSS>/
```

Ví dụ: `debug/words/Hoi_gio_trial_01_2026-11-04_203000/`

Bên trong bao gồm:
* `session.json`: Chứa thông tin cấu hình phiên test (từ mục tiêu, class, trial, thời gian bắt đầu/kết thúc, checkpoint mô hình sử dụng, tổng số sequence ghi được).
* `events.jsonl`: Từng dòng là một sự kiện theo chuẩn JSON Line (kết quả dự đoán Top-1, sự kiện reset buffer khi mất dấu tay `SEQUENCE_RESET`, thời gian chi tiết).
* `sequences/`: Thư mục chứa các file ma trận nhị phân `sequence_001.npy`, `sequence_002.npy`, ... (shape `[30, 75, 3]`), dùng để phân tích mô hình ngoại tuyến hoặc tái hiện lỗi.

---

## 5. Hướng dẫn gỡ bỏ hoàn toàn Debug khi hoàn thiện dự án

Sau khi hoàn tất nghiên cứu và nghiệm thu, bạn có thể xóa hoàn toàn mã nguồn debug theo các bước đơn giản sau:

### 1. Xóa các file tài liệu và log:
- Xóa file `DEBUG_README.md` (chính là file này).
- Xóa thư mục lưu dữ liệu log: `debug/`

### 2. Xóa các file chức năng Debug:
- Xóa component giao diện: `src/components/WordDebugPanel.jsx`
- Xóa service gọi API debug: `src/services/wordDebugApi.js`
- Xóa module logging backend: `backend/vision/word_debugger.py`

### 3. Dọn dẹp liên kết trong mã nguồn chính:
* **Trong `src/components/CameraSignRecognizer.jsx`:**
  - Xóa dòng import:
    ```javascript
    import WordDebugPanel from './WordDebugPanel';
    import { isWordDebugEnabled, logWordDebugReset } from '../services/wordDebugApi';
    ```
  - Xóa dòng render (khoảng dòng 1066):
    ```jsx
    {conversionMode === 'words' && isWordDebugEnabled() && <WordDebugPanel />}
    ```
  - Xóa các lời gọi `logWordDebugReset(...)` (nếu có).

* **Trong `backend/app.py`:**
  - Xóa phần `# API WORD DEBUG` (từ dòng 394 đến khoảng dòng 446 định nghĩa các endpoint `/api/debug/word/*`).

Sau các bước trên, toàn bộ hệ thống trở về trạng thái thuần sản phẩm (production-ready) sạch sẽ 100%.

