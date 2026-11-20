# VSL Connect AI

Ứng dụng React + FastAPI hỗ trợ nhận diện và học ngôn ngữ ký hiệu Việt Nam.

**Huấn luyện SSTCN 102 lớp:** xem [hướng dẫn dữ liệu, preprocessing, train và đánh giá](docs/sstcn/README.md)
và [kết quả kiểm chứng cùng nguồn gốc dữ liệu](docs/sstcn/STATUS.md).
Dataset đầy đủ: [GitHub Releases — SSTCN dataset v1](https://github.com/ThanhHai151/SIGN_LANGUAGE/releases/tag/sstcn-dataset-v1).

Từ thư mục gốc repo, chạy `./start.sh` để khởi động frontend và backend.

[Triển khai trên Render Free](https://render.com/deploy?repo=https://github.com/ThanhHai151/SIGN_LANGUAGE/tree/cloud-deploy) · [Hướng dẫn triển khai và huấn luyện trên cloud](docs/CLOUD_DEPLOYMENT.md)

## Chạy bản build trên máy cá nhân

```bash
npm run build
./start.sh --production
```

Mở <http://127.0.0.1:8000>. Một tiến trình FastAPI phục vụ giao diện và API. Build lại sau khi sửa frontend. Chế độ này mặc định đặt 2 luồng tính toán PyTorch trên CPU qua `OMP_NUM_THREADS` và `MKL_NUM_THREADS`.

## Luyện ký hiệu từ QIPEDC bằng camera

Trong **Camera → Từ vựng**, chọn **Tự do** và chủ đề **Tất cả**. Camera thử so khớp với các mẫu QIPEDC đã chuẩn bị ngay khi bạn thực hiện cử chỉ. Nút **Mẫu và hiệu chỉnh** ở thanh bên trái mở danh sách ký hiệu đang sẵn sàng, gồm `mấy giờ?`, `ăn đủ` và các mẫu gần giống để phân biệt. Chủ đề **Số đếm và Thời gian** cũng hỗ trợ `mấy giờ?`.

Để mở rộng danh sách, vào **Từ điển QIPEDC**, mở video của mục bạn muốn và bấm **Thêm vào Tự do**. Quay lại camera với chủ đề **Tất cả** để thực hiện bất kỳ ký hiệu nào trong danh sách đang hoạt động. Danh sách thêm được lưu theo trình duyệt. Lần đầu cần kết nối QIPEDC để tải mẫu. Khi hai mẫu quá gần nhau, camera báo thử lại.

Với ký hiệu hay che khuất bàn tay như `năn nỉ`, có thể bật **Bám tay chi tiết** trong chế độ **Từ vựng**. Chế độ này xử lý tới 640 px bằng MediaPipe Holistic Full và có thể giảm FPS trên máy yếu. Khi bảng chẩn đoán báo mất dấu một tay, đặt camera hơi lệch để cả hai cổ tay và đường viền của hai bàn tay cùng hiện trong khung hình.

Từ màn hình camera, bấm **Từ điển QIPEDC**, tìm `mấy giờ?` hoặc một mục có video và bấm **Luyện bằng camera**. Chuẩn bị mẫu, bật camera rồi bấm **Ghi 3 giây và so khớp**. Giữ hai vai cùng bàn tay trong khung. Nếu chủ động đổi tay so với người trong video, chọn **Tôi dùng tay ngược với người trong video**. Lần đầu cần kết nối QIPEDC để tải video. Các lần sau dùng mẫu đã lưu trên máy.

Bộ trích mẫu cần Python có `mediapipe==0.10.21`. `start.sh` dùng `.venv-sstcn` nếu có và chưa kích hoạt môi trường ảo khác. Dùng `VSL_PYTHON` để chỉ định trình thông dịch khác.

Để xem vì sao phụ đề bị sai, mở **Chẩn đoán video** từ thanh bên trái, nhập ký hiệu định làm và bấm **Ghi chẩn đoán 5 giây**. Bấm mốc thời gian trong báo cáo để tua đến lúc mất dấu tay hoặc xem cửa sổ nào sinh dự đoán. Với `mấy giờ?`, chế độ **Từ vựng → Tự do** ghi thêm hai pha chỉ cổ tay và đưa tay lên. Video không có âm thanh và nằm trong trình duyệt. Bạn có thể tải clip hoặc báo cáo JSON riêng.

# 🚀 Hướng dẫn cài đặt SIGN_LANGUAGE

## 📦 Cài đặt lần đầu

### 1. Kiểm tra môi trường

```bash
python --version
node --version
npm --version
git --version
```
Cài đặt các môi trường chưa có

### 2. Tạo môi trường ảo Python

```bash
python -m venv .venv
```

### 3. Kích hoạt môi trường ảo

```bash
.\.venv\Scripts\Activate.ps1
```

Sau khi kích hoạt thành công, Terminal sẽ xuất hiện `(.venv)`.

### 4. Cập nhật pip

```bash
python -m pip install --upgrade pip
```

### 5. Cài đặt thư viện Backend

```bash
python -m pip install fastapi uvicorn pydantic numpy opencv-python torch
```

### 6. Chạy Backend

```bash
python -m uvicorn backend.app:app --reload
```

Backend mặc định:

- API: `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`

> Giữ Terminal Backend đang chạy và mở một Terminal mới để chạy Frontend.

### 7. Cài đặt Frontend

```bash
npm install
```

### 8. Chạy Frontend

```bash
npm run dev
```

Mở địa chỉ Local được hiển thị trên Terminal sau khi Vite khởi động.

---

# 🔄 Các lần chạy sau

Sau khi đã cài đặt dự án lần đầu, chỉ cần chạy Backend và Frontend.

## Terminal 1 — Backend

Kích hoạt môi trường ảo:

```bash
.\.venv\Scripts\Activate.ps1
```

Chạy Backend:

```bash
python -m uvicorn backend.app:app --reload
```

## Terminal 2 — Frontend

Chạy Frontend:

```bash
npm run dev
```

---

# 📚 Bộ từ vựng VSL thống nhất

Cả hai chiều **Camera → Văn bản** và **Văn bản → Ký hiệu** dùng chung một bộ từ: `backend/data/vsl_lexicon.json` (3.363 mục từ, 4.464 video).

- **Danh mục NNKH QIPEDC (Bộ GD&ĐT)**: ~3.300 từ, video người Điếc thể hiện, nhiều từ có 3 biến thể **Bắc / Trung / Nam**, kèm từ loại và giải nghĩa.
- **102 từ giáo trình**: mỗi lớp của mô hình camera SSTCN được gắn vào đúng một mục từ (`sstcn_index`).
- **Robot 3D**: 50 mục từ có tư thế robot (`pose_key`); từ chưa có tư thế thì hiện video mẫu người thật.
- **Mức thông dụng**: ~800 từ được đánh dấu `common`, xếp hạng theo tần suất âm tiết trong hội thoại tiếng Việt.

API: `GET /api/lexicon?filter=common|all|camera|robot|alphabet&q=...` · chi tiết nguồn: `backend/data/sources/README.md` · dựng lại: `python -m backend.training.build_lexicon`.

---

### Video mẫu trong Từ điển

Video gốc của QIPEDC đặt chỉ mục ở cuối file và không cho trình duyệt lưu đệm, nên phát trực tiếp hay bị giật.
Backend tải mỗi video **một lần**, chuyển chỉ mục lên đầu file (không mã hóa lại hình ảnh) và lưu ở
`backend/data/media_cache/`, rồi phát qua `GET /api/media/videos/{mã}.mp4` (có tua, cache dài hạn).
Từ điển còn tải trước video của các ô đầu danh sách và của ô đang di chuột tới.

# 🤖 Robot học ký hiệu từ video người thật

Robot 3D không dùng tư thế vẽ tay cho từng từ nữa mà **bắt chước người Điếc trong video QIPEDC**:

1. **Trích chuyển động** — `backend/training/extract_motion_browser.js` chạy MediaPipe Holistic ngay trong trình duyệt trên trang QIPEDC, lấy tọa độ 3D vai/khuỷu/cổ tay và 21 khớp mỗi bàn tay (8 khung/giây).
2. **Chuyển sang góc khớp robot (retargeting)** — `backend/training/retarget_motion.py`: IK cánh tay 2 đoạn, tách xoay cẳng tay/cổ tay (swing–twist), góc gập từng đốt ngón; tự cắt phần đứng yên đầu/cuối video. Sai lệch hướng cẳng tay so với video ≈ 1,5°.
3. **Phát lại** — `GET /api/motion/{mã video}`; robot chỉ diễn đoạn có động tác của ký hiệu. Tỉ lệ cơ thể robot (vai : cánh tay : cẳng tay) lấy theo người thật để tay chạm đúng ngực/trán.
4. **Diễn tả ký hiệu** — `backend/training/describe_signs.py` phân tích cùng dữ liệu video để viết cách làm từng ký hiệu: số tay, hình dạng bàn tay, hướng lòng bàn tay, hướng đầu ngón tay, vị trí và chuyển động (lưu ở `backend/data/sign_descriptions.json`, trả kèm `sign_description` trong `/api/text_to_sign` và `/api/lexicon/{id}`).

**Từ chưa được học** (vd. "làm bài tập"): khi người dùng nhập, trình duyệt tự chạy MediaPipe trên video QIPEDC của từ đó
(`src/services/motionLearner.js`), gửi điểm khớp lên `POST /api/motion/learn` → backend chuyển sang chuyển động robot,
lưu vào `backend/data/motions/` và viết phần diễn tả. Trong lúc học (~10 giây), robot đánh vần từ bằng bảng chữ cái
ngón tay; học xong robot diễn lại đúng ký hiệu. Lần sau từ đó có sẵn. Retargeting chỉ cần numpy (`backend/models/rotation.py`).

Thêm từ mới hàng loạt:

```bash
# 1) Mở https://qipedc.moet.gov.vn/dictionary → F12 → Console → dán extract_motion_browser.js
#    await vslExtract(['W00489', 'D0002', ...])   → tải về qipedc_motion_raw.json.gz
# 2) Chuyển sang góc khớp robot
python -m backend.training.batch_retarget qipedc_motion_raw.json.gz
# 3) Viết phần diễn tả ký hiệu
python -m backend.training.describe_signs qipedc_motion_raw.json.gz
```
