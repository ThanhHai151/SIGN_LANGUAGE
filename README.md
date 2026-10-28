# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some Oxlint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the Oxlint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and Oxlint's TypeScript related rules in your project.


cd /home/thanhhai/Documents/UNIVERSITY
./start.sh

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
