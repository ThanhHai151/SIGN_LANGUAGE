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
