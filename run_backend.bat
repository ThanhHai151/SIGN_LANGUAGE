@echo off
title VSL Backend (FastAPI)
set PYTHONUTF8=1
cd /d "e:\NAM4\SEMINAR\SIGN_LANGUAGE"
".venv\Scripts\python.exe" -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
pause

