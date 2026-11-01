@echo off
title VSL Connect AI Starter
cd /d "%~dp0"
echo Starting VSL Backend...
start "VSL Backend (FastAPI)" cmd /c "run_backend.bat"
echo Starting VSL Frontend...
start "VSL Frontend (Vite)" cmd /c "run_frontend.bat"
echo Done! Both servers are starting in separate windows.

