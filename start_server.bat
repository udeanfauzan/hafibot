@echo off
TITLE Hafibot Laptop AI Server
echo ============================================================
echo       HAFIBOT - Laptop AI Server (NVIDIA GPU CUDA)
echo ============================================================
echo.
echo Starting AI Server (YOLOv8-Pose + 1D CNN-LSTM + Web Monitoring)...
echo Server will be accessible at http://localhost:5000
echo.
if not defined HF_HOME set "HF_HOME=%~dp0.hf_cache"
if not defined HAFIBOT_QWEN_DEVICE set "HAFIBOT_QWEN_DEVICE=cpu"
if not defined HAFIBOT_DBERT_DEVICE set "HAFIBOT_DBERT_DEVICE=cpu"
call .\env_falldetect\Scripts\activate
python src\server_app.py --host 0.0.0.0 --port 5000
pause
