@echo off
chcp 65001 >nul
cd /d "%~dp0"

:: Добавляем пути к CUDA-библиотекам NVIDIA
set "CUDA_PATH=%~dp0.venv\Lib\site-packages\nvidia\cublas\bin"
set "CUDNN_PATH=%~dp0.venv\Lib\site-packages\nvidia\cudnn\bin"
set "CUBLAS_PATH=%~dp0.venv\Lib\site-packages\nvidia\cublas\bin"
set "NVRTC_PATH=%~dp0.venv\Lib\site-packages\nvidia\cuda_nvrtc\bin"
set "PATH=%CUDA_PATH%;%CUDNN_PATH%;%NVRTC_PATH%;%CUBLAS_PATH%;%PATH%"

set "PATH=%PATH%;C:\Program Files\OpenSSL-Win64\bin"

.venv\Scripts\python fserver.py
pause