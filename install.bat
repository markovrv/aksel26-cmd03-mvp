@echo off
chcp 65001 >nul
title Voice Stream — Установка
cd /d "%~dp0"

echo ========================================
echo   Voice Stream — Установка
echo ========================================
echo.

:: ─── 1. Проверка Python ────────────────────────────────
:check_python
python --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PY_VER=%%v"
    echo [OK] Python %PY_VER%
    goto check_cuda
)

echo [..] Python не найден. Скачиваю Python 3.13...

:: Определяем архитектуру
if "%PROCESSOR_ARCHITECTURE%"=="AMD64" (
    set PY_URL=https://www.python.org/ftp/python/3.13.2/python-3.13.2-amd64.exe
    set PY_INSTALLER=python-3.13.2-amd64.exe
) else (
    set PY_URL=https://www.python.org/ftp/python/3.13.2/python-3.13.2.exe
    set PY_INSTALLER=python-3.13.2.exe
)

:: Скачиваем установщик
echo Скачиваю %PY_INSTALLER%...
powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PY_URL%' -OutFile '%TEMP%\%PY_INSTALLER%' }"
if %errorlevel% neq 0 (
    echo [ERROR] Не удалось скачать Python. Проверьте интернет-соединение.
    pause
    exit /b 1
)

:: Устанавливаем
echo Устанавливаю Python...
start /wait "" "%TEMP%\%PY_INSTALLER%" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
if %errorlevel% neq 0 (
    echo [WARN] Установка Python могла завершиться с кодом %errorlevel%
)

:: Обновляем PATH для текущей сессии
for /f "tokens=2*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') do set "PATH=%%b;%PATH%"
:: Проверяем ещё раз
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python не обнаружен после установки. Перезапустите скрипт.
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PY_VER=%%v"
echo [OK] Python %PY_VER% установлен

:: ─── 2. Проверка NVIDIA CUDA (GPU) ─────────────────────
:check_cuda
echo.
echo --[ Проверка NVIDIA CUDA ]--

:: Проверяем NVIDIA GPU через nvidia-smi
echo [..] Проверяю NVIDIA CUDA...

:: Пробуем найти nvidia-smi — сначала System32 (для 64-битного cmd),
:: затем Sysnative (для 32-битного cmd на 64-битной ОС)
set "NVSMI_CMD="
if exist "%SystemRoot%\System32\nvidia-smi.exe" set "NVSMI_CMD=%SystemRoot%\System32\nvidia-smi.exe"
if not defined NVSMI_CMD if exist "%SystemRoot%\Sysnative\nvidia-smi.exe" set "NVSMI_CMD=%SystemRoot%\Sysnative\nvidia-smi.exe"
if not defined NVSMI_CMD if exist "%SystemRoot%\SysWOW64\nvidia-smi.exe" set "NVSMI_CMD=%SystemRoot%\SysWOW64\nvidia-smi.exe"
if not defined NVSMI_CMD if exist "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe" set "NVSMI_CMD=C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"

if defined NVSMI_CMD (
    "%NVSMI_CMD%" --query-gpu=name,driver_version,cuda_version --format=csv,noheader > "%TEMP%\nvsmi_out.txt" 2>&1
    if %errorlevel% equ 0 (
        for /f "tokens=1,2,3 delims=, " %%a in ('type "%TEMP%\nvsmi_out.txt"') do (
            set "GPU_NAME=%%a"
            set "DRIVER_VER=%%b"
            set "CUDA_VER=%%c"
        )
        del "%TEMP%\nvsmi_out.txt" 2>nul
        echo [OK] GPU: %GPU_NAME% ^| Драйвер: %DRIVER_VER% ^| CUDA: %CUDA_VER%
        echo [OK] CUDA-ускорение доступно
        goto check_openssl
    )
    del "%TEMP%\nvsmi_out.txt" 2>nul
) else (
    echo [..] nvidia-smi не найден ни в одном из стандартных путей.
)

:: CUDA не найдена — даём инструкции
echo [WARN] NVIDIA CUDA не обнаружена.
echo.
echo Voice Stream использует NVIDIA CUDA 12.x для ускорения Whisper на GPU.
echo.
echo Варианты:
echo   1) Установите/обновите драйверы NVIDIA с CUDA:
echo      https://developer.nvidia.com/cuda-downloads
echo.
echo   2) Переключитесь на CPU-режим (измените в fserver.py: DEVICE = "cpu", COMPUTE_TYPE = "int8")
echo.
choice /C YN /M "Продолжить установку без CUDA"
if %errorlevel% equ 2 (
    echo [INFO] Установка прервана пользователем.
    pause
    exit /b 1
)
echo [INFO] Продолжаю без CUDA-ускорения.

:: ─── 3. Проверка OpenSSL ──────────────────────────────
:check_openssl
echo.
echo --[ Проверка OpenSSL ]--
set "OPENSSL_FOUND="

:: Ищем в PATH
openssl version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=1,2" %%a in ('openssl version') do set "OPENSSL_VER=%%b"
    echo [OK] OpenSSL %OPENSSL_VER% (найден в PATH)
    set "OPENSSL_FOUND=1"
    set "OPENSSL_CMD=openssl"
    goto check_venv
)

:: Ищем в Program Files
if exist "C:\Program Files\OpenSSL-Win64\bin\openssl.exe" (
    for /f "tokens=1,2" %%a in ('"C:\Program Files\OpenSSL-Win64\bin\openssl.exe" version') do set "OPENSSL_VER=%%b"
    echo [OK] OpenSSL %OPENSSL_VER% (C:\Program Files\OpenSSL-Win64\bin\)
    set "OPENSSL_FOUND=1"
    set "OPENSSL_CMD=C:\Program Files\OpenSSL-Win64\bin\openssl.exe"
    goto check_venv
)

if exist "C:\Program Files (x86)\OpenSSL-Win32\bin\openssl.exe" (
    for /f "tokens=1,2" %%a in ('"C:\Program Files (x86)\OpenSSL-Win32\bin\openssl.exe" version') do set "OPENSSL_VER=%%b"
    echo [OK] OpenSSL %OPENSSL_VER% (C:\Program Files (x86)\OpenSSL-Win32\bin\)
    set "OPENSSL_FOUND=1"
    set "OPENSSL_CMD=C:\Program Files (x86)\OpenSSL-Win32\bin\openssl.exe"
    goto check_venv
)

echo [WARN] OpenSSL не найден.
echo.
echo OpenSSL необходим для генерации самоподписанного SSL-сертификата
echo (без него HTTPS/WSS не будет работать).
echo.
echo Скачайте и установите OpenSSL:
echo   1) Перейдите на https://slproweb.com/products/Win32OpenSSL.html
echo   2) Скачайте "Win64 OpenSSL v3.x.x" (Light-версия подойдёт)
echo   3) Установите в C:\Program Files\OpenSSL-Win64
echo   4) После установки перезапустите этот скрипт
echo.
choice /C YN /M "Продолжить установку без OpenSSL (сертификат не будет создан)"
if %errorlevel% equ 2 (
    echo [INFO] Установка прервана пользователем.
    pause
    exit /b 1
)
echo [INFO] Продолжаю без OpenSSL. Сертификат нужно будет создать вручную.
echo   Команда: openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"

:: ─── 4. Виртуальное окружение ──────────────────────────
:check_venv
echo.
echo --[ Виртуальное окружение ]--
if exist ".venv\Scripts\python.exe" (
    echo [OK] Виртуальное окружение уже создано
    goto install_deps
)

echo [..] Создаю виртуальное окружение...
python -m venv .venv
if %errorlevel% neq 0 (
    echo [ERROR] Не удалось создать виртуальное окружение
    pause
    exit /b 1
)
echo [OK] Виртуальное окружение создано

:: ─── 5. Установка зависимостей Python ─────────────────
:install_deps
echo.
echo --[ Установка зависимостей Python ]--
echo [..] Обновляю pip...
.venv\Scripts\python -m pip install --upgrade pip -q
if %errorlevel% equ 0 (
    echo [OK] pip обновлён
) else (
    echo [WARN] Не удалось обновить pip
)

echo [..] Устанавливаю зависимости из requirements.txt...
echo [..] Будут установлены:
echo    - fast-whisper (распознавание речи)
echo    - torch + silero (синтез речи TTS)
echo    - websockets, httpx (сервер)
echo    - CUDA-библиотеки nvidia (ускорение GPU)
echo.
.venv\Scripts\pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Не удалось установить зависимости
    echo.
    echo Попробуйте вручную:
    echo   .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)
echo [OK] Зависимости установлены

:: ─── 6. Создание SSL-сертификата (если есть OpenSSL) ──
echo.
echo --[ SSL-сертификат ]--
if not defined OPENSSL_FOUND (
    echo [SKIP] OpenSSL не найден — пропускаем создание сертификата.
    echo   Создайте вручную: openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"
    goto final
)

if exist "cert.pem" if exist "key.pem" (
    echo [OK] Сертификат уже существует
) else (
    echo [..] Генерирую самоподписанный сертификат...
    "%OPENSSL_CMD%" req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost" >nul 2>&1
    if exist "cert.pem" (
        echo [OK] Сертификат создан: cert.pem, key.pem
    ) else (
        echo [WARN] Не удалось создать сертификат. Создайте вручную:
        echo   openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"
    )
)

:: ─── 7. Финальное сообщение ──────────────────────────
:final
echo.
echo ========================================
echo   Установка завершена!
echo ========================================
echo.
echo Для запуска сервера выполните:
echo   run_server.bat
echo.
echo Или вручную:
echo   .venv\Scripts\python fserver.py
echo.
echo ╔══ Сервисы ═══════════════════════════════════════╗
echo ║  Распознавание речи: Whisper (faster-whisper)    ║
echo ║  Синтез речи (TTS):   Silero (локально)          ║
echo ║  LLM чат:             через WebSocket             ║
echo ╚═══════════════════════════════════════════════════╝
echo.
echo Веб-интерфейс доступен по адресу:
echo   https://localhost:8765
echo.

pause