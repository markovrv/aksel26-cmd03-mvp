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
    python --version
    echo [OK] Python найден
    goto check_venv
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
echo [OK] Python установлен

:: ─── 2. Виртуальное окружение ──────────────────────────
:check_venv
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

:: ─── 3. Установка зависимостей ─────────────────────────
:install_deps
echo [..] Устанавливаю зависимости...
.venv\Scripts\python -m pip install --upgrade pip -q
.venv\Scripts\pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Не удалось установить зависимости
    pause
    exit /b 1
)
echo [OK] Зависимости установлены

:: Добавляем пути CUDA
set "CUDA_PATH=%~dp0.venv\Lib\site-packages\nvidia\cublas\bin"
set "CUDNN_PATH=%~dp0.venv\Lib\site-packages\nvidia\cudnn\bin"
set "CUBLAS_PATH=%~dp0.venv\Lib\site-packages\nvidia\cublas\bin"
set "NVRTC_PATH=%~dp0.venv\Lib\site-packages\nvidia\cuda_nvrtc\bin"
set "PATH=%CUDA_PATH%;%CUDNN_PATH%;%NVRTC_PATH%;%CUBLAS_PATH%;%PATH%"
set "PATH=%PATH%;C:\Program Files\OpenSSL-Win64\bin"

echo.
echo ========================================
echo   Установка завершена. Запускаю сервер...
echo ========================================
echo.

:: ─── 4. Запуск сервера ────────────────────────────────
.venv\Scripts\python fserver.py

pause