@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

echo [1/4] Kiem tra Python...
where py >nul 2>&1
if errorlevel 1 (
  echo Chua co Python Launcher.
  echo Cai Python 3.12 tu https://www.python.org/downloads/windows/
  echo Khi cai, danh dau "Add python.exe to PATH", sau do chay lai file nay.
  pause
  exit /b 1
)

if not exist ".venv-pilot\Scripts\python.exe" (
  py -3.12 -m venv .venv-pilot 2>nul
  if errorlevel 1 py -3 -m venv .venv-pilot
  if errorlevel 1 (
    echo Khong tao duoc moi truong Python. Can Python 3.10 tro len.
    pause
    exit /b 1
  )
)

echo [2/4] Cai thu vien pilot nhe, khong cai PaddleOCR...
".venv-pilot\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :install_error
".venv-pilot\Scripts\python.exe" -m pip install -r requirements-gemini-pilot.txt
if errorlevel 1 goto :install_error

echo [3/4] Gan source dang editable...
".venv-pilot\Scripts\python.exe" -m pip install --no-deps -e .
if errorlevel 1 goto :install_error

echo [4/4] Tao cau hinh rieng tren may tram...
set "CONFIG_DIR=%LOCALAPPDATA%\TramCanQR"
if not exist "%CONFIG_DIR%" mkdir "%CONFIG_DIR%"
if not exist "%CONFIG_DIR%\config.env" copy /Y "config.env.example" "%CONFIG_DIR%\config.env" >nul

echo.
echo CAI DAT XONG.
echo Notepad se mo file cau hinh. Dien API key moi, luu file, sau do chay CHAY-TRAM-CAN.cmd.
start "" notepad.exe "%CONFIG_DIR%\config.env"
pause
exit /b 0

:install_error
echo.
echo CAI DAT THAT BAI. Kiem tra Internet roi chay lai file nay.
pause
exit /b 1
