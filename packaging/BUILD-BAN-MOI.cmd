@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

if not exist "tools\build_windows.ps1" (
  echo Khong tim thay tools\build_windows.ps1. Hay chay file nay tai thu muc goc source.
  pause
  exit /b 1
)

echo Dang chay test...
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m pytest -q
) else if exist ".venv-pilot\Scripts\python.exe" (
  ".venv-pilot\Scripts\python.exe" -m pip install "pytest==8.4.2" "qrcode[pil]==8.2"
  if errorlevel 1 goto :failed
  ".venv-pilot\Scripts\python.exe" -m pytest tests\test_gemini_weight.py tests\test_ui.py tests\test_windows_app.py -q
) else (
  echo Chua co moi truong Python. Hay chay CAI-DAT-LAN-DAU.cmd truoc.
  pause
  exit /b 1
)
if errorlevel 1 goto :failed

echo Dang tao EXE va bo cai...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\tools\build_windows.ps1"
if errorlevel 1 goto :failed

echo.
echo BUILD THANH CONG.
if exist "dist\installer" (
  echo Bo cai nam trong: dist\installer
  start "" explorer.exe "dist\installer"
) else (
  echo Ban portable nam trong: dist\TramCanQR
  start "" explorer.exe "dist\TramCanQR"
)
pause
exit /b 0

:failed
echo.
echo BUILD THAT BAI. Xem loi phia tren; khong gui file cu cho khach.
pause
exit /b 1
