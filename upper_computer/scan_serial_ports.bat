@echo off
setlocal
chcp 65001 >nul
set "APP_DIR=%~dp0"
set "PYTHON_EXE=%APP_DIR%.venv\Scripts\python.exe"
set "PYTHONDONTWRITEBYTECODE=1"

if not exist "%PYTHON_EXE%" (
  echo [ERROR] E-drive project Python environment not found: %PYTHON_EXE%
  pause
  exit /b 1
)

echo 只读扫描 COM 端口，不会打开串口或发送数据。
"%PYTHON_EXE%" "%APP_DIR%tools\list_serial_ports.py"
pause
endlocal
