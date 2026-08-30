@echo off
setlocal
set "APP_DIR=%~dp0"
set "PROJECT_ROOT=%APP_DIR%.."
set "PYTHON_EXE=%APP_DIR%.venv\Scripts\python.exe"
set "PYTHONDONTWRITEBYTECODE=1"
set "TEMP=%APP_DIR%work\temp"
set "TMP=%APP_DIR%work\temp"
set "YOLO_CONFIG_DIR=%APP_DIR%work\ultralytics"
set "TORCH_HOME=%APP_DIR%work\torch"
set "MPLCONFIGDIR=%APP_DIR%work\matplotlib"
set "BRIDGE_PORT=8765"

if not exist "%PYTHON_EXE%" (
  echo [ERROR] E-drive project Python environment not found: %PYTHON_EXE%
  pause
  exit /b 1
)

if not exist "%TEMP%" mkdir "%TEMP%"

set "PORT_PID="
for /f "tokens=5" %%P in ('netstat -ano -p tcp ^| findstr /R /C:":%BRIDGE_PORT% .*LISTENING"') do set "PORT_PID=%%P"
if defined PORT_PID (
  echo [ERROR] WebSocket port %BRIDGE_PORT% is already in use by PID %PORT_PID%.
  echo Close the previous bridge window first, then run this file again.
  pause
  exit /b 2
)

cd /d "%PROJECT_ROOT%"
echo [INFO] Official YCB scans + strict official YOLOv8-Seg, no RGB fallback.
echo [INFO] MuJoCo and this console stay open until you close the viewer or press Ctrl+C.
echo [INFO] WebSocket: ws://127.0.0.1:%BRIDGE_PORT%
"%PYTHON_EXE%" "scripts\vision\ycb_upper_bridge.py" --viewer --host 127.0.0.1 --port %BRIDGE_PORT% --fps 8 --confidence 0.06 --speed-scale 0.40 --playback-rate 1.80
echo.
echo [INFO] Bridge stopped. This window is intentionally kept open so errors remain visible.
pause
endlocal
