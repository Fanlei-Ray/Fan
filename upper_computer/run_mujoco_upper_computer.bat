@echo off
setlocal
set "APP_DIR=%~dp0"
set "PYTHON_EXE=%APP_DIR%.venv\Scripts\python.exe"
set "PYTHONDONTWRITEBYTECODE=1"
set "TEMP=%APP_DIR%work\temp"
set "TMP=%APP_DIR%work\temp"

if not exist "%PYTHON_EXE%" (
  echo [ERROR] E-drive project Python environment not found: %PYTHON_EXE%
  pause
  exit /b 1
)

if not exist "%TEMP%" mkdir "%TEMP%"
"%PYTHON_EXE%" "%APP_DIR%run.py" --config "%APP_DIR%config\mujoco_vision_ws.json"
if errorlevel 1 pause
endlocal
