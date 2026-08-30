@echo off
setlocal
set "APP_DIR=%~dp0"
set "PROJECT_PYTHON=%APP_DIR%.venv\Scripts\python.exe"
set "FALLBACK_PYTHON=E:\Anaconda\envs\openarm\python.exe"
set "PYTHONDONTWRITEBYTECODE=1"

if exist "%PROJECT_PYTHON%" (
  set "PYTHON_EXE=%PROJECT_PYTHON%"
) else (
  set "PYTHON_EXE=%FALLBACK_PYTHON%"
)

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python environment not found.
  pause
  exit /b 1
)

"%PYTHON_EXE%" "%APP_DIR%run.py" --config "%APP_DIR%config\legacy_vision_ws.json"
if errorlevel 1 pause
endlocal
