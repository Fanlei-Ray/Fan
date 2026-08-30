@echo off
setlocal
set "APP_DIR=%~dp0"
set "PYTHON_EXE=E:\Anaconda\envs\openarm\python.exe"
set "PYTHONDONTWRITEBYTECODE=1"

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python not found: %PYTHON_EXE%
  echo Edit PYTHON_EXE in this file after confirming the environment path.
  pause
  exit /b 1
)

"%PYTHON_EXE%" "%APP_DIR%run.py"
if errorlevel 1 pause
endlocal
