@echo off
setlocal
set "APP_DIR=%~dp0"

:MENU
cls
echo ============================================================
echo OpenArm Course Demo Launcher
echo ============================================================
echo [1] Start official YCB + YOLOv8-Seg full demo
echo [2] Start legacy MuJoCo vision bridge
echo [3] Start legacy MuJoCo upper computer
echo [4] Start real-vision monitor (observe only)
echo [5] Scan serial ports (read only)
echo [6] Run all upper-computer tests
echo [7] Open report folder
echo [0] Exit
echo.
set /p "CHOICE=Select: "

if "%CHOICE%"=="1" start "OpenArm Official YCB Demo" "%APP_DIR%run_ycb_real_objects_demo.bat"
if "%CHOICE%"=="2" start "OpenArm MuJoCo Bridge" "%APP_DIR%run_mujoco_vision_bridge.bat"
if "%CHOICE%"=="3" start "OpenArm MuJoCo Upper" "%APP_DIR%run_mujoco_upper_computer.bat"
if "%CHOICE%"=="4" start "OpenArm Real Vision Monitor" "%APP_DIR%run_vision_monitor.bat"
if "%CHOICE%"=="5" start "OpenArm Serial Scan" "%APP_DIR%scan_serial_ports.bat"
if "%CHOICE%"=="6" call :RUN_TESTS
if "%CHOICE%"=="7" start "" "%APP_DIR%report"
if "%CHOICE%"=="0" exit /b 0
goto MENU

:RUN_TESTS
if not exist "%APP_DIR%.venv\Scripts\python.exe" (
  echo [ERROR] Project Python environment not found.
  pause
  goto :eof
)
pushd "%APP_DIR%.."
"%APP_DIR%.venv\Scripts\python.exe" -m unittest discover -s "%APP_DIR%tests" -v
popd
pause
goto :eof
