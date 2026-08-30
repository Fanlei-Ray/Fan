@echo off
setlocal
set "APP_DIR=%~dp0"
start "OpenArm YCB MuJoCo + YOLOv8-Seg bridge" cmd /k call "%APP_DIR%run_ycb_real_objects_bridge.bat"
timeout /t 3 /nobreak >nul
call "%APP_DIR%run_ycb_real_objects_upper_computer.bat"
endlocal
