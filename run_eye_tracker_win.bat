@echo off
cd /d "%~dp0"
echo Starting Eye Tracker (Windows)...
echo.

REM Use the project's Python venv if it exists
if exist "eye_tracker_venv\Scripts\python.exe" (
  "eye_tracker_venv\Scripts\python.exe" eye_tracker_win.py
) else (
  python eye_tracker_win.py
)

if errorlevel 1 (
  echo.
  echo If you see "No module named cv2", activate the venv and install:
  echo   eye_tracker_venv\Scripts\activate
  echo   pip install -r requirements.txt
  echo.
)
pause
