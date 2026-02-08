@echo off
REM Use the Windows venv (fixes "No pyvenv.cfg" when .venv was created on Mac)
cd /d "%~dp0"
if not exist "venv\Scripts\activate.bat" (
    echo venv not found. Create it with: C:\python.exe -m venv venv
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo pip install failed. On Windows, dlib needs Visual Studio C++ to build.
    echo Install with: winget install -e --id Microsoft.VisualStudio.2022.BuildTools --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools;includeRecommended"
    echo Then close ALL terminals, open a new one, and run this batch again.
    pause
    exit /b 1
)
python main.py
if errorlevel 1 pause
