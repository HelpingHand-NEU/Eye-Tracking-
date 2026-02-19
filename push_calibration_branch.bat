@echo off
cd /d "%~dp0"

git status >nul 2>&1
if errorlevel 1 (
  echo Not a git repo.
  pause
  exit /b 1
)

git checkout calibration 2>nul
if errorlevel 1 (
  git checkout -b calibration
)

git config user.email "huaqingwang.com@gmail.com" 2>nul
git config user.name "peterwang133" 2>nul

git add -A
git status
git commit -m "Calibration branch: current version"
if errorlevel 1 (
  echo Nothing to commit or commit failed.
  pause
  exit /b 1
)

git push -u origin calibration
echo Done.
pause
