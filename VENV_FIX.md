# Fix: "No pyvenv.cfg" on Windows

## What was wrong
The `.venv` folder was created on **macOS** (paths like `/Library/Frameworks/Python.framework/...`). On Windows that venv is invalid, so you get "No pyvenv.cfg file" when the IDE or terminal uses it.

## What we did
- Created a **Windows-native** virtual environment in the `venv` folder using `C:\python.exe`.
- This `venv` works on your machine; use it instead of `.venv`.

## How to run

### Option A: Use the run script (easiest)
```batch
run_main.bat
```
This activates `venv`, installs dependencies, and runs `main.py`.

### Option B: Use Cursor/VS Code interpreter
1. Press `Ctrl+Shift+P` → **Python: Select Interpreter**
2. Choose **Enter interpreter path...**
3. Select: `c:\Users\aXun\Desktop\Capstone\Eye-Tracking-\venv\Scripts\python.exe`
4. Run `main.py` from the editor (F5 or Run button).

### Option C: Manual in terminal
```powershell
cd "c:\Users\aXun\Desktop\Capstone\Eye-Tracking-"
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

## If you don't have C:\python.exe
Install Python from [python.org](https://www.python.org/downloads/) or ensure your system Python is on PATH, then recreate the venv:
```powershell
"C:\Path\To\Your\python.exe" -m venv venv
```
