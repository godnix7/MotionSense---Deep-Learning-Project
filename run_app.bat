@echo off
REM Launch MotionSense with the project's virtual environment (CUDA PyTorch).
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m streamlit run app.py
) else (
    python -m streamlit run app.py
)
