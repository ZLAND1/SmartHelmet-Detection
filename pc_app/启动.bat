@echo off
cd /d "%~dp0src"
echo Starting Helmet Detection App...
echo.
python app.py
if errorlevel 1 (
    echo.
    echo [ERROR] Launch failed! Please install dependencies first:
    echo    pip install -r ..\requirements-pc.txt
    pause
)
