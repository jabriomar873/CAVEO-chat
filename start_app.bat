@echo off
REM Enhanced PDF Chat Application Launcher
REM This script starts the application with optimized settings

echo ====================================
echo  Enhanced PDF Chat Application
echo ====================================
echo.
echo Starting application with optimized settings...
echo.

REM Set environment variables to reduce warnings
set TORCH_LOGS=0
set PYTHONWARNINGS=ignore::UserWarning:torch,ignore::DeprecationWarning:langchain

REM Start the application
cd /d "%~dp0"
streamlit run app.py --theme.base="light" --server.headless=false

echo.
echo Application stopped.
pause
