@echo off
chcp 65001 >nul
cd /d "%~dp0\.."

set "PYTHON_EXE="
for %%P in (
    "E:\AI\cuda_env\python.exe"
    "C:\Users\28382\Miniconda3\envs\torch_env\python.exe"
    "D:\anaconda3\envs\torch_env\python.exe"
    "C:\anaconda3\envs\torch_env\python.exe"
) do (
    if not defined PYTHON_EXE if exist %%~P set "PYTHON_EXE=%%~P"
)

if not defined PYTHON_EXE (
    echo Could not find a usable Python environment.
    echo Expected one of:
    echo   E:\AI\cuda_env\python.exe
    echo   C:\Users\28382\Miniconda3\envs\torch_env\python.exe
    pause
    exit /b 1
)

"%PYTHON_EXE%" main_program\predict_gui.py
pause
