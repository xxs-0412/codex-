@echo off
setlocal

set "ROOT_DIR=%~dp0"
set "APP_DIR="
set "PYTHON_EXE="

for /d %%D in ("%ROOT_DIR%*") do (
    if exist "%%~fD\main_program\predict_gui.py" (
        set "APP_DIR=%%~fD"
    )
)

if not defined APP_DIR (
    echo Could not find app directory.
    pause
    exit /b 1
)

for %%P in (
    "E:\AI\cuda_env\python.exe"
    "C:\Users\28382\Miniconda3\envs\torch_env\python.exe"
    "D:\anaconda3\envs\torch_env\python.exe"
    "C:\anaconda3\envs\torch_env\python.exe"
    "C:\Users\28382\AppData\Local\Programs\Python\Python313\python.exe"
    "C:\Users\28382\AppData\Local\Programs\Python\Python312\python.exe"
    "C:\Users\28382\AppData\Local\Programs\Python\Python311\python.exe"
    "C:\Users\28382\AppData\Local\Programs\Python\Python310\python.exe"
) do (
    if not defined PYTHON_EXE if exist %%~P set "PYTHON_EXE=%%~P"
)

if not defined PYTHON_EXE (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Could not find Python.
        pause
        exit /b 1
    )
    set "PYTHON_EXE=python"
)

pushd "%APP_DIR%"
"%PYTHON_EXE%" main_program\predict_gui.py
set "EXIT_CODE=%ERRORLEVEL%"
popd

if not "%EXIT_CODE%"=="0" (
    echo Launch failed. Exit code: %EXIT_CODE%
)
pause
