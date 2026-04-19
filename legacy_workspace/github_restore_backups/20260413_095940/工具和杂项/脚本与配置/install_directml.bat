@echo off
setlocal
set "PYTHON=E:\AI\cuda_env\python.exe"
if not exist "%PYTHON%" (
    echo ERROR: Python not found at %PYTHON%
    pause
    exit /b 1
)
echo Installing PyTorch with CUDA support...
"%PYTHON%" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch CUDA
    pause
    exit /b 1
)
echo.
echo Verifying installation...
"%PYTHON%" -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No CUDA device')"
if errorlevel 1 (
    echo ERROR: PyTorch CUDA import failed
    pause
    exit /b 1
)
echo.
echo SUCCESS: PyTorch with CUDA installed and working!
pause
