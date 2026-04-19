@echo off
chcp 65001 >nul
cd /d "%~dp0\..\.."

set "TORCH_PY="
set "BENCHMARK_SCRIPT=软件主体\main_program\benchmark_network_architectures.py"

for %%P in (
    "E:\AI\cuda_env\python.exe"
    "C:\Users\28382\Miniconda3\envs\torch_env\python.exe"
    "D:\anaconda3\envs\torch_env\python.exe"
    "C:\anaconda3\envs\torch_env\python.exe"
) do (
    if not defined TORCH_PY if exist %%~P set "TORCH_PY=%%~P"
)

if not defined TORCH_PY (
    echo Could not find torch Python environment.
    pause
    exit /b 1
)

echo 正在运行网络结构对比...
set "MPLBACKEND=Agg"
"%TORCH_PY%" "%BENCHMARK_SCRIPT%"
if errorlevel 1 (
    echo 网络结构对比失败。
    pause
    exit /b 1
)

echo.
echo 网络结构对比完成。
echo 结果已保存到：结果\网络结构对比
pause
