@echo off
cd /d "%~dp0"

set "TORCH_PY=D:\anaconda3\envs\torch_env\python.exe"
set "BENCHMARK_SCRIPT=..\recursive_physics_surrogate\main_program\benchmark_network_architectures.py"

echo Running neural architecture benchmark on the held-out test case...
"%TORCH_PY%" "%BENCHMARK_SCRIPT%"
if errorlevel 1 (
    echo Benchmark failed.
    pause
    exit /b 1
)

echo.
echo Benchmark completed successfully.
echo Results saved to comparison_results\network_architecture_benchmark
pause
