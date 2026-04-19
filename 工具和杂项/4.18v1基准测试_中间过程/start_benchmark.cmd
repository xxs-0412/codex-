@echo off
setlocal
chcp 65001 >nul

cd /d "%~dp0\..\.."

if not exist ".codex-runlogs" mkdir ".codex-runlogs"
set "LOG_FILE=.codex-runlogs\4_18v1_benchmark.log"

echo [%date% %time%] Starting 4.18v1 benchmark > "%LOG_FILE%"
"E:\AI\cuda_env\python.exe" "results_ascii\4.18v1基准测试\run_benchmark.py" >> "%LOG_FILE%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"
echo [%date% %time%] 4.18v1 benchmark finished with exit code %EXIT_CODE% >> "%LOG_FILE%"

exit /b %EXIT_CODE%
