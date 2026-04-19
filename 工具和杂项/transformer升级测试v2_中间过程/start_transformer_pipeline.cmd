@echo off
setlocal
chcp 65001 >nul

cd /d "%~dp0\..\.."

if not exist ".codex-runlogs" mkdir ".codex-runlogs"
set "LOG_FILE=.codex-runlogs\transformer_v2_pipeline.log"

echo [%date% %time%] Starting transformer upgrade v2 pipeline > "%LOG_FILE%"

echo [%date% %time%] Step 1: candidate rounds >> "%LOG_FILE%"
"E:\AI\cuda_env\python.exe" "tools_ascii\transformer升级测试v2_中间过程\run_all_rounds.py" >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto :fail

echo [%date% %time%] Step 2: final benchmark >> "%LOG_FILE%"
"E:\AI\cuda_env\python.exe" "results_ascii\transformer升级测试v2\run_benchmark.py" >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto :fail

echo [%date% %time%] Step 3: write docs >> "%LOG_FILE%"
"E:\AI\cuda_env\python.exe" "tools_ascii\transformer升级测试v2_中间过程\write_upgrade_docs.py" >> "%LOG_FILE%" 2>&1
if errorlevel 1 goto :fail

echo [%date% %time%] Transformer upgrade v2 pipeline finished successfully >> "%LOG_FILE%"
exit /b 0

:fail
set "EXIT_CODE=%ERRORLEVEL%"
echo [%date% %time%] Transformer upgrade v2 pipeline failed with exit code %EXIT_CODE% >> "%LOG_FILE%"
exit /b %EXIT_CODE%
