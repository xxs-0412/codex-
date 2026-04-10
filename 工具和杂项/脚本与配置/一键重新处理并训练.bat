@echo off
chcp 65001 >nul
cd /d "%~dp0\..\.."

set "BASE_PY=D:\anaconda3\python.exe"
set "TORCH_PY=D:\anaconda3\envs\torch_env\python.exe"
set "PROCESS_SCRIPT=软件主体\main_program\process_real_wear_data.py"
set "TRAIN_SCRIPT=软件主体\main_program\train_real_wear_models.py"

echo [1/2] 正在把原始磨损数据整理成修改后数据...
"%BASE_PY%" "%PROCESS_SCRIPT%"
if errorlevel 1 (
    echo 修改数据处理失败。
    pause
    exit /b 1
)

echo.
echo [2/2] 正在训练递推模型并生成对比结果...
"%TORCH_PY%" "%TRAIN_SCRIPT%"
if errorlevel 1 (
    echo 模型训练失败。
    pause
    exit /b 1
)

echo.
echo 处理和训练已完成。
echo 修改后的数据在：磨损数据（改）
echo 对比结果在：结果
echo 训练得到的模型在：工具和杂项\训练模型
pause
