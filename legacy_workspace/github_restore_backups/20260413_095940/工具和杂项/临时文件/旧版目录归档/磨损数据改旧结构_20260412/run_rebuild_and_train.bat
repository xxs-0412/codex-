@echo off
cd /d "%~dp0"

set "BASE_PY=D:\anaconda3\python.exe"
set "TORCH_PY=D:\anaconda3\envs\torch_env\python.exe"
set "PROCESS_SCRIPT=..\recursive_physics_surrogate\main_program\process_real_wear_data.py"
set "TRAIN_SCRIPT=..\recursive_physics_surrogate\main_program\train_real_wear_models.py"

echo [1/2] Processing raw xlsx files into standard dataset...
"%BASE_PY%" "%PROCESS_SCRIPT%"
if errorlevel 1 (
    echo Data processing failed.
    pause
    exit /b 1
)

echo.
echo [2/2] Training recursive StressNet and baseline static model...
"%TORCH_PY%" "%TRAIN_SCRIPT%"
if errorlevel 1 (
    echo Model training failed.
    pause
    exit /b 1
)

echo.
echo Pipeline completed successfully.
echo Processed data, trained model, and comparison figures have been saved into this folder.
pause
