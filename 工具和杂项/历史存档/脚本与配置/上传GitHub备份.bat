@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%push_backup_branch.ps1"

if errorlevel 1 (
    echo.
    echo Backup upload failed.
    exit /b 1
)

echo.
echo Backup upload finished.
exit /b 0
