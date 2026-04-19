$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path (Join-Path $scriptDir "..\..")).Path
$envPy = "E:\AI\cuda_env\python.exe"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"

if (-not (Test-Path $envPy)) {
    throw "Python not found at $envPy. Please ensure E:\AI\cuda_env exists."
}

Write-Host "[1/4] Upgrading pip ..."
& $envPy -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip."
}

Write-Host "[2/4] Installing PyTorch with CUDA 12.8 support ..."
& $envPy -m pip install torch --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install PyTorch CUDA."
}

Write-Host "[3/4] Installing training dependencies ..."
& $envPy -m pip install numpy pandas matplotlib scipy scikit-learn openpyxl
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install training dependencies."
}

Write-Host "[4/4] Validating CUDA environment ..."
& $envPy -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.get_device_name(0)}'); print('validate_cuda_env: ok')"
if ($LASTEXITCODE -ne 0) {
    throw "CUDA environment validation failed."
}

Write-Host ""
Write-Host "Environment is ready."
Write-Host "Python: $envPy"
Write-Host "Device: CUDA (auto-detected by device_runtime.py)"
