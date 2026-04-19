$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$probeDir = $scriptDir
$rootDir = $null
while ($probeDir) {
    if (Test-Path -LiteralPath (Join-Path $probeDir "result_benchmark_418_v1")) {
        $rootDir = $probeDir
        break
    }
    $parentDir = Split-Path -Parent $probeDir
    if ($parentDir -eq $probeDir) {
        break
    }
    $probeDir = $parentDir
}

if (-not $rootDir) {
    throw "Unable to locate workspace root from launcher path: $($MyInvocation.MyCommand.Path)"
}

Set-Location -LiteralPath $rootDir

$logDir = Join-Path $rootDir ".codex-runlogs"
if (-not (Test-Path -LiteralPath $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

$logFile = Join-Path $logDir "4_18v1_benchmark.log"
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting 4.18v1 benchmark" | Set-Content -LiteralPath $logFile -Encoding utf8

& "E:\AI\cuda_env\python.exe" "result_benchmark_418_v1\run_benchmark.py" >> $logFile 2>&1
$exitCode = $LASTEXITCODE

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 4.18v1 benchmark finished with exit code $exitCode" | Add-Content -LiteralPath $logFile -Encoding utf8
exit $exitCode
