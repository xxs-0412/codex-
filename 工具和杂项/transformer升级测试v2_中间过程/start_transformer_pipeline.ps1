$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$probeDir = $scriptDir
$rootDir = $null
while ($probeDir) {
    if (Test-Path -LiteralPath (Join-Path $probeDir "result_transformer_upgrade_v2")) {
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

$logFile = Join-Path $logDir "transformer_v2_pipeline.log"
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting transformer upgrade v2 pipeline" | Set-Content -LiteralPath $logFile -Encoding utf8

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$ScriptPath
    )

    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Label" | Add-Content -LiteralPath $logFile -Encoding utf8
    & "E:\AI\cuda_env\python.exe" $ScriptPath >> $logFile 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

Invoke-Step -Label "Step 1: candidate rounds" -ScriptPath "tool_transformer_upgrade_v2_intermediate\run_all_rounds.py"
Invoke-Step -Label "Step 2: final benchmark" -ScriptPath "result_transformer_upgrade_v2\run_benchmark.py"
Invoke-Step -Label "Step 3: write docs" -ScriptPath "tool_transformer_upgrade_v2_intermediate\write_upgrade_docs.py"

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Transformer upgrade v2 pipeline finished successfully" | Add-Content -LiteralPath $logFile -Encoding utf8
exit 0
