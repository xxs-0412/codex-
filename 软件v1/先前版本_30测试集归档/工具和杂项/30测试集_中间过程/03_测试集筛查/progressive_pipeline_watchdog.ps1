$ErrorActionPreference = "Continue"

# Keep this file ASCII-only. Windows PowerShell 5.1 may misread UTF-8 files
# without BOM, so all paths are derived from the script location.
$ScreenDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$MidRoot = Split-Path -Parent $ScreenDir
$ToolsRoot = Split-Path -Parent $MidRoot
$Root = Split-Path -Parent $ToolsRoot

$Python = "E:\AI\cuda_env\python.exe"
$Script = Join-Path $ScreenDir "progressive_dual_module_pipeline.py"
$StdoutLog = Join-Path $ScreenDir "progressive_pipeline_stdout.log"
$StderrLog = Join-Path $ScreenDir "progressive_pipeline_stderr.log"
$WatchdogLog = Join-Path $ScreenDir "progressive_pipeline_watchdog.log"
$PidFile = Join-Path $ScreenDir "progressive_pipeline_pid.txt"

function Write-WatchdogLog {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $WatchdogLog -Value "[$timestamp] $Message" -Encoding UTF8
}

function Get-StateJson {
    try {
        return Get-ChildItem -LiteralPath $Root -Filter "pipeline_state.json" -Recurse -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
    }
    catch {
        Write-WatchdogLog "state search failed: $($_.Exception.Message)"
        return $null
    }
}

function Test-PipelineFinished {
    $stateFile = Get-StateJson
    if ($null -eq $stateFile) {
        return $false
    }
    try {
        $state = Get-Content -LiteralPath $stateFile.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
        return ($state.last_stage -eq "final_doc_done")
    }
    catch {
        Write-WatchdogLog "state read failed: $($_.Exception.Message)"
        return $false
    }
}

function Get-PipelineProcess {
    Get-CimInstance Win32_Process | Where-Object {
        ($_.Name -like "python*") -and ($_.CommandLine -like "*progressive_dual_module_pipeline.py*")
    }
}

function Start-Pipeline {
    if (-not (Test-Path -LiteralPath $Script)) {
        Write-WatchdogLog "script missing: $Script"
        return
    }

    $argLine = '-u "' + $Script + '"'
    $proc = Start-Process -FilePath $Python -ArgumentList $argLine -WorkingDirectory $ScreenDir -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog -PassThru
    Set-Content -LiteralPath $PidFile -Value $proc.Id -Encoding ASCII
    Write-WatchdogLog "started pipeline pid=$($proc.Id)"
}

Write-WatchdogLog "watchdog started"

while ($true) {
    if (Test-PipelineFinished) {
        Write-WatchdogLog "pipeline finished; watchdog exits"
        break
    }

    $pipeline = @(Get-PipelineProcess)
    if ($pipeline.Count -eq 0) {
        Write-WatchdogLog "pipeline not running; restarting"
        Start-Pipeline
    }
    else {
        Set-Content -LiteralPath $PidFile -Value $pipeline[0].ProcessId -Encoding ASCII
        Write-WatchdogLog "pipeline alive pid=$($pipeline[0].ProcessId)"
    }

    Start-Sleep -Seconds 120
}
