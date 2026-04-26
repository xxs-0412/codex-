$ErrorActionPreference = "Continue"

$ToolDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ToolsRoot = Split-Path -Parent $ToolDir
$Root = Split-Path -Parent $ToolsRoot
$Python = "E:\AI\cuda_env\python.exe"
$Script = Join-Path $ToolDir "run_paper_experiments.py"
$StdoutLog = Join-Path $ToolDir "paper_pipeline_stdout.log"
$StderrLog = Join-Path $ToolDir "paper_pipeline_stderr.log"
$WatchdogLog = Join-Path $ToolDir "paper_pipeline_watchdog.log"
$PidFile = Join-Path $ToolDir "paper_pipeline_pid.txt"
$ResultRoot = Join-Path (Join-Path $Root "结果") "4.25测试集选定_论文结构重跑"
$FinalMarker = Join-Path $ResultRoot "05_最终横向对比\追加seed判断.json"

function Write-WatchdogLog {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $WatchdogLog -Value "[$timestamp] $Message" -Encoding UTF8
}

function Test-PipelineFinished {
    return (Test-Path -LiteralPath $FinalMarker)
}

function Get-PipelinePidFromFile {
    if (-not (Test-Path -LiteralPath $PidFile)) { return $null }
    try {
        return [int](Get-Content -LiteralPath $PidFile -ErrorAction Stop | Select-Object -First 1)
    } catch {
        return $null
    }
}

function Test-PipelineAlive {
    $pidValue = Get-PipelinePidFromFile
    if ($null -eq $pidValue) { return $false }
    $proc = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
    if ($null -eq $proc) { return $false }
    return ($proc.ProcessName -like "python*")
}

function Start-Pipeline {
    if (-not (Test-Path -LiteralPath $Script)) {
        Write-WatchdogLog "script missing: $Script"
        return
    }
    $argLine = '-u "' + $Script + '" --stage all'
    $argLine = $argLine.Replace('\"', '"')
    $proc = Start-Process -FilePath $Python -ArgumentList $argLine -WorkingDirectory $ToolDir -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog -PassThru -WindowStyle Hidden
    Set-Content -LiteralPath $PidFile -Value $proc.Id -Encoding ASCII
    Write-WatchdogLog "started pipeline pid=$($proc.Id)"
}

Write-WatchdogLog "watchdog started"
while ($true) {
    if (Test-PipelineFinished) {
        Write-WatchdogLog "pipeline finished; watchdog exits"
        break
    }
    if (Test-PipelineAlive) {
        $pidValue = Get-PipelinePidFromFile
        Write-WatchdogLog "pipeline alive pid=$pidValue"
    } else {
        Write-WatchdogLog "pipeline not running; restarting"
        Start-Pipeline
    }
    Start-Sleep -Seconds 180
}
