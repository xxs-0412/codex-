param(
    [string]$Remote = "origin",
    [string]$Prefix = "backup"
)

$ErrorActionPreference = "Stop"

function Run-Git {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Args
    )

    & git @Args
    if ($LASTEXITCODE -ne 0) {
        throw "git command failed: git $($Args -join ' ')"
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repoRoot

$status = (& git status --porcelain)
if ($LASTEXITCODE -ne 0) {
    throw "Unable to read git status."
}
if ($status) {
    throw "There are uncommitted changes. Commit them first, then run the backup upload script."
}

$head = (& git rev-parse --verify HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or -not $head) {
    throw "Unable to resolve HEAD."
}

$shortHead = (& git rev-parse --short HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or -not $shortHead) {
    throw "Unable to resolve short HEAD."
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$branchName = "{0}/{1}_{2}" -f $Prefix, $timestamp, $shortHead

$existingLocal = (& git show-ref --verify --quiet ("refs/heads/" + $branchName)); $localExit = $LASTEXITCODE
if ($localExit -eq 0) {
    throw "Local branch already exists: $branchName"
}

$existingRemote = (& git ls-remote --exit-code --heads $Remote $branchName *> $null); $remoteExit = $LASTEXITCODE
if ($remoteExit -eq 0) {
    throw "Remote branch already exists: $branchName"
}

Run-Git branch $branchName $head
Run-Git push -u $Remote $branchName

Write-Host ""
Write-Host "Backup uploaded successfully."
Write-Host "Branch: $branchName"
Write-Host "Commit: $shortHead"
Write-Host "Remote: $Remote"
