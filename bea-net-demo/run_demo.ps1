param(
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [string]$Repo = "$PSScriptRoot\vendor\BEA-Net",
    [string]$Python = "$PSScriptRoot\.venv\Scripts\python.exe",
    [ValidateSet('cpu', 'cuda')][string]$Device = 'cpu',
    [int]$Limit = 4
)
$ErrorActionPreference = 'Stop'
& $Python "$PSScriptRoot\run_demo.py" --repo $Repo --checkpoint $Checkpoint --device $Device --limit $Limit
if ($LASTEXITCODE -ne 0) { throw "BEA-Net demo failed (exit $LASTEXITCODE)." }
