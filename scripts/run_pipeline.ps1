param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Stage2Args
)

$ErrorActionPreference = "Stop"

& (Join-Path $PSScriptRoot "run_stage1.ps1")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& (Join-Path $PSScriptRoot "run_stage2.ps1") @Stage2Args
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
