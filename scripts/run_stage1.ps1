param(
    [string]$Dataset = $(if ($env:DATASET) { $env:DATASET } else { "Luxury" }),
    [int]$GpuId = $(if ($env:GPU_ID) { [int]$env:GPU_ID } else { 0 }),
    [string]$PythonBin = $(if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }),
    [string]$Model = $(if ($env:STAGE1_MODEL) { $env:STAGE1_MODEL } else { "meta-llama/Meta-Llama-3.1-8B" }),
    [string]$CacheDir = $env:HF_CACHE_DIR,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$datasetDir = Join-Path $projectRoot "data\processed\$Dataset"

foreach ($requiredFile in @(
    (Join-Path $datasetDir "$Dataset.item.json"),
    (Join-Path $datasetDir "$Dataset.train.inter")
)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Missing Stage 1 input: $requiredFile"
    }
}

$commandArgs = @(
    "-m", "stage1.run",
    "--dataset", $Dataset,
    "--root", (Join-Path $projectRoot "data\processed"),
    "--gpu_id", "$GpuId",
    "--model_name_or_path", $Model
)
if ($CacheDir) {
    $commandArgs += @("--model_cache_dir", $CacheDir)
}
$commandArgs += $ExtraArgs

Push-Location $projectRoot
try {
    & $PythonBin @commandArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
