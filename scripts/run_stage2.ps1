param(
    [string]$Dataset = $(if ($env:DATASET) { $env:DATASET } else { "Luxury" }),
    [int]$GpuId = $(if ($env:GPU_ID) { [int]$env:GPU_ID } else { 0 }),
    [string]$PythonBin = $(if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }),
    [string]$Model = $(if ($env:STAGE2_MODEL) { $env:STAGE2_MODEL } else { "meta-llama/Llama-3.2-1B" }),
    [string]$CacheDir = $env:HF_CACHE_DIR,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$datasetDir = Join-Path $projectRoot "data\processed\$Dataset"
$sidPath = Join-Path $projectRoot "data\sids\$Dataset\$Dataset.sid-set-token-weight-stage1.json"

foreach ($split in @("train", "valid", "test")) {
    $splitPath = Join-Path $datasetDir "$Dataset.$split.inter"
    if (-not (Test-Path -LiteralPath $splitPath -PathType Leaf)) {
        throw "Missing Stage 2 input: $splitPath"
    }
}
if (-not (Test-Path -LiteralPath $sidPath -PathType Leaf)) {
    throw "Missing Stage 1 SID artifact: $sidPath"
}

$commandArgs = @(
    "-m", "stage2.run",
    "--dataset_path", $datasetDir,
    "--train_file", "$Dataset.train.inter",
    "--valid_file", "$Dataset.valid.inter",
    "--test_file", "$Dataset.test.inter",
    "--sid_json", $sidPath,
    "--model_name_or_path", $Model,
    "--device", "cuda:$GpuId",
    "--log_path", (Join-Path $projectRoot "logs\LLMSetSID\$Dataset\itemrank.log"),
    "--save_path", (Join-Path $projectRoot "checkpoints\LLMSetSID\$Dataset\itemrank"),
    "--topm_tokens", "0"
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
