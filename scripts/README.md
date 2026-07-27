# VaLiDRec Run Scripts

Run commands from any directory; each script resolves the repository root
automatically.

## Stage 1

```bash
bash scripts/run_stage1.sh
```

This reads `data/processed/Luxury/` and writes:

```text
data/sids/Luxury/Luxury.sid-set-token-weight-stage1.json
```

## Stage 2

```bash
bash scripts/run_stage2.sh
```

Top-M approximation is disabled by default (`--topm_tokens 0`).

## Full Pipeline

```bash
bash scripts/run_pipeline.sh
```

## Environment Overrides

```bash
DATASET=Luxury \
GPU_ID=0 \
STAGE1_MODEL=meta-llama/Meta-Llama-3.1-8B \
STAGE2_MODEL=meta-llama/Llama-3.2-1B \
HF_CACHE_DIR=/path/to/huggingface/cache \
bash scripts/run_pipeline.sh
```

Additional command-line arguments are forwarded by the individual stage
scripts. For example:

```bash
bash scripts/run_stage2.sh --batch_size 8 --num_epochs 100
```

## Windows PowerShell

Equivalent PowerShell entry points are available:

```powershell
.\scripts\run_stage1.ps1
.\scripts\run_stage2.ps1
.\scripts\run_pipeline.ps1
```

Extra Stage 2 arguments can be appended directly:

```powershell
.\scripts\run_stage2.ps1 --batch_size 8 --num_epochs 100
```
