# run_fixed_pipeline.ps1  -  run from inside unlearning_eval\
#   powershell -ExecutionPolicy Bypass -File .\run_fixed_pipeline.ps1            # fresh run
#   powershell -ExecutionPolicy Bypass -File .\run_fixed_pipeline.ps1 -Resume    # continue after a crash
#
# 0) pre-flight: refuses to start unless every fix is installed
# 1) backs up old checkpoints + eval caches (skipped with -Resume, so finished checkpoints are kept)
# 2) config lines   3) train vanilla + retrained   4) memorisation gate
# 5) unlearn + evaluate (gate-passing datasets only)   6) independent check (non-fatal)
# Everything printed is also written to logs\pipeline_<timestamp>.log
param([switch]$Resume)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$datasets = @("mmubench", "mllmu_bench")
$ts  = Get-Date -Format "yyyyMMdd_HHmmss"
New-Item -ItemType Directory -Path ".\logs" -Force | Out-Null
$log = ".\logs\pipeline_$ts.log"
$t0  = Get-Date

function Say($msg, $color = "Green") {
    $line = "[{0:HH:mm:ss}] {1}" -f (Get-Date), $msg
    Write-Host $line -ForegroundColor $color
    Add-Content $log $line
}

# Runs python, shows + logs its output, returns the exit code. Python logging writes to stderr;
# under "Stop" Windows PowerShell 5.1 would turn that into a terminating error, so relax it here.
function Run-Py([string[]]$pyArgs) {
    Say ("py " + ($pyArgs -join " ")) "Cyan"
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & py @pyArgs 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append | Out-Host
    $code = $LASTEXITCODE
    $ErrorActionPreference = $old
    return $code
}

# --- 0. pre-flight: every fix must be in place ------------------------------------------------
$checks = @(
    @{ f = "data\loader.py";          p = "max_bad_frac";                 what = "loader: tolerant empty-row check" },
    @{ f = "data\loader.py";          p = "_fail_or_synthetic";           what = "loader: no silent synthetic data" },
    @{ f = "data\dataset.py";         p = "-100";                         what = "dataset: prompt tokens masked" },
    @{ f = "models\llava_model.py";   p = "grad_check=\(epoch == 1\)";    what = "llava_model: grad check epoch 1 only" },
    @{ f = "models\llava_model.py";   p = "enable_input_require_grads";   what = "llava_model: gradients through checkpointing" },
    @{ f = "evaluation\metrics.py";   p = "_name_match";                  what = "metrics: MLLMU name scoring" },
    @{ f = "unlearning\methods.py";   p = "T_max=10\*\*9";                what = "methods: unlearning LR schedule fixed" },
    @{ f = "config.py";               p = "def apply_dataset";            what = "config: per-dataset settings" },
    @{ f = "main.py";                 p = "config\.apply_dataset";        what = "main: per-dataset settings applied" },
    @{ f = "main.py";                 p = "lr=config\.FINETUNE_LR";       what = "main: lr passed to full_train" },
    @{ f = "memorization_gate.py";    p = "forget_gap";                   what = "memorization_gate.py present" },
    @{ f = "main.py";                 p = "lora_snapshot";                what = "main: one model on GPU (patch_memory.py)" },
    @{ f = "models\lora_swap.py";     p = "class LoRAView";               what = "models\lora_swap.py installed" }
)
$missing = @()
foreach ($c in $checks) {
    if (-not (Test-Path $c.f) -or -not (Select-String -Path $c.f -Pattern $c.p -Quiet)) { $missing += "$($c.what)  [$($c.f)]" }
}
if ($missing.Count) {
    Say "Pre-flight FAILED - these fixes are not installed:" "Red"
    $missing | ForEach-Object { Say "   - $_" "Red" }
    Say "Install the latest files (unlearning_eval_fixes.zip, metrics.py, loader.py, llava_model.py) and run apply_tuned_settings.ps1 + apply_dataset_settings.ps1 first." "Red"
    exit 1
}
Say "[0] Pre-flight OK: all $($checks.Count) fixes present."
$null = Run-Py @("-c", "import config; [config.apply_dataset(d) for d in ['mmubench','mllmu_bench']]")

# --- 1. back up old checkpoints and caches (main.py skips anything that already exists) -------
if ($Resume) {
    Say "[1] -Resume: keeping existing checkpoints and caches." "Yellow"
} else {
    $bk = ".\backup_before_fix_$ts"
    New-Item -ItemType Directory -Path "$bk\checkpoints", "$bk\outputs" -Force | Out-Null
    if (Test-Path ".\checkpoints") { Get-ChildItem ".\checkpoints\*.pt" -ErrorAction SilentlyContinue | Move-Item -Destination "$bk\checkpoints" }
    if (Test-Path ".\outputs") {
        Get-ChildItem ".\outputs\eval_*.json", ".\outputs\metrics_*.json", ".\outputs\multimodal_results.json", ".\outputs\memorization_gate_*.json" -ErrorAction SilentlyContinue |
            Move-Item -Destination "$bk\outputs"
    }
    Say "[1] Old checkpoints and caches moved to $bk"
}

# --- 2. config lines (COCO_DIR deliberately left out: unlok_vqa is skipped) --------------------
$cfg = Get-Content ".\config.py" -Raw -Encoding UTF8
$add = @()
if ($cfg -notmatch "ALLOW_SYNTHETIC")    { $add += "ALLOW_SYNTHETIC = False" }
if ($cfg -notmatch "MAX_INPUT_LEN_HARD") { $add += "MAX_INPUT_LEN_HARD = 2048" }
if ($cfg -notmatch "FREE_TEXT_METRIC")   { $add += 'FREE_TEXT_METRIC = "f1"' }
if ($add.Count) { Add-Content ".\config.py" ("`n# --- added by run_fixed_pipeline.ps1 ---`n" + ($add -join "`n")) -Encoding UTF8 }
Say "[2] config.py: $(if ($add.Count) { 'added ' + ($add -join ', ') } else { 'nothing to add' })"

# --- 3. train vanilla + retrained ----------------------------------------------------------------
$code = Run-Py (@("main.py", "--stage", "train", "--dataset") + $datasets)
if ($code -ne 0) { Say "Training failed (exit $code). Fix, then rerun with -Resume." "Red"; exit 1 }
Say "[3] Training done ($([int]((Get-Date) - $t0).TotalMinutes) min elapsed)."

# --- 4. memorisation gate -------------------------------------------------------------------------
$pass = @()
foreach ($d in $datasets) {
    $code = Run-Py @("memorization_gate.py", "--dataset", $d)
    if ($code -eq 0) { $pass += $d; Say "[4] GATE PASS: $d" }
    else { Say "[4] GATE FAIL: $d  -> raise its FINETUNE_EPOCHS / FINETUNE_LR in DATASET_SETTINGS (config.py), delete checkpoints\*_${d}_*.pt, rerun with -Resume." "Red" }
}
if ($pass.Count -eq 0) { Say "No dataset passed the gate. Stopping before unlearning." "Red"; exit 1 }

# --- 5. unlearn + evaluate (passing datasets only) ----------------------------------------------
$code = Run-Py (@("main.py", "--stage", "unlearn", "--dataset") + $pass)
if ($code -ne 0) { Say "Unlearning failed (exit $code). Fix, then rerun with -Resume." "Red"; exit 1 }
Say "[5] Unlearning done ($([int]((Get-Date) - $t0).TotalMinutes) min elapsed)."
$code = Run-Py (@("main.py", "--stage", "evaluate", "--dataset") + $pass)
if ($code -ne 0) { Say "Evaluation failed (exit $code). Fix, then rerun with -Resume." "Red"; exit 1 }
Say "[5] Evaluation done ($([int]((Get-Date) - $t0).TotalMinutes) min elapsed)."

# --- 6. independent check (non-fatal: uses its own scoring, so a failure here does not block results)
if (Test-Path ".\fixed_eval.py") {
    $code = Run-Py @("fixed_eval.py", "--datasets", ($pass -join ","))
    if ($code -ne 0) { Say "[6] fixed_eval.py failed (exit $code) - main results are still valid." "Yellow" }
}

# --- summary -----------------------------------------------------------------------------------------
Say "================ SUMMARY ================"
foreach ($d in $datasets) {
    $g = Get-ChildItem ".\outputs\memorization_gate_${d}_seed*.json" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($g) {
        $j = Get-Content $g.FullName -Raw | ConvertFrom-Json
        Say ("{0,-12} gate={1}  forget gap={2:N3}  vanilla forget={3:N3}  retrained forget={4:N3}" -f $d,
             $(if ($j.pass) { "PASS" } else { "FAIL" }), $j.forget_gap, $j.scores.vanilla.forget, $j.scores.retrained.forget)
    }
}
$n = (Get-ChildItem ".\outputs\metrics_*.json" -ErrorAction SilentlyContinue).Count
Say "Metrics files written: $n   Total time: $([int]((Get-Date) - $t0).TotalMinutes) min"
Say "Log: $log   Send me this log + outputs\metrics_*.json"
