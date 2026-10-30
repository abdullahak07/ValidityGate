# run_steps234.ps1 - unattended run of steps 3, 2 and 4 (in that order, safest first). ~20-26 h in total.
# From unlearning_eval\ , in a NEW PowerShell window:
#     powershell -ExecutionPolicy Bypass -File .\run_steps234.ps1
#   STEP 3  BakLLaVA seeds 123 + 5508 (unlearn + evaluate + held-out)            -> checkpoints_bakllava\, outputs_bakllava\
#   STEP 2  LLaVA, option-shuffled training, seed 42 (train, gate, unlearn,
#           evaluate, reference, held-out)                                       -> checkpoints_shuffle\, outputs_shuffle\
#   STEP 4  third backbone llava-interleave-qwen-7b, seed 42 (same stages)      -> checkpoints_qwen\, outputs_qwen\
# Each step is independent: if one fails, the script logs it and moves on to the next.
# Safe to re-run with the same command: finished models / files are skipped.
# Log: logs\steps234_<time>.log   Status of every stage: logs\steps234_status.txt
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Path ".\logs" -Force | Out-Null
$log = ".\logs\steps234_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log"
$status = ".\logs\steps234_status.txt"
$t0 = Get-Date
$datasets = @("mmbench1k", "scienceqa")

function Say($m, $c = "Green") {
    $l = "[{0:HH:mm:ss} +{1,5:N0} min] {2}" -f (Get-Date), ((Get-Date) - $t0).TotalMinutes, $m
    Write-Host $l -ForegroundColor $c; Add-Content $log $l
}
function Run-Py([string[]]$a) {
    Say ("py " + ($a -join " ")) "Cyan"
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    & py @a 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append | Out-Host
    $code = $LASTEXITCODE; $ErrorActionPreference = $old; return $code
}
function Mark($m) { Add-Content $status ("[{0:yyyy-MM-dd HH:mm}] {1}" -f (Get-Date), $m); Say $m "Yellow" }
function Clear-Modes { Remove-Item Env:UNL_BACKBONE -ErrorAction SilentlyContinue; Remove-Item Env:UNL_VARIANT -ErrorAction SilentlyContinue }
$env:TRANSFORMERS_OFFLINE = "1"; $env:HF_DATASETS_OFFLINE = "1"

# full pipeline for one mode (train -> gate -> unlearn -> evaluate -> reference -> held-out), seed 42
function Run-Full($name, $outdir) {
    if ((Run-Py (@("main.py", "--stage", "train", "--dataset") + $datasets)) -ne 0) { Mark "$name : TRAIN FAILED"; return }
    Mark "$name : training done"
    $pass = @()
    foreach ($d in $datasets) {
        if ((Run-Py @("memorization_gate.py", "--dataset", $d)) -eq 0) { $pass += $d; Mark "$name : GATE PASS $d" }
        else { Mark "$name : GATE FAIL $d (no unlearning for it; held-out eval still runs)" }
    }
    if ($pass.Count -gt 0) {
        if ((Run-Py (@("main.py", "--stage", "unlearn", "--seed", "42", "--dataset") + $pass)) -ne 0) { Mark "$name : UNLEARN FAILED" }
        else {
            Mark "$name : unlearning done"
            if ((Run-Py (@("main.py", "--stage", "evaluate", "--seed", "42", "--dataset") + $pass)) -ne 0) { Mark "$name : EVALUATE FAILED" }
            else { Mark "$name : evaluation done" }
        }
    }
    foreach ($d in $datasets) {
        if ((Run-Py @("eval_reference.py", "--dataset", $d)) -ne 0) { Mark "$name : reference failed for $d" }
    }
    if ((Run-Py (@("heldout_eval.py", "--datasets") + $datasets)) -ne 0) { Mark "$name : HELDOUT FAILED" }
    else { Mark "$name : held-out done -> $outdir\heldout\heldout_summary.json" }
}

Mark "=== START steps 3, 2, 4 ==="
if ((Run-Py @("patch_step2_step4.py")) -ne 0) { Mark "patch_step2_step4 failed - stopping"; exit 1 }

# ---------------- STEP 3: BakLLaVA seeds 123 + 5508 ----------------
Clear-Modes; $env:UNL_BACKBONE = "bakllava"
if ((Run-Py (@("main.py", "--stage", "unlearn", "--seed", "123", "5508", "--dataset") + $datasets)) -ne 0) { Mark "STEP 3 : UNLEARN FAILED" }
else {
    Mark "STEP 3 : unlearning done"
    if ((Run-Py (@("main.py", "--stage", "evaluate", "--seed", "42", "123", "5508", "--dataset") + $datasets)) -ne 0) { Mark "STEP 3 : EVALUATE FAILED" }
    else { Mark ("STEP 3 : evaluation done ({0} eval files)" -f (Get-ChildItem ".\outputs_bakllava\eval_*.json").Count) }
    if ((Run-Py (@("heldout_eval.py", "--seeds", "123", "5508", "--datasets") + $datasets)) -ne 0) { Mark "STEP 3 : HELDOUT FAILED" }
    else { Mark "STEP 3 : held-out done" }
}

# ---------------- STEP 2: option-shuffled training (LLaVA) ----------------
Clear-Modes; $env:UNL_VARIANT = "shuffle"
Run-Full "STEP 2 (shuffle)" "outputs_shuffle"

# ---------------- STEP 4: third backbone ----------------
Clear-Modes; $env:UNL_BACKBONE = "qwen"
Run-Full "STEP 4 (qwen)" "outputs_qwen"

Clear-Modes
Mark ("=== ALL DONE in {0:N1} h ===" -f ((Get-Date) - $t0).TotalHours)
Say "Send Claude: logs\steps234_status.txt, the log $log, and these folders zipped:"
Say "  outputs_bakllava\  outputs_shuffle\  outputs_qwen\   (json files only are enough)"
