# overnight_backbone2.ps1 - second backbone (BakLLaVA = LLaVA-1.5 architecture on Mistral-7B), unattended.
# Start it from unlearning_eval\ in a NEW PowerShell window, now, and leave:
#     powershell -ExecutionPolicy Bypass -File .\backbone2\overnight_backbone2.ps1
# 1. waits until bug_injection.py / scorer_audit.py have finished and the GPU is free
# 2. patches config so this backbone writes ONLY to checkpoints_bakllava\ and outputs_bakllava\ (LLaVA results untouched)
# 3. downloads llava-hf/bakLlava-v1-hf (~15 GB, once)      4. 3-minute smoke test - stops here if anything is wrong
# 5. trains vanilla + retrained on mmbench1k and scienceqa   6. memorisation gate
# 7. unlearning sweep (6 methods x 3 learning rates, seed 42) + evaluation + reference rows, gate-passing datasets only
# Everything is logged to logs\backbone2_<time>.log . Safe to re-run: finished steps are skipped.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$ts  = Get-Date -Format "yyyyMMdd_HHmmss"
New-Item -ItemType Directory -Path ".\logs" -Force | Out-Null
$log = ".\logs\backbone2_$ts.log"
$t0  = Get-Date
function Say($m, $c = "Green") { $l = "[{0:HH:mm:ss}] {1}" -f (Get-Date), $m; Write-Host $l -ForegroundColor $c; Add-Content $log $l }
function Run-Py([string[]]$a) {
    Say ("py " + ($a -join " ")) "Cyan"
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    & py @a 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append | Out-Host
    $code = $LASTEXITCODE; $ErrorActionPreference = $old; return $code
}

# --- 1. wait for the GPU ---------------------------------------------------------------------------
Say "[1] Waiting for bug_injection.py / scorer_audit.py / main.py to finish..."
$idle = 0
while ($true) {
    $busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='py.exe'" |
              Where-Object { $_.CommandLine -match "bug_injection|scorer_audit|main\.py|quick_check" })
    $auditDone = (Test-Path ".\outputs\scorer_audit.json") -and
                 ((Get-Content ".\outputs\scorer_audit.json" -Raw) -match '"summary"')
    if ($busy.Count -gt 0) { $idle = 0 } else { $idle++ }
    if ($busy.Count -eq 0 -and ($auditDone -or $idle -ge 10)) { break }
    Start-Sleep -Seconds 60
}
Say "[1] GPU is free (scorer audit finished: $auditDone). Starting."

# --- 2. patch (idempotent) + backbone switch ---------------------------------------------------------
if ((Run-Py @("backbone2\patch_backbone.py")) -ne 0) { Say "patch_backbone failed - stopping." "Red"; exit 1 }
$env:UNL_BACKBONE = "bakllava"

# --- 3. download the model once (online only for this step) -----------------------------------------
$dl = "import os; os.environ['HF_HUB_OFFLINE']='0'; os.environ['TRANSFORMERS_OFFLINE']='0'; " +
      "from huggingface_hub import snapshot_download; import config; " +
      "print(snapshot_download('llava-hf/bakLlava-v1-hf', cache_dir=config.DATA_DIR))"
$ok = $false
foreach ($try in 1..3) { if ((Run-Py @("-c", $dl)) -eq 0) { $ok = $true; break }; Say "download attempt $try failed, retrying in 2 min" "Yellow"; Start-Sleep 120 }
if (-not $ok) { Say "Model download failed 3 times - stopping (LLaVA results untouched)." "Red"; exit 1 }

# --- 4. smoke test -----------------------------------------------------------------------------------
if ((Run-Py @("backbone2\smoke_backbone.py")) -ne 0) { Say "SMOKE TEST FAILED - stopping before training. Send this log to Claude." "Red"; exit 1 }

# --- 5-6. train + gate ---------------------------------------------------------------------------------
$datasets = @("mmbench1k", "scienceqa")
if ((Run-Py (@("main.py", "--stage", "train", "--dataset") + $datasets)) -ne 0) { Say "Training failed." "Red"; exit 1 }
$pass = @()
foreach ($d in $datasets) {
    if ((Run-Py @("memorization_gate.py", "--dataset", $d)) -eq 0) { $pass += $d; Say "[6] GATE PASS: $d" } else { Say "[6] GATE FAIL: $d" "Yellow" }
}
if ($pass.Count -eq 0) { Say "No dataset passed the gate on BakLLaVA - that is itself a reportable result. Stopping." "Yellow"; exit 0 }

# --- 7. unlearn + evaluate (seed 42) + reference rows ---------------------------------------------------
if ((Run-Py (@("main.py", "--stage", "unlearn", "--seed", "42", "--dataset") + $pass)) -ne 0) { Say "Unlearning failed." "Red"; exit 1 }
if ((Run-Py (@("main.py", "--stage", "evaluate", "--seed", "42", "--dataset") + $pass)) -ne 0) { Say "Evaluation failed." "Red"; exit 1 }
foreach ($d in $pass) { $null = Run-Py @("eval_reference.py", "--dataset", $d) }

# --- summary -------------------------------------------------------------------------------------------
Say "================ BACKBONE-2 SUMMARY ================"
foreach ($d in $datasets) {
    $g = ".\outputs_bakllava\memorization_gate_${d}_seed42.json"
    if (Test-Path $g) { $j = Get-Content $g -Raw | ConvertFrom-Json
        Say ("{0,-10} gate={1} gap={2:N3} vanilla F={3:N2} retrained F={4:N2}" -f $d, $(if ($j.pass) {"PASS"} else {"FAIL"}), $j.forget_gap, $j.scores.vanilla.forget, $j.scores.retrained.forget) }
}
$n = (Get-ChildItem ".\outputs_bakllava\eval_*.json" -ErrorAction SilentlyContinue).Count
Say "Evaluated models: $n   Total time: $([int]((Get-Date) - $t0).TotalMinutes) min   Log: $log"
