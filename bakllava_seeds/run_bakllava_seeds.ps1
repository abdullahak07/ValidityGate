# run_bakllava_seeds.ps1 - BakLLaVA seeds 123 and 5508 (same 3-seed design as LLaVA) + NaN fix.
# From unlearning_eval\ :   powershell -ExecutionPolicy Bypass -File .\bakllava_seeds\run_bakllava_seeds.ps1
# ~5-6 h. Writes only to checkpoints_bakllava\ and outputs_bakllava\. Safe to re-run (finished steps skipped).
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Path ".\logs" -Force | Out-Null
$log = ".\logs\bakllava_seeds_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log"
$t0 = Get-Date
function Say($m, $c = "Green") { $l = "[{0:HH:mm:ss}] {1}" -f (Get-Date), $m; Write-Host $l -ForegroundColor $c; Add-Content $log $l }
function Run-Py([string[]]$a) {
    Say ("py " + ($a -join " ")) "Cyan"
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    & py @a 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append | Out-Host
    $code = $LASTEXITCODE; $ErrorActionPreference = $old; return $code
}
if ((Run-Py @("bakllava_seeds\fix_js_nan.py")) -ne 0) { Say "fix_js_nan failed" "Red"; exit 1 }
$env:UNL_BACKBONE = "bakllava"
$ds = @("mmbench1k", "scienceqa")
if ((Run-Py (@("main.py", "--stage", "unlearn", "--seed", "123", "5508", "--dataset") + $ds)) -ne 0) { Say "Unlearning failed" "Red"; exit 1 }
if ((Run-Py (@("main.py", "--stage", "evaluate", "--seed", "42", "123", "5508", "--dataset") + $ds)) -ne 0) { Say "Evaluation failed" "Red"; exit 1 }
$null = Run-Py @("eval_reference.py", "--dataset", "mmbench1k")
$n = (Get-ChildItem ".\outputs_bakllava\eval_*.json").Count
Say "DONE. BakLLaVA evaluated models: $n (expected 108). Total $([int]((Get-Date) - $t0).TotalMinutes) min. Log: $log"
