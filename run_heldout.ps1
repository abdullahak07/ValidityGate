# run_heldout.ps1 - held-out evaluation for LLaVA and BakLLaVA (seed 42). Evaluation only, ~2-2.5 h.
# From unlearning_eval\ :   powershell -ExecutionPolicy Bypass -File .\run_heldout.ps1
# Safe to re-run (finished models are skipped). Add more seeds later with:  py heldout_eval.py --seeds 123 5508
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Path ".\logs" -Force | Out-Null
$log = ".\logs\heldout_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log"
$t0 = Get-Date
Remove-Item Env:UNL_BACKBONE -ErrorAction SilentlyContinue
py heldout_eval.py 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
$env:UNL_BACKBONE = "bakllava"
py heldout_eval.py 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
Remove-Item Env:UNL_BACKBONE
Write-Host ("DONE in {0:N0} min. Send outputs\heldout\heldout_summary.json and outputs_bakllava\heldout\heldout_summary.json. Log: {1}" -f ((Get-Date) - $t0).TotalMinutes, $log) -ForegroundColor Green
