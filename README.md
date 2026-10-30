# Evaluation Reliability in Multimodal Machine Unlearning (VGUA)

Code and results for "Evaluation Reliability in Multimodal Machine Unlearning: Pipeline Validity,
Retraining Alignment, and Operating-Regime Effects" (A. A. Khan, H. Laga, F. Sohel).

## What is where
| Paper item | File(s) |
|---|---|
| Unlearning pipeline (6 methods x 3 LRs, LoRA) | `main.py`, `unlearning/methods.py`, `models/`, `data/`, `config.py` |
| C1 memorization gate and de-duplication | `memorization_gate.py`, `data/loader.py` |
| C2 scorer audit | `scorer_audit.py`, `scorer_fixes/scorer_fixes.py` |
| C3 bug injection | `bug_injection.py` |
| Criteria (FA, RA, MIA advantage, AD, JS, dW) | `evaluation/metrics.py` |
| Held-out and option-shuffled evaluation | `heldout_eval.py`, `variants.py` |
| Run scripts (LLaVA, BakLLaVA, Qwen, shuffle) | `run_fixed_pipeline.ps1`, `run_steps234.ps1`, `run_heldout.ps1`, `backbone2/`, `bakllava_seeds/` |
| Per-model results | `outputs/` (LLaVA), `outputs_bakllava/`, `outputs_qwen/`, `outputs_shuffle/` |
| Correlations, pooling, dose-response, forest plot | `paper_analysis/analysis_figures.py` -> `stats.json` |
| Do-nothing comparison (vs. original model) | `paper_analysis/original_comparison.py` -> `stats_original.json` |
| Collapse-stratified analysis | `paper_analysis/noncollapse.py` -> `stats_noncollapse.json` |
| Figure 1 | `paper_analysis/make_figure1.py` |

## Reproduce the paper's statistics (CPU, minutes)
```
pip install -r requirements.txt
cd paper_analysis
python analysis_figures.py
python original_comparison.py
python noncollapse.py
```

## Re-run the experiments (GPU)
Base models and datasets download from Hugging Face on first run (`download_models.py`).
Exact package versions: `requirements_tnnls_freeze.txt`. Experiments ran on one RTX 4090.

## Note on the earlier version
An earlier version of this work (arXiv) used a pipeline that fails the validity conditions; its code is
available at https://github.com/abdullahak07/UnifiedUnl. All results in this repository come from the rebuilt pipeline.
