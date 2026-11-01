# Validity-Gated Evaluation of Multimodal Machine Unlearning (VGUA)

Code, per-model results, and analysis scripts for "Validity-Gated Evaluation of Multimodal Machine Unlearning" (A. A. Khan, H. Laga, F. Sohel).

## Reproduce the paper results (one command, CPU only)

```
pip install numpy scipy matplotlib
python reproduce_paper.py
```

The script recomputes every statistic and figure in the paper from the released result files, checks them against the raw per-model files, and prints each headline number next to the value reported in the paper. It takes about 5-15 minutes and writes everything to `reproduced/`:

| Output | Paper item |
|---|---|
| `stats.json`, `figures/fig_forest.pdf`, `figures/fig_dose_response.pdf` | Criterion-target correlations, pooled estimates, dose response (Tables IV-V, Fig. 2, Fig. S1) |
| `stats_noncollapse.json` | Collapse-stratified analysis (Table S10) |
| `stats_original.json`, `figures/fig_vs_original.pdf` | Do-nothing baseline (Fig. 3, Table S11) |
| `figures/figure_1_main.pdf` | Framework overview (Fig. 1) |
| `REPORT.txt` | Headline numbers vs. the paper, recovery rates, raw-file consistency check |

## Repository layout

| Paper item | File(s) |
|---|---|
| Unlearning pipeline (6 methods x 3 learning rates, LoRA) | `main.py`, `unlearning/methods.py`, `models/`, `data/`, `config.py` |
| C1 memorization gate and de-duplication | `memorization_gate.py`, `data/loader.py` |
| C2 scorer audit | `scorer_audit.py`, `scorer_fixes/scorer_fixes.py` |
| C3 bug injection | `bug_injection.py` |
| Criteria (FA, RA, MIA advantage, AD, JS, dW) | `evaluation/metrics.py` |
| Held-out and option-shuffled evaluation | `heldout_eval.py`, `variants.py` |
| Per-model results | `outputs/` (LLaVA-1.5), `outputs_bakllava/`, `outputs_qwen/`, `outputs_shuffle/` |
| Aggregated results and analysis scripts | `paper_analysis/` |

## Re-running the GPU experiments

Base models and datasets are downloaded from Hugging Face on first use (`download_models.py`). Exact package versions are in `requirements_tnnls_freeze.txt`. All experiments ran on one NVIDIA RTX 4090.

```
pip install -r requirements.txt
.\run_fixed_pipeline.ps1                     # LLaVA-1.5: training, memorization gate, unlearning sweep, evaluation
python scorer_audit.py                       # C2 scorer audit
python bug_injection.py                      # C3 bug injection
.\backbone2\overnight_backbone2.ps1          # BakLLaVA, seed 42
.\bakllava_seeds\run_bakllava_seeds.ps1      # BakLLaVA, seeds 123 and 5508
.\run_steps234.ps1                           # option-shuffled LLaVA, Qwen-based LLaVA, held-out evaluation
```

Each script describes its stages in its header comment and skips models that already exist.

## License
See `LICENSE`.