"""
eval_vanilla.py
===============
Evaluates vanilla (pre-unlearning) model FA and RA on each benchmark.
Uses existing vanilla checkpoints (seed 42 only â€” one per dataset).

Run from unlearning_eval: py eval_vanilla.py

Output: outputs/vanilla_results.json  +  printed table for paper.
"""

import os, sys, json
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "0"   # allow dataset download if needed

import config
from models.llava_model import load_llava, apply_lora
from utils import load_checkpoint, get_logger, save_json

logger = get_logger(__name__)

CKPT_DIR = config.CHECKPOINT_DIR
OUT_PATH = os.path.join(config.RESULTS_DIR, "vanilla_results.json")

# Vanilla checkpoint for each dataset (seed 42 only)
VANILLA_CKPTS = {
    "mllmu_bench": os.path.join(CKPT_DIR, "vanilla_mllmu_bench_seed42.pt"),
    "unlok_vqa":   os.path.join(CKPT_DIR, "vanilla_unlok_vqa_seed42.pt"),
    "mmubench":    os.path.join(CKPT_DIR, "vanilla_mmubench_seed42.pt"),
}

DATASET_LABELS = {
    "mllmu_bench": "MLLMU-Bench",
    "unlok_vqa":   "UnLOK-VQA",
    "mmubench":    "MMUBench",
}

def run():
    print("=" * 60)
    print("Vanilla (pre-unlearning) baseline evaluation")
    print("Computes FA and RA on each benchmark before any unlearning")
    print("=" * 60)

    from data.loader import load_dataset_by_name
    from evaluation.metrics import compute_fa, compute_ra

    # Load processor once
    print("\nLoading LLaVA processor...")
    _, processor = load_llava()
    print("Processor loaded âœ“")

    results = []

    for dataset, ckpt_path in VANILLA_CKPTS.items():
        print(f"\n{'â”€'*55}")
        print(f"Dataset: {DATASET_LABELS[dataset]}")
        print(f"{'â”€'*55}")

        if not os.path.exists(ckpt_path):
            print(f"  SKIP â€” checkpoint not found: {ckpt_path}")
            continue

        # Load dataset
        print(f"  Loading {dataset}...")
        os.environ["HF_DATASETS_OFFLINE"] = "0"
        data = load_dataset_by_name(dataset)
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        print(f"  forget={len(data['forget'])}  retain={len(data['retain'])}")

        # Load vanilla model
        print(f"  Loading vanilla checkpoint...")
        base, _ = load_llava()
        base     = apply_lora(base)
        model, _ = load_checkpoint(base, ckpt_path)
        model.to(config.DEVICE)
        print(f"  Loaded âœ“")

        # Evaluate FA on forget set
        print(f"  Computing FA (forget accuracy on vanilla model)...")
        fa = compute_fa(model, processor, data["forget"])
        print(f"  FA = {fa:.4f}  (high = model knows the forget content = good for unlearning to work)")

        # Evaluate RA on retain set
        print(f"  Computing RA (retain accuracy on vanilla model)...")
        ra = compute_ra(model, processor, data["retain"])
        print(f"  RA = {ra:.4f}  (high = model performs well on retain set = correct baseline)")

        row = {
            "dataset": dataset,
            "model":   "vanilla",
            "seed":    42,
            "FA":      fa,
            "RA":      ra,
        }
        results.append(row)

        # Interpretation
        if fa < 0.1:
            print(f"\n  âš   WARNING: Vanilla FA={fa:.3f} is near zero.")
            print(f"     The model did NOT learn the forget-set content before unlearning.")
            print(f"     This means {dataset} is a degenerate benchmark â€” unlearning")
            print(f"     trivially achieves FAâ‰ˆ0 because there was nothing to forget.")
        elif fa > 0.9:
            print(f"\n  âœ“  Vanilla FA={fa:.3f}: model clearly knows the forget content.")
            print(f"     Unlearning is working on a non-trivial task.")
        else:
            print(f"\n  âœ“  Vanilla FA={fa:.3f}: partial memorization.")

        import torch; torch.cuda.empty_cache()
        del model, base

    # Save results
    save_json(results, OUT_PATH)

    # Print table
    print("\n\n" + "=" * 60)
    print("PRE-UNLEARNING BASELINE TABLE (vanilla model, seed 42)")
    print("=" * 60)
    print(f"{'Dataset':20s}  {'Vanilla FAâ†‘':12s}  {'Vanilla RAâ†‘':12s}  {'Verdict'}")
    print("-" * 70)
    for r in results:
        fa, ra = r["FA"], r["RA"]
        verdict = ("âš  degenerate â€” model did not learn forget content"
                   if fa < 0.1 else
                   "âœ“ model clearly knows forget content"
                   if fa > 0.8 else
                   "partial memorization")
        print(f"  {DATASET_LABELS[r['dataset']]:18s}  {fa:.4f}        {ra:.4f}        {verdict}")

    print(f"\nNote: Vanilla FA should be HIGH for unlearning to be meaningful.")
    print(f"If Vanilla FA â‰ˆ 0, the model never learned the forget content")
    print(f"and 'unlearning' is trivially successful â€” not a meaningful benchmark.")

    print(f"\nSaved to {OUT_PATH}")
    print("\nLaTeX rows for paper table:")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"\textbf{Dataset} & \textbf{Vanilla FA}$\uparrow$ & \textbf{Vanilla RA}$\uparrow$ & \textbf{Interpretation} \\")
    print(r"\midrule")
    for r in results:
        fa, ra = r["FA"], r["RA"]
        interp = ("Model did not learn forget content" if fa < 0.1 else
                  "Model clearly knows forget content" if fa > 0.8 else
                  "Partial memorization")
        print(f"  {DATASET_LABELS[r['dataset']]} & {fa:.3f} & {ra:.3f} & {interp} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

if __name__ == "__main__":
    run()

