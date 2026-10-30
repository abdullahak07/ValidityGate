r"""
memorization_gate.py - run AFTER  `py main.py --stage train`  and BEFORE any unlearning.

    py memorization_gate.py --dataset mmubench
    py memorization_gate.py --dataset mllmu_bench --seed 42 --min_gap 0.15

Unlearning is only meaningful if the vanilla model knows the forget set better than a model that never
saw it. This script loads vanilla_<ds>_seed<seed>.pt and retrained_<ds>_seed<seed>.pt, scores both with the
corrected metric (evaluation.metrics._accuracy: MCQ letter-logit scoring or token-F1), and passes only if

    vanilla(forget) - retrained(forget) >= min_gap   and   vanilla(retain) - retrained(retain) is small

Exit code 0 = PASS (run unlearning), 1 = FAIL (fix training first: more epochs / check loss masking).
"""
import os
import sys
import json
import argparse

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_gap", type=float, default=0.15)
    ap.add_argument("--n_retain", type=int, default=100)
    args = ap.parse_args()

    import torch
    import config
    config.ACTIVE_MODEL = "llava"
    from data.loader import load_dataset_by_name
    from models.llava_model import load_llava, apply_lora
    from utils import load_checkpoint
    from evaluation.metrics import _accuracy

    data = load_dataset_by_name(args.dataset)
    splits = {"forget": data["forget"], "retain": data["retain"][: args.n_retain], "test": data["test"][:100]}
    res = {}
    for tag in ("vanilla", "retrained"):
        path = os.path.join(config.CHECKPOINT_DIR, f"{tag}_{args.dataset}_seed{args.seed}.pt")
        if not os.path.exists(path):
            print(f"Missing checkpoint: {path}")
            sys.exit(1)
        model, proc = load_llava()
        model = apply_lora(model)
        model, _ = load_checkpoint(model, path)
        model.to(config.DEVICE)
        res[tag] = {sp: _accuracy(model, proc, s, desc=f"{tag}-{sp}") for sp, s in splits.items()}
        print(tag, {k: round(v, 3) for k, v in res[tag].items()}, flush=True)
        del model
        torch.cuda.empty_cache()

    gap_f = res["vanilla"]["forget"] - res["retrained"]["forget"]
    gap_r = res["vanilla"]["retain"] - res["retrained"]["retain"]
    ok = gap_f >= args.min_gap
    out = {"dataset": args.dataset, "seed": args.seed, "scores": res,
           "forget_gap": gap_f, "retain_gap": gap_r, "min_gap": args.min_gap, "pass": ok}
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, f"memorization_gate_{args.dataset}_seed{args.seed}.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nforget gap (vanilla - retrained) = {gap_f:+.3f}   retain gap = {gap_r:+.3f}")
    print("PASS: the vanilla model has forget-specific knowledge; unlearning can be evaluated."
          if ok else
          "FAIL: no forget-specific knowledge. Do NOT run unlearning; train longer / check label masking.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
