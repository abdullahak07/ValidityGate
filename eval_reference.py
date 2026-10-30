r"""
eval_reference.py - reference rows for the results table.   Run from unlearning_eval\:
    py eval_reference.py --dataset mmubench

The per-method numbers (AD, JS, weight distance, MIA) cannot be interpreted without knowing what they are
for the two anchors:
  * vanilla   vs retrained  -> the value for "no unlearning at all" (the starting point)
  * retrained vs retrained  -> the value for "perfect unlearning"   (AD/JS/dist = 0; MIA = retrained's own)
Writes outputs\reference_<dataset>.json. ~10-15 min, one model on the GPU.
"""
import os
import sys
import json
import argparse

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mmubench")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    import config
    config.ACTIVE_MODEL = "llava"
    from data.loader import load_dataset_by_name
    from models.llava_model import load_llava, apply_lora
    from models.lora_swap import LoRAView
    from utils import load_checkpoint
    from evaluation.metrics import (compute_fa, compute_ra, compute_mia, compute_ad, compute_js,
                                    compute_retrained_distance)

    data = load_dataset_by_name(args.dataset)
    forget, retain = data["forget"], data["retain"]
    ck = lambda tag: os.path.join(config.CHECKPOINT_DIR, f"{tag}_{args.dataset}_seed{args.seed}.pt")

    model, proc = load_llava()
    model = apply_lora(model)
    load_checkpoint(model, ck("retrained"))
    model.to(config.DEVICE)
    retrained = LoRAView(model, name="retrained")
    load_checkpoint(model, ck("vanilla"))
    model.to(config.DEVICE)
    vanilla = LoRAView(model, name="vanilla")

    out = {"dataset": args.dataset, "seed": args.seed}
    for tag, m in (("vanilla", vanilla), ("retrained", retrained)):
        r = {"FA": compute_fa(m, proc, forget), "RA": compute_ra(m, proc, retain),
             "MIA": compute_mia(m, proc, forget, retain)}
        if tag == "vanilla":   # distance of the untouched starting point from the retrained target
            r["AD"] = compute_ad(m, retrained, proc, forget)
            r["JS"] = compute_js(m, retrained, proc, forget)
            r["retrained_distance"] = compute_retrained_distance(m, retrained)
        else:
            r.update({"AD": 0.0, "JS": 0.0, "retrained_distance": 0.0})
        out[tag] = r
        print(tag, {k: round(v, 4) for k, v in r.items()}, flush=True)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    p = os.path.join(config.RESULTS_DIR, f"reference_{args.dataset}.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)
    print("Saved", p)


if __name__ == "__main__":
    main()
