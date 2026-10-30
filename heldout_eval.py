r"""
heldout_eval.py - held-out evaluation for the TNNLS revision (reviewer point: GEMMU's "no held-out evaluation").
EVALUATION ONLY - no training, nothing is overwritten. Run from unlearning_eval\ :
    py heldout_eval.py                       (LLaVA, seed 42, both datasets)
    $env:UNL_BACKBONE="bakllava"; py heldout_eval.py; Remove-Item Env:UNL_BACKBONE
or let run_heldout.ps1 do both.

For every model (retrained, vanilla, the 18 unlearned models of each seed) it records, per question:
  forget_orig   : predicted letter on the original forget question (as used in unlearning)
  forget_shift1 : the same question with the options rotated by 1 (new letters, same contents)  -> held-out variant
  forget_shift2 : rotated by 2 (only when the question has >= 3 options)                       -> held-out variant
  retain_shift1 : first --retain_n retain questions, rotated by 1                              -> held-out retain
  forget_rev / retain_rev : options in REVERSED order (>= 3 options). In shuffle mode (UNL_VARIANT=shuffle) the
                  models were trained on rotations, so only the reversed order is truly held out.
  test          : predicted letter on the held-out test questions                              -> agreement with retrained
Gold letters are recomputed for every variant. Output: <RESULTS_DIR>/heldout/heldout_<dataset>_<model>.json
Safe to re-run: finished models are skipped. At the end it prints a summary table and writes heldout_summary.json.
"""
import os
import sys
import json
import time
import argparse

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import torch
import config
from data.loader import load_dataset_by_name
from models.llava_model import load_llava, apply_lora
from utils import load_checkpoint
from evaluation.metrics import _letter_ids, _LETTERS
from variants import reverse as _reverse

INSTR = "Answer with the option's letter from the given choices directly."


def rotate(s, k):
    """Return a copy of sample s with its options rotated by k positions, or None if the question text
    does not contain the option lines in the expected 'L. text' form."""
    opts = s.get("options") or {}
    letters = [L for L in _LETTERS if L in opts]
    n = len(letters)
    if n < 2 or k % n == 0:
        return None
    lines = str(s["question"]).split("\n")
    opt_lines = [f"{L}. {opts[L]}" for L in letters]
    try:
        pos = [lines.index(o) for o in opt_lines]
    except ValueError:
        return None
    if pos != list(range(pos[0], pos[0] + n)):
        return None
    texts = [opts[L] for L in letters]
    new_texts = texts[k % n:] + texts[:k % n]
    gold_text = opts[str(s["answer"]).strip().upper()]
    new_opts = dict(zip(letters, new_texts))
    new_answer = [L for L in letters if new_opts[L] == gold_text]
    if len(new_answer) != 1:          # duplicate option texts - ambiguous, skip
        return None
    new_lines = lines[:pos[0]] + [f"{L}. {new_opts[L]}" for L in letters] + lines[pos[0] + n:]
    out = dict(s)
    out.update(question="\n".join(new_lines), options=new_opts, answer=new_answer[0])
    return out


@torch.inference_mode()
def predict(model, processor, samples, ids):
    preds = []
    for s in samples:
        prompt = f"USER: <image>\n{s['question']}\nASSISTANT:"
        inputs = processor(text=prompt, images=s["image"], return_tensors="pt").to(config.DEVICE)
        logits = model(**inputs).logits[0, -1].float()
        avail = [L for L in _LETTERS if L in s["options"]]
        preds.append(max(avail, key=lambda L: max(float(logits[i]) for i in ids[L])))
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["mmbench1k", "scienceqa"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--retain_n", type=int, default=100)
    a = ap.parse_args()

    out_dir = os.path.join(config.RESULTS_DIR, "heldout")
    os.makedirs(out_dir, exist_ok=True)
    model, processor = load_llava()
    model = apply_lora(model)
    model.to(config.DEVICE)
    model.eval()
    ids = _letter_ids(processor.tokenizer)
    t0 = time.time()

    for ds in a.datasets:
        config.apply_dataset(ds)
        data = load_dataset_by_name(ds)
        forget, retain, test = data["forget"], data["retain"][:a.retain_n], data["test"]
        sets = {"forget_orig": forget,
                "forget_shift1": [rotate(s, 1) for s in forget],
                "forget_shift2": [rotate(s, 2) if len(s.get("options") or {}) >= 3 else None for s in forget],
                "retain_shift1": [rotate(s, 1) for s in retain],
                "forget_rev": [_reverse(s) for s in forget],
                "retain_rev": [_reverse(s) for s in retain],
                "test": test}
        gold = {k: [None if s is None else str(s["answer"]).strip().upper() for s in v] for k, v in sets.items()}
        skipped = {k: sum(s is None for s in v) for k, v in sets.items()}
        print(f"\n=== {ds}: forget {len(forget)}, retain_n {len(retain)}, test {len(test)}; "
              f"variants skipped (no clean rotation): {skipped}", flush=True)

        jobs = [("retrained", f"retrained_{ds}_seed42.pt"), ("vanilla", f"vanilla_{ds}_seed42.pt")]
        for seed in a.seeds:
            jobs += [(f"{m}_seed{seed}", f"unlearned_{ds}_{m}_seed{seed}.pt") for m in config.METHODS]
        for name, ck in jobs:
            fp = os.path.join(out_dir, f"heldout_{ds}_{name}.json")
            if os.path.exists(fp):
                continue
            path = os.path.join(config.CHECKPOINT_DIR, ck)
            if not os.path.exists(path):
                print(f"  [missing] {ck}", flush=True)
                continue
            load_checkpoint(model, path)
            model.eval()
            rec = {"dataset": ds, "model": name, "checkpoint": ck, "gold": gold, "pred": {}}
            for k, v in sets.items():
                keep = [s for s in v if s is not None]
                p = iter(predict(model, processor, keep, ids))
                rec["pred"][k] = [None if s is None else next(p) for s in v]
            with open(fp + ".tmp", "w") as f:
                json.dump(rec, f)
            os.replace(fp + ".tmp", fp)
            acc = lambda k: _acc(rec["pred"][k], gold[k])
            print(f"  {name:32s} FA {acc('forget_orig'):.2f}  FA_heldout {acc('forget_shift1'):.2f}  "
                  f"RA_heldout {acc('retain_shift1'):.2f}  test {acc('test'):.2f}   [{(time.time() - t0) / 60:.0f} min]",
                  flush=True)
    summarize(out_dir)


def _acc(p, g):
    pairs = [(x, y) for x, y in zip(p, g) if x is not None and y is not None]
    return sum(x == y for x, y in pairs) / len(pairs) if pairs else float("nan")


def summarize(out_dir):
    files = sorted(f for f in os.listdir(out_dir) if f.startswith("heldout_") and f.endswith(".json")
                   and f != "heldout_summary.json")
    recs = [json.load(open(os.path.join(out_dir, f))) for f in files]
    ref = {r["dataset"]: r for r in recs if r["model"] == "retrained"}
    rows = []
    for r in recs:
        g, p = r["gold"], r["pred"]
        R = ref.get(r["dataset"])
        row = {"dataset": r["dataset"], "model": r["model"]}
        for k in p:
            row[f"acc_{k}"] = _acc(p[k], g[k])
        variants = [k for k in ("forget_shift1", "forget_shift2", "forget_rev") if k in p]
        if R is not None:   # per-question agreement with the retrained model
            for k in ("test", "forget_orig", "forget_shift1", "forget_rev"):
                if k not in p or k not in R["pred"]:
                    continue
                pairs = [(x, y) for x, y in zip(p[k], R["pred"][k]) if x is not None and y is not None]
                row[f"agree_{k}"] = sum(x == y for x, y in pairs) / len(pairs) if pairs else float("nan")
        # recoverability: forget questions answered WRONG in the original order but RIGHT in every rotation
        wrong = [i for i, (x, y) in enumerate(zip(p["forget_orig"], g["forget_orig"])) if x != y]
        rec_ok = 0
        for i in wrong:
            vs = [(p[k][i], g[k][i]) for k in variants if p[k][i] is not None]
            rec_ok += int(bool(vs) and all(x == y for x, y in vs))
        row["n_forget_wrong"] = len(wrong)
        row["recovered_frac"] = rec_ok / len(wrong) if wrong else float("nan")
        rows.append(row)
    with open(os.path.join(out_dir, "heldout_summary.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("\nSUMMARY (acc = accuracy; agree = same letter as the retrained model; recovered = wrong originally, "
          "right on every rotation)")
    for r in rows:
        print(f"{r['dataset']:10s} {r['model']:32s} FA {r.get('acc_forget_orig', float('nan')):.2f}  "
              f"FA_ho {r.get('acc_forget_shift1', float('nan')):.2f}  FA_rev {r.get('acc_forget_rev', float('nan')):.2f}  "
              f"RA_ho {r.get('acc_retain_shift1', float('nan')):.2f}  RA_rev {r.get('acc_retain_rev', float('nan')):.2f}  "
              f"agree_test {r.get('agree_test', float('nan')):.2f}  recovered {r['recovered_frac']:.2f} "
              f"(of {r['n_forget_wrong']})")
    print(f"\nWrote {os.path.join(out_dir, 'heldout_summary.json')} - send this file (and the heldout folder zipped) to Claude.")


if __name__ == "__main__":
    main()
