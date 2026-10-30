r"""
scorer_audit.py - claim C2: the SAME models scored three ways give different unlearning conclusions.
Run from unlearning_eval\ :   py scorer_audit.py            (~80-100 min, one model on the GPU)

For every dataset (mmbench1k, scienceqa) and model (vanilla, retrained, 6 methods x 3 lrs, seed 42):
  letter  : option-letter logit scoring (what the pipeline now uses)
  strict  : generate an answer, accept it only if it starts with the gold letter or equals the gold option text
  lenient : generate an answer, accept it if the gold letter appears ANYWHERE in it (the original pipeline)
Part B (MLLMU, free-text biographies): token-F1 vs subject-identity (does the answer name the right person),
for the vanilla and retrained MLLMU models.
Writes outputs\scorer_audit.json and prints the summary. Safe to re-run: finished models are skipped.
"""
import os
import re
import sys
import json
import time

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import torch
import config
config.ACTIVE_MODEL = "llava"
from data.loader import load_dataset_by_name
from models.llava_model import load_llava, apply_lora, generate_answers
from utils import load_checkpoint
from evaluation.metrics import _mcq_accuracy, _token_f1, _name_match, _bio_name

OUT = os.path.join(BASE, "outputs", "scorer_audit.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
METHODS = ["ga", "graddiff", "random_labels", "finetune_retain", "salun", "npo"]
LRS = {"1e-04": "", "3e-05": "__lr3e-05", "1e-05": "__lr1e-05"}


def save():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT + ".tmp", "w"), indent=1)
    os.replace(OUT + ".tmp", OUT)


def strict_ok(pred, s):
    p = pred.strip()
    # a bare option letter: "B", "B.", "(B)", "B) south", "B: south" - NOT the article in "A beautiful scene"
    m = re.match(r"^\W*\(?([A-E])(?:[\.\):]|\s*$)", p)
    if m:
        return m.group(1) == s["answer"]
    gold_text = str(s["options"][s["answer"]]).strip().lower()
    return p.lower().rstrip(". ") == gold_text


def lenient_ok(pred, s):
    return s["answer"].lower() in pred.lower()   # the original _flexible_accuracy rule


def score_mcq(model, proc, samples):
    old = config.MAX_NEW_TOKENS
    config.MAX_NEW_TOKENS = 16
    try:
        preds = generate_answers(model, proc, samples, batch_size=8, desc="gen")
    finally:
        config.MAX_NEW_TOKENS = old
    n = len(samples)
    return {"letter": _mcq_accuracy(model, proc, samples),
            "strict": sum(strict_ok(p, s) for p, s in zip(preds, samples)) / n,
            "lenient": sum(lenient_ok(p, s) for p, s in zip(preds, samples)) / n,
            "example_outputs": preds[:5]}


t0 = time.time()
model, proc = load_llava()
model = apply_lora(model)
model.to(config.DEVICE)
CK = config.CHECKPOINT_DIR

# ---------------- Part A: MCQ scorers ----------------
for ds in ["mmbench1k", "scienceqa"]:
    config.apply_dataset(ds)
    data = load_dataset_by_name(ds)
    F, R = data["forget"], data["retain"][:100]
    todo = [("vanilla", f"vanilla_{ds}_seed42.pt"), ("retrained", f"retrained_{ds}_seed42.pt")]
    todo += [(f"{m}@{lr}", f"unlearned_{ds}_{m}{suf}_seed42.pt") for lr, suf in LRS.items() for m in METHODS]
    res.setdefault(ds, {})
    for name, f in todo:
        if name in res[ds]:
            continue
        p = os.path.join(CK, f)
        if not os.path.exists(p):
            print("  missing", f)
            continue
        load_checkpoint(model, p)
        model.eval()
        r = {"forget": score_mcq(model, proc, F), "retain": score_mcq(model, proc, R)}
        res[ds][name] = r
        save()
        print(f"[{(time.time() - t0) / 60:5.1f} min] {ds:9s} {name:22s} " +
              "  ".join(f"{k}: F={r['forget'][k]:.2f} R={r['retain'][k]:.2f}" for k in ("letter", "strict", "lenient")),
              flush=True)

# ---------------- Part B: MLLMU free-text ----------------
if "mllmu" not in res:
    config.apply_dataset("mllmu_bench")
    data = load_dataset_by_name("mllmu_bench")
    F = [s for s in data["forget"] if _bio_name(s["answer"])]
    R = [s for s in data["retain"] if _bio_name(s["answer"])][:100]
    out = {}
    for tag in ("vanilla", "retrained"):
        load_checkpoint(model, os.path.join(CK, f"{tag}_mllmu_bench_seed42.pt"))
        model.eval()
        out[tag] = {}
        for split, S in (("forget", F), ("retain", R)):
            preds = generate_answers(model, proc, S, batch_size=8, desc=f"mllmu-{tag}-{split}")
            out[tag][split] = {"token_f1": sum(_token_f1(p, s["answer"]) for p, s in zip(preds, S)) / len(S),
                               "identity": sum(_name_match(p, s["answer"]) for p, s in zip(preds, S)) / len(S),
                               "n": len(S)}
        print(f"MLLMU {tag}: {out[tag]}", flush=True)
    res["mllmu"] = out
    save()

# ---------------- summary ----------------
from scipy.stats import kendalltau
print("\n================ SCORER AUDIT SUMMARY ================")
summary = {}
for ds in ["mmbench1k", "scienceqa"]:
    d = res.get(ds, {})
    if "vanilla" not in d or "retrained" not in d:
        continue
    print(f"\n{ds}")
    s = {}
    for sc in ("letter", "strict", "lenient"):
        gap = d["vanilla"]["forget"][sc] - d["retrained"]["forget"][sc]
        s[sc] = {"vanilla_F": d["vanilla"]["forget"][sc], "retrained_F": d["retrained"]["forget"][sc], "gate_gap": gap}
        print(f"  {sc:8s} vanilla F={d['vanilla']['forget'][sc]:.2f}  retrained F={d['retrained']['forget'][sc]:.2f}"
              f"  gate gap={gap:+.2f} ({'PASS' if gap >= 0.15 else 'FAIL'})")
    models = [k for k in d if "@" in k]
    for a, b in (("letter", "strict"), ("letter", "lenient"), ("strict", "lenient")):
        ta = kendalltau([d[m]["forget"][a] for m in models], [d[m]["forget"][b] for m in models]).correlation
        dif = sum(1 for m in models if abs(d[m]["forget"][a] - d[m]["forget"][b]) >= 0.10)
        print(f"  forget-acc ranking agreement {a} vs {b}: tau={ta:+.2f}; models differing by >=0.10: {dif}/{len(models)}")
        s[f"tau_{a}_{b}"] = ta
    summary[ds] = s
if "mllmu" in res:
    m = res["mllmu"]
    print("\nMLLMU (biographies)")
    for sc in ("token_f1", "identity"):
        print(f"  {sc:9s} vanilla F={m['vanilla']['forget'][sc]:.2f}  retrained F={m['retrained']['forget'][sc]:.2f}"
              f"  gap={m['vanilla']['forget'][sc] - m['retrained']['forget'][sc]:+.2f}")
res["summary"] = summary
save()
print(f"\nDone in {(time.time() - t0) / 60:.0f} min -> {OUT}")
