r"""
scorer_fixes.py - two follow-up experiments for claim C2 (scoring). LLaVA-1.5-7B only. ~60-75 min.
Run (or let run_after_gpu.ps1 run it):   py scorer_fixes\scorer_fixes.py

PART 1 - when does the original lenient scorer break?  (MMBench + ScienceQA)
  The original pipeline dropped the answer options from the prompt, so models answered in free text, and the
  scorer counted an answer correct if the gold letter ("a"/"b"/"c"/"d") appeared ANYWHERE in it.
  For the untrained base model, vanilla, retrained and the 6 methods at 1e-4 and 3e-5 (seed 42), with the
  ORIGINAL no-options prompt we report:
    lenient_gold    : lenient rule against the TRUE letter            (what the original pipeline reported)
    lenient_placebo : lenient rule against a WRONG letter of the same question  (a scorer that carries information
                      must score much lower here; if both are ~equal the scorer measures nothing)
    text_match      : the gold OPTION TEXT appears as a whole phrase in the answer (a sensible free-text scorer)
PART 2 - free-text identity scoring on a model that really memorised the MLLMU people
  "vanilla-mini": fresh LoRA trained on the MLLMU forget biographies + 60 retain biographies;
  "retrained-mini": same recipe on the 60 retain biographies only (never sees the forget people).
  Both scored on the forget biographies with token F1 and with subject identity (names the right person).
Writes outputs\scorer_fixes.json and prints a SUMMARY. Safe to re-run: finished parts are skipped.
"""
import os
import re
import sys
import json
import time
import random

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import config
config.ACTIVE_MODEL = "llava"
from data.loader import load_dataset_by_name
from data.dataset import make_dataloader
from models.llava_model import load_llava, apply_lora, generate_answers
from models.lora_swap import lora_snapshot, lora_restore
from utils import load_checkpoint
from evaluation.metrics import _token_f1, _name_match, _bio_name, _norm

OUT = os.path.join(BASE, "outputs", "scorer_fixes.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
t0 = time.time()


def save():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT + ".tmp", "w"), indent=1)
    os.replace(OUT + ".tmp", OUT)


def log(m):
    print(f"[{(time.time() - t0) / 60:5.1f} min] {m}", flush=True)


model, proc = load_llava()
model = apply_lora(model)
model.to(config.DEVICE)
init = lora_snapshot(model)          # fresh LoRA (B = 0) == the untrained base model
CK = config.CHECKPOINT_DIR

# ======================= PART 1 =======================
def no_option_version(s):
    q = dict(s)
    q["question"] = s.get("raw_question", s["question"])      # original prompt: question only, no options
    return q


def part1_scores(preds, samples):
    rng = random.Random(0)
    lg = lp = tm = 0
    for p, s in zip(preds, samples):
        pl = p.lower()
        lg += s["answer"].lower() in pl
        wrong = rng.choice([L for L in s["options"] if L != s["answer"]])
        lp += wrong.lower() in pl
        gold_txt = _norm(s["options"][s["answer"]])
        tm += bool(gold_txt) and re.search(r"(?:^|\s)" + re.escape(gold_txt) + r"(?:\s|$)", _norm(p)) is not None
    n = len(samples)
    return {"lenient_gold": lg / n, "lenient_placebo": lp / n, "text_match": tm / n}


METHODS = ["ga", "graddiff", "random_labels", "finetune_retain", "salun", "npo"]
for ds in ["mmbench1k", "scienceqa"]:
    key = f"part1_{ds}"
    res.setdefault(key, {})
    config.apply_dataset(ds)
    data = load_dataset_by_name(ds)
    F = [no_option_version(s) for s in data["forget"]]
    R = [no_option_version(s) for s in data["retain"][:100]]
    todo = [("base", None), ("vanilla", f"vanilla_{ds}_seed42.pt"), ("retrained", f"retrained_{ds}_seed42.pt")]
    todo += [(f"{m}@{lr}", f"unlearned_{ds}_{m}{suf}_seed42.pt")
             for lr, suf in (("1e-04", ""), ("3e-05", "__lr3e-05")) for m in METHODS]
    for name, f in todo:
        if name in res[key]:
            continue
        if f is None:
            lora_restore(model, init)
        elif os.path.exists(os.path.join(CK, f)):
            load_checkpoint(model, os.path.join(CK, f))
        else:
            log(f"missing {f}")
            continue
        model.eval()
        old = config.MAX_NEW_TOKENS
        config.MAX_NEW_TOKENS = 32
        try:
            pf = generate_answers(model, proc, F, batch_size=8, desc="gen")
            pr = generate_answers(model, proc, R, batch_size=8, desc="gen")
        finally:
            config.MAX_NEW_TOKENS = old
        r = {"forget": part1_scores(pf, F), "retain": part1_scores(pr, R), "examples": pf[:4]}
        res[key][name] = r
        save()
        log(f"{ds:9s} {name:22s} forget: lenient={r['forget']['lenient_gold']:.2f} "
            f"placebo={r['forget']['lenient_placebo']:.2f} text={r['forget']['text_match']:.2f}   e.g. {pf[0][:60]!r}")

# ======================= PART 2 =======================
def train(loader, epochs, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=lr, weight_decay=0.01)
    sched = CosineAnnealingLR(opt, T_max=epochs * len(loader))
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for b in loader:
            with torch.autocast("cuda", dtype=torch.float16, enabled=config.MIXED_PRECISION):
                out = model(**{k: b[k].to(config.DEVICE) for k in ("input_ids", "attention_mask", "pixel_values", "labels")})
            opt.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            tot += float(out.loss)
        log(f"    epoch {ep + 1}/{epochs} loss={tot / len(loader):.4f}")


if "part2_mllmu" not in res:
    config.apply_dataset("mllmu_bench")
    data = load_dataset_by_name("mllmu_bench")
    F = [s for s in data["forget"] if _bio_name(s["answer"])]
    R60 = [s for s in data["retain"] if _bio_name(s["answer"])][:60]
    E, LR = config.FINETUNE_EPOCHS, config.FINETUNE_LR
    out = {"n_forget": len(F), "epochs": E, "lr": LR}
    for tag, train_set in (("vanilla_mini", F + R60), ("retrained_mini", R60)):
        log(f"PART 2: training {tag} on {len(train_set)} biographies, {E} epochs @ {LR}")
        torch.manual_seed(42)
        lora_restore(model, init)
        train(make_dataloader(train_set, proc, batch_size=config.FINETUNE_BATCH, shuffle=True), E, LR)
        model.eval()
        preds = generate_answers(model, proc, F, batch_size=8, desc=f"{tag}-forget")
        out[tag] = {"token_f1": sum(_token_f1(p, s["answer"]) for p, s in zip(preds, F)) / len(F),
                    "identity": sum(_name_match(p, s["answer"]) for p, s in zip(preds, F)) / len(F),
                    "examples": [{"gold": s["answer"][:120], "pred": p[:120]} for p, s in zip(preds[:4], F[:4])]}
        log(f"  {tag}: token_f1={out[tag]['token_f1']:.3f} identity={out[tag]['identity']:.3f}")
    res["part2_mllmu"] = out
    save()

# ======================= SUMMARY =======================
print("\n================ SCORER FIXES SUMMARY ================")
for ds in ["mmbench1k", "scienceqa"]:
    d = res.get(f"part1_{ds}", {})
    if not d:
        continue
    print(f"\nPART 1 {ds} (original no-options prompt), forget set:")
    print(f"  {'model':22s} {'lenient(gold)':>14s} {'lenient(WRONG)':>15s} {'text match':>11s}")
    for name, r in d.items():
        f = r["forget"]
        print(f"  {name:22s} {f['lenient_gold']:14.2f} {f['lenient_placebo']:15.2f} {f['text_match']:11.2f}")
if "part2_mllmu" in res:
    p = res["part2_mllmu"]
    v, r = p["vanilla_mini"], p["retrained_mini"]
    print(f"\nPART 2 MLLMU memorised mini-models ({p['n_forget']} forget biographies):")
    print(f"  token F1 : vanilla={v['token_f1']:.3f} retrained={r['token_f1']:.3f} gap={v['token_f1'] - r['token_f1']:+.3f}")
    print(f"  identity : vanilla={v['identity']:.3f} retrained={r['identity']:.3f} gap={v['identity'] - r['identity']:+.3f}")
print(f"\nDone in {(time.time() - t0) / 60:.0f} min -> {OUT}")
