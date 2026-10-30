r"""
bug_injection.py - claim C5: each silent pipeline bug, switched back on alone, destroys the memorisation that any
unlearning result depends on, while the run still "completes" normally.   Run from unlearning_eval\ :
    py bug_injection.py            (~60-75 min, one model on the GPU)

Setup (ScienceQA, same as quick_check): fresh LoRA trained on the 100 forget questions + 60 retain questions,
then forget accuracy and held-out test accuracy are measured. "Forget-specific gain" = forget gain - max(0, test gain).
Conditions (each starts from the same fresh LoRA):
  correct        : the fixed pipeline (8 epochs, answer-only loss, gradients reach LoRA, per-batch cosine schedule)
  one_epoch      : FINETUNE_EPOCHS = 1 (the original config)
  whole_sequence : loss on prompt + image tokens + answer instead of the answer only
  no_gradient    : reentrant gradient checkpointing without input grads (LoRA receives no gradient)
  lr_schedule    : CosineAnnealingLR(T_max = epochs) stepped every batch (LR hits 0 after 8 batches, then oscillates)
Part B (no GPU): forget/retain leakage from MMBench's option-shuffled duplicates under the original split.
Writes outputs\bug_injection.json. Safe to re-run: finished conditions are skipped.
"""
import os
import sys
import json
import time
import random

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import config
config.ACTIVE_MODEL = "llava"
from data.loader import load_dataset_by_name, _content_key, make_forget_retain_split
from data.dataset import make_dataloader
from models.llava_model import load_llava, apply_lora
from models.lora_swap import lora_snapshot, lora_restore
from evaluation.metrics import _accuracy

OUT = os.path.join(BASE, "outputs", "bug_injection.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}


def save():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT + ".tmp", "w"), indent=1)
    os.replace(OUT + ".tmp", OUT)


def train(model, loader, epochs, lr, whole_sequence=False, sched_bug=False):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=lr, weight_decay=0.01)
    sched = CosineAnnealingLR(opt, T_max=epochs if sched_bug else epochs * len(loader))
    model.train()
    first_grad, losses, lrs = None, [], []
    for ep in range(epochs):
        for b in loader:
            ids = b["input_ids"].to(config.DEVICE)
            mask = b["attention_mask"].to(config.DEVICE)
            pix = b["pixel_values"].to(config.DEVICE)
            lab = b["labels"].to(config.DEVICE)
            if whole_sequence:
                lab = ids.masked_fill(mask == 0, -100)
            with torch.autocast("cuda", dtype=torch.float16, enabled=config.MIXED_PRECISION):
                out = model(input_ids=ids, attention_mask=mask, pixel_values=pix, labels=lab)
            opt.zero_grad()
            out.loss.backward()
            if first_grad is None:
                first_grad = float(sum(p.grad.float().norm() for p in params if p.grad is not None))
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            losses.append(float(out.loss))
            lrs.append(opt.param_groups[0]["lr"])
        print(f"    epoch {ep + 1}/{epochs}  loss={sum(losses[-len(loader):]) / len(loader):.4f}", flush=True)
    return {"first_step_grad_norm": first_grad, "final_epoch_loss": sum(losses[-len(loader):]) / len(loader),
            "mean_lr_over_run": sum(lrs) / len(lrs), "lr_peak": lr}


t0 = time.time()
config.apply_dataset("scienceqa")
data = load_dataset_by_name("scienceqa")
forget, retain60, test = data["forget"], data["retain"][:60], data["test"][:100]
model, proc = load_llava()
model = apply_lora(model)
model.to(config.DEVICE)
init = lora_snapshot(model)
loader = make_dataloader(forget + retain60, proc, batch_size=config.FINETUNE_BATCH, shuffle=True)


def score(tag):
    model.eval()
    r = {"forget": _accuracy(model, proc, forget), "test": _accuracy(model, proc, test)}
    print(f"  {tag:15s} forget={r['forget']:.3f}  test={r['test']:.3f}", flush=True)
    return r


if "base" not in res:
    lora_restore(model, init)
    res["base"] = score("base")
    save()

E, LR = config.FINETUNE_EPOCHS, config.FINETUNE_LR
conds = {"correct": dict(epochs=E), "one_epoch": dict(epochs=1), "whole_sequence": dict(epochs=E, whole_sequence=True),
         "no_gradient": dict(epochs=E), "lr_schedule": dict(epochs=E, sched_bug=True)}
for name, kw in conds.items():
    if name in res:
        continue
    print(f"\n=== {name} ===", flush=True)
    torch.manual_seed(42)
    random.seed(42)
    lora_restore(model, init)
    if name == "no_gradient":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        model.disable_input_require_grads()
    try:
        info = train(model, loader, kw.pop("epochs"), LR, **kw)
    finally:
        if name == "no_gradient":
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
    r = score(name)
    fg = r["forget"] - res["base"]["forget"]
    tg = r["test"] - res["base"]["test"]
    r.update(info, forget_gain=fg, test_gain=tg, forget_specific_gain=fg - max(0.0, tg))
    res[name] = r
    save()
    print(f"  forget-specific gain = {r['forget_specific_gain']:+.3f}   [{(time.time() - t0) / 60:.0f} min]", flush=True)

# ---------------- Part B: duplicate leakage (CPU) ----------------
if "dup_leakage" not in res:
    from data.loader import load_mmubench
    raw = load_mmubench(cap=10 ** 9, dedupe_shuffle=False, name="raw")["train"]   # all labelled rows, original order
    keys = [_content_key(s) for s in raw]
    out = {"rows": len(raw), "unique_questions": len(set(keys))}
    for label, rows_ in (("original_pipeline_first300", list(range(300))),
                         ("random_1000_no_dedup", random.Random(0).sample(range(len(raw)), 1000))):
        sub = [raw[i] for i in rows_]
        f, r_ = make_forget_retain_split(sub, config.FORGET_RATIO)
        rk = {_content_key(s) for s in r_}
        leaked = sum(_content_key(s) in rk for s in f)
        out[label] = {"forget": len(f), "forget_with_copy_in_retain": leaked, "fraction": leaked / len(f)}
    res["dup_leakage"] = out
    save()

print("\n================ BUG INJECTION SUMMARY ================")
print(f"base (untrained LoRA): forget={res['base']['forget']:.3f} test={res['base']['test']:.3f}")
for name in conds:
    if name in res:
        r = res[name]
        print(f"{name:15s} forget={r['forget']:.3f} test={r['test']:.3f}  forget-specific gain={r['forget_specific_gain']:+.3f}"
              f"  first-step grad norm={r['first_step_grad_norm']:.2e}  mean lr={r['mean_lr_over_run']:.1e}")
if "dup_leakage" in res:
    d = res["dup_leakage"]
    print(f"\nMMBench: {d['rows']} labelled rows = {d['unique_questions']} unique questions")
    for k in ("original_pipeline_first300", "random_1000_no_dedup"):
        print(f"  {k}: {d[k]['forget_with_copy_in_retain']}/{d[k]['forget']} forget questions have a copy in retain "
              f"({d[k]['fraction']:.0%})")
print(f"\nDone in {(time.time() - t0) / 60:.0f} min -> {OUT}")
