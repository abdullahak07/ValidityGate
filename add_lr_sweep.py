r"""
add_lr_sweep.py - run once from unlearning_eval\:   py add_lr_sweep.py
Then:  powershell -ExecutionPolicy Bypass -File .\tnnls_v2\run_v2.ps1

Adds an unlearning-strength sweep. Every method is also run at lr 1e-5 and 3e-5 (the finished 1e-4 runs are
kept and reused). Method names become e.g. "ga__lr3e-05"; checkpoints/results get those names too.
Why: at 1e-4 four of six methods collapse the model (forget acc ~= retain acc ~= chance), so the analysis would
only cover broken models. The sweep gives each method a weak / medium / strong setting.
Also caches SalUn's saliency map (it depends only on the vanilla model + forget set), so the slow 17-min
saliency pass runs once per dataset instead of once per run.
Backs up unlearning/methods.py and config.py.
"""
import os
import shutil
import sys
import time
import py_compile

B = os.path.dirname(os.path.abspath(__file__))
ts = time.strftime("%Y%m%d_%H%M%S")
mp, cp = os.path.join(B, "unlearning", "methods.py"), os.path.join(B, "config.py")
m = open(mp, encoding="utf-8", errors="surrogateescape").read()
c = open(cp, encoding="utf-8", errors="surrogateescape").read()
if "__lr" in m:
    sys.exit("Already applied.")
NL = "\r\n" if "\r\n" in m else "\n"
m = m.replace("\r\n", "\n")

# 1) method__lr<value> support in the dispatcher
old = '''    assert method_name in dispatch, \\
        f"Unknown method: {method_name}. Choose from {list(dispatch)}"

    logger.info(f"=== Running unlearning method: {method_name} (seed={seed}) ===")
    return dispatch[method_name](model, forget_set, retain_set, processor, seed=seed)'''
new = '''    base, lr = method_name, None
    if "__lr" in method_name:                      # e.g. "ga__lr3e-05" -> method ga at lr 3e-5
        base, lr_s = method_name.split("__lr", 1)
        lr = float(lr_s)
    assert base in dispatch, f"Unknown method: {base}. Choose from {list(dispatch)}"
    old_lr = config.UNLEARN_LR
    if lr is not None:
        config.UNLEARN_LR = lr
    logger.info(f"=== Running unlearning method: {base} (lr={config.UNLEARN_LR}, seed={seed}) ===")
    try:
        return dispatch[base](model, forget_set, retain_set, processor, seed=seed)
    finally:
        config.UNLEARN_LR = old_lr'''
if old not in m:
    sys.exit("NOT PATCHED: dispatcher not found in the expected form. Send unlearning/methods.py to Claude.")
m = m.replace(old, new)

# 2) SalUn saliency cache
old_s = '''    model.eval()
    sal = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in params.items()}
    fl = make_dataloader(forget_set, processor, batch_size=config.UNLEARN_BATCH, shuffle=False)
    for b in tqdm(fl, desc="[SalUn] saliency", unit="batch"):
        model.zero_grad()
        model(**_batch_to(b, device)).loss.backward()
        for n, p in params.items():
            if p.grad is not None:
                sal[n] += p.grad.detach().abs().float()
    model.zero_grad()'''
new_s = '''    import hashlib, os as _os
    # the saliency map depends only on the starting (vanilla) LoRA weights and the forget set -> cache it
    fp = hashlib.md5(("|".join(str(s["question"]) for s in forget_set) +
                      f"|{sum(float(p.detach().float().sum()) for p in params.values()):.6e}").encode()).hexdigest()
    cache = _os.path.join(config.CHECKPOINT_DIR, "_salun_cache", f"{fp}.pt")
    if _os.path.exists(cache):
        sal = {n: v.to(device) for n, v in torch.load(cache, map_location="cpu").items()}
        logger.info(f"[SalUn] saliency loaded from cache {_os.path.basename(cache)}")
    else:
        model.eval()
        sal = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in params.items()}
        fl = make_dataloader(forget_set, processor, batch_size=config.UNLEARN_BATCH, shuffle=False)
        for b in tqdm(fl, desc="[SalUn] saliency", unit="batch"):
            model.zero_grad()
            model(**_batch_to(b, device)).loss.backward()
            for n, p in params.items():
                if p.grad is not None:
                    sal[n] += p.grad.detach().abs().float()
        model.zero_grad()
        _os.makedirs(_os.path.dirname(cache), exist_ok=True)
        torch.save({n: v.cpu() for n, v in sal.items()}, cache + ".tmp")
        _os.replace(cache + ".tmp", cache)'''
if old_s not in m:
    sys.exit("NOT PATCHED: SalUn saliency block not found. Send unlearning/methods.py to Claude.")
m = m.replace(old_s, new_s)

# 3) config: methods at three strengths
if "lr sweep" not in c:
    c += '''

# --- lr sweep (add_lr_sweep.py): every method at 1e-4 (plain names, already run), 3e-5 and 1e-5 ---
_BASE_METHODS = ["ga", "graddiff", "random_labels", "finetune_retain", "salun", "npo"]
METHODS = _BASE_METHODS + [f"{m}__lr{lr}" for lr in ("3e-05", "1e-05") for m in _BASE_METHODS]
'''
shutil.copy2(mp, mp + f".bak_{ts}")
shutil.copy2(cp, cp + f".bak_{ts}")
open(mp, "w", encoding="utf-8", errors="surrogateescape", newline="").write(m.replace("\n", NL))
open(cp, "w", encoding="utf-8", errors="surrogateescape", newline="").write(c)
py_compile.compile(mp, doraise=True)
py_compile.compile(cp, doraise=True)
print("Applied. 18 method settings per dataset (6 methods x 3 learning rates). Backups: *.bak_" + ts)
print("Next:  powershell -ExecutionPolicy Bypass -File .\\tnnls_v2\\run_v2.ps1")
