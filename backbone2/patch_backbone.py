r"""
patch_backbone.py - makes the pipeline runnable on a second backbone WITHOUT touching the LLaVA results.
Idempotent. Called by overnight_backbone2.ps1 (you do not need to run it yourself).

* config.py: if the environment variable UNL_BACKBONE=bakllava is set, use llava-hf/bakLlava-v1-hf (LLaVA-1.5
  architecture on a Mistral-7B language model instead of Vicuna/Llama-2) and write to checkpoints_bakllava\ and
  outputs_bakllava\ . Without the variable nothing changes.
* memorization_gate.py / eval_reference.py: write their JSON into config.RESULTS_DIR instead of a hard-coded
  outputs\ (so the second backbone cannot overwrite the LLaVA gate/reference files).
"""
import os
import re
import shutil
import time
import py_compile

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ts = time.strftime("%Y%m%d_%H%M%S")


def patch(path, fn):
    p = os.path.join(B, path)
    s = open(p, encoding="utf-8", errors="surrogateescape").read()
    new = fn(s)
    if new != s:
        shutil.copy2(p, p + ".bak_" + ts)
        open(p, "w", encoding="utf-8", errors="surrogateescape", newline="").write(new)
        print("  patched", path)
    py_compile.compile(p, doraise=True)


def cfg(s):
    if "UNL_BACKBONE" in s:
        return s
    return s + '''

# --- second backbone (patch_backbone.py): only active when UNL_BACKBONE=bakllava is set ---
if os.environ.get("UNL_BACKBONE", "llava").lower() == "bakllava":
    MODEL_NAME     = "llava-hf/bakLlava-v1-hf"
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_bakllava")
    RESULTS_DIR    = os.path.join(BASE_DIR, "outputs_bakllava")
    for _d in (CHECKPOINT_DIR, RESULTS_DIR):
        os.makedirs(_d, exist_ok=True)
    print(f"[config] BACKBONE = {MODEL_NAME} -> {CHECKPOINT_DIR}, {RESULTS_DIR}")
'''


def gate(s):
    s = s.replace('os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)',
                  'os.makedirs(config.RESULTS_DIR, exist_ok=True)')
    return s.replace('os.path.join(BASE_DIR, "outputs", f"memorization_gate_',
                     'os.path.join(config.RESULTS_DIR, f"memorization_gate_')


def ref(s):
    s = s.replace('os.makedirs(os.path.join(BASE, "outputs"), exist_ok=True)', 'os.makedirs(config.RESULTS_DIR, exist_ok=True)')
    return s.replace('os.path.join(BASE, "outputs", f"reference_', 'os.path.join(config.RESULTS_DIR, f"reference_')


patch("config.py", cfg)
patch("memorization_gate.py", gate)
patch("eval_reference.py", ref)
for f, needle in (("memorization_gate.py", "config.RESULTS_DIR"), ("eval_reference.py", "config.RESULTS_DIR"),
                  ("config.py", "UNL_BACKBONE")):
    assert needle in open(os.path.join(B, f), encoding="utf-8", errors="ignore").read(), f"patch failed: {f}"
print("patch_backbone: OK")
