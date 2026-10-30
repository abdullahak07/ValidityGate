# Force offline mode â€” no HTTP pings if models/data already cached
import os as _os
_os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
_os.environ.setdefault('HF_DATASETS_OFFLINE', '1')

"""
config.py â€” Central configuration for all experiments.
Edit ONLY this file to change datasets, methods, seeds, paths.
"""

import os

# â”€â”€â”€ Reproducibility â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SEEDS = [42, 123, 5508]

# â”€â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
# data_cache/ is inside the project folder â€” model and datasets live here.
DATA_DIR = os.path.join(BASE_DIR, "data_cache")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR     = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR     = os.path.join(BASE_DIR, "outputs", "figures")

for d in [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# â”€â”€â”€ Model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
MODEL_NAME      = "llava-hf/llava-1.5-7b-hf"   # HuggingFace model ID
MAX_NEW_TOKENS  = 50
MAX_INPUT_LEN   = 768    # 576 image tokens + 192 text. NEVER go below 640 (LLaVA image=576).

# â”€â”€â”€ Memory optimisation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
GRAD_CHECKPOINT = True   # Trades compute for memory â€” essential for 7B+LoRA
MIXED_PRECISION = True   # Use torch autocast fp16 during forward pass

# â”€â”€â”€ LoRA â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
LORA_R          = 8
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.05
LORA_TARGET     = ["q_proj", "v_proj", "k_proj", "o_proj"]

# â”€â”€â”€ Training â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
FINETUNE_LR         = 2e-5
FINETUNE_EPOCHS     = 8       # quick_check: forget 0.63 -> 0.93 at 8 epochs (1 epoch learnt nothing)
MAX_TRAIN_SAMPLES   = 300     # cap per dataset â€” enough for ranking analysis, avoids 9hr runs
FINETUNE_BATCH      = 2       # RTX 4090 24GB: keep at 2 for LLaVA-7B+LoRA
UNLEARN_LR          = 1e-4    # quick_check: GA at 1e-5 / 3e-5 removed nothing; 1e-4 forget -0.20, test -0.03
UNLEARN_EPOCHS      = 1       # the validated GA setting was 1 epoch at 1e-4
UNLEARN_BATCH       = 2
WARMUP_RATIO        = 0.03

# â”€â”€â”€ Datasets â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DATASETS = ["mllmu_bench", "unlok_vqa", "mmubench"]
FORGET_RATIO = 0.1          # 10% of training set becomes forget set

# â”€â”€â”€ Unlearning Methods â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
METHODS = ["gradient_ascent", "random_labels", "finetune_retain", "salun"]

# â”€â”€â”€ SalUn specific â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SALUN_THRESHOLD = 0.5       # saliency mask threshold

# â”€â”€â”€ MIA specific â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
MIA_N_SHADOWS   = 4
MIA_BATCH       = 32

# â”€â”€â”€ Unimodal Baseline (CIFAR-10) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CIFAR_FORGET_CLASS  = 0     # forget class "airplane"
CIFAR_EPOCHS        = 30
CIFAR_LR            = 0.01
CIFAR_BATCH         = 128

# â”€â”€â”€ UQS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
METRIC_NAMES = ["FA", "RA", "MIA", "AD", "JS"]
# Weights derived empirically from retrained-model correlation (Week 4)
# Set to equal weights initially; updated after correlation analysis
UQS_WEIGHTS  = [0.2, 0.2, 0.2, 0.2, 0.2]

# â”€â”€â”€ Device â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[config] Using device: {DEVICE}")


# Multi-model support
ACTIVE_MODEL     = "llava"
BLIP2_MODEL_NAME = "Salesforce/blip2-opt-2.7b"
BLIP2_LORA_TARGET = ["q_proj", "v_proj", "k_proj", "out_proj"]
# â”€â”€â”€ NegGrad+ hyperparameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
NEGGRAD_ALPHA = 1.0   # weight for forget ascent
NEGGRAD_BETA  = 1.0   # weight for retain descent

# â”€â”€â”€ SCRUB hyperparameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SCRUB_GAMMA    = 1.0   # scale for forget ascent
SCRUB_ALPHA_KL = 0.5   # blend: KL(student||teacher) vs CE on retain


# --- added by run_fixed_pipeline.ps1 ---
ALLOW_SYNTHETIC = False
MAX_INPUT_LEN_HARD = 2048
FREE_TEXT_METRIC = "f1"


# --- per-dataset settings (added by apply_dataset_settings.ps1; validated with quick_check.py) ---
DATASET_SETTINGS = {
    "mmubench":    {"FINETUNE_EPOCHS": 8,  "FINETUNE_LR": 2e-5, "UNLEARN_LR": 1e-4, "UNLEARN_EPOCHS": 1},
    "mllmu_bench": {"FINETUNE_EPOCHS": 16, "FINETUNE_LR": 2e-4, "UNLEARN_LR": 3e-5, "UNLEARN_EPOCHS": 1},
}


def apply_dataset(name):
    import sys as _sys
    _mod = _sys.modules[__name__]
    s = DATASET_SETTINGS.get(name, {})
    for k, v in s.items():
        setattr(_mod, k, v)
    print(f"[config] settings for {name}: {s if s else 'defaults'}")


# --- TNNLS v2 (install_v2.py) ---
METHODS = ["ga", "graddiff", "random_labels", "finetune_retain", "salun", "npo"]
DATASET_SETTINGS.update({
    "mmbench1k": {"FINETUNE_EPOCHS": 8, "FINETUNE_LR": 2e-5, "UNLEARN_LR": 1e-4, "UNLEARN_EPOCHS": 1},
    "scienceqa": {"FINETUNE_EPOCHS": 8, "FINETUNE_LR": 2e-5, "UNLEARN_LR": 1e-4, "UNLEARN_EPOCHS": 1},
})


# --- lr sweep (add_lr_sweep.py): every method at 1e-4 (plain names, already run), 3e-5 and 1e-5 ---
_BASE_METHODS = ["ga", "graddiff", "random_labels", "finetune_retain", "salun", "npo"]
METHODS = _BASE_METHODS + [f"{m}__lr{lr}" for lr in ("3e-05", "1e-05") for m in _BASE_METHODS]


# --- second backbone (patch_backbone.py): only active when UNL_BACKBONE=bakllava is set ---
if os.environ.get("UNL_BACKBONE", "llava").lower() == "bakllava":
    MODEL_NAME     = "llava-hf/bakLlava-v1-hf"
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_bakllava")
    RESULTS_DIR    = os.path.join(BASE_DIR, "outputs_bakllava")
    for _d in (CHECKPOINT_DIR, RESULTS_DIR):
        os.makedirs(_d, exist_ok=True)
    print(f"[config] BACKBONE = {MODEL_NAME} -> {CHECKPOINT_DIR}, {RESULTS_DIR}")


# --- STEP2/STEP4 modes (patch_step2_step4.py) ---
_BACKBONE = os.environ.get("UNL_BACKBONE", "llava").lower()
if _BACKBONE == "qwen":
    MODEL_NAME     = "llava-hf/llava-interleave-qwen-7b-hf"
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_qwen")
    RESULTS_DIR    = os.path.join(BASE_DIR, "outputs_qwen")
    print(f"[config] BACKBONE = {MODEL_NAME} -> {CHECKPOINT_DIR}, {RESULTS_DIR}")
if os.environ.get("UNL_VARIANT", "").lower() == "shuffle":
    CHECKPOINT_DIR = CHECKPOINT_DIR.rstrip("\\/") + "_shuffle"
    RESULTS_DIR    = RESULTS_DIR.rstrip("\\/") + "_shuffle"
    SHUFFLE_EPOCH_DIV = 4
    _apply_dataset_unshuffled = apply_dataset

    def apply_dataset(name):
        _apply_dataset_unshuffled(name)
        globals()["FINETUNE_EPOCHS"] = max(1, round(globals()["FINETUNE_EPOCHS"] / SHUFFLE_EPOCH_DIV))
        print(f"[config] UNL_VARIANT=shuffle: 4 option-order copies per question, FINETUNE_EPOCHS={FINETUNE_EPOCHS}")
    print(f"[config] VARIANT = shuffle -> {CHECKPOINT_DIR}, {RESULTS_DIR}")
for _d in (CHECKPOINT_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)
