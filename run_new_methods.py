"""
run_new_methods.py
==================
Runs NegGrad+ and SCRUB using existing checkpoints from:
  C:/Users/34998855/AppData/.../unlearning_eval/

Uses:
  - retrained_mmubench_seed42.pt  as oracle M*
  - vanilla_mmubench_seed42.pt    as base model (if exists)
  - Otherwise re-trains from scratch (takes ~2-3h)

Run: py run_new_methods.py
ETA: ~8-12 hours for unlearning + eval (training skipped if vanilla exists)
"""

import os, sys, json, time, glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# â”€â”€ Point to the original project that has all checkpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ORIG_CKPT_DIR  = r"C:\Users\34998855\AppData\Roaming\JetBrains\PyCharm2025.1\extensions\com.intellij.database\unlearning_eval\checkpoints"
ORIG_DATA_DIR  = r"C:\Users\34998855\AppData\Roaming\JetBrains\PyCharm2025.1\extensions\com.intellij.database\unlearning_eval\data_cache"

os.environ["TRANSFORMERS_OFFLINE"]          = "1"
os.environ["HF_DATASETS_OFFLINE"]           = "0"   # allow MMUBench download if needed
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Patch config BEFORE importing it
import config
config.CHECKPOINT_DIR = ORIG_CKPT_DIR
config.DATA_DIR       = ORIG_DATA_DIR

NEW_METHODS = ["neggrad_plus", "scrub"]
DATASET     = "mmubench"
SEEDS       = config.SEEDS   # [42, 123, 5508]

print("""
â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
â•‘  run_new_methods.py â€” NegGrad+ and SCRUB on MMUBench         â•‘
â•‘  Using existing checkpoints from unlearning_eval/            â•‘
â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
""")

# â”€â”€ Show available checkpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
all_pts = sorted(glob.glob(os.path.join(ORIG_CKPT_DIR, "*.pt")))
print(f"Original checkpoint dir: {ORIG_CKPT_DIR}")
print(f"PT files found: {len(all_pts)}")
mmubench_pts = [p for p in all_pts if "mmubench" in os.path.basename(p).lower()]
print(f"MMUBench-related: {len(mmubench_pts)}")
for p in mmubench_pts:
    mb = os.path.getsize(p) / 1e6
    print(f"  {os.path.basename(p):<55} {mb:.0f} MB")
print()

# â”€â”€ Detect naming pattern â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Check if files have 'llava_' prefix or not
has_prefix = any("llava_" in os.path.basename(p) for p in all_pts
                 if "mmubench" in os.path.basename(p).lower()
                 and "blip" not in os.path.basename(p).lower())
prefix = "llava_" if has_prefix else ""
print(f"Detected naming prefix: '{prefix}' (llava_ prefix: {has_prefix})")
print()

def find_ckpt(kind, dataset=DATASET, seed=42):
    """Find checkpoint: kind = 'vanilla', 'retrained', 'unlearned_METHOD'"""
    candidates = [
        os.path.join(ORIG_CKPT_DIR, f"{prefix}{kind}_{dataset}_seed{seed}.pt"),
        os.path.join(ORIG_CKPT_DIR, f"{kind}_{dataset}_seed{seed}.pt"),
        os.path.join(ORIG_CKPT_DIR, f"llava_{kind}_{dataset}_seed{seed}.pt"),
        # seed fallback for retrained/vanilla
        os.path.join(ORIG_CKPT_DIR, f"{prefix}{kind}_{dataset}_seed42.pt"),
        os.path.join(ORIG_CKPT_DIR, f"{kind}_{dataset}_seed42.pt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

# â”€â”€ Verify required checkpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("Checking required checkpoints for seed=42:")
vanilla_42   = find_ckpt("vanilla", seed=42)
retrained_42 = find_ckpt("retrained", seed=42)
print(f"  vanilla    seed42 : {'âœ“ ' + os.path.basename(vanilla_42) if vanilla_42 else 'âœ— NOT FOUND'}")
print(f"  retrained  seed42 : {'âœ“ ' + os.path.basename(retrained_42) if retrained_42 else 'âœ— NOT FOUND'}")
print()

if not retrained_42:
    print("ERROR: retrained (oracle) checkpoint not found â€” cannot evaluate AD/JS.")
    print("Run the original pipeline first: py main.py --stage train --dataset mmubench")
    sys.exit(1)

if not vanilla_42:
    print("WARNING: vanilla checkpoint not found â€” will retrain from scratch (~2-3h)")
    print("         If you want to skip this, copy vanilla_mmubench_seed42.pt to:")
    print(f"         {ORIG_CKPT_DIR}")
    print()
    NEED_TRAIN = True
else:
    NEED_TRAIN = False
    print("Vanilla checkpoint found â€” skipping training âœ“")
    print()


# â”€â”€ Step 1: Train (only if vanilla missing) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if NEED_TRAIN:
    print("="*65)
    print("STEP 1/4 â€” Training vanilla model (~2-3h)")
    print("="*65)
    from main import stage_train
    stage_train(datasets=[DATASET], seeds=[42])
    vanilla_42 = find_ckpt("vanilla", seed=42)
    if not vanilla_42:
        print("ERROR: Training failed â€” vanilla checkpoint still not found.")
        sys.exit(1)
    print("Step 1 done âœ“\n")
else:
    print("STEP 1/4 â€” SKIPPED (vanilla checkpoint exists)")


# â”€â”€ Step 2: Unlearn with NegGrad+ and SCRUB â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("="*65)
print("STEP 2/4 â€” Unlearning with NegGrad+ and SCRUB")
print(f"           {len(NEW_METHODS)} methods Ã— {len(SEEDS)} seeds = {len(NEW_METHODS)*len(SEEDS)} runs")
print("           ETA: ~8-12 hours")
print("="*65)

from models.llava_model import load_llava, apply_lora, load_checkpoint
from utils import set_seed, save_json, save_checkpoint
from data.loader import load_dataset_by_name

print("Loading MMUBench dataset...")
os.environ["HF_DATASETS_OFFLINE"] = "0"
data = load_dataset_by_name(DATASET)
os.environ["HF_DATASETS_OFFLINE"] = "1"
print(f"  forget={len(data['forget'])}  retain={len(data['retain'])} âœ“\n")

_, processor = load_llava()

for method_name in NEW_METHODS:
    for seed in SEEDS:
        out_name = f"{prefix}unlearned_{DATASET}_{method_name}_seed{seed}.pt"
        out_path = os.path.join(ORIG_CKPT_DIR, out_name)

        if os.path.exists(out_path):
            print(f"  [SKIP] {out_name} (already exists)")
            continue

        # Find vanilla base for this seed (fall back to seed42)
        van = find_ckpt("vanilla", seed=seed) or vanilla_42
        print(f"\n  [{method_name}] seed={seed}")
        print(f"  Loading base: {os.path.basename(van)} ...")
        base, _ = load_llava()
        base = apply_lora(base)
        model, _ = load_checkpoint(base, van)
        model.to(config.DEVICE)

        set_seed(seed)
        extra = {}
        if method_name == "neggrad_plus":
            extra = {"alpha": getattr(config,"NEGGRAD_ALPHA",1.0),
                     "beta":  getattr(config,"NEGGRAD_BETA", 1.0)}
        elif method_name == "scrub":
            extra = {"gamma":    getattr(config,"SCRUB_GAMMA",   1.0),
                     "alpha_kl": getattr(config,"SCRUB_ALPHA_KL",0.5)}

        from unlearning.methods import neggrad_plus, scrub
        fn = {"neggrad_plus": neggrad_plus, "scrub": scrub}[method_name]

        print(f"  Unlearning {method_name} (ETA ~2-3h) ...")
        t0 = time.time()
        unlearned = fn(model, data["forget"], data["retain"],
                       processor, seed=seed, **extra)

        save_checkpoint(unlearned, out_path,
                        metadata={"dataset":DATASET,"method":method_name,"seed":seed})
        elapsed = (time.time()-t0)/60
        print(f"  Saved â†’ {out_name}  ({elapsed:.1f} min) âœ“")

        del model, unlearned
        import torch; torch.cuda.empty_cache()

print("\nStep 2 done âœ“\n")


# â”€â”€ Step 3: Evaluate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("="*65)
print("STEP 3/4 â€” Computing metrics (FA/RA/MIA/AD/JS)")
print("="*65)

from evaluation.metrics import evaluate_all_metrics, compute_retrained_distance
from utils import load_json

new_results = []

for method_name in NEW_METHODS:
    for seed in SEEDS:
        cache_path = os.path.join(
            config.RESULTS_DIR,
            f"eval_{DATASET}_{method_name}_seed{seed}.json"
        )
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 10:
            print(f"  [SKIP] {os.path.basename(cache_path)} (cached)")
            new_results.append(load_json(cache_path))
            continue

        unlearned_name = f"{prefix}unlearned_{DATASET}_{method_name}_seed{seed}.pt"
        unlearned_path = os.path.join(ORIG_CKPT_DIR, unlearned_name)
        if not os.path.exists(unlearned_path):
            print(f"  [MISSING] {unlearned_name} â€” skipping eval")
            continue

        # Oracle for this seed
        ret_path = find_ckpt("retrained", seed=seed) or retrained_42
        print(f"\n  Evaluating {method_name} seed={seed} ...")
        print(f"    oracle : {os.path.basename(ret_path)}")
        print(f"    model  : {unlearned_name}")

        # Load models
        base1, _ = load_llava(); base1 = apply_lora(base1)
        unl_model, _ = load_checkpoint(base1, unlearned_path)
        unl_model.to(config.DEVICE)

        base2, _ = load_llava(); base2 = apply_lora(base2)
        ret_model, _ = load_checkpoint(base2, ret_path)
        ret_model.to(config.DEVICE)

        print("    Computing FA/RA/MIA/AD/JS ...")
        metrics = evaluate_all_metrics(
            unlearned_model=unl_model,
            retrained_model=ret_model,
            processor=processor,
            forget_set=data["forget"],
            retain_set=data["retain"],
            metric_cache_path=os.path.join(
                config.RESULTS_DIR,
                f"metrics_{DATASET}_{method_name}_seed{seed}.json"
            ),
        )
        ret_dist = compute_retrained_distance(
            unl_model, ret_model, processor, data["forget"]
        )

        result = {"dataset":DATASET,"method":method_name,"seed":seed,
                  **metrics,"retrained_distance":ret_dist}

        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        save_json(result, cache_path)
        new_results.append(result)

        print(f"    FA={metrics.get('FA',0):.4f}  RA={metrics.get('RA',0):.4f}  "
              f"MIA={metrics.get('MIA',0):.4f}  AD={metrics.get('AD',0):.2f}  "
              f"JS={metrics.get('JS',0):.6f}  âœ“")

        del unl_model, ret_model
        import torch; torch.cuda.empty_cache()

print(f"\nStep 3 done â€” {len(new_results)} results âœ“\n")


# â”€â”€ Step 4: Merge â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("="*65)
print("STEP 4/4 â€” Merging into multimodal_results.json")
print("="*65)

mm_path = os.path.join(config.RESULTS_DIR, "multimodal_results.json")
existing = load_json(mm_path) if os.path.exists(mm_path) else []
keys_new = {(r["method"],r["dataset"],r["seed"]) for r in new_results}
merged   = [r for r in existing
            if (r["method"],r["dataset"],r["seed"]) not in keys_new] + new_results

with open(mm_path, "w") as f:
    json.dump(merged, f, indent=2)

print(f"  {len(new_results)} new + {len(merged)-len(new_results)} existing = {len(merged)} total âœ“")

# â”€â”€ Summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if new_results:
    print(f"\n{'Method':20s} {'Seed':5s} {'FA':6s} {'RA':6s} {'MIA':6s} {'AD':8s} {'JS':8s}")
    print("-"*62)
    for r in sorted(new_results, key=lambda x:(x['method'],x['seed'])):
        print(f"{r['method']:20s} {r['seed']:5d} {r.get('FA',0):.4f} {r.get('RA',0):.4f} "
              f"{r.get('MIA',0):.4f} {r.get('AD',0):8.2f} {r.get('JS',0):.6f}")

print(f"""
â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
â•‘  DONE                                                        â•‘
â•‘  Next: py main.py --stage analyse                            â•‘
â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
""")

