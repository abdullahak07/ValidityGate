"""
data/loader.py - Load MLLMU-Bench, UnLOK-VQA, MMUBench, and CIFAR-10.
Each dataset returns a dict:
    {
        "train":   [{"image": PIL, "question": str, "answer": str}, ...],
        "forget":  [...],
        "retain":  [...],
        "test":    [...],
    }
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Any
from PIL import Image
import numpy as np

from utils import get_logger, set_seed
import config

logger = get_logger(__name__)

# FIX: never silently substitute synthetic data. A failed load must stop the run.
ALLOW_SYNTHETIC = bool(getattr(config, "ALLOW_SYNTHETIC", False))
LETTERS = ["A", "B", "C", "D"]


def _fail_or_synthetic(name, err):
    if ALLOW_SYNTHETIC:
        logger.warning(f"{name} load failed ({err}). ALLOW_SYNTHETIC=True -> synthetic fallback.")
        return _synthetic_vqa_dataset(name)
    raise RuntimeError(f"{name} failed to load: {err}. Refusing to fall back to synthetic data "
                       f"(set config.ALLOW_SYNTHETIC=True only for pipeline smoke tests).") from err


def _valid(v):
    if v is None:
        return False
    if isinstance(v, float) and v != v:  # NaN
        return False
    s = str(v).strip()
    return s != "" and s.lower() not in ("nan", "none")


def _require_text(samples, name, split, max_bad_frac=0.05):
    """Drop rows with an empty question/answer. A few are genuine gaps in the dataset; many means the
    loader is reading the wrong fields, so raise if more than max_bad_frac are empty."""
    bad = [i for i, s in enumerate(samples) if not str(s["question"]).strip() or not str(s["answer"]).strip()]
    if samples and len(bad) > max_bad_frac * len(samples):
        raise RuntimeError(f"{name}/{split}: {len(bad)} of {len(samples)} rows have an empty question or "
                           f"answer. The loader is reading the wrong fields.")
    if bad:
        logger.warning(f"{name}/{split}: dropped {len(bad)} of {len(samples)} rows with an empty "
                       f"question/answer (rows {bad[:10]}).")
    keep = set(range(len(samples))) - set(bad)
    return [s for i, s in enumerate(samples) if i in keep]


# - Generic split helper -
def make_forget_retain_split(samples: List[dict], forget_ratio: float, seed: int = 42):
    """Randomly split samples into forget / retain sets."""
    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)
    n_forget = max(1, int(len(shuffled) * forget_ratio))
    forget  = shuffled[:n_forget]
    retain  = shuffled[n_forget:]
    return forget, retain


# - MLLMU-Bench -
def load_mllmu_bench(split: str = "all") -> dict:
    """
    Load MLLMU-Bench from the HF arrow cache by finding .arrow files directly.
    Bypasses lock files entirely (Windows 260-char path limit workaround).
    """
    import os, glob
    from datasets import Dataset

    def _find_arrow_files(config_name: str):
        """Walk the cache and return all .arrow files for a given config."""
        base = os.path.join(config.DATA_DIR, "MLLMMU___mllmu-bench", config_name)
        if not os.path.isdir(base):
            return []
        arrows = glob.glob(os.path.join(base, "**", "*.arrow"), recursive=True)
        # Filter out index files, keep data files
        arrows = [a for a in arrows if not a.endswith("_indices.arrow")]
        return sorted(arrows)

    def parse(example):
        img = example.get("image", None)
        if img is None or not isinstance(img, Image.Image):
            img = Image.new("RGB", (336, 336), color=(128, 128, 128))
        else:
            img = img.convert("RGB")
        question = str(example.get("question") or example.get("Question") or "")
        answer   = str(example.get("answer")   or example.get("Answer")   or "")
        return {"image": img, "question": question,
                "answer": answer, "source": "mllmu_bench"}

    def load_config(config_name: str):
        arrows = _find_arrow_files(config_name)
        if not arrows:
            raise FileNotFoundError(f"No .arrow files found for {config_name}")
        logger.info(f"  {config_name}: found {len(arrows)} arrow file(s)")
        # Load each arrow file as a Dataset and concatenate
        from datasets import concatenate_datasets
        parts = [Dataset.from_file(a) for a in arrows]
        return concatenate_datasets(parts) if len(parts) > 1 else parts[0]

    try:
        logger.info("Loading MLLMU-Bench from arrow cache...")

        forget_ds = load_config("forget_10")
        retain_ds = load_config("retain_90")
        test_ds   = load_config("Test_Set")

        forget_samples = [parse(e) for e in forget_ds]
        retain_samples = [parse(e) for e in retain_ds]
        forget_samples = _require_text(forget_samples, "mllmu_bench", "forget")
        retain_samples = _require_text(retain_samples, "mllmu_bench", "retain")
        # FIX: Test_Set has columns ['ID','images','Classification_Task','Generation_Task','Mask_Task'];
        # it has no top-level question/answer, so the old parse produced empty text + grey image.
        test_samples = _parse_mllmu_test(test_ds)

        logger.info(f"MLLMU-Bench loaded: forget={len(forget_samples)}, "
                    f"retain={len(retain_samples)}, test={len(test_samples)}")
        return {
            "train":  forget_samples + retain_samples,
            "forget": forget_samples,
            "retain": retain_samples,
            "test":   test_samples,
        }

    except Exception as e:
        return _fail_or_synthetic("mllmu_bench", e)


def _parse_mllmu_test(test_ds):
    """Build (image, question, answer) items from Test_Set's nested Generation_Task.
    If the structure is not recognised, return [] with a warning (FA/RA do not use the test split)."""
    out = []
    for e in test_ds:
        imgs = e.get("images") or e.get("image")
        img = imgs[0] if isinstance(imgs, list) and imgs else imgs
        img = img.convert("RGB") if isinstance(img, Image.Image) else None
        tasks = e.get("Generation_Task") or []
        if isinstance(tasks, dict):
            tasks = [tasks]
        for t in tasks if isinstance(tasks, list) else []:
            if not isinstance(t, dict):
                continue
            low = {str(k).lower(): v for k, v in t.items()}
            q = low.get("question") or low.get("query")
            a = low.get("answer") or low.get("ground_truth") or low.get("gt")
            if _valid(q) and _valid(a) and img is not None:
                out.append({"image": img, "question": str(q), "answer": str(a), "source": "mllmu_bench"})
    if not out:
        logger.warning("MLLMU Test_Set: Generation_Task structure not recognised -> empty test split.")
    return out


def load_unlok_vqa() -> Dict[str, List[dict]]:
    """
    Load UnLOK-VQA from arrow cache or download if not present.
    Correct HF path: vaidehi99/UnLOK-VQA
    Paper: Patil et al., arXiv 2505.01456
    """
    import glob
    from datasets import Dataset, concatenate_datasets

    HF_PATH    = "vaidehi99/UnLOK-VQA"
    # HF cache folder name: "/" -> "---" and special chars encoded
    CACHE_NAME = "vaidehi99___un_lok-vqa"   # HF encodes hyphen differently

    def _find_arrows():
        base = os.path.join(config.DATA_DIR, CACHE_NAME)
        if not os.path.isdir(base):
            return []
        arrows = glob.glob(os.path.join(base, "**", "*.arrow"), recursive=True)
        return [a for a in sorted(arrows) if not a.endswith("_indices.arrow")]

    # FIX: UnLOK-VQA columns are ['id','src','pred','rephrase','loc','loc_ans','image_id'].
    # Question = src, answer = pred, and there is NO image column: images must be read from COCO
    # (OK-VQA images) using image_id. The old parse read 'question'/'answer' (absent -> empty strings)
    # and used a blue placeholder image for every sample.
    coco_dir = getattr(config, "COCO_DIR", None)

    def _coco_image(image_id):
        if not coco_dir:
            raise RuntimeError("UnLOK-VQA needs COCO images: set config.COCO_DIR to the folder containing "
                               "COCO train2014/val2014 jpgs (OK-VQA uses COCO).")
        iid = int(image_id)
        for sub in ("val2014", "train2014", ""):
            for name in (f"COCO_val2014_{iid:012d}.jpg", f"COCO_train2014_{iid:012d}.jpg", f"{iid:012d}.jpg"):
                p = os.path.join(coco_dir, sub, name)
                if os.path.exists(p):
                    return Image.open(p).convert("RGB")
        raise FileNotFoundError(f"COCO image for image_id={image_id} not found under {coco_dir}")

    def parse(example):
        img = example.get("image", None)
        img = img.convert("RGB") if isinstance(img, Image.Image) else _coco_image(example["image_id"])
        question = str(example.get("src") or "")
        answer   = str(example.get("pred") or "")
        return {"image": img, "question": question, "answer": answer, "source": "unlok_vqa",
                "rephrase": example.get("rephrase"), "loc": example.get("loc"),
                "loc_ans": example.get("loc_ans")}

    try:
        logger.info("Loading UnLOK-VQA from arrow cache...")
        arrows = _find_arrows()

        if arrows:
            logger.info(f"  Found {len(arrows)} arrow file(s) in cache")
            parts = [Dataset.from_file(a) for a in arrows]
            ds_all = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
            all_samples = [parse(e) for e in ds_all]
        else:
            logger.info(f"  Not cached - downloading {HF_PATH}...")
            from datasets import load_dataset
            import os as _os
            # Temporarily re-enable network for this download
            _os.environ["HF_DATASETS_OFFLINE"] = "0"
            ds = load_dataset(HF_PATH, cache_dir=config.DATA_DIR,
                              token=_os.environ.get("HF_TOKEN"))
            _os.environ["HF_DATASETS_OFFLINE"] = "1"
            split_key = "train" if "train" in ds else list(ds.keys())[0]
            all_samples = [parse(e) for e in ds[split_key]]

        cap = getattr(config, "MAX_TRAIN_SAMPLES", len(all_samples))
        all_samples = all_samples[:cap]
        all_samples = _require_text(all_samples, "unlok_vqa", "all")
        forget, retain = make_forget_retain_split(all_samples, config.FORGET_RATIO)
        # FIX: the old test split was all_samples[:200], i.e. it overlapped the training data.
        # UnLOK provides held-out locality questions (loc / loc_ans) on the same images; use those.
        test_samples = [{"image": s["image"], "question": str(s["loc"]), "answer": str(s["loc_ans"]),
                         "source": "unlok_vqa_loc"} for s in all_samples[:200]
                        if _valid(s.get("loc")) and _valid(s.get("loc_ans"))]

        logger.info(f"UnLOK-VQA loaded: total={len(all_samples)}, "
                    f"forget={len(forget)}, retain={len(retain)}")
        return {"train": all_samples, "forget": forget,
                "retain": retain, "test": test_samples}

    except Exception as e:
        return _fail_or_synthetic("unlok_vqa", e)


# - MMUBench -
def load_mmubench(cap=None, dedupe_shuffle=False, name="mmubench") -> Dict[str, List[dict]]:
    """
    NOTE (FIX): despite the function name, this loads lmms-lab/MMBench_EN (MMBench, Liu et al. 2023),
    a general multiple-choice VQA benchmark - NOT MMUBench (Li et al., NeurIPS 2024). Name it MMBench-EN
    in the paper. The dataset key "mmubench" is kept only so existing scripts keep working.
    """
    import glob
    from datasets import Dataset, concatenate_datasets

    HF_PATH    = "lmms-lab/MMBench_EN"
    CACHE_NAME = "lmms-lab___mm_bench_en"   # actual folder in data_cache

    def _find_arrows():
        base = os.path.join(config.DATA_DIR, CACHE_NAME)
        if not os.path.isdir(base):
            return []
        all_arrows = glob.glob(os.path.join(base, "**", "*.arrow"), recursive=True)
        return [a for a in sorted(all_arrows) if not a.endswith("_indices.arrow")]

    def parse(example):
        img = example.get("image", None)
        if img is None or not isinstance(img, Image.Image):
            raise RuntimeError(f"MMBench sample {example.get('index')} has no image")
        img = img.convert("RGB")
        # FIX: keep the hint and the A-D options and put them IN the question text (standard LLaVA
        # MMBench prompt), so training, generation and scoring all see the choices. The old parse
        # dropped them, so the model was asked e.g. "Which term matches the picture?" with no options
        # and a single-letter target.
        q = str(example.get("question") or "")
        options = {L: str(example[L]).strip() for L in LETTERS if L in example and _valid(example[L])}
        hint = str(example.get("hint")).strip() if _valid(example.get("hint")) else ""
        parts = ([hint] if hint else []) + [q] + [f"{L}. {options[L]}" for L in LETTERS if L in options]
        parts.append("Answer with the option's letter from the given choices directly.")
        answer = str(example.get("answer") or "").strip().upper()
        if answer == "":
            return None   # FIX: MMBench_EN's test split has no public answers -> skip unlabelled rows
        if answer not in options:
            raise RuntimeError(f"MMBench sample {example.get('index')}: answer {answer!r} not among options")
        return {"image": img, "question": "\n".join(parts), "raw_question": q,
                "answer": answer, "options": options, "source": "mmbench_en"}

    try:
        logger.info("Loading MMUBench from arrow cache...")
        arrows = _find_arrows()

        if arrows:
            logger.info(f"  Found {len(arrows)} arrow file(s) in cache")
            parts = [Dataset.from_file(a) for a in arrows]
            ds_all = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
            all_samples = [x for x in (parse(e) for e in ds_all) if x is not None]
        else:
            logger.info(f"  Not cached - downloading {HF_PATH}...")
            from datasets import load_dataset
            import os as _os
            _os.environ["HF_DATASETS_OFFLINE"] = "0"
            ds = load_dataset(HF_PATH, cache_dir=config.DATA_DIR,
                              token=_os.environ.get("HF_TOKEN"))
            _os.environ["HF_DATASETS_OFFLINE"] = "1"
            split_key = "train" if "train" in ds else list(ds.keys())[0]
            all_samples = [x for x in (parse(e) for e in ds[split_key]) if x is not None]

        logger.info(f"  MMBench labelled rows: {len(all_samples)}")
        if dedupe_shuffle:
            all_samples = _dedupe_and_shuffle(all_samples, name)
        cap = cap or getattr(config, "MAX_TRAIN_SAMPLES", len(all_samples))
        held_out = all_samples[cap:cap + 200]      # FIX: test = held-out questions never trained on
        all_samples = all_samples[:cap]
        forget, retain = make_forget_retain_split(all_samples, config.FORGET_RATIO)
        test_samples   = held_out

        logger.info(f"MMUBench loaded: total={len(all_samples)}, "
                    f"forget={len(forget)}, retain={len(retain)}")
        return {"train": all_samples, "forget": forget,
                "retain": retain, "test": test_samples}

    except Exception as e:
        return _fail_or_synthetic(name, e)


# - CIFAR-10 (Unimodal baseline) -
def load_cifar10():
    """Load CIFAR-10 with forget class = config.CIFAR_FORGET_CLASS."""
    import torchvision
    import torchvision.transforms as T

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)),
    ])
    train_ds = torchvision.datasets.CIFAR10(
        root=config.DATA_DIR, train=True,  download=True, transform=transform)
    test_ds  = torchvision.datasets.CIFAR10(
        root=config.DATA_DIR, train=False, download=True, transform=transform)

    forget_idx = [i for i, (_, y) in enumerate(train_ds)
                  if y == config.CIFAR_FORGET_CLASS]
    retain_idx = [i for i, (_, y) in enumerate(train_ds)
                  if y != config.CIFAR_FORGET_CLASS]

    from torch.utils.data import Subset
    forget_set = Subset(train_ds, forget_idx)
    retain_set = Subset(train_ds, retain_idx)

    logger.info(f"CIFAR-10: forget={len(forget_set)}, retain={len(retain_set)}, "
                f"test={len(test_ds)}")
    return {
        "train":  train_ds,
        "forget": forget_set,
        "retain": retain_set,
        "test":   test_ds,
    }


# - v2 helpers (TNNLS) -
def _content_key(s):
    """Questions that differ only by option ORDER (MMBench CircularEval copies) or appear twice must not be
    split across forget and retain, otherwise the retain-only model still sees the forget content."""
    import hashlib
    opts = sorted(str(v).strip().lower() for v in (s.get("options") or {}).values())
    img = s["image"].resize((32, 32)).convert("L").tobytes()
    return hashlib.md5((str(s.get("raw_question", s["question"])).strip().lower() + "||" + "|".join(opts)).encode()
                       + img).hexdigest()


def _dedupe_and_shuffle(samples, name, seed=42):
    seen, out = set(), []
    for s in samples:
        k = _content_key(s)
        if k not in seen:
            seen.add(k)
            out.append(s)
    random.Random(seed).shuffle(out)
    logger.info(f"  {name}: {len(samples)} rows -> {len(out)} unique questions after de-duplication; shuffled (seed {seed})")
    return out


def load_scienceqa(cap=1000) -> Dict[str, List[dict]]:
    """ScienceQA (Lu et al., NeurIPS 2022), image questions only. Multiple choice with 2-5 options, scored by
    option-letter logits exactly like MMBench. train split -> forget/retain; validation split -> held-out test."""
    try:
        import glob
        from datasets import Dataset, concatenate_datasets
        letters = ["A", "B", "C", "D", "E"]
        # Read the cached arrow files directly. load_dataset() in offline mode builds a lock-file name from
        # the full cache path, which exceeds the Windows 260-character path limit in this project folder.
        base = os.path.join(config.DATA_DIR, "derek-thomas___science_qa")
        arrows = [a for a in sorted(glob.glob(os.path.join(base, "**", "*.arrow"), recursive=True))
                  if not a.endswith("_indices.arrow")]

        def _split(tag):
            fs = [a for a in arrows if f"-{tag}" in os.path.basename(a)]
            if not fs:
                raise RuntimeError(f"ScienceQA {tag} arrow files not found under {base}: {arrows[:5]}")
            parts = [Dataset.from_file(a) for a in fs]
            return concatenate_datasets(parts) if len(parts) > 1 else parts[0]

        ds = {"train": _split("train"), "validation": _split("validation")}

        def parse(e):
            img = e.get("image")
            if img is None or not isinstance(img, Image.Image):
                return None
            ch = list(e["choices"])
            if not (2 <= len(ch) <= 5):
                return None
            options = {letters[i]: str(c).strip() for i, c in enumerate(ch)}
            hint = str(e.get("hint") or "").strip()
            q = str(e["question"]).strip()
            parts = ([f"Context: {hint}"] if hint else []) + [q] + [f"{L}. {options[L]}" for L in options]
            parts.append("Answer with the option's letter from the given choices directly.")
            return {"image": img.convert("RGB"), "question": "\n".join(parts), "raw_question": q,
                    "answer": letters[int(e["answer"])], "options": options, "source": "scienceqa"}

        train = [x for x in (parse(e) for e in ds["train"]) if x is not None]
        val = [x for x in (parse(e) for e in ds["validation"]) if x is not None]
        logger.info(f"  ScienceQA image questions: train={len(train)}, validation={len(val)}")
        train = _dedupe_and_shuffle(train, "scienceqa")[:cap]
        test = _dedupe_and_shuffle(val, "scienceqa-val")[:200]
        forget, retain = make_forget_retain_split(train, config.FORGET_RATIO)
        logger.info(f"ScienceQA loaded: total={len(train)}, forget={len(forget)}, retain={len(retain)}, test={len(test)}")
        return {"train": train, "forget": forget, "retain": retain, "test": test}
    except Exception as e:
        return _fail_or_synthetic("scienceqa", e)


# - Master loader -
def load_dataset_by_name(name: str) -> Dict[str, List[dict]]:
    loaders = {
        "mllmu_bench": load_mllmu_bench,
        "unlok_vqa":   load_unlok_vqa,
        "mmubench":    load_mmubench,
        # v2 (TNNLS): de-duplicated, shuffled, 1000 train samples -> 100 forget
        "mmbench1k":   lambda: load_mmubench(cap=1000, dedupe_shuffle=True, name="mmbench1k"),
        "scienceqa":   lambda: load_scienceqa(cap=1000),
    }
    assert name in loaders, f"Unknown dataset: {name}. Choose from {list(loaders)}"
    return loaders[name]()


# - Synthetic fallback (when datasets not yet released / no internet) -
def _synthetic_vqa_dataset(name: str, n_train=1000, n_test=200) -> Dict[str, List[dict]]:
    """
    Generate a small synthetic VQA dataset for testing your pipeline
    before real datasets are available. Replace with real data for paper.
    """
    logger.warning(f"Using SYNTHETIC data for {name} - replace with real dataset!")

    TEMPLATES = [
        ("What color is the object?", ["red", "blue", "green", "yellow"]),
        ("How many objects are there?", ["1", "2", "3", "4"]),
        ("What is the shape?", ["circle", "square", "triangle", "rectangle"]),
        ("Is the object large or small?", ["large", "small"]),
    ]

    def make_sample(idx):
        q_tmpl, answers = TEMPLATES[idx % len(TEMPLATES)]
        color = (np.random.randint(50,200), np.random.randint(50,200), np.random.randint(50,200))
        img = Image.new("RGB", (224, 224), color=color)
        return {
            "image":    img,
            "question": q_tmpl,
            "answer":   answers[idx % len(answers)],
            "source":   name,
        }

    train = [make_sample(i) for i in range(n_train)]
    test  = [make_sample(i) for i in range(n_train, n_train + n_test)]
    forget, retain = make_forget_retain_split(train, config.FORGET_RATIO)

    return {"train": train, "forget": forget, "retain": retain, "test": test}


# --- STEP2/STEP4 modes (patch_step2_step4.py) ---
import os as _os
if _os.environ.get("UNL_VARIANT", "").lower() == "shuffle":
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from variants import augment4 as _augment4
    _load_dataset_unshuffled = load_dataset_by_name

    def load_dataset_by_name(name):
        d = _load_dataset_unshuffled(name)
        d["train"] = _augment4(d["forget"]) + _augment4(d["retain"])     # vanilla: F u R, 4 orders each
        d["retain_train"] = _augment4(d["retain"])                       # retrained: R only, 4 orders each
        return d
