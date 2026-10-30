"""
utils.py â€” Shared utility functions.
"""

import os
import random
import json
import logging
import numpy as np
import torch

# â”€â”€â”€ Logging â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    return logging.getLogger(name)

logger = get_logger(__name__)


# â”€â”€â”€ Reproducibility â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Seed set to {seed}")


# â”€â”€â”€ Checkpoint helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _lora_state(model):
    return {n: p.detach().to("cpu").clone() for n, p in model.named_parameters() if "lora_" in n}


def save_checkpoint(model, path: str, metadata: dict = None):
    """Save ONLY the LoRA weights (~40 MB). The 7B base is frozen and identical for every checkpoint, so
    saving the full state dict (13 GB) was pure waste and filled the disk. Written to a temp file and renamed,
    so a crash can never leave a half-written file that looks finished."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"state_dict": _lora_state(model), "lora_only": True}
    if not payload["state_dict"]:
        raise RuntimeError("save_checkpoint: model has no LoRA parameters")
    if metadata:
        payload["metadata"] = metadata
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    logger.info(f"Saved checkpoint ({len(payload['state_dict'])} LoRA tensors) -> {path}")


def load_checkpoint(model, path: str):
    """Load the LoRA weights from a checkpoint into `model` IN PLACE (weights stay on their current device).
    Works for both new LoRA-only checkpoints and old full-state checkpoints (only lora_ tensors are used;
    the frozen base weights in old files are identical to the loaded base model)."""
    try:
        payload = torch.load(path, map_location="cpu", mmap=True)
    except Exception:
        payload = torch.load(path, map_location="cpu")
    sd = payload["state_dict"]
    params = dict(model.named_parameters())
    n = 0
    with torch.no_grad():
        for k, v in sd.items():
            if "lora_" in k and k in params:
                params[k].copy_(v.to(device=params[k].device, dtype=params[k].dtype))
                n += 1
    n_model = sum(1 for k in params if "lora_" in k)
    if n != n_model:
        raise RuntimeError(f"load_checkpoint: loaded {n} LoRA tensors but the model has {n_model} ({path})")
    meta = payload.get("metadata", {})
    logger.info(f"Loaded checkpoint ({n} LoRA tensors) <- {path}")
    return model, meta


# â”€â”€â”€ JSON helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    logger.info(f"Saved JSON â†’ {path}")


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


# â”€â”€â”€ Accuracy â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def compute_accuracy(predictions: list, labels: list) -> float:
    """Simple exact-match accuracy."""
    assert len(predictions) == len(labels), "Length mismatch"
    correct = sum(p.strip().lower() == l.strip().lower()
                  for p, l in zip(predictions, labels))
    return correct / len(labels) if labels else 0.0


# â”€â”€â”€ Softmax / probs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.float(), dim=-1)


# â”€â”€â”€ Flatten nested dict â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def flatten_dict(d: dict, parent_key="", sep="_") -> dict:
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def get_dataloader(samples, processor, batch_size=2, shuffle=True,
                   corrupt_labels=False, corrupt_answers=None):
    import config as _cfg
    active = getattr(_cfg, "ACTIVE_MODEL", "llava").lower()
    if active == "blip2":
        from data.blip2_dataset import make_dataloader_blip2
        return make_dataloader_blip2(
            samples, processor,
            batch_size=batch_size, shuffle=shuffle,
            corrupt_labels=corrupt_labels, corrupt_answers=corrupt_answers,
        )
    else:
        from data.dataset import make_dataloader
        return make_dataloader(
            samples, processor,
            batch_size=batch_size, shuffle=shuffle,
            corrupt_labels=corrupt_labels, corrupt_answers=corrupt_answers,
        )
