r"""
models/lora_swap.py - several LoRA "models" that share ONE frozen LLaVA base on the GPU.

All checkpoints in this project differ only in their LoRA tensors; the 7B base is identical and frozen.
Holding two full models (retrained + unlearned) needs ~28 GB, more than a 24 GB card, and on Windows the
overflow silently spills into system RAM (training/eval ~150x slower). A LoRAView behaves like a model,
but before any use it copies its own LoRA tensors into the shared base.

Self-test (no checkpoints needed, ~1-2 min):   py -m models.lora_swap
"""
import weakref

import torch


def lora_snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}


def lora_restore(model, snap):
    with torch.no_grad():
        params = dict(model.named_parameters())
        for n, v in snap.items():
            params[n].copy_(v)
    return model


class LoRAView:
    # id(shared model) -> WEAK reference to the view whose LoRA weights are loaded. A strong reference here
    # kept every old view (and so every old 14 GB model) alive after main.py dropped it -> GPU spill.
    _active = {}

    def __init__(self, shared, snap=None, name="view"):
        # snap must be the LoRA weights CURRENTLY in `shared` (default: take them now)
        object.__setattr__(self, "_shared", shared)
        object.__setattr__(self, "_snap", snap if snap is not None else lora_snapshot(shared))
        object.__setattr__(self, "_name", name)
        # a new view always describes the weights currently loaded in the shared model
        LoRAView._active[id(shared)] = weakref.ref(self)

    def _activate(self):
        key = id(self._shared)
        ref = LoRAView._active.get(key)
        if ref is None or ref() is not self:
            lora_restore(self._shared, self._snap)
            LoRAView._active[key] = weakref.ref(self)
        return self._shared

    def __getattr__(self, name):
        return getattr(self._activate(), name)

    def __setattr__(self, name, value):
        setattr(self._activate(), name, value)

    def __call__(self, *args, **kwargs):
        return self._activate()(*args, **kwargs)

    def __repr__(self):
        return f"LoRAView({self._name})"


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import config
    from models.llava_model import load_llava, apply_lora
    from PIL import Image

    model, proc = load_llava()
    model = apply_lora(model)
    a = LoRAView(model, name="A")
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.add_(torch.randn_like(p) * 0.05)
    b = LoRAView(model, lora_snapshot(model), name="B")

    img = Image.new("RGB", (336, 336), (90, 140, 200))
    inp = proc(text="USER: <image>\nWhat colour is this?\nASSISTANT:", images=img, return_tensors="pt").to(config.DEVICE)
    with torch.no_grad():
        la1 = a(**inp).logits[0, -1].float()
        lb = b(**inp).logits[0, -1].float()
        la2 = a(**inp).logits[0, -1].float()
    d_ab, d_aa = float((la1 - lb).abs().max()), float((la1 - la2).abs().max())
    print(f"max |A - B| = {d_ab:.4f} (must be > 0)   max |A - A again| = {d_aa:.6f} (must be ~0)")
    print(f"GPU memory in use: {torch.cuda.memory_allocated() / 2**30:.1f} GB (one model)")
    ok = d_ab > 1e-3 and d_aa < 1e-2
    print("LoRA swap self-test:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
