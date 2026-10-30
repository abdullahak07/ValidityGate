"""
unlearning/methods.py â€” All 4 unlearning methods.

Methods:
  1. Gradient Ascent (GA)       â€” flip loss sign on forget set
  2. Random Labels (RL)         â€” corrupt forget set labels, fine-tune
  3. Fine-tune on Retain (FT)   â€” train only on retain, rely on catastrophic forgetting
  4. SalUn                      â€” saliency-based weight masking + unlearning

Each method takes:
    model       â€” vanilla fine-tuned model (already trained on full dataset)
    forget_set  â€” list of VQA samples to forget
    retain_set  â€” list of VQA samples to retain
    processor   â€” LLaVA processor
    seed        â€” for reproducibility

Returns: unlearned model
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import List, Dict
import copy
from tqdm import tqdm

from utils import get_dataloader as make_dataloader  # model-aware routing (LLaVA or BLIP-2)
from models.model_factory import full_train  # model-aware routing

def _get_train_one_epoch():
    """Return correct train_one_epoch for active model."""
    import config as _cfg
    if getattr(_cfg, 'ACTIVE_MODEL', 'llava').lower() == 'blip2':
        from models.blip2_model import train_one_epoch_blip2 as _fn
    else:
        from models.llava_model import train_one_epoch as _fn
    return _fn
from utils import get_logger, set_seed
import config

logger = get_logger(__name__)


# â”€â”€â”€ Helper: deep copy model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _copy(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)


# â”€â”€â”€ 1. Gradient Ascent â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def gradient_ascent(
    model: nn.Module,
    forget_set: List[Dict],
    retain_set: List[Dict],
    processor,
    seed: int = 42,
) -> nn.Module:
    """
    Flip the cross-entropy loss sign on the forget set.
    Maximises loss on forget â†’ degrades performance on forgotten data.
    Optionally adds retain regularisation to prevent total model collapse.
    """
    set_seed(seed)
    model = _copy(model)
    logger.info("[GA] Starting Gradient Ascent unlearning...")

    forget_loader = make_dataloader(
        forget_set, processor,
        batch_size=config.UNLEARN_BATCH,
        shuffle=True,
    )
    retain_loader = make_dataloader(
        retain_set, processor,
        batch_size=config.UNLEARN_BATCH,
        shuffle=True,
    )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.UNLEARN_LR,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=10**9)
    retain_iter = iter(retain_loader)

    model.train()
    device = config.DEVICE

    for epoch in range(1, config.UNLEARN_EPOCHS + 1):
        total_loss = 0.0

        pbar = tqdm(forget_loader, desc=f"[GA] Epoch {epoch}/{config.UNLEARN_EPOCHS}", unit="batch")
        for batch in pbar:
            optimizer.zero_grad()

            # â”€â”€ Forget loss (ascent = negative cross-entropy) â”€â”€
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values   = batch["pixel_values"].to(device)
            labels         = batch["labels"].to(device)

            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        labels=labels)
            forget_loss = -out.loss   # flip sign

            # â”€â”€ Retain regularisation (optional but stabilises training) â”€â”€
            try:
                r_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                r_batch = next(retain_iter)

            r_out = model(
                input_ids      = r_batch["input_ids"].to(device),
                attention_mask = r_batch["attention_mask"].to(device),
                pixel_values   = r_batch["pixel_values"].to(device),
                labels         = r_batch["labels"].to(device),
            )
            retain_loss = r_out.loss

            loss = forget_loss + 0.5 * retain_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += out.loss.item()
            pbar.set_postfix(forget_loss=f"{out.loss.item():.4f}")

        scheduler.step()
        logger.info(f"[GA] Epoch {epoch}/{config.UNLEARN_EPOCHS} â€” "
                    f"forget_loss: {total_loss/len(forget_loader):.4f}")

    logger.info("[GA] Unlearning complete.")
    return model


# â”€â”€â”€ 2. Random Labels â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def random_labels(
    model: nn.Module,
    forget_set: List[Dict],
    retain_set: List[Dict],
    processor,
    seed: int = 42,
) -> nn.Module:
    """
    Replace forget set answers with random wrong answers, then fine-tune.
    Teaches model to associate forgotten questions with wrong answers.
    """
    set_seed(seed)
    model = _copy(model)
    logger.info("[RL] Starting Random Labels unlearning...")

    # Collect all possible answers from retain set
    all_answers = list({s["answer"] for s in retain_set})
    if len(all_answers) < 2:
        all_answers = ["unknown", "none", "other", "N/A"]

    # Build corrupted forget loader
    corrupt_loader = make_dataloader(
        forget_set, processor,
        batch_size=config.UNLEARN_BATCH,
        shuffle=True,
        corrupt_labels=True,
        corrupt_answers=all_answers,
    )

    # Fine-tune on corrupted forget set
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.UNLEARN_LR,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=10**9)

    model.train()
    for epoch in range(1, config.UNLEARN_EPOCHS + 1):
        loss = _get_train_one_epoch()(
            model, corrupt_loader, optimizer, scheduler,
            loss_sign=1,   # normal descent on wrong labels
        )
        logger.info(f"[RL] Epoch {epoch}/{config.UNLEARN_EPOCHS} â€” loss: {loss:.4f}")

    logger.info("[RL] Unlearning complete.")
    return model


# â”€â”€â”€ 3. Fine-tune on Retain â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def finetune_retain(
    model: nn.Module,
    forget_set: List[Dict],
    retain_set: List[Dict],
    processor,
    seed: int = 42,
) -> nn.Module:
    """
    Continue training only on the retain set.
    Catastrophic forgetting handles the forget set passively.
    """
    set_seed(seed)
    model = _copy(model)
    logger.info("[FT] Starting Fine-tune on Retain unlearning...")

    retain_loader = make_dataloader(
        retain_set, processor,
        batch_size=config.UNLEARN_BATCH,
        shuffle=True,
    )

    model = full_train(
        model, retain_loader,
        n_epochs=config.UNLEARN_EPOCHS,
        lr=config.UNLEARN_LR,
        loss_sign=1,
    )

    logger.info("[FT] Unlearning complete.")
    return model


# â”€â”€â”€ 4. SalUn â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def salun_noise(
    model: nn.Module,
    forget_set: List[Dict],
    retain_set: List[Dict],
    processor,
    seed: int = 42,
    threshold: float = config.SALUN_THRESHOLD,
) -> nn.Module:
    """
    Saliency Unlearning (SalUn, Fan et al. ICLR 2024).
    Steps:
      1. Compute gradient saliency map on forget set
      2. Create binary mask: top threshold% most salient weights = 1
      3. Apply random perturbation ONLY to masked weights
      4. Fine-tune unmasked (retained) weights on retain set
    """
    set_seed(seed)
    model = _copy(model)
    logger.info("[SalUn] Starting Saliency Unlearning...")

    device = config.DEVICE

    # â”€â”€ Step 1: Compute saliency map â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    logger.info("[SalUn] Computing saliency map on forget set...")
    forget_loader = make_dataloader(
        forget_set, processor,
        batch_size=config.UNLEARN_BATCH,
        shuffle=False,
    )

    model.eval()
    grad_accum = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            grad_accum[name] = torch.zeros_like(param.data)

    for batch in tqdm(forget_loader, desc="[SalUn] Saliency map", unit="batch"):
        model.zero_grad()
        out = model(
            input_ids      = batch["input_ids"].to(device),
            attention_mask = batch["attention_mask"].to(device),
            pixel_values   = batch["pixel_values"].to(device),
            labels         = batch["labels"].to(device),
        )
        out.loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_accum[name] += param.grad.data.abs()

    # Average across batches
    n = len(forget_loader)
    for name in grad_accum:
        grad_accum[name] /= max(n, 1)

    # â”€â”€ Step 2: Build binary saliency mask â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    logger.info(f"[SalUn] Building mask with threshold={threshold}...")
    all_grads = torch.cat([g.flatten() for g in grad_accum.values()])
    cutoff    = torch.quantile(all_grads, 1.0 - threshold)

    mask = {}
    for name, grad in grad_accum.items():
        mask[name] = (grad >= cutoff).float()

    # â”€â”€ Step 3: Perturb masked (forget-salient) weights â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    logger.info("[SalUn] Perturbing salient weights...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask:
                noise = torch.randn_like(param.data) * 0.01
                param.data += mask[name] * noise

    # â”€â”€ Step 4: Fine-tune on retain set (unmasked weights) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    logger.info("[SalUn] Fine-tuning on retain set...")

    # Freeze salient weights, only update non-salient
    _salun_hook_handles = []
    for name, param in model.named_parameters():
        if name in mask:
            _salun_hook_handles.append(param.register_hook(
                lambda grad, m=mask[name]: grad * (1.0 - m.to(grad.device))
            ))

    # Cap retain set â€” SalUn only needs a small subset to restore utility
    import random as _random
    _random.seed(seed)
    salun_retain = retain_set if len(retain_set) <= 50 else _random.sample(retain_set, 50)
    logger.info(f"[SalUn] Using {len(salun_retain)} retain samples (capped at 50)")

    retain_loader = make_dataloader(
        salun_retain, processor,
        batch_size=config.UNLEARN_BATCH,
        shuffle=True,
    )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.UNLEARN_LR,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=10**9)

    model.train()
    salun_epochs = 1  # SalUn only needs 1 epoch â€” weight perturbation does the heavy lifting
    for epoch in range(1, salun_epochs + 1):
        loss = _get_train_one_epoch()(model, retain_loader, optimizer, scheduler)
        logger.info(f"[SalUn] Epoch {epoch}/{salun_epochs} â€” loss: {loss:.4f}")

    # Remove the freeze hooks explicitly (the model object is reused by the next method / seed)
    for h in _salun_hook_handles:
        h.remove()
    _salun_hook_handles.clear()

    logger.info("[SalUn] Unlearning complete.")
    return model





# â”€â”€â”€ 5. NegGrad+ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def neggrad_plus(
    model,
    forget_set,
    retain_set,
    processor,
    seed: int = 42,
    alpha: float = 1.0,
    beta: float = 1.0,
):
    """
    NegGrad+ â€” BALANCED unlearning (Jia et al. NeurIPS 2023).
    Simultaneously: gradient ASCENT on forget + gradient DESCENT on retain.
    loss = -alpha * L_forget + beta * L_retain
    Prevents model collapse seen in vanilla GA. Addresses reviewer criticism
    that all original methods are one-sided.
    """
    import copy
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from tqdm import tqdm
    set_seed(seed)
    model = copy.deepcopy(model)
    device = config.DEVICE
    logger.info(f"[NegGrad+] Starting (alpha={alpha}, beta={beta})...")

    forget_loader = make_dataloader(forget_set, processor,
                                    batch_size=config.UNLEARN_BATCH, shuffle=True)
    retain_loader = make_dataloader(retain_set, processor,
                                    batch_size=config.UNLEARN_BATCH, shuffle=True)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=config.UNLEARN_LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=10**9)
    model.train()

    for epoch in range(1, config.UNLEARN_EPOCHS + 1):
        total = 0.0
        retain_iter = iter(retain_loader)
        pbar = tqdm(forget_loader, desc=f"[NegGrad+] Epoch {epoch}/{config.UNLEARN_EPOCHS}")
        for f_batch in pbar:
            optimizer.zero_grad()
            f_out = model(input_ids=f_batch["input_ids"].to(device),
                          attention_mask=f_batch["attention_mask"].to(device),
                          pixel_values=f_batch["pixel_values"].to(device),
                          labels=f_batch["labels"].to(device))
            L_forget = -alpha * f_out.loss
            try:
                r_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                r_batch = next(retain_iter)
            r_out = model(input_ids=r_batch["input_ids"].to(device),
                          attention_mask=r_batch["attention_mask"].to(device),
                          pixel_values=r_batch["pixel_values"].to(device),
                          labels=r_batch["labels"].to(device))
            L_retain = beta * r_out.loss
            loss = L_forget + L_retain
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            pbar.set_postfix(f=f"{f_out.loss.item():.3f}", r=f"{r_out.loss.item():.3f}")
        scheduler.step()
        logger.info(f"[NegGrad+] Epoch {epoch} loss={total/max(len(forget_loader),1):.4f}")
    logger.info("[NegGrad+] Done.")
    return model


# â”€â”€â”€ 6. SCRUB â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def scrub(
    model,
    forget_set,
    retain_set,
    processor,
    seed: int = 42,
    gamma: float = 1.0,
    alpha_kl: float = 0.5,
):
    """
    SCRUB â€” teacher-student KL unlearning (Kurmanji et al. NeurIPS 2024).
    Teacher = frozen original model.
    Loss = gamma*(-L_forget) + alpha_kl*KL(student||teacher) + (1-alpha_kl)*L_retain_CE
    Balanced: prevents retain-set degradation via KL constraint.
    Addresses reviewer criticism that original methods are one-sided.
    """
    import copy
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    import torch.nn.functional as F
    from tqdm import tqdm
    set_seed(seed)
    student = copy.deepcopy(model)
    teacher = copy.deepcopy(model)
    device  = config.DEVICE
    logger.info(f"[SCRUB] Starting (gamma={gamma}, alpha_kl={alpha_kl})...")

    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    teacher.to(device)

    forget_loader = make_dataloader(forget_set, processor,
                                    batch_size=config.UNLEARN_BATCH, shuffle=True)
    retain_loader = make_dataloader(retain_set, processor,
                                    batch_size=config.UNLEARN_BATCH, shuffle=True)
    optimizer = AdamW([p for p in student.parameters() if p.requires_grad],
                      lr=config.UNLEARN_LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=10**9)
    student.train()

    for epoch in range(1, config.UNLEARN_EPOCHS + 1):
        total = 0.0
        retain_iter = iter(retain_loader)
        pbar = tqdm(forget_loader, desc=f"[SCRUB] Epoch {epoch}/{config.UNLEARN_EPOCHS}")
        for f_batch in pbar:
            optimizer.zero_grad()
            s_f = student(input_ids=f_batch["input_ids"].to(device),
                          attention_mask=f_batch["attention_mask"].to(device),
                          pixel_values=f_batch["pixel_values"].to(device),
                          labels=f_batch["labels"].to(device))
            L_forget = -gamma * s_f.loss
            try:
                r_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                r_batch = next(retain_iter)
            r_ids  = r_batch["input_ids"].to(device)
            r_mask = r_batch["attention_mask"].to(device)
            r_pix  = r_batch["pixel_values"].to(device)
            r_lbl  = r_batch["labels"].to(device)
            s_r = student(input_ids=r_ids, attention_mask=r_mask,
                          pixel_values=r_pix, labels=r_lbl)
            with torch.no_grad():
                t_r = teacher(input_ids=r_ids, attention_mask=r_mask,
                              pixel_values=r_pix, labels=r_lbl)
            if hasattr(s_r, "logits") and s_r.logits is not None:
                L_kl = F.kl_div(F.log_softmax(s_r.logits.float(), dim=-1),
                                 F.softmax(t_r.logits.float(), dim=-1),
                                 reduction="batchmean")
            else:
                L_kl = torch.tensor(0.0, device=device)
            L_retain = alpha_kl * L_kl + (1.0 - alpha_kl) * s_r.loss
            loss = L_forget + L_retain
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            pbar.set_postfix(f=f"{s_f.loss.item():.3f}", kl=f"{L_kl.item():.4f}")
        scheduler.step()
        logger.info(f"[SCRUB] Epoch {epoch} loss={total/max(len(forget_loader),1):.4f}")
    logger.info("[SCRUB] Done.")
    return student



# ============================================================================================
# v2 methods (TNNLS). Names now match what each method actually does:
#   ga          - pure gradient ascent on the forget set
#   graddiff    - gradient ascent on forget + 0.5 x retain loss  (the old "gradient_ascent")
#   salun       - SalUn (Fan et al., ICLR 2024): random-label forgetting + retain loss, updating ONLY the
#                 forget-salient weights (top SALUN_THRESHOLD by |grad| on the forget set)
#   salun_noise - the old implementation (Gaussian noise on salient weights + retain fine-tune), kept only
#                 to reproduce earlier numbers; it is NOT SalUn
#   npo         - Negative Preference Optimisation (Zhang et al., 2024) + retain loss (NPO+RT)
# ============================================================================================
import torch.nn.functional as _F


def _batch_to(batch, device):
    return {k: batch[k].to(device) for k in ("input_ids", "attention_mask", "pixel_values", "labels")}


def _seq_logp(model, b):
    """Sum of log-probabilities of the supervised (answer) tokens, per sample."""
    out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], pixel_values=b["pixel_values"])
    logits = out.logits[:, :-1, :].float()
    labels = b["labels"][:, 1:]
    nll = _F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                           ignore_index=-100, reduction="none").view(labels.shape)
    return -(nll * (labels != -100)).sum(dim=1)


def _cycle(loader):
    while True:
        for b in loader:
            yield b


def ga(model, forget_set, retain_set, processor, seed=42):
    """Pure gradient ascent: maximise the loss on the forget set, nothing else."""
    set_seed(seed)
    device = config.DEVICE
    fl = make_dataloader(forget_set, processor, batch_size=config.UNLEARN_BATCH, shuffle=True)
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=config.UNLEARN_LR)
    model.train()
    for epoch in range(1, config.UNLEARN_EPOCHS + 1):
        for b in tqdm(fl, desc=f"[GA] Epoch {epoch}/{config.UNLEARN_EPOCHS}", unit="batch"):
            opt.zero_grad()
            out = model(**_batch_to(b, device))
            (-out.loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    logger.info("[GA] done.")
    return model


def graddiff(model, forget_set, retain_set, processor, seed=42):
    """Gradient difference = the previous 'gradient_ascent' (-L_forget + 0.5 L_retain)."""
    return gradient_ascent(model, forget_set, retain_set, processor, seed=seed)


def salun(model, forget_set, retain_set, processor, seed=42, threshold=config.SALUN_THRESHOLD, alpha=1.0):
    """SalUn (Fan et al., ICLR 2024) adapted to LoRA: saliency mask from |grad L_forget|, then random-label
    forgetting + alpha * retain loss, with gradients outside the mask zeroed."""
    set_seed(seed)
    device = config.DEVICE
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}

    # 1) saliency on the forget set
    import hashlib, os as _os
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
        _os.replace(cache + ".tmp", cache)
    flat = torch.cat([v.flatten() for v in sal.values()])
    k = max(1, int(threshold * flat.numel()))
    cutoff = torch.topk(flat, k, sorted=False).values.min()
    mask = {n: (v >= cutoff).to(params[n].dtype) for n, v in sal.items()}
    frac = sum(float(m.sum()) for m in mask.values()) / flat.numel()
    logger.info(f"[SalUn] mask keeps {frac:.1%} of LoRA weights (threshold={threshold})")
    del sal, flat
    handles = [p.register_hook(lambda g, m=mask[n]: g * m) for n, p in params.items()]

    # 2) random-label forgetting + retain loss, salient weights only
    all_answers = sorted({s["answer"] for s in retain_set})
    rl = make_dataloader(forget_set, processor, batch_size=config.UNLEARN_BATCH, shuffle=True,
                         corrupt_labels=True, corrupt_answers=all_answers)
    rt = _cycle(make_dataloader(retain_set, processor, batch_size=config.UNLEARN_BATCH, shuffle=True))
    opt = AdamW(list(params.values()), lr=config.UNLEARN_LR, weight_decay=0.0)   # decay would move masked-out weights
    model.train()
    try:
        for epoch in range(1, config.UNLEARN_EPOCHS + 1):
            for b in tqdm(rl, desc=f"[SalUn] Epoch {epoch}/{config.UNLEARN_EPOCHS}", unit="batch"):
                opt.zero_grad()
                loss = model(**_batch_to(b, device)).loss + alpha * model(**_batch_to(next(rt), device)).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
    finally:
        for h in handles:
            h.remove()
    logger.info("[SalUn] done.")
    return model


def npo(model, forget_set, retain_set, processor, seed=42, beta=0.1, retain_w=1.0):
    """NPO + retain loss: L = -(2/beta) * mean log sigmoid(-beta * (log p_theta - log p_ref)) + retain_w * L_retain.
    p_ref = the model at the start of unlearning (the vanilla model), computed once before any update."""
    set_seed(seed)
    device = config.DEVICE
    fl = make_dataloader(forget_set, processor, batch_size=config.UNLEARN_BATCH, shuffle=False)
    model.eval()
    with torch.no_grad():
        ref = [_seq_logp(model, _batch_to(b, device)).detach() for b in tqdm(fl, desc="[NPO] reference", unit="batch")]
    rt = _cycle(make_dataloader(retain_set, processor, batch_size=config.UNLEARN_BATCH, shuffle=True))
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=config.UNLEARN_LR)
    model.train()
    for epoch in range(1, config.UNLEARN_EPOCHS + 1):
        for i, b in enumerate(tqdm(fl, desc=f"[NPO] Epoch {epoch}/{config.UNLEARN_EPOCHS}", unit="batch")):
            opt.zero_grad()
            logr = _seq_logp(model, _batch_to(b, device)) - ref[i]
            l_npo = -(2.0 / beta) * _F.logsigmoid(-beta * logr).mean()
            loss = l_npo + retain_w * model(**_batch_to(next(rt), device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    logger.info("[NPO] done.")
    return model


# â”€â”€â”€ Master dispatcher â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def run_unlearning(
    method_name: str,
    model: nn.Module,
    forget_set: List[Dict],
    retain_set: List[Dict],
    processor,
    seed: int = 42,
) -> nn.Module:
    """
    Call the correct unlearning method by name.
    method_name: one of config.METHODS
    """
    dispatch = {
        "gradient_ascent":  gradient_ascent,
        "random_labels":    random_labels,
        "finetune_retain":  finetune_retain,
        "salun":            salun,
        "salun_noise":      salun_noise,
        "ga":               ga,
        "graddiff":         graddiff,
        "npo":              npo,
        "neggrad_plus":     neggrad_plus,
        "scrub":            scrub,
    }
    base, lr = method_name, None
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
        config.UNLEARN_LR = old_lr


# --- patch_memory.py: no deep copy of the 7B model (2 x 14 GB > 24 GB GPU). main.py restores the
# vanilla LoRA weights before every method, so unlearning the model in place is equivalent.
# NOTE: SCRUB needs a frozen teacher copy and is NOT valid with this override (it is not in METHODS).
def _copy(model):
    return model
