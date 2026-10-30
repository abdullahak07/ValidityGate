"""
evaluation/metrics.py - All 5 evaluation metrics.

Metrics:
  FA  - Forget Accuracy       (lower = better unlearning)
  RA  - Retain Accuracy       (higher = better utility)
  MIA - Membership Inference  (lower = better privacy)
  AD  - Activation Distance   (lower = closer to retrained model)
  JS  - JS Divergence         (lower = closer to retrained model)
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

from models.model_factory import generate_answers, extract_activations, extract_output_probs  # model-aware routing
from utils import get_logger, compute_accuracy
import config

# FIX: _flexible_accuracy counted a sample correct whenever the gold string appeared ANYWHERE in the
# output. With single-letter gold answers (MMBench) any sentence containing "a"/"b"/"c"/"d" scored as
# correct, and an empty gold answer (UnLOK) made every output "correct". It is replaced by:
#   * multiple choice (samples carry "options"): option-letter logit scoring - no text parsing;
#   * free text: SQuAD-style token F1 (default) or normalized whole-phrase match (config.FREE_TEXT_METRIC).
import re as _re
from collections import Counter as _Counter
_LETTERS = ["A", "B", "C", "D", "E"]   # ScienceQA has up to 5 options


def _norm(s):
    s = _re.sub(r"[^\w\s]", " ", str(s).lower())
    s = _re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _token_f1(pred, gold):
    p, g = _norm(pred).split(), _norm(gold).split()
    if not p or not g:
        return 0.0
    k = sum((_Counter(p) & _Counter(g)).values())
    if k == 0:
        return 0.0
    pr, rc = k / len(p), k / len(g)
    return 2 * pr * rc / (pr + rc)


def _whole_phrase(pred, gold):
    g = _norm(gold)
    return 1.0 if g and _re.search(r"(?:^|\s)" + _re.escape(g) + r"(?:\s|$)", _norm(pred)) else 0.0


# MLLMU-Bench forget/retain answers are biographies that open with the person's name
# ("Liam Falkner, born in Oslo, ..."), and every question is the same "tell me about this person" prompt.
# Token F1 against a biography rewards the template (born in / currently residing / annual salary) even when
# the model describes the WRONG person, so it cannot separate a model that knows the person from one that
# does not. For those samples we score whether the generated answer names the right person.
_NAME_RE = _re.compile(r"^\s*([A-Z][\w'\-\.]+(?:\s+[A-Z][\w'\-\.]+){1,3})\s*,")


def _bio_name(gold):
    m = _NAME_RE.match(str(gold))
    return m.group(1) if m and len(str(gold).split()) >= 8 else None


def _name_match(pred, gold):
    name = _bio_name(gold)
    if name is None:
        return None
    return 1.0 if _norm(name) in _norm(pred) else 0.0


def _letter_ids(tokenizer):
    ids = {}
    for L in _LETTERS:
        cand = set()
        for t in (L, "\u2581" + L):
            i = tokenizer.convert_tokens_to_ids(t)
            if i is not None and i != tokenizer.unk_token_id:
                cand.add(int(i))
        for t in (L, " " + L):
            enc = tokenizer(t, add_special_tokens=False).input_ids
            if enc:
                cand.add(int(enc[-1]))
        ids[L] = sorted(cand)
    return ids


@torch.no_grad()
def _mcq_accuracy(model, processor, samples, device=config.DEVICE):
    model.eval()
    ids = _letter_ids(processor.tokenizer)
    correct = 0
    for s in samples:
        prompt = f"USER: <image>\n{s['question']}\nASSISTANT:"
        inputs = processor(text=prompt, images=s["image"], return_tensors="pt").to(device)
        logits = model(**inputs).logits[0, -1].float()
        avail = [L for L in _LETTERS if L in s["options"]]
        pred = max(avail, key=lambda L: max(float(logits[i]) for i in ids[L]))
        correct += int(pred == str(s["answer"]).strip().upper())
    return correct / len(samples) if samples else 0.0


def _accuracy(model, processor, samples, batch_size=8, desc="acc"):
    if not samples:
        return 0.0
    if all(isinstance(s.get("options"), dict) and s["options"] for s in samples):
        return _mcq_accuracy(model, processor, samples)
    preds = generate_answers(model, processor, samples, batch_size=batch_size, desc=desc)
    golds = [s["answer"] for s in samples]
    if any(not str(g).strip() for g in golds):
        raise RuntimeError("Empty gold answers in evaluation set - fix the data loader.")
    fn = _whole_phrase if getattr(config, "FREE_TEXT_METRIC", "f1") == "exact" else _token_f1
    use_name = getattr(config, "BIO_NAME_METRIC", True)
    total = 0.0
    for p, g in zip(preds, golds):
        v = _name_match(p, g) if use_name else None
        total += fn(p, g) if v is None else v
    return total / len(golds)

logger = get_logger(__name__)


# --- FA: Forget Accuracy ------------------------------------------------------
def compute_fa(
    model: nn.Module,
    processor,
    forget_set: List[Dict],
    batch_size: int = 8,
) -> float:
    """
    Forget Accuracy - accuracy on the forget set.
    Lower = better (model has forgotten the data).
    Perfect unlearning = FA close to random chance.
    """
    logger.info("[FA] Computing Forget Accuracy...")
    fa     = _accuracy(model, processor, forget_set, batch_size=batch_size, desc="[FA]")
    logger.info(f"[FA] FA = {fa:.4f}")
    return fa


# --- RA: Retain Accuracy ------------------------------------------------------
def compute_ra(
    model: nn.Module,
    processor,
    retain_set: List[Dict],
    batch_size: int = 8,
) -> float:
    """
    Retain Accuracy - accuracy on the retain set.
    Higher = better (model preserved useful knowledge).
    """
    logger.info("[RA] Computing Retain Accuracy...")
    # Cap to 100 - sufficient statistical power while keeping compute tractable
    subset = retain_set[:100]
    ra     = _accuracy(model, processor, subset, batch_size=batch_size, desc="[RA]")
    logger.info(f"[RA] RA = {ra:.4f}")
    return ra


# --- MIA: Membership Inference Attack -----------------------------------------
def compute_mia(
    model: nn.Module,
    processor,
    forget_set: List[Dict],
    retain_set: List[Dict],
    device: str = config.DEVICE,
    n_shadow: int = config.MIA_N_SHADOWS,
) -> float:
    """
    Confidence-based Membership Inference Attack.
    
    Strategy:
      - Extract model's confidence (max softmax prob) on forget & retain samples
      - Train a simple binary classifier: 1=member (retain), 0=non-member (forget)
      - Attack success rate = classifier accuracy on forget set
      - Lower = better (harder to tell if data was in training set = better forgetting)
    
    Returns MIA attack success rate [0, 1].
    """
    logger.info("[MIA] Running Membership Inference Attack...")
    model.eval()

    def get_features(samples: List[Dict], desc="MIA") -> np.ndarray:
        """
        Extract 3 features per sample for improved MIA robustness:
          1. Max softmax confidence (standard)
          2. Prediction entropy (captures uncertainty)
          3. Negative log-likelihood / perplexity proxy
        Using multiple features makes the attack harder to game and
        addresses the reviewer concern of "minimal protocol".
        """
        from tqdm import tqdm
        features = []
        for i in tqdm(range(0, len(samples), config.MIA_BATCH), desc=f"[{desc}]", unit="batch", leave=False):
            batch = samples[i: i + config.MIA_BATCH]
            for sample in batch:
                prompt = f"USER: <image>\n{sample['question']}\nASSISTANT:"
                inputs = processor(
                    text=prompt,
                    images=sample["image"],
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                logits = outputs.logits[:, -1, :].float()
                probs  = torch.softmax(logits, dim=-1)
                # Feature 1: max confidence
                conf = probs.max().item()
                # Feature 2: entropy (negative  higher means more certain)
                entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                # Feature 3: top-2 probability gap (margin)
                top2 = probs.topk(2).values
                margin = (top2[0] - top2[1]).item() if len(top2) > 1 else 0.0
                features.append([conf, -entropy, margin])  # negate entropy so high=more certain
        return np.array(features)

    # Use subset for speed
    max_samples = min(100, len(forget_set), len(retain_set))  # 100 samples for more reliable LR classifier
    f_subset = forget_set[:max_samples]
    r_subset = retain_set[:max_samples]

    forget_feat = get_features(f_subset, desc="MIA-forget")
    retain_feat = get_features(r_subset, desc="MIA-retain")

    # Build classifier dataset using multi-feature MIA
    # retain=member(1), forget=non-member(0)
    X = np.vstack([retain_feat, forget_feat])
    y = np.concatenate([np.ones(len(retain_feat)), np.zeros(len(forget_feat))])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validated LR for more robust attack estimate
    # (addresses reviewer concern about "minimal protocol")
    from sklearn.model_selection import cross_val_predict
    try:
        clf = LogisticRegression(max_iter=1000, C=1.0)
        # 5-fold cross-validated predictions
        cv_preds = cross_val_predict(clf, X_scaled, y, cv=min(5, len(y)//4))
        # Balanced accuracy: accounts for both member and non-member classification
        # TNR = fraction of forget (non-member) samples correctly identified
        # TPR = fraction of retain (member) samples correctly identified
        forget_idx = np.where(y == 0)[0]
        retain_idx  = np.where(y == 1)[0]
        tnr = (cv_preds[forget_idx] == 0).mean()   # non-member recall
        tpr = (cv_preds[retain_idx] == 1).mean()   # member recall
        ba  = (tpr + tnr) / 2                      # balanced accuracy
        # MIA advantage: 0=no signal=best, 1=perfect attack=worst
        # Computed per-checkpoint so absolute value is applied before seed averaging
        mia_adv = 2 * abs(ba - 0.5)
    except Exception:
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_scaled, y)
        preds = clf.predict(X_scaled)
        forget_idx = np.where(y == 0)[0]
        retain_idx  = np.where(y == 1)[0]
        tnr = (preds[forget_idx] == 0).mean()
        tpr = (preds[retain_idx] == 1).mean()
        ba  = (tpr + tnr) / 2
        mia_adv = 2 * abs(ba - 0.5)

    logger.info(f"[MIA] Balanced accuracy = {ba:.4f}, "
                f"MIA_adv = {mia_adv:.4f} (0=no signal=best)")
    return float(mia_adv)


# --- AD: Activation Distance --------------------------------------------------
def compute_ad(
    unlearned_model: nn.Module,
    retrained_model: nn.Module,
    processor,
    forget_set: List[Dict],
    layer_idx: int = -2,
    n_samples: int = 100,
) -> float:
    """
    Activation Distance - L2 distance between unlearned and retrained model
    activations on the forget set.
    Lower = unlearned model is closer to retrained (ideal) model.
    """
    logger.info("[AD] Computing Activation Distance...")
    subset = forget_set[:n_samples]

    act_unlearned = extract_activations(unlearned_model, processor, subset, layer_idx)
    act_retrained = extract_activations(retrained_model, processor, subset, layer_idx)

    # Mean L2 distance per sample
    distances = torch.norm(act_unlearned - act_retrained, dim=1)
    ad        = distances.mean().item()

    logger.info(f"[AD] AD = {ad:.4f}")
    return ad


# --- JS: Jensen-Shannon Divergence --------------------------------------------
def compute_js(
    unlearned_model: nn.Module,
    retrained_model: nn.Module,
    processor,
    forget_set: List[Dict],
    n_samples: int = 100,
) -> float:
    """
    JS Divergence between output probability distributions of
    unlearned model vs retrained model on forget set.
    Lower = unlearned model outputs are closer to retrained model.
    """
    logger.info("[JS] Computing JS Divergence...")
    subset = forget_set[:n_samples]

    probs_unlearned = extract_output_probs(unlearned_model, processor, subset)
    probs_retrained = extract_output_probs(retrained_model, processor, subset)

    # Compute JS divergence per sample, take mean
    js_values = []
    n_bad = 0
    for p, q in zip(probs_unlearned.numpy(), probs_retrained.numpy()):
        if not (np.isfinite(p).all() and np.isfinite(q).all()):
            n_bad += 1          # fp16 overflow on this sample (seen with BakLLaVA) - skip, reported below
            continue
        # Ensure valid probability distributions
        p = np.abs(p) + 1e-10
        q = np.abs(q) + 1e-10
        p /= p.sum()
        q /= q.sum()
        js = jensenshannon(p, q) ** 2   # squared JSD in [0, 1]
        js_values.append(js)

    if n_bad:
        logger.warning(f"[JS] skipped {n_bad}/{n_bad + len(js_values)} samples with non-finite probabilities")
    js_mean = float(np.nanmean(js_values)) if js_values else float("nan")
    logger.info(f"[JS] JS = {js_mean:.4f}")
    return js_mean


# --- Run all 5 metrics at once ------------------------------------------------
def evaluate_all_metrics(
    unlearned_model: nn.Module,
    retrained_model: nn.Module,
    processor,
    forget_set: List[Dict],
    retain_set: List[Dict],
    n_mia_samples: int = 200,
    n_dist_samples: int = 100,
    metric_cache_path: str = None,   # per-metric cache file
) -> Dict[str, float]:
    """
    Run all 5 metrics and return a dict.
    Saves each metric immediately after computing - crash-safe.
    Keys: FA, RA, MIA, AD, JS
    """
    import json, os
    logger.info("=== Evaluating all 5 metrics ===")

    # Load partial cache if it exists
    partial = {}
    if metric_cache_path and os.path.exists(metric_cache_path):
        with open(metric_cache_path) as f:
            partial = json.load(f)
        logger.info(f"  Resuming from partial cache: {list(partial.keys())}")

    def _save(d):
        if metric_cache_path:
            os.makedirs(os.path.dirname(metric_cache_path), exist_ok=True)
            with open(metric_cache_path, "w") as f:
                json.dump(d, f, indent=2)

    results = dict(partial)  # start from what we have

    if "FA" not in results:
        results["FA"] = compute_fa(unlearned_model, processor, forget_set)
        _save(results)
    else:
        logger.info(f"[FA] Skipped (cached) = {results['FA']:.4f}")

    if "RA" not in results:
        results["RA"] = compute_ra(unlearned_model, processor, retain_set)
        _save(results)
    else:
        logger.info(f"[RA] Skipped (cached) = {results['RA']:.4f}")

    if "MIA" not in results:
        results["MIA"] = compute_mia(unlearned_model, processor, forget_set, retain_set)
        _save(results)
    else:
        logger.info(f"[MIA] Skipped (cached) = {results['MIA']:.4f}")

    if "AD" not in results:
        results["AD"] = compute_ad(unlearned_model, retrained_model, processor,
                                   forget_set, n_samples=n_dist_samples)
        _save(results)
    else:
        logger.info(f"[AD] Skipped (cached) = {results['AD']:.4f}")

    if "JS" not in results:
        results["JS"] = compute_js(unlearned_model, retrained_model, processor,
                                   forget_set, n_samples=n_dist_samples)
        _save(results)
    else:
        logger.info(f"[JS] Skipped (cached) = {results['JS']:.4f}")

    logger.info(f"Results: {results}")
    return results


# --- Retrained model distance (for UQS weight derivation) ---------------------
def compute_retrained_distance(
    unlearned_model,
    retrained_model,
    processor=None,
    forget_set=None,
    n_samples: int = 100,
) -> float:
    """
    Oracle proximity target: mean normalised Frobenius distance between
    effective LoRA weight updates Delta W = (alpha/r)*BA of the unlearned model
    and the retain-only counterfactual reference M*.

    Uses effective updates Delta W rather than raw A/B tensors to avoid
    LoRA-factorisation non-identifiability: B\'=BS, A\'=SA gives
    B\'A\'=BA (same model, different tensors, different raw L2 distance).

    Formula:
        d(M, M*) = (1/L) _l  Delta W_l(M) - Delta W_l(M*)_F
                                -------------------------
                                 Delta W_l(M*)_F + 

    where L = number of LoRA layers,  = 1e-8.
    Lower = unlearned model is closer to M* in effective weight space.
    """
    import re as _re
    EPS = 1e-8

    def get_lora_updates(model):
        """Return dict of effective Delta W = (alpha/r) * B @ A per layer."""
        params = {n: p.data.cpu().float()
                  for n, p in model.named_parameters() if p.requires_grad}
        updates = {}
        # Find paired lora_A / lora_B tensors
        a_keys = [k for k in params if "lora_A" in k]
        for a_key in a_keys:
            b_key = a_key.replace("lora_A", "lora_B")
            if b_key not in params:
                continue
            A = params[a_key]  # shape (r, d_in)
            B = params[b_key]  # shape (d_out, r)
            r = A.shape[0]
            # Try to detect alpha from module name; default alpha=r (scale=1)
            alpha = float(r)
            try:
                layer_name = a_key.replace(".lora_A.default.weight", "")
                module = dict(model.named_modules()).get(layer_name)
                if module is not None and hasattr(module, "scaling"):
                    alpha = float(list(module.scaling.values())[0]) * r
                elif module is not None and hasattr(module, "lora_alpha"):
                    alpha = float(module.lora_alpha)
            except Exception:
                pass
            scale = alpha / r
            delta_W = scale * (B @ A)  # (d_out, d_in)
            updates[a_key] = delta_W
        return updates

    updates_hat  = get_lora_updates(unlearned_model)
    updates_star = get_lora_updates(retrained_model)

    common = set(updates_hat) & set(updates_star)
    if not common:
        logger.warning("[Oracle] No matched LoRA layers - falling back to raw param distance")
        # Fallback: raw param distance (original behaviour)
        total, n = 0.0, 0
        r_params = {nm: p.data.cpu().float()
                    for nm, p in retrained_model.named_parameters()
                    if p.requires_grad}
        for nm, p in unlearned_model.named_parameters():
            if nm in r_params and p.requires_grad:
                total += torch.norm(p.data.cpu().float() - r_params[nm]).item()
                n += 1
        return total / max(n, 1)

    total = 0.0
    for key in common:
        dW_hat  = updates_hat[key]
        dW_star = updates_star[key]
        num  = torch.norm(dW_hat - dW_star, p="fro").item()
        denom = torch.norm(dW_star, p="fro").item() + EPS
        total += num / denom

    return total / len(common)


# _SAFE_JS_PATCH
from scipy.special import rel_entr as _rel_entr
_JS_SKIPPED = [0, 0]
def jensenshannon(p, q, *a, **k):
    p = np.asarray(p, dtype=np.float64).ravel(); q = np.asarray(q, dtype=np.float64).ravel()
    _JS_SKIPPED[1] += 1
    if not (np.isfinite(p).all() and np.isfinite(q).all()) or p.sum() <= 0 or q.sum() <= 0:
        _JS_SKIPPED[0] += 1
        print(f"[JS] skipped non-finite sample ({_JS_SKIPPED[0]}/{_JS_SKIPPED[1]} so far)")
        return float("nan")
    p = p / p.sum(); q = q / q.sum(); m = 0.5 * (p + q)
    d = 0.5 * (_rel_entr(p, m).sum() + _rel_entr(q, m).sum())
    return float(np.sqrt(max(d, 0.0)))
