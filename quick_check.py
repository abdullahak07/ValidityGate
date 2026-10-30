r"""
quick_check.py - 20-40 min smoke test BEFORE the full pipeline. Uses the FIXED files (loader, dataset,
llava_model, metrics). Saves nothing to checkpoints\.

    py quick_check.py --dataset mmubench
    py quick_check.py --dataset mllmu_bench --epochs 5

Stage A (memorisation): score the untouched base model, LoRA-train it on forget + a small retain slice,
                        score again.  PASS-A if the forget set improves clearly MORE than held-out test:
                        (forget_after - forget_before) - max(0, test_after - test_before) >= --min_gap.
                        (Comparing forget_after with test_after directly is wrong: MMBench's held-out
                        questions are easier - base 0.90 vs forget 0.63.)
Stage B (unlearning):   gradient ascent on the forget set only, short.
                        PASS-B if forget drops by >= --min_drop while held-out test drops by <= --max_test_drop.
Also prints seconds per training epoch so the full run can be timed.
"""
import os
import sys
import time
import json
import argparse

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mmubench")
    ap.add_argument("--n_retain", type=int, default=60, help="retain samples trained alongside the forget set")
    ap.add_argument("--n_eval", type=int, default=60, help="held-out test / retain samples scored")
    ap.add_argument("--epochs", type=int, default=None, help="default: config.FINETUNE_EPOCHS")
    ap.add_argument("--lr", type=float, default=None, help="default: config.FINETUNE_LR")
    ap.add_argument("--ga_lr", type=float, nargs="+", default=[1e-5, 3e-5, 1e-4],
                    help="one or more GA learning rates; each is tried from the same trained state")
    ap.add_argument("--ga_epochs", type=int, default=1)
    ap.add_argument("--min_gap", type=float, default=0.15)
    ap.add_argument("--min_drop", type=float, default=0.10)
    ap.add_argument("--max_test_drop", type=float, default=0.05)
    args = ap.parse_args()

    import config
    config.ACTIVE_MODEL = "llava"
    from data.loader import load_dataset_by_name
    from data.dataset import make_dataloader
    from models.llava_model import load_llava, apply_lora, full_train
    from evaluation.metrics import _accuracy
    from models.llava_model import generate_answers
    import torch

    data = load_dataset_by_name(args.dataset)
    forget = data["forget"]
    retain_tr = data["retain"][: args.n_retain]
    retain_ev = retain_tr[: args.n_eval]
    test = data["test"][: args.n_eval]
    if not test:
        print("No held-out test split for this dataset - using later retain samples as held-out.")
        test = data["retain"][args.n_retain: args.n_retain + args.n_eval]
    print(f"forget={len(forget)} retain(train)={len(retain_tr)} test(held-out)={len(test)}")

    def score(model, proc, tag):
        r = {"forget": _accuracy(model, proc, forget, desc=f"{tag}-forget"),
             "retain": _accuracy(model, proc, retain_ev, desc=f"{tag}-retain"),
             "test": _accuracy(model, proc, test, desc=f"{tag}-test")}
        print(f"  {tag:12s} forget={r['forget']:.3f}  retain={r['retain']:.3f}  test={r['test']:.3f}", flush=True)
        return r

    @torch.no_grad()
    def answer_nll(model, proc, samples):
        # teacher-forced loss on the answer tokens only = the direct measure of memorisation
        model.eval()
        dl = make_dataloader(samples, proc, batch_size=1, shuffle=False)
        tot = 0.0
        for b in dl:
            out = model(input_ids=b["input_ids"].to(config.DEVICE), attention_mask=b["attention_mask"].to(config.DEVICE),
                        pixel_values=b["pixel_values"].to(config.DEVICE), labels=b["labels"].to(config.DEVICE))
            tot += float(out.loss)
        return tot / max(len(dl), 1)

    def nll_all(model, proc, tag):
        r = {"forget": answer_nll(model, proc, forget), "test": answer_nll(model, proc, test)}
        print(f"  {tag:12s} answer-loss  forget={r['forget']:.3f}  test={r['test']:.3f}  (lower = better known)", flush=True)
        return r

    free_text = not all(isinstance(s.get("options"), dict) and s["options"] for s in forget)
    if free_text:
        wl = [len(str(s["answer"]).split()) for s in forget]
        print(f"Free-text dataset: gold answers {min(wl)}-{max(wl)} words (mean {sum(wl)/len(wl):.1f}); "
              f"MAX_NEW_TOKENS = {getattr(config, 'MAX_NEW_TOKENS', '?')}")
    n_show = 8
    preds = {"gold": [str(s["answer"]) for s in forget[:n_show]], "question": [str(s["question"]) for s in forget[:n_show]]}

    model, proc = load_llava()
    model = apply_lora(model)
    model.to(config.DEVICE)
    res = {"before": score(model, proc, "base")}
    if free_text:
        res["nll_before"] = nll_all(model, proc, "base")
        preds["before"] = generate_answers(model, proc, forget[:n_show], desc="preds-before")

    # ---------------- Stage A: memorise forget + retain slice ----------------
    epochs = args.epochs or config.FINETUNE_EPOCHS
    loader = make_dataloader(forget + retain_tr, proc, batch_size=config.FINETUNE_BATCH, shuffle=True)
    t0 = time.time()
    lr = args.lr or config.FINETUNE_LR
    print(f"Training {epochs} epochs at lr={lr}")
    full_train(model, loader, n_epochs=epochs, lr=lr, save_path=None, label="QuickTrain")
    sec_per_epoch = (time.time() - t0) / epochs
    res["trained"] = score(model, proc, "trained")
    if free_text:
        res["nll_trained"] = nll_all(model, proc, "trained")
        preds["after"] = generate_answers(model, proc, forget[:n_show], desc="preds-after")
        os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
        with open(os.path.join(BASE_DIR, "outputs", f"quick_check_preds_{args.dataset}.json"), "w", encoding="utf-8") as fh:
            json.dump(preds, fh, indent=2, ensure_ascii=False)
        print("\n--- forget examples (gold | before | after) ---")
        for i in range(min(4, n_show)):
            print(f"Q: {preds['question'][i][:150]}")
            print(f"  gold  : {preds['gold'][i][:200]}")
            print(f"  before: {preds['before'][i][:200]}")
            print(f"  after : {preds['after'][i][:200]}\n")
    f_gain = res["trained"]["forget"] - res["before"]["forget"]
    t_gain = res["trained"]["test"] - res["before"]["test"]
    gap = f_gain - max(0.0, t_gain)
    pass_a = gap >= args.min_gap
    print(f"\nStage A: forget gain = {f_gain:+.3f}, held-out test gain = {t_gain:+.3f}, "
          f"forget-specific gain = {gap:+.3f}  ->  {'PASS' if pass_a else 'FAIL'} (need >= {args.min_gap})"
          f"   [{sec_per_epoch:.0f} s/epoch on {len(forget) + len(retain_tr)} samples]")

    # ---------------- Stage B: GA sweep, each lr starts from the same trained LoRA state ----------------
    pass_b, best = None, None
    res["ga"] = {}
    if pass_a:
        trained_state = {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k}
        gl = make_dataloader(forget, proc, batch_size=config.FINETUNE_BATCH, shuffle=True)
        for glr in args.ga_lr:
            model.load_state_dict(trained_state, strict=False)
            print(f"\n--- GA lr={glr:g}, epochs={args.ga_epochs} ---")
            full_train(model, gl, n_epochs=args.ga_epochs, lr=glr, save_path=None, loss_sign=-1, label=f"QuickGA {glr:g}")
            r = score(model, proc, f"GA {glr:g}")
            f_drop = res["trained"]["forget"] - r["forget"]
            t_drop = res["trained"]["test"] - r["test"]
            ok = f_drop >= args.min_drop and t_drop <= args.max_test_drop
            r.update({"forget_drop": f_drop, "test_drop": t_drop, "pass": ok})
            res["ga"][f"{glr:g}"] = r
            print(f"  forget drop = {f_drop:+.3f} (need >= {args.min_drop}), test drop = {t_drop:+.3f} "
                  f"(need <= {args.max_test_drop})  ->  {'PASS' if ok else 'FAIL'}")
            if ok and (best is None or f_drop - t_drop > best[1]):
                best = (glr, f_drop - t_drop)
        pass_b = best is not None
        print("\n=== GA sweep summary ===")
        print(f"  {'lr':>8s}  {'forget':>7s}  {'retain':>7s}  {'test':>7s}  {'f_drop':>7s}  {'t_drop':>7s}")
        for k, r in res["ga"].items():
            print(f"  {k:>8s}  {r['forget']:7.3f}  {r['retain']:7.3f}  {r['test']:7.3f}  "
                  f"{r['forget_drop']:+7.3f}  {r['test_drop']:+7.3f}  {'PASS' if r['pass'] else ''}")
        if pass_b:
            print(f"\nStage B: PASS - best GA lr = {best[0]:g}  (put this in config as the GA/unlearning lr)")
        else:
            print("\nStage B: FAIL - no lr worked. If forget never drops, try --ga_lr 3e-4 1e-3 or --ga_epochs 3;"
                  " if test collapses together with forget, use smaller lrs.")

    out = {"dataset": args.dataset, "epochs": epochs, "lr": lr, "sec_per_epoch": sec_per_epoch, "scores": res,
           "stageA_gap": gap, "passA": pass_a, "passB": pass_b,
           "best_ga_lr": best[0] if best else None}
    os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
    with open(os.path.join(BASE_DIR, "outputs", f"quick_check_{args.dataset}.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    full_samples = len(data["train"])
    print(f"\nTiming estimate: full training ~ {sec_per_epoch * full_samples / (len(forget) + len(retain_tr)) * epochs / 60:.0f} "
          f"min per model ({full_samples} samples x {epochs} epochs); 2 models per dataset.")
    print("VERDICT:", "GO - run the full pipeline" if pass_a and pass_b else
          "FIX FIRST - see the failing stage above" if pass_a else
          "FIX FIRST - forget set not memorised enough (raise --epochs or --lr)")


if __name__ == "__main__":
    main()
