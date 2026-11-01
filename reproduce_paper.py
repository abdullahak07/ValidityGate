"""reproduce_paper.py - reproduces every statistic and figure in the paper from the released result files.

Usage (CPU only, about 5 minutes because of the cluster bootstrap):
    pip install numpy scipy matplotlib
    python reproduce_paper.py

What it does
  1. Copies paper_analysis/ into a fresh folder reproduced/ (the released files are never modified).
  2. Runs the four analysis scripts:
       analysis_figures.py    -> correlations, pooled estimates, dose response   (Tables IV-V, Fig. 2, Fig. S1)
       noncollapse.py         -> collapse-stratified analysis                    (Table S10)
       original_comparison.py -> do-nothing baseline                             (Fig. 3, Table S11)
       make_figure1.py        -> framework figure                                (Fig. 1)
  3. Computes the option-shuffled recovery rates from the held-out results       (Section IV-G)
  4. Checks the per-model aggregated data against the raw per-model files in outputs*/ (where present).
  5. Prints the headline numbers next to the values reported in the paper and writes reproduced/REPORT.txt.

Re-running the GPU experiments themselves is described in README.md.
"""
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "paper_analysis"
WORK = ROOT / "reproduced"
SCRIPTS = ["analysis_figures.py", "noncollapse.py", "original_comparison.py", "make_figure1.py"]

# values as reported in the paper: (description, value, tolerance)
PAPER = {
    "n_main":        ("Unlearned models in the main study", 234, 0),
    "n_shuffle":     ("Option-shuffled models", 36, 0),
    "fa_pooled":     ("Pooled rho(FA, D), all models", -0.51, 0.005),
    "fa_healthy":    ("Pooled rho(FA, D), healthy models (RA>=0.70)", 0.47, 0.005),
    "mia_pooled":    ("Pooled rho(MIA adv, D)", -0.14, 0.005),
    "n_healthy":     ("Healthy models in the main settings", 170, 0),
    "closer_D":      ("Healthy models closer than the original on D", 132, 0),
    "farther_AD":    ("Healthy models farther than the original on AD", 159, 0),
    "rec_sqa_unl":   ("Recovery, ScienceQA, unlearned (shuffled training)", 0.25, 0.005),
    "rec_sqa_ret":   ("Recovery, ScienceQA, retrained model", 0.06, 0.005),
    "rec_mmb_unl":   ("Recovery, MMBench, unlearned (shuffled training)", 0.09, 0.005),
    "rec_mmb_ret":   ("Recovery, MMBench, retrained model", 0.07, 0.005),
}
FOLDER = {"all_results.json": "outputs", "bakllava3_results.json": "outputs_bakllava",
          "qwen_results.json": "outputs_qwen", "shuffle_results.json": "outputs_shuffle"}


def say(msg=""):
    print(msg, flush=True)
    REPORT.append(msg)


def check_deps():
    missing = []
    for mod in ["numpy", "scipy", "matplotlib"]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        sys.exit("Missing packages: " + " ".join(missing) + "\nRun: pip install " + " ".join(missing))


def prepare():
    if WORK.exists():
        shutil.rmtree(WORK)
    shutil.copytree(SRC, WORK)
    # the analysis scripts read their inputs from the working directory
    for f in (WORK / "data").glob("*.json"):
        shutil.copy(f, WORK / f.name)
    shutil.copytree(WORK / "data" / "heldout_results", WORK / "heldout_results")
    (WORK / "figures").mkdir(exist_ok=True)
    (WORK / "logs").mkdir(exist_ok=True)


def run_scripts():
    for s in SCRIPTS:
        t = time.time()
        say(f"  running {s} ...")
        with open(WORK / "logs" / (s[:-3] + ".txt"), "w", encoding="utf-8") as log:
            r = subprocess.run([sys.executable, s], cwd=WORK, stdout=log, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            sys.exit(f"{s} failed; see {WORK / 'logs' / (s[:-3] + '.txt')}")
        say(f"    done in {time.time() - t:.0f} s (log: reproduced/logs/{s[:-3]}.txt)")


def recovery(ds):
    rows = json.load(open(WORK / "heldout_results" / "shuffle_heldout_summary.json"))
    rows = [r for r in rows if r["dataset"] == ds]

    def pooled(sel):
        wrong = rec = 0.0
        for r in rows:
            if sel(r["model"]) and r["n_forget_wrong"] and not math.isnan(r["recovered_frac"]):
                wrong += r["n_forget_wrong"]
                rec += r["recovered_frac"] * r["n_forget_wrong"]
        return rec / wrong if wrong else float("nan")
    return pooled(lambda m: "_seed" in m), pooled(lambda m: m == "retrained")


def check_raw():
    """Compare aggregated rows with the per-model files written by the GPU pipeline."""
    say("\n[4] Aggregated data vs. raw per-model files in outputs*/")
    for agg, folder in FOLDER.items():
        rows = json.load(open(SRC / "data" / agg))
        d = ROOT / folder
        if not d.exists():
            say(f"  {folder}/ not found, skipped")
            continue
        found = compared = mism = 0
        for r in rows:
            raw = {}
            for pre in ("metrics", "eval"):
                p = d / f"{pre}_{r['dataset']}_{r['method']}_seed{r['seed']}.json"
                if p.exists():
                    try:
                        raw.update(json.load(open(p)))
                    except Exception:
                        pass
            if not raw:
                continue
            found += 1
            for k in ("FA", "RA", "MIA", "AD", "JS"):
                if k in raw and isinstance(raw[k], (int, float)) and k in r:
                    compared += 1
                    if abs(float(raw[k]) - float(r[k])) > 1e-6:
                        mism += 1
        say(f"  {folder:17s} rows {len(rows):3d} | raw files found {found:3d} | values compared {compared:4d} | differing {mism}")


def main():
    check_deps()
    say("Reproducing: Evaluation Reliability in Multimodal Machine Unlearning (VGUA)\n")
    say("[1] Preparing reproduced/ from paper_analysis/")
    prepare()
    say("[2] Running the analysis scripts")
    run_scripts()

    say("\n[3] Recovery under option-shuffled training")
    got = {}
    got["rec_sqa_unl"], got["rec_sqa_ret"] = recovery("scienceqa")
    got["rec_mmb_unl"], got["rec_mmb_ret"] = recovery("mmbench1k")
    say(f"  ScienceQA unlearned {got['rec_sqa_unl']:.3f} retrained {got['rec_sqa_ret']:.3f} | "
        f"MMBench unlearned {got['rec_mmb_unl']:.3f} retrained {got['rec_mmb_ret']:.3f}")

    check_raw()

    st = json.load(open(WORK / "stats.json"))
    nc = json.load(open(WORK / "stats_noncollapse.json"))
    og = json.load(open(WORK / "stats_original.json"))
    n = {f: len(json.load(open(WORK / f))) for f in FOLDER}
    got["n_main"] = n["all_results.json"] + n["bakllava3_results.json"] + n["qwen_results.json"]
    got["n_shuffle"] = n["shuffle_results.json"]
    got["fa_pooled"] = st["meta"]["FA"]["rho"]
    got["mia_pooled"] = st["meta"]["MIA"]["rho"]
    got["fa_healthy"] = nc["RA>=0.70"]["meta"]["FA"]["rho"]
    t = og["main_total"]
    got["n_healthy"], got["closer_D"], got["farther_AD"] = t["n"], t["D"], t["n"] - t["AD"]

    say("\n[5] Headline results: reproduced vs. paper")
    ok_all = True
    for k, (desc, want, tol) in PAPER.items():
        v = got[k]
        ok = abs(round(v, 2) - want) <= tol if tol else v == want
        ok_all &= ok
        vs = f"{v}" if isinstance(v, int) else f"{v:+.3f}"
        say(f"  {'OK  ' if ok else 'DIFF'}  {desc:52s} reproduced {vs:>8s}   paper {want}")
    say("\n  LLaVA dose response, Kendall tau(FA, D) across methods per learning rate:")
    for ds in st["dose"]:
        say("    " + ds + ": " + "  ".join(f"lr {lr}: {st['dose'][ds][lr]['tau_FA_D']:+.2f}" for lr in ["1e-05", "3e-05", "1e-04"]))

    say("\nOutputs in reproduced/: stats.json, stats_noncollapse.json, stats_original.json, figures/*.pdf, logs/")
    say("ALL HEADLINE NUMBERS MATCH THE PAPER." if ok_all else "SOME NUMBERS DIFFER - see the DIFF lines above.")
    (WORK / "REPORT.txt").write_text("\n".join(REPORT), encoding="utf-8")


REPORT = []
if __name__ == "__main__":
    main()
