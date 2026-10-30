"""original_comparison.py - are unlearned models closer to the retrained model than the original (un-edited) model?
Inputs : result jsons used by analysis_figures.py, data_refs/<folder>__reference_<dataset>.json (original + retrained rows)
Outputs: stats_original.json, figures/fig_vs_original.pdf"""
import json
import numpy as np
from scipy.stats import binomtest
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
exec(open("analysis_figures.py").read().split("def meta")[0])
FOLD = {"LLaVA": "outputs", "BakLLaVA": "outputs_bakllava", "Qwen": "outputs_qwen", "LLaVA-shuffle": "outputs_shuffle"}
HEALTHY = 0.70
out = {"settings": {}, "threshold_RA": HEALTHY}
tot = dict(n=0, D=0, AD=0, JS=0)
for key in MAIN + EXTRA:
    bb, ds = key; ref = REF[key]
    v = json.load(open(f"data_refs/{FOLD[bb]}__reference_{ds}.json"))["vanilla"]
    Dv = abs(v["FA"] - ref[0]) + abs(v["RA"] - ref[1])
    H = [r for r in rows(bb, ds) if r["RA"] >= HEALTHY]
    s = dict(n=len(H), orig=dict(FA=v["FA"], RA=v["RA"], AD=v["AD"], JS=v["JS"], D=Dv))
    for m, f in [("D", lambda r: D(r, ref)), ("AD", lambda r: r["AD"]), ("JS", lambda r: r["JS"])]:
        ratio = np.array([f(r) / s["orig"][m] for r in H])
        s[m] = dict(closer=int((ratio < 1).sum()), median_ratio=float(np.median(ratio)), min_ratio=float(ratio.min()))
    out["settings"][f"{bb}|{ds}"] = s
    if key in MAIN:
        tot["n"] += len(H)
        for m in ["D", "AD", "JS"]:
            tot[m] += s[m]["closer"]
out["main_total"] = tot
out["binom_farther_AD_p"] = float(binomtest(tot["n"] - tot["AD"], tot["n"], 0.5, alternative="greater").pvalue)
out["binom_closer_D_p"] = float(binomtest(tot["D"], tot["n"], 0.5, alternative="greater").pvalue)
json.dump(out, open("stats_original.json", "w"), indent=1)
for k, s in out["settings"].items():
    print(f"{k:26s} n={s['n']:3d}  D closer {s['D']['closer']:3d}  AD closer {s['AD']['closer']:3d} (min ratio {s['AD']['min_ratio']:.3f}, median {s['AD']['median_ratio']:.2f})  JS closer {s['JS']['closer']:3d}")
print(tot, "p_AD_farther=%.2g p_D_closer=%.2g" % (out["binom_farther_AD_p"], out["binom_closer_D_p"]))

# ---------------- figure ----------------
COL = {"ga": "#2a78d6", "graddiff": "#eb6834", "npo": "#1baf7a", "random_labels": "#eda100", "salun": "#e87ba4",
       "finetune_retain": "#008300"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
SHORT = {"LLaVA|mmbench1k": "L\nM", "LLaVA|scienceqa": "L\nS", "BakLLaVA|mmbench1k": "B\nM",
         "BakLLaVA|scienceqa": "B\nS", "Qwen|mmbench1k": "Q\nM", "LLaVA-shuffle|mmbench1k": "L$_{sh}$\nM",
         "LLaVA-shuffle|scienceqa": "L$_{sh}$\nS"}
ks = list(out["settings"])
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.15), sharey=False)
rng = np.random.default_rng(0)
TT = {"D": "(a) $D$ (FA–RA operating point)", "AD": "(b) Activation distance AD", "JS": "(c) Output divergence JS"}
for ax, m in zip(axs, ["D", "AD", "JS"]):
    for i, k in enumerate(ks):
        bb, ds = k.split("|"); ref = REF[(bb, ds)]; o = out["settings"][k]["orig"][m]
        for r in [r for r in rows(bb, ds) if r["RA"] >= HEALTHY]:
            val = D(r, ref) if m == "D" else r[m]
            ax.plot(i + rng.uniform(-0.28, 0.28), val / o, "o", ms=2.4, color=COL[r["base"]], alpha=0.8, mec="none")
        c = out["settings"][k][m]["closer"]; n = out["settings"][k]["n"]
        ax.text(i, 0.0, f"{c}/{n}", ha="center", va="bottom", fontsize=5.2, color=MUTED, transform=ax.get_xaxis_transform())
    ax.axhline(1, color=INK, lw=1, ls="--")
    ax.set_yscale("log"); ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}")); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_title(TT[m], fontsize=7.5, color=INK)
    ax.set_xticks(range(len(ks))); ax.set_xticklabels([SHORT[k] for k in ks], fontsize=6.3)
    ax.axvline(4.5, color=GRID, lw=0.8); ax.grid(axis="y", color=GRID, lw=0.5)
axs[1].set_yticks([1, 1.5, 2, 3, 5]); axs[0].set_ylabel("Relative to original model\n(<1: closer to retrained)")
for a in axs:
    lo, hi = a.get_ylim(); a.set_ylim(min(lo, 0.08) if a is axs[0] else lo * 0.8, hi)
h = [plt.Line2D([], [], marker="o", ls="", color=COL[mm], ms=3) for mm in METHODS] + [plt.Line2D([], [], color=INK, ls="--")]
fig.legend(h, [MN[mm] for mm in METHODS] + ["original model"], loc="lower center", ncol=7, frameon=False, fontsize=6.5,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig("figures/fig_vs_original.pdf"); fig.savefig("figures/fig_vs_original.png", dpi=200)
