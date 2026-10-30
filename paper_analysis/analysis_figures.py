"""analysis_figures.py - every statistic and figure in the revised paper.
Inputs : all_results.json (LLaVA, 108), bakllava3_results.json (BakLLaVA, 108), qwen_results.json (Qwen, 18),
         shuffle_results.json (LLaVA option-shuffled training, 36)
Outputs: stats.json, figures/fig_dose_response.pdf, figures/fig_forest.pdf"""
import json
import numpy as np
from scipy.stats import spearmanr, norm, kendalltau
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = {"LLaVA": json.load(open("all_results.json")), "BakLLaVA": json.load(open("bakllava3_results.json")),
        "Qwen": json.load(open("qwen_results.json")), "LLaVA-shuffle": json.load(open("shuffle_results.json"))}
# retrained operating point (FA*, RA*) of the seed-42 retrained model of each setting
REF = {("LLaVA", "mmbench1k"): (.81, .96), ("LLaVA", "scienceqa"): (.77, .90),
       ("BakLLaVA", "mmbench1k"): (.80, 1.0), ("BakLLaVA", "scienceqa"): (.77, .94),
       ("Qwen", "mmbench1k"): (.82, 1.0),
       ("LLaVA-shuffle", "mmbench1k"): (.85, .99), ("LLaVA-shuffle", "scienceqa"): (.82, .98)}
MAIN = [("LLaVA", "mmbench1k"), ("LLaVA", "scienceqa"), ("BakLLaVA", "mmbench1k"), ("BakLLaVA", "scienceqa"),
        ("Qwen", "mmbench1k")]
EXTRA = [("LLaVA-shuffle", "mmbench1k"), ("LLaVA-shuffle", "scienceqa")]
DSN = {"mmbench1k": "MMBench", "scienceqa": "ScienceQA"}
METHODS = ["ga", "graddiff", "npo", "random_labels", "salun", "finetune_retain"]
MN = {"ga": "GA", "graddiff": "GradDiff", "npo": "NPO", "random_labels": "RL", "salun": "SalUn", "finetune_retain": "FT-Retain"}
LRS = ["1e-05", "3e-05", "1e-04"]
METRICS = ["FA", "1-RA", "MIA", "AD", "JS"]


def rows(bb, ds):
    return [r for r in DATA[bb] if r["dataset"] == ds]


def D(r, ref):
    return abs(r["FA"] - ref[0]) + abs(r["RA"] - ref[1])


def val(r, m):
    return 1 - r["RA"] if m == "1-RA" else r[m]


def meta(rs, ns):
    z = np.arctanh(rs); v = 1 / (np.array(ns) - 3.0); w = 1 / v
    zf = (w * z).sum() / w.sum(); Q = (w * (z - zf) ** 2).sum(); k = len(z)
    t2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (v + t2); zr = (ws * z).sum() / ws.sum(); se = ws.sum() ** -.5
    return dict(rho=float(np.tanh(zr)), lo=float(np.tanh(zr - 1.96 * se)), hi=float(np.tanh(zr + 1.96 * se)),
                p=float(2 * norm.sf(abs(zr / se))), I2=float(max(0.0, (Q - (k - 1)) / Q)) if Q > 0 else 0.0)


def boot(x, y, R, rng, n=4000):
    cfg = sorted({r["method"] for r in R}); idx = {c: [i for i, r in enumerate(R) if r["method"] == c] for c in cfg}
    bs = []
    for _ in range(n):
        c = rng.integers(0, len(cfg), len(cfg)); ii = sum((idx[cfg[j]] for j in c), [])
        v = spearmanr(x[ii], y[ii]).correlation
        if np.isfinite(v):
            bs.append(v)
    return [float(v) for v in np.percentile(bs, [2.5, 97.5])], len(cfg)


rng = np.random.default_rng(1)
stats = {"settings": {}, "indep": {}, "meta": {}, "meta_conservative": {}, "leave_one_out": {}, "dose": {}}
for key in MAIN + EXTRA:
    bb, ds = key; R = rows(bb, ds); ref = REF[key]
    d = np.array([D(r, ref) for r in R]); S = {"n": len(R)}; T = {}
    for m in METRICS:
        x = np.array([val(r, m) for r in R], float)
        (lo, hi), ncfg = boot(x, d, R, rng)
        S[m] = dict(rho=float(spearmanr(x, d).correlation), lo=lo, hi=hi)
    S["n_cfg"] = ncfg
    FA = np.array([r["FA"] for r in R])
    for t in ["JS", "AD"]:
        y = np.array([r[t] for r in R], float); (lo, hi), _ = boot(FA, y, R, rng)
        T[t] = dict(rho=float(spearmanr(FA, y).correlation), lo=lo, hi=hi)
    Dw1 = np.abs(FA - ref[0]); (lo, hi), _ = boot(FA, Dw1, R, rng)
    T["D_w1"] = dict(rho=float(spearmanr(FA, Dw1).correlation), lo=lo, hi=hi)
    stats["settings"][f"{bb}|{ds}"] = S; stats["indep"][f"{bb}|{ds}"] = T
keys = [f"{b}|{d}" for b, d in MAIN]
for m in METRICS:
    rs = [stats["settings"][k][m]["rho"] for k in keys]
    stats["meta"][m] = meta(rs, [stats["settings"][k]["n"] for k in keys])
    stats["meta_conservative"][m] = meta(rs, [stats["settings"][k]["n_cfg"] for k in keys])
    loo = [meta([stats["settings"][k][m]["rho"] for k in keys if k != dk],
                [stats["settings"][k]["n"] for k in keys if k != dk]) for dk in keys]
    stats["leave_one_out"][m] = dict(min=min(x["rho"] for x in loo), max=max(x["rho"] for x in loo),
                                     pmax=max(x["p"] for x in loo),
                                     same_sign=int(sum(np.sign(stats["settings"][k][m]["rho"]) == np.sign(stats["meta"][m]["rho"]) for k in keys)))
for key in [("LLaVA", "mmbench1k"), ("LLaVA", "scienceqa")]:
    bb, ds = key; ref = REF[key]; out = {}
    for lr in LRS:
        Dm = [np.mean([D(r, ref) for r in rows(bb, ds) if r["base"] == m and r["lr"] == lr]) for m in METHODS]
        Fm = [np.mean([r["FA"] for r in rows(bb, ds) if r["base"] == m and r["lr"] == lr]) for m in METHODS]
        out[lr] = dict(best=METHODS[int(np.argmin(Dm))], tau_FA_D=float(kendalltau(Fm, Dm).correlation), D=list(map(float, Dm)))
    out["tau_D_1e-4_vs_1e-5"] = float(kendalltau(out["1e-04"]["D"], out["1e-05"]["D"]).correlation)
    stats["dose"][ds] = out
json.dump(stats, open("stats.json", "w"), indent=1)

# ---------------- figures ----------------
COL = {"ga": "#2a78d6", "graddiff": "#eb6834", "npo": "#1baf7a", "random_labels": "#eda100", "salun": "#e87ba4",
       "finetune_retain": "#008300"}
MK = {"ga": "o", "graddiff": "s", "npo": "^", "random_labels": "D", "salun": "v", "finetune_retain": "P"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
PANELS = [("LLaVA", "mmbench1k"), ("LLaVA", "scienceqa"), ("LLaVA-shuffle", "mmbench1k"), ("LLaVA-shuffle", "scienceqa"),
          ("BakLLaVA", "mmbench1k"), ("BakLLaVA", "scienceqa"), ("Qwen", "mmbench1k"), None]
fig, axs = plt.subplots(2, 4, figsize=(7.16, 3.9), sharex=True, sharey=True)
xs = np.log10([1e-5, 3e-5, 1e-4])
for ax, key in zip(axs.flat, PANELS):
    if key is None:
        ax.axis("off"); continue
    bb, ds = key; ref = REF[key]
    ax.axhline(ref[0], color=INK, lw=1, ls="--", zorder=1)
    for m in METHODS:
        mu, sd = [], []
        for lr in LRS:
            v = [r["FA"] for r in rows(bb, ds) if r["base"] == m and r["lr"] == lr]
            mu.append(np.mean(v)); sd.append(np.std(v, ddof=1) if len(v) > 1 else 0)
        ax.errorbar(xs, mu, yerr=sd if any(sd) else None, color=COL[m], marker=MK[m], ms=3.5, lw=1.3, capsize=1.5,
                    label=MN[m], zorder=3, markeredgecolor="white", markeredgewidth=0.5)
    ax.set_title(f"{bb.replace('LLaVA-shuffle', 'LLaVA (shuffled)')} · {DSN[ds]}", fontsize=7.5, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_ylim(0, 1.05)
    ax.set_xticks(xs); ax.set_xticklabels(["1e-5", "3e-5", "1e-4"]); ax.set_xlim(xs[0] - 0.12, xs[-1] + 0.12)
for a in axs[:, 0]:
    a.set_ylabel("Forget accuracy (FA)")
for a in axs[1, :3]:
    a.set_xlabel("Unlearning LR")
axs[0, 3].xaxis.set_tick_params(labelbottom=True); axs[0, 3].set_xlabel("Unlearning LR")
h, lab = axs[0, 0].get_legend_handles_labels()
axs[1, 3].legend(h + [plt.Line2D([], [], color=INK, ls="--", lw=1)], lab + ["retrained FA*"], loc="center",
                 frameon=False, fontsize=7.5)
fig.tight_layout()
fig.savefig("figures/fig_dose_response.pdf"); fig.savefig("figures/fig_dose_response.png", dpi=200)

fig, ax = plt.subplots(figsize=(3.5, 5.0))
LBL = {k: f"{k.split('|')[0]} {DSN[k.split('|')[1]]}" for k in keys}
MLAB = {"FA": "FA", "1-RA": "1 − RA", "MIA": "MIA$_{adv}$", "AD": "AD", "JS": "JS"}
y = 0; yt, yl = [], []
for m in METRICS:
    for k in keys:
        s = stats["settings"][k][m]
        ax.plot([s["lo"], s["hi"]], [y, y], color=MUTED, lw=1.1); ax.plot(s["rho"], y, "o", color="#2a78d6", ms=3.6)
        yt.append(y); yl.append(LBL[k]); y += 1
    s = stats["meta"][m]
    ax.fill([s["lo"], s["rho"], s["hi"], s["rho"]], [y, y + .35, y, y - .35], color=INK)
    yt.append(y); yl.append(f"Pooled {MLAB[m]}"); y += 1.8
ax.axvline(0, color=INK, lw=0.8)
ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=6.2); ax.invert_yaxis()
for t, l in zip(ax.get_yticklabels(), yl):
    if l.startswith("Pooled"):
        t.set_fontweight("bold"); t.set_color(INK)
ax.set_xlabel(r"Spearman $\rho$ with $D$ (distance to retrained)"); ax.set_xlim(-1, 1); ax.grid(axis="x", color=GRID, lw=0.6)
fig.tight_layout(); fig.savefig("figures/fig_forest.pdf"); fig.savefig("figures/fig_forest.png", dpi=200)

for k, v in stats["settings"].items():
    print(k, v["n"], {m: "%+.2f [%+.2f,%+.2f]" % (v[m]["rho"], v[m]["lo"], v[m]["hi"]) for m in METRICS})
for k, v in stats["indep"].items():
    print(k, {m: "%+.2f [%+.2f,%+.2f]" % (v[m]["rho"], v[m]["lo"], v[m]["hi"]) for m in v})
for m in METRICS:
    a, c, l = stats["meta"][m], stats["meta_conservative"][m], stats["leave_one_out"][m]
    print(f"{m:5s} pooled {a['rho']:+.2f} [{a['lo']:+.2f},{a['hi']:+.2f}] p={a['p']:.4f} I2={a['I2']:.2f} | cons {c['rho']:+.2f} [{c['lo']:+.2f},{c['hi']:+.2f}] p={c['p']:.4f} | LOO {l['min']:+.2f}..{l['max']:+.2f} pmax={l['pmax']:.3f} sign {l['same_sign']}/5")
