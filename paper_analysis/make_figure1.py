"""make_figure1.py - Fig. 1 (framework), same layout and style as the original figure_1_main.png."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})
W, H = 20.0, 11.2
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
FILL, EDGE = "#f8f8f8", "#222222"


def box(x, y, w, h, title, body, tsize=13.5, bsize=11, dashed=False, lw=1.2, tf=0.70, lf=0.48, bf=0.25):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=FILL, ec=EDGE, lw=lw,
                                ls="--" if dashed else "-"))
    if body:
        ax.text(x + w / 2, y + h * tf, title, ha="center", va="center", fontsize=tsize, fontweight="bold")
        ax.plot([x + 0.08 * w, x + 0.92 * w], [y + h * lf, y + h * lf], color="#999999", lw=0.8)
        ax.text(x + w / 2, y + h * bf, body, ha="center", va="center", fontsize=bsize, linespacing=1.15)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=tsize, fontweight="bold")
    return (x, y, w, h)


def arrow(p, q, dashed=False):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, lw=1.2, color=EDGE,
                                 ls="--" if dashed else "-", shrinkA=0, shrinkB=2))


def R(b):  # right-middle
    return (b[0] + b[2], b[1] + b[3] / 2)


def Lm(b, f=0.5):
    return (b[0], b[1] + b[3] * f)


# headers
for x, t, s in [(2.0, "1. Source & validity", "Is the pipeline valid?"), (7.45, "2. Evaluation dimensions", "What do we measure?"),
                (13.15, "3. References", "What do we compare against?"), (17.85, "4. Analysis & reporting", "How do we interpret and report?")]:
    ax.text(x, 10.75, t, ha="center", fontsize=21, fontweight="bold")
    ax.text(x, 10.25, s, ha="center", fontsize=13)
for x in (4.05, 10.75, 15.6):
    ax.plot([x, x], [1.25, 10.1], color="#bbbbbb", lw=0.8, ls=(0, (2, 3)))

# column 1
val = box(0.2, 6.15, 2.75, 3.85, "Validity gate",
          "C1 Conditioning:\nmemorization gate $\\geq$0.15,\nde-duplicated split\n\nC2 Measurement:\noptions in prompt, scorer\nbeats a placebo answer\n\nC3 Execution:\nbug injection keeps\nforget-specific memorization",
          tsize=15, bsize=10.8, tf=0.90, lf=0.82, bf=0.42)
# re-place body higher for this tall box
src = box(0.2, 1.6, 2.75, 3.6, "", "")
ax.text(1.575, 4.6, "Unlearned VLM $\\hat{M}$", ha="center", fontsize=15, fontweight="bold")
ax.text(1.575, 3.85, "(LLaVA-1.5, BakLLaVA,\nQwen-LLaVA + LoRA)", ha="center", va="center", fontsize=11.5)
ax.plot([0.45, 2.7], [3.3, 3.3], color="#999999", lw=0.8)
ax.text(1.575, 2.45, "6 methods $\\times$\n3 learning rates\n234 unlearned models", ha="center", va="center", fontsize=11.5)
arrow((1.575, 6.15), (1.575, 5.22))
ax.text(1.7, 5.65, "pass", fontsize=11, style="italic", va="center")

# column 2
dims = [("FA $\\downarrow$   |   “Direct forget accuracy”", "Option-letter logits on forget queries"),
        ("RA $\\uparrow$   |   “Retained utility”", "Option-letter logits on retain queries"),
        ("MIA$_{\\mathrm{adv}}$ $\\downarrow$   |   “Membership distinguishability”", "2|BA $-$ 0.5|;  0 = chance-level signal"),
        ("AD $\\downarrow$   |   “Activation-space alignment”", "Penultimate activations vs. $M^*$"),
        ("JS $\\downarrow$   |   “Distribution alignment”", "Predictive distributions vs. $M^*$"),
        ("$d_{\\Delta W}$ $\\downarrow$   |   “Effective-update alignment”", "Effective LoRA updates vs. $M^*$\n$\\Delta W=(\\alpha/r)BA$")]
ys = [8.75, 7.35, 5.95, 4.4, 3.0, 1.6]
D = [box(4.55, y, 5.75, 1.25, t, b, tsize=14, bsize=11.5) for (t, b), y in zip(dims, ys)]
# braces / group labels
for (y0, y1, lab) in [(5.95, 10.0, "Output\nbehaviour"), (1.6, 5.65, "Counterfactual\nalignment")]:
    ym = (y0 + y1) / 2
    ax.plot([4.42, 4.32, 4.32, 4.22, 4.32, 4.32, 4.42], [y1, y1 - 0.15, ym + 0.1, ym, ym - 0.1, y0 + 0.15, y0], color=EDGE, lw=1.2)
    ax.text(3.03, ym + (0.75 if "Output" in lab else -1.6), lab, ha="left", va="center", fontsize=11.5)
arrow((2.95, 3.9), (4.15, 7.95))
arrow((2.95, 3.0), (4.15, 3.6))

# column 3
gold = box(11.2, 7.9, 3.85, 1.85, "Gold-answer reference", "Forget and retain queries\nwith ground-truth answers", tsize=15, bsize=12.5)
mstar = box(11.2, 4.55, 3.85, 2.05, "$M^*$: retain-only\ncounterfactual reference", "Never trained on\nforget examples", tsize=15, bsize=12.5, tf=0.72, lf=0.48, bf=0.25)
tgt = box(11.1, 1.6, 4.05, 2.35, "Target $D$", "Distance to the retrained\noperating point\n$|\\mathrm{FA}-\\mathrm{FA}^*|+|\\mathrm{RA}-\\mathrm{RA}^*|$",
          tsize=15, bsize=12, tf=0.82, lf=0.68, bf=0.36)
arrow(R(D[0]), Lm(gold, 0.72)); arrow(R(D[1]), Lm(gold, 0.30))
arrow(R(D[2]), Lm(mstar, 0.80))
for d, f in zip(D[3:], (0.55, 0.35, 0.18)):
    arrow(R(d), Lm(mstar, f))
arrow((13.1, 4.55), (13.1, 3.97))

# column 4
A = [box(16.0, 8.55, 3.85, 1.55, "Retraining alignment",
         "Spearman $\\rho$ of each criterion with $D$\n(retrained FA–RA operating point)", tsize=14, bsize=11),
     box(16.0, 6.75, 3.85, 1.55, "Dose-dependent\nmethod selection",
         "Learning-rate sweep: rankings\nchange with update strength", tsize=14, bsize=11),
     box(16.0, 4.95, 3.85, 1.55, "Reference-definition check",
         "Effective $\\Delta W=(\\alpha/r)BA$\nvs. raw LoRA factors", tsize=14, bsize=11),
     box(16.0, 3.15, 3.85, 1.55, "Scalarization is declared",
         "Deployment utility and weights stated\nMetric vector remains primary", tsize=14, bsize=11),
     box(16.0, 1.35, 3.85, 1.55, "No single scalar is\nan erasure certificate",
         "Evaluation is multi-dimensional\nand context-dependent", tsize=14, bsize=11, lw=1.6)]
arrow(R(tgt), Lm(A[0], 0.25)); arrow(R(tgt), Lm(A[1], 0.3))
arrow(R(gold), Lm(A[0], 0.6)); arrow(R(mstar), Lm(A[2], 0.5)); arrow(R(mstar), Lm(A[3], 0.6))
arrow((15.15, 2.2), Lm(A[4], 0.5), dashed=True)

ax.add_patch(FancyBboxPatch((0.2, 0.15), 19.65, 0.85, boxstyle="round,pad=0,rounding_size=0.05", fc=FILL, ec=EDGE, lw=1.2))
ax.text(10.0, 0.575, "Evaluation conclusions are conditioned on pipeline validity, operating regime, criterion, and reference definition.",
        ha="center", va="center", fontsize=16, style="italic")
fig.savefig("figures/figure_1_main.png", dpi=150)
fig.savefig("figures/figure_1_main.pdf")
