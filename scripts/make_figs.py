"""Paper-3 Phase-1 figures. Reads the four result tables' numbers (hard-coded from
the finalized JSONs) and renders publication-quality PDFs into paper3/figs/.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11, "font.family": "serif",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150})
OUT = "/root/autodl-tmp/dnagpt/bio-trans/paper3/figs"
os.makedirs(OUT, exist_ok=True)

CK = ["Base\n(it)", "Base\n(it-bio)", "BioCPT", "BioCPT\n+SFT"]
COL = ["#8c8c8c", "#b0b0b0", "#c0392b", "#2c6fbb"]

# ---- Fig 1: the U-curve across all axes (normalized to base=1.0) ----------
# metrics chosen as the headline per axis
axes = {
    "MMLU (5s)":        [0.646, 0.642, 0.776, 0.635],
    "MBPP (3s)":        [0.332, 0.350, 0.630, 0.348],
    "BixBench (MCC)":   [0.232, 0.232, 0.924, 0.361],
    "ARC-C (25s)":      [0.367, 0.367, 0.700, 0.382],
    "HellaSwag (norm)": [0.488, 0.486, 0.855, 0.486],
}
fig, ax = plt.subplots(figsize=(7, 4.2))
x = np.arange(4)
for (name, v), m in zip(axes.items(), ["o", "s", "^", "D", "v"]):
    ax.plot(x, np.array(v), marker=m, lw=1.8, ms=6, label=name)
ax.set_xticks(x); ax.set_xticklabels(CK)
ax.set_ylabel("Score (raw)")
ax.set_title("Biological CPT lifts every held-out axis; SFT narrows it back", fontsize=11)
ax.axvspan(1.5, 2.5, color="#c0392b", alpha=0.06)
ax.legend(fontsize=8.5, ncol=2, frameon=False, loc="upper right")
ax.grid(axis="y", ls=":", alpha=0.4)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_ucurve.pdf"); plt.close(fig)

# ---- Fig 2: capability radar, base vs BioCPT vs SFT -----------------------
labels = ["MMLU", "ARC-C", "HellaSwag", "MBPP", "BixBench"]
base = [0.646, 0.367, 0.488, 0.332, 0.232]
cpt  = [0.776, 0.700, 0.855, 0.630, 0.924]
sft  = [0.635, 0.382, 0.486, 0.348, 0.361]
ang = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
ang += ang[:1]
fig = plt.figure(figsize=(5.2, 5.2)); ax = plt.subplot(111, polar=True)
for vals, c, lab in [(base, COL[0], "Base (it)"), (cpt, COL[2], "BioCPT"),
                     (sft, COL[3], "BioCPT+SFT")]:
    d = vals + vals[:1]
    ax.plot(ang, d, color=c, lw=2, label=lab)
    ax.fill(ang, d, color=c, alpha=0.10)
ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels)
ax.set_ylim(0, 1.0)
ax.set_title("Capability profile", pad=18)
ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), fontsize=9, frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_radar.pdf"); plt.close(fig)

# ---- Fig 3: reasoning-behavior shift --------------------------------------
length = [108.8, 107.5, 64.4, 125.8]
back   = [0.370, 0.324, 0.006, 0.396]
acc    = [0.667, 0.652, 0.689, 0.703]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.8))
x = np.arange(4)
b1 = a1.bar(x, length, color=COL, width=0.6)
a1.set_xticks(x); a1.set_xticklabels(CK, fontsize=9)
a1.set_ylabel("CoT chain length (tokens)")
a1.set_title("CPT shortens the reasoning chain")
for xi, v in zip(x, length): a1.text(xi, v+2, f"{v:.0f}", ha="center", fontsize=9)
a2b = a2.twinx()
a2.bar(x-0.18, back, width=0.36, color="#c0392b", label="backtrack/gen")
a2b.plot(x, acc, "o-", color="#222", lw=1.6, label="accuracy")
a2.set_xticks(x); a2.set_xticklabels(CK, fontsize=9)
a2.set_ylabel("Self-correction (backtracks/gen)", color="#c0392b")
a2b.set_ylabel("Accuracy"); a2b.set_ylim(0.5, 0.8)
a2.set_title("Fewer backtracks, accuracy preserved")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_reasoning.pdf"); plt.close(fig)

print("wrote:", os.listdir(OUT))
