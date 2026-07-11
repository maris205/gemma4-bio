"""Phase-2 figures for the capability-shaping paper.

fig1_mixture_trajectory: as bio% rises (0/5/20/50), plot General (MMLU, GSM8K)
    vs Biology (homology_std MCC, BixBench MCC) -> shows general flat, bio up.
fig2_composition: bio-share fixed 20%, compare Protein/DNA/Prot+DNA on each
    bio metric -> which scientific data type is most effective.
fig3_shaping_2d: 2D trajectory General-axis (mean of mmlu,gsm8k normalized) vs
    Biology-axis (mean of the 3 bio MCCs), one point per mixture model.
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3})
RES = "/root/autodl-tmp/dnagpt/bio-trans2/results"
PAPER = "/root/autodl-tmp/dnagpt/bio-trans/paper3/figs"
rows = list(csv.DictReader(open(f"{RES}/phase2_summary.csv")))
D = {r["model"]: r for r in rows}
f = lambda m, k: float(D[m][k])

# ---- fig1: mixture-ratio trajectory ----
mix = ["M0-base", "M1-bio5", "M2-bio20", "M3-bio50"]
x = [f(m, "bio_share") for m in mix]
fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(x, [f(m, "mmlu") for m in mix], "o-", color="#1f77b4", label="MMLU (general)")
ax1.plot(x, [f(m, "gsm8k") for m in mix], "s-", color="#2ca02c", label="GSM8K (reasoning)")
ax1.set_xlabel("Biology data share in CPT mixture (%)")
ax1.set_ylabel("General accuracy", color="#1f77b4")
ax1.set_ylim(0.5, 1.0)
ax2 = ax1.twinx()
ax2.plot(x, [f(m, "protein_homology_std_mcc") for m in mix], "^--", color="#d62728", label="Homology-std (MCC)")
ax2.plot(x, [f(m, "f8_bixbench_tf_mcc") for m in mix], "D--", color="#ff7f0e", label="BixBench (MCC)")
ax2.set_ylabel("Biology capability (MCC / acc)", color="#d62728")
ax2.set_ylim(0, 1.05)
ax2.grid(False)
l1, la1 = ax1.get_legend_handles_labels()
l2, la2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, la1 + la2, loc="center left", fontsize=8.5, framealpha=0.9)
ax1.set_title("Capability shaping by biology mixture ratio\n(general flat, biology rises — no catastrophic forgetting)")
fig.tight_layout()
fig.savefig(f"{RES}/fig1_mixture_trajectory.png", dpi=150)
fig.savefig(f"{PAPER}/p2_fig1_mixture.pdf")
print("wrote fig1_mixture_trajectory.png")

# ---- fig2: composition ablation (bio share = 20%) ----
comp = ["C1-protein", "C2-dna", "C3-protdna", "M2-bio20"]
labels = ["Protein", "DNA", "Prot+DNA", "Prot+DNA+Lit"]
metrics = [("protein_homology_std_mcc", "Homology-std"),
           ("protein_homology_remote_mcc", "Homology-remote"),
           ("f8_bixbench_tf_mcc", "BixBench")]
fig, ax = plt.subplots(figsize=(7.5, 4.5))
nb = len(metrics)
w = 0.2
import numpy as np
xc = np.arange(len(comp))
for j, (mk, ml) in enumerate(metrics):
    ax.bar(xc + (j - 1) * w, [f(m, mk) for m in comp], w, label=ml)
ax.set_xticks(xc)
ax.set_xticklabels(labels)
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("MCC")
ax.set_title("Which scientific data type shapes biology capability?\n(bio share fixed at 20%)")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(f"{RES}/fig2_composition.png", dpi=150)
fig.savefig(f"{PAPER}/p2_fig2_composition.pdf")
print("wrote fig2_composition.png")

# ---- fig3: 2D shaping trajectory ----
def gen_axis(m):
    return (f(m, "mmlu") + f(m, "gsm8k")) / 2
def bio_axis(m):
    return (f(m, "protein_homology_std_mcc")
            + f(m, "protein_homology_remote_mcc")
            + f(m, "f8_bixbench_tf_mcc")) / 3
fig, ax = plt.subplots(figsize=(6, 5))
gx = [gen_axis(m) for m in mix]
by = [bio_axis(m) for m in mix]
ax.plot(gx, by, "-", color="gray", alpha=0.5, zorder=1)
for m in mix:
    ax.scatter(gen_axis(m), bio_axis(m), s=90, zorder=2)
    ax.annotate(f"{m}\n({int(f(m,'bio_share'))}% bio)",
                (gen_axis(m), bio_axis(m)), textcoords="offset points",
                xytext=(8, 4), fontsize=8)
ax.set_xlabel("General axis  (mean MMLU, GSM8K)")
ax.set_ylabel("Biology axis  (mean of 3 bio MCCs)")
ax.set_title("Capability-shaping trajectory")
fig.tight_layout()
fig.savefig(f"{RES}/fig3_shaping_2d.png", dpi=150)
fig.savefig(f"{PAPER}/p2_fig3_shaping2d.pdf")
print("wrote fig3_shaping_2d.png")
