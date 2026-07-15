"""Part IV figures: robustness (scaling) + replay control + router mechanism.

fig4_scaling: general (mmlu) delta and bixbench delta (bio50-text100) at 3
    scales (E2B/E4B/26B) -> shows general-preserving property is scale-robust.
fig5_replay: bar chart of homology_std/remote/bixbench MCC across
    no-replay/10%replay/20%replay at fixed bio50 mixture.
fig6_router: layer-averaged routing JS divergence, it-bio->bio (full CPT) vs
    the Phase-2 LoRA mixture sweep (M0->M3) -> dissociation of capability
    shaping from routing reorganization at LoRA scale.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "serif", "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3})
PAPER = "/root/autodl-tmp/dnagpt/bio-trans/paper3/figs"

# ---- fig4: scaling robustness ----
sizes = ["E2B\n(2.3B)", "E4B\n(4.5B)", "26B-A4B\n(4B active)"]
mmlu_text = [0.576, 0.694, 0.802]
mmlu_bio = [0.578, 0.695, 0.796]
bb_text = [0.479, 0.680, 0.850]
bb_bio = [0.623, 0.781, 1.000]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
x = np.arange(3)
w = 0.32
ax1.bar(x - w/2, mmlu_text, w, label="text-only", color="#1f77b4")
ax1.bar(x + w/2, mmlu_bio, w, label="+50% biology", color="#d62728")
ax1.set_xticks(x); ax1.set_xticklabels(sizes)
ax1.set_ylabel("MMLU accuracy")
ax1.set_title("General axis: preserved at every scale")
ax1.set_ylim(0, 1.0)
ax1.legend(fontsize=9)
for i in range(3):
    d = mmlu_bio[i] - mmlu_text[i]
    ax1.annotate(f"{d:+.3f}", (x[i], max(mmlu_text[i], mmlu_bio[i]) + 0.03),
                ha="center", fontsize=8)

ax2.bar(x - w/2, bb_text, w, label="text-only", color="#1f77b4")
ax2.bar(x + w/2, bb_bio, w, label="+50% biology", color="#d62728")
ax2.set_xticks(x); ax2.set_xticklabels(sizes)
ax2.set_ylabel("BixBench MCC")
ax2.set_title("Biology axis: lifted at every scale")
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=9)
for i in range(3):
    d = bb_bio[i] - bb_text[i]
    ax2.annotate(f"{d:+.3f}", (x[i], max(bb_text[i], bb_bio[i]) + 0.03),
                ha="center", fontsize=8)
fig.suptitle("Scale robustness of the capability-shaping pattern (2.3B → 26B, 10× range)")
fig.tight_layout()
fig.savefig(f"{PAPER}/p4_fig4_scaling.pdf")
print("wrote p4_fig4_scaling.pdf")

# ---- fig5: replay control ----
labels = ["no replay\n(M3-bio50)", "10% replay\n(R1)", "20% replay\n(R2)"]
hom_std = [0.375, 0.318, 0.266]
hom_rem = [0.088, 0.000, -0.004]
bixb = [1.000, 1.000, 0.873]
mmlu = [0.796, 0.800, 0.795]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4.2))
x = np.arange(3)
w = 0.26
axL.bar(x - w, hom_std, w, label="Homology-std", color="#d62728")
axL.bar(x, hom_rem, w, label="Homology-remote", color="#ff7f0e")
axL.bar(x + w, bixb, w, label="BixBench", color="#2ca02c")
axL.axhline(0, color="k", lw=0.6)
axL.set_xticks(x); axL.set_xticklabels(labels, fontsize=8.5)
axL.set_ylabel("MCC")
axL.set_title("Biology axis: replay does not help\n(more replay → less effective bio exposure)")
axL.legend(fontsize=8)

axR.plot(x, mmlu, "o-", color="#1f77b4", markersize=8)
axR.set_xticks(x); axR.set_xticklabels(labels, fontsize=8.5)
axR.set_ylabel("MMLU accuracy")
axR.set_ylim(0.75, 0.85)
axR.set_title("General axis: already flat without replay\n(nothing to recover)")
fig.suptitle("Replay control at fixed 50% biological share (26B, 100M tokens)")
fig.tight_layout()
fig.savefig(f"{PAPER}/p4_fig5_replay.pdf")
print("wrote p4_fig5_replay.pdf")

# ---- fig6: router mechanism ----
fig, ax = plt.subplots(figsize=(7, 4.5))
lineage_x = ["it-bio\n(no CPT)", "bio\n(8.7B tok,\nfull CPT)"]
lineage_y = [0.2187, 0.4505]  # mean(dna_vs_nl, prot_vs_nl)
ax.plot([0, 1], lineage_y, "o-", color="#d62728", markersize=10,
       label="Part I lineage (full-parameter CPT)")

sweep_x = [0, 5, 20, 50]
sweep_y = [0.3755, 0.3847, 0.3841, 0.3788]  # M0..M3 mean bio-vs-nl JS
ax2 = ax.twiny()
ax2.plot(sweep_x, sweep_y, "s--", color="#1f77b4", markersize=7,
        label="Phase-2 mixture sweep (100M-token LoRA)")
ax2.set_xlabel("Biological share in Phase-2 LoRA sweep (%)")
ax.set_xticks([0, 1]); ax.set_xticklabels(lineage_x)
ax.set_ylabel("Mean routing JS divergence\n(biology vs. natural-language prompts)")
ax.set_title("Routing reorganization: full CPT reshapes it,\nLoRA mixture ratio does not (despite lifting capability)")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
fig.tight_layout()
fig.savefig(f"{PAPER}/p4_fig6_router.pdf")
print("wrote p4_fig6_router.pdf")
