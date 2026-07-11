"""Collect Phase-2 eval JSONs into one table + tidy csv for plotting.

Reads results/<tag>__<task>.json (general) and results/bioLL_<tag>__<task>.json
(bio), extracts the headline metric per (model, task), prints a markdown table
and writes results/phase2_summary.{json,csv}.
"""
import glob, json, os, csv

RES = "/root/autodl-tmp/dnagpt/bio-trans2/results"
MODELS = ["M0-base", "M1-bio5", "M2-bio20", "M3-bio50",
          "C1-protein", "C2-dna", "C3-protdna"]
# bio share (%) per model for trajectory x-axis
BIO_SHARE = {"M0-base": 0, "M1-bio5": 5, "M2-bio20": 20, "M3-bio50": 50,
             "C1-protein": 20, "C2-dna": 20, "C3-protdna": 20}

# how to pull the headline number out of each task's results dict
def headline(task, results):
    r = results.get(task) or next(iter(results.values()), {})
    for key in ("acc,none", "acc_norm,none", "exact_match,none",
                "exact_match,strict-match", "exact_match,flexible-extract",
                "mc2,none", "acc"):
        if key in r:
            return round(float(r[key]), 4)
    # fall back: first non-stderr float
    for k, v in r.items():
        if "stderr" not in k and isinstance(v, (int, float)):
            return round(float(v), 4)
    return None


def main():
    table = {m: {} for m in MODELS}
    # general tasks
    for f in glob.glob(f"{RES}/*__*.json"):
        base = os.path.basename(f)
        if base.startswith("bioLL_"):
            continue
        d = json.load(open(f))
        tag, task = d["tag"], d["task"]
        if tag in table:
            table[tag][task] = headline(task, d.get("results", {}))
    # bio tasks (acc + mcc)
    for f in glob.glob(f"{RES}/bioLL_*__*.json"):
        d = json.load(open(f))
        tag, task = d["tag"], d["task"]
        if tag in table:
            table[tag][task + "_acc"] = d.get("acc")
            table[tag][task + "_mcc"] = d.get("mcc")

    # column order
    cols = ["mmlu", "gsm8k", "truthfulqa_mc2",
            "protein_homology_std_mcc", "protein_homology_remote_mcc",
            "f8_bixbench_tf_mcc"]
    print(f"\n{'model':12s} bio% " + " ".join(f"{c[:16]:>16s}" for c in cols))
    for m in MODELS:
        row = table[m]
        cells = " ".join(f"{(row.get(c) if row.get(c) is not None else '--'):>16}" for c in cols)
        print(f"{m:12s} {BIO_SHARE[m]:>3d}% {cells}")

    json.dump(table, open(f"{RES}/phase2_summary.json", "w"), indent=2)
    with open(f"{RES}/phase2_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "bio_share"] + cols)
        for m in MODELS:
            w.writerow([m, BIO_SHARE[m]] + [table[m].get(c, "") for c in cols])
    print(f"\nwrote {RES}/phase2_summary.json + .csv")


if __name__ == "__main__":
    main()
