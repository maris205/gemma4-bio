"""Bio2NL Tier-1 collector: matched-compute continuation experiment.

For each of 3 seeds, nl_arm and bio_arm share the same nl_base start and the
same token/step budget; only the continuation content differs. We report per
seed delta = bio_arm - nl_arm, then mean +/- sd across seeds. A task shows a
transfer effect if the mean delta is clearly outside noise (|mean| > sd).
"""
import glob, json, os
import statistics as st

RES = "/root/autodl-tmp/bio-trans/bio2nl/tier1_gpt2/results"
SEEDS = [0, 1, 2]
TASKS = ["paws_en", "hellaswag", "lambada_openai",
          "anagrams1_local", "anagrams2_local", "cycle_letters_local",
          "random_insertion_local", "reversed_words_local"]
H1 = {"paws_en", "anagrams1_local", "anagrams2_local", "cycle_letters_local",
      "random_insertion_local", "reversed_words_local"}
H2 = {"hellaswag", "lambada_openai"}


def headline(results, task):
    r = results.get(task, next(iter(results.values()), {}))
    for key in ("acc,none", "exact_match,none", "acc_norm,none"):
        if key in r:
            return float(r[key])
    for k, v in r.items():
        if "stderr" not in k and isinstance(v, (int, float)):
            return float(v)
    return None


def load(tag, task):
    f = f"{RES}/{tag}__{task}.json"
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    return headline(d.get("results", {}), task)


def main():
    print(f"{'task':24s}{'hyp':4s}" +
          "".join(f"  s{s}(nl->bio)" for s in SEEDS) +
          f"{'mean_delta':>12s}{'sd':>8s}  verdict")
    rows = []
    for task in TASKS:
        deltas = []
        cellstrs = []
        for s in SEEDS:
            nl = load(f"nl_seed{s}", task)
            bio = load(f"bio_seed{s}", task)
            if nl is None or bio is None:
                cellstrs.append("   --/--  ")
                continue
            d = bio - nl
            deltas.append(d)
            cellstrs.append(f"{nl:.3f}->{bio:.3f}")
        hyp = "H1" if task in H1 else "H2"
        if deltas:
            m = st.mean(deltas)
            sd = st.pstdev(deltas) if len(deltas) > 1 else 0.0
            sig = abs(m) > sd and abs(m) > 0.005
            expect_pos = task in H1
            verdict = ("bio LIFTS" if m > 0 else "bio HURTS") if sig else "noise"
            if sig and ((expect_pos and m > 0) or (not expect_pos and m <= 0)):
                verdict += "  [as predicted]"
            elif sig:
                verdict += "  [AGAINST prediction]"
        else:
            m, sd, verdict = float("nan"), float("nan"), "NO DATA"
        print(f"{task:24s}{hyp:4s}" + "".join(f"  {c:>12s}" for c in cellstrs) +
              f"{m:>+12.4f}{sd:>8.4f}  {verdict}")
        rows.append({"task": task, "hyp": hyp, "mean_delta": m, "sd": sd,
                     "deltas": deltas, "verdict": verdict})

    json.dump(rows, open(f"{RES}/tier1_summary.json", "w"), indent=2)
    print(f"\nwrote {RES}/tier1_summary.json")

    # dissociation check
    h1_hits = [r for r in rows if r["hyp"] == "H1" and "LIFTS" in r["verdict"]]
    h2_hits = [r for r in rows if r["hyp"] == "H2" and "HURTS" in r["verdict"]]
    print(f"\nH1 (structural) tasks showing bio-lift : {len(h1_hits)}/{sum(1 for r in rows if r['hyp']=='H1')}")
    print(f"H2 (knowledge) tasks showing bio-hurt  : {len(h2_hits)}/{sum(1 for r in rows if r['hyp']=='H2')}")


if __name__ == "__main__":
    main()
