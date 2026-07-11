"""Phase-2 bio-axis eval: reconstruct a CPT model by TAG, score the 3 bio
loglikelihood tasks (homology_std, homology_remote, BixBench) reusing the
Phase-1 scoring helpers (ll_of_continuation, mcc_score, PROMPT).

Writes results/bioLL_<tag>__<task>.json, same schema as Phase-1.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sm120_guard  # noqa: F401
import torch

# reuse Phase-1 scoring logic
P1 = "/root/autodl-tmp/dnagpt/bio-trans/paper3/scripts"
sys.path.insert(0, P1)
from run_bio_loglik import ll_of_continuation, load_test, PROMPT  # noqa: E402
sys.path.insert(0, "/root/autodl-tmp/dnagpt/biopaws2")
from eval.score import mcc_score  # noqa: E402
from bt2_load import load_cpt_model  # noqa: E402

D = "/root/autodl-tmp/dnagpt/biopaws2/data"
BIO_TASKS = [f"{D}/protein_homology_std.jsonl",
             f"{D}/protein_homology_remote.jsonl",
             f"{D}/f8_bixbench_tf.jsonl"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--task-files", nargs="+", default=BIO_TASKS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default="/root/autodl-tmp/dnagpt/bio-trans2/results")
    a = ap.parse_args()

    t0 = time.time()
    model, tok, meta = load_cpt_model(a.tag, device=0, merge=True)
    print(f"[bioLL:{a.tag}] loaded {time.time()-t0:.1f}s mix={meta['mix']}", flush=True)

    os.makedirs(a.out_dir, exist_ok=True)
    for tf in a.task_files:
        tt = time.time()
        rows = load_test(tf, a.limit)
        preds = {}
        for r in rows:
            user = [m for m in r["messages"] if m["role"] != "assistant"][-1]["content"]
            choices = r.get("choices") or ["Yes", "No"]
            prompt = PROMPT.format(instr=user)
            best, best_ll = None, -1e9
            for c in choices:
                ll = ll_of_continuation(model, tok, prompt, " " + c)
                if ll > best_ll:
                    best_ll, best = ll, c
            preds[r["id"]] = best
        gold = {r["id"]: r["answer_short"] for r in rows}
        acc = sum(preds[i] == gold[i] for i in preds) / len(preds)
        mcc, _ = mcc_score(rows, preds, {})
        task = os.path.basename(tf).replace(".jsonl", "")
        out = os.path.join(a.out_dir, f"bioLL_{a.tag}__{task}.json")
        json.dump({"tag": a.tag, "task": task, "mix": meta["mix"], "n": len(rows),
                   "acc": round(acc, 4), "mcc": round(mcc, 4),
                   "elapsed_s": round(time.time()-tt, 1), "predictions": preds},
                  open(out, "w"), ensure_ascii=False, indent=2)
        print(f"[bioLL:{a.tag}] {task} (n={len(rows)}, {time.time()-tt:.0f}s): "
              f"acc={acc:.4f} mcc={mcc:.4f}", flush=True)
    print(f"[bioLL:{a.tag}] DONE {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
