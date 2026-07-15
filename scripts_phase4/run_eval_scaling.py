"""Scaling experiment eval: load an E2B/E4B CPT model by TAG, run lm-eval tasks.

Mirrors run_eval.py exactly but uses scaling_load (dense model, no router,
base_model path read per-tag from cpt_meta.json) instead of bt2_load.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sm120_guard  # noqa: F401

import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from scaling_load import load_scaling_model

FEWSHOT = {"mmlu": 5, "gsm8k": 5, "truthfulqa_mc2": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tasks", nargs="+", default=["mmlu", "gsm8k", "truthfulqa_mc2"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--out-dir", default="/root/autodl-tmp/dnagpt/bio-trans2/results_scaling")
    a = ap.parse_args()

    t0 = time.time()
    model, tok, meta = load_scaling_model(a.tag, device=0, merge=True)
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=a.batch_size)
    print(f"[eval:{a.tag}] loaded in {time.time()-t0:.1f}s "
          f"mix={meta['mix']} | VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    os.makedirs(a.out_dir, exist_ok=True)
    for task in a.tasks:
        fs = FEWSHOT.get(task, 0)
        tt = time.time()
        try:
            res = simple_evaluate(model=lm, tasks=[task], limit=a.limit,
                                  num_fewshot=fs, bootstrap_iters=0)
        except Exception as e:
            print(f"[eval:{a.tag}] {task} FAILED: {e}", flush=True)
            continue
        out = os.path.join(a.out_dir, f"{a.tag}__{task}.json")
        json.dump({"tag": a.tag, "task": task, "mix": meta["mix"],
                   "num_fewshot": fs, "limit": a.limit,
                   "results": res["results"], "n-samples": res.get("n-samples"),
                   "elapsed_s": round(time.time()-tt, 1)},
                  open(out, "w"), indent=2, default=str)
        r = res["results"].get(task, {})
        head = {k: v for k, v in r.items() if "stderr" not in k and "alias" not in k}
        print(f"[eval:{a.tag}] {task} ({fs}s, {time.time()-tt:.0f}s): "
              f"{json.dumps(head, default=str)}", flush=True)
    print(f"[eval:{a.tag}] DONE | {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
