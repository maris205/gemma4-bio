"""Phase-1 ①General full eval: load a Gemma4-MM checkpoint once, run each
benchmark at its community-standard few-shot, write one JSON per (ckpt,task).

sm_120 grouped-mm guard installed before lm_eval import. MC tasks use the
loglikelihood path (forward logits), so Instruct greedy-degeneration is irrelevant.
"""
import argparse, json, os, time

import transformers.integrations.moe as _moe
_moe._can_use_grouped_mm = lambda *a, **k: False

import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer

# community-standard few-shot per benchmark
FEWSHOT = {
    "mmlu": 5,
    "arc_challenge": 25,
    "hellaswag": 10,
    "truthfulqa_mc2": 0,
    "arc_easy": 0,
    "bbh": 3,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True, help="short name for output files")
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        a.ckpt, torch_dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=a.batch_size)
    print(f"[eval:{a.tag}] loaded in {time.time()-t0:.1f}s "
          f"| VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    os.makedirs(a.out_dir, exist_ok=True)
    for task in a.tasks:
        fs = FEWSHOT.get(task, None)
        tt = time.time()
        try:
            res = simple_evaluate(model=lm, tasks=[task], limit=a.limit,
                                  num_fewshot=fs, bootstrap_iters=0)
        except Exception as e:
            print(f"[eval:{a.tag}] {task} FAILED: {e}", flush=True)
            continue
        out = os.path.join(a.out_dir, f"{a.tag}__{task}.json")
        json.dump({"ckpt": a.ckpt, "tag": a.tag, "task": task,
                   "num_fewshot": fs, "limit": a.limit,
                   "results": res["results"], "n-samples": res.get("n-samples"),
                   "elapsed_s": round(time.time()-tt, 1)},
                  open(out, "w"), indent=2, default=str)
        # print headline metric
        r = res["results"].get(task, {})
        head = {k: v for k, v in r.items() if "stderr" not in k and "alias" not in k}
        print(f"[eval:{a.tag}] {task} ({fs}-shot, {time.time()-tt:.0f}s): "
              f"{json.dumps(head, default=str)}", flush=True)

    print(f"[eval:{a.tag}] DONE all tasks | total {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
