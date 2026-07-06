"""Phase-1 ③Coding / reasoning: generative tasks (HumanEval/MBPP/GSM8K).

Unlike ①General these are generation+execution tasks. sm_120 grouped-mm guard
installed before lm_eval import. HumanEval/MBPP require executing generated code,
so lm_eval needs confirm_run_unsafe_code=True.
"""
import argparse, json, os, time

import transformers.integrations.moe as _moe
_moe._can_use_grouped_mm = lambda *a, **k: False

import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer

FEWSHOT = {"humaneval": 0, "mbpp": 3, "gsm8k": 5, "gsm8k_cot": 8,
           "humaneval_instruct": 0, "mbpp_instruct": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        a.ckpt, torch_dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=a.batch_size)
    print(f"[code:{a.tag}] loaded in {time.time()-t0:.1f}s "
          f"| VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    os.makedirs(a.out_dir, exist_ok=True)
    for task in a.tasks:
        fs = FEWSHOT.get(task, 0)
        tt = time.time()
        try:
            res = simple_evaluate(model=lm, tasks=[task], limit=a.limit,
                                  num_fewshot=fs, bootstrap_iters=0,
                                  confirm_run_unsafe_code=True)
        except Exception as e:
            print(f"[code:{a.tag}] {task} FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        out = os.path.join(a.out_dir, f"{a.tag}__{task}.json")
        json.dump({"ckpt": a.ckpt, "tag": a.tag, "task": task, "num_fewshot": fs,
                   "limit": a.limit, "results": res["results"],
                   "n-samples": res.get("n-samples"),
                   "elapsed_s": round(time.time()-tt, 1)},
                  open(out, "w"), indent=2, default=str)
        r = res["results"].get(task, {})
        head = {k: v for k, v in r.items() if "stderr" not in k and "alias" not in k}
        print(f"[code:{a.tag}] {task} ({fs}-shot, {time.time()-tt:.0f}s): "
              f"{json.dumps(head, default=str)}", flush=True)

    print(f"[code:{a.tag}] DONE | total {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
