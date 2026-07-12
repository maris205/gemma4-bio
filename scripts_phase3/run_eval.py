"""Bio2NL Tier-1 eval: load a GPT-2 arm checkpoint by tag, run the H1/H2
task battery via lm-eval HFLM. No adapters -- these are full finetune saves.

H1 (structural/character, predict bio-arm > nl-arm):
  paws_en (remote surface-confusion association), anagrams1/2, cycle_letters,
  random_insertion, reversed_words (character-level, from the GPT-3 paper suite)
H2 (knowledge, predict bio-arm <= nl-arm):
  hellaswag, lambada_openai
"""
import argparse, json, os, time
import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from transformers import AutoModelForCausalLM, AutoTokenizer

FEWSHOT = {"paws_en": 0, "hellaswag": 0, "lambada_openai": 0,
           "anagrams1_local": 0, "anagrams2_local": 0, "cycle_letters_local": 0,
           "random_insertion_local": 0, "reversed_words_local": 0}

MODELS_ROOT = "/root/autodl-tmp/bio-trans/bio2nl/tier1_gpt2/models"
LOCAL_TASKS = "/root/autodl-tmp/bio-trans/bio2nl/tier1_gpt2/lm_eval_tasks"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default="/root/autodl-tmp/bio-trans/bio2nl/tier1_gpt2/results")
    a = ap.parse_args()

    ckpt = os.path.join(MODELS_ROOT, a.tag)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16,
                                                 device_map={"": 0})
    model.eval()
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size="auto")
    print(f"[eval:{a.tag}] loaded in {time.time()-t0:.1f}s "
          f"VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
    tm = TaskManager(include_path=LOCAL_TASKS)

    os.makedirs(a.out_dir, exist_ok=True)
    for task in a.tasks:
        fs = FEWSHOT.get(task, 0)
        tt = time.time()
        try:
            res = simple_evaluate(model=lm, tasks=[task], limit=a.limit,
                                  num_fewshot=fs, bootstrap_iters=0, task_manager=tm)
        except Exception as e:
            print(f"[eval:{a.tag}] {task} FAILED: {e}", flush=True)
            continue
        out = os.path.join(a.out_dir, f"{a.tag}__{task}.json")
        json.dump({"tag": a.tag, "task": task, "num_fewshot": fs, "limit": a.limit,
                   "results": res["results"], "n-samples": res.get("n-samples"),
                   "elapsed_s": round(time.time()-tt, 1)},
                  open(out, "w"), indent=2, default=str)
        r = res["results"].get(task, {})
        head = {k: v for k, v in r.items() if "stderr" not in k and "alias" not in k}
        print(f"[eval:{a.tag}] {task} ({time.time()-tt:.0f}s): {json.dumps(head, default=str)}",
              flush=True)
    print(f"[eval:{a.tag}] DONE | {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
