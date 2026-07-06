"""Phase-1 ②Biology: Mode-A zero-shot QA on biological tasks across checkpoints.

Reuses biopaws2 task jsonl + eval.score.score_task. Adds --prompt-style so
completion-style checkpoints (BioCPT, no chat_template) get an Alpaca prompt,
matching the fair regime used in ①General/③Coding, while Instruct checkpoints
use their chat_template.
"""
import argparse, json, os, sys, time

import transformers.integrations.moe as _moe
_moe._can_use_grouped_mm = lambda *a, **k: False

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, "/root/autodl-tmp/dnagpt/biopaws2")
from eval.score import score_task, mcc_score  # noqa: E402


def load_golds(path):
    g, cm = [], {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("split") != "test":
            continue
        g.append(r)
        if r.get("choices"):
            cm[r["task_id"]] = r["choices"]
    return g, cm

ALPACA = ("Below is an instruction that describes a task. "
          "Write a response that appropriately completes the request.\n\n"
          "### Instruction:\n{instr}\n\n### Answer:\n")


def load_test(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("split") == "test":
                rows.append(r)
    return rows[:limit] if limit else rows


def build_prompt(r, tok, style):
    user = [m for m in r["messages"] if m["role"] != "assistant"]
    if style == "alpaca":
        return ALPACA.format(instr=user[-1]["content"])
    # chat style
    try:
        return tok.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    except Exception:
        return user[-1]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--task-files", nargs="+", required=True)
    ap.add_argument("--prompt-style", choices=["chat", "alpaca"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.ckpt, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        a.ckpt, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    print(f"[bio:{a.tag}] loaded in {time.time()-t0:.1f}s style={a.prompt_style} "
          f"VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    os.makedirs(a.out_dir, exist_ok=True)
    for tf in a.task_files:
        tt = time.time()
        rows = load_test(tf, a.limit)
        preds = {}
        for i in range(0, len(rows), a.batch_size):
            chunk = rows[i:i + a.batch_size]
            prompts = [build_prompt(r, tok, a.prompt_style) for r in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=2048).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=a.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            for r, o, inp in zip(chunk, out, enc["input_ids"]):
                preds[r["id"]] = tok.decode(o[len(inp):], skip_special_tokens=True).strip()
        res = score_task(tf, preds)
        # add MCC for binary tasks (homology_std is 90% Yes -> accuracy misleads)
        try:
            g, cm = load_golds(tf)
            mcc, nm = mcc_score(g, preds, cm)
            res["mcc"] = round(mcc, 4)
        except Exception as e:
            res["mcc_err"] = str(e)
        task = os.path.basename(tf).replace(".jsonl", "")
        out = os.path.join(a.out_dir, f"bio_{a.tag}__{task}.json")
        json.dump({"ckpt": a.ckpt, "tag": a.tag, "task": task,
                   "prompt_style": a.prompt_style, "n_test": len(rows),
                   "result": res, "elapsed_s": round(time.time()-tt, 1),
                   "predictions": preds},
                  open(out, "w"), ensure_ascii=False, indent=2)
        print(f"[bio:{a.tag}] {task} (n={len(rows)}, {time.time()-tt:.0f}s): {res}", flush=True)
    print(f"[bio:{a.tag}] DONE | total {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
