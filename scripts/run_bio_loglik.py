"""Phase-1 ②Biology (format-robust): score binary bio tasks by loglikelihood.

For each example, compare LL of appending "Yes" vs "No" (or the task's two
choices) to the SAME neutral prompt, pick argmax. This mirrors the ①General
loglikelihood-MC regime, removing the generation / instruction-following /
chat-vs-alpaca confound so all 4 checkpoints are compared fairly on pure
discrimination. Reports accuracy + MCC.
"""
import argparse, json, os, sys, time
import transformers.integrations.moe as _moe
_moe._can_use_grouped_mm = lambda *a, **k: False
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, "/root/autodl-tmp/dnagpt/biopaws2")
from eval.score import mcc_score  # noqa: E402

# neutral completion prompt: task instruction (from jsonl) + "Answer:" then Yes/No
PROMPT = "{instr}\n\nAnswer (Yes or No):"


def load_test(path, limit=None):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("split") == "test":
            rows.append(r)
    return rows[:limit] if limit else rows


@torch.no_grad()
def ll_of_continuation(model, tok, prompt, cont):
    """Mean log-prob of `cont` tokens given `prompt`."""
    p_ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    c_ids = tok(cont, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    ids = torch.cat([p_ids, c_ids], dim=1)
    logits = model(ids).logits
    # logits for predicting c_ids start at position len(p)-1
    lp = torch.log_softmax(logits[0, p_ids.shape[1]-1:-1], dim=-1)
    tok_lp = lp[range(c_ids.shape[1]), c_ids[0]]
    return tok_lp.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--task-files", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.ckpt, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.ckpt, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    print(f"[bioLL:{a.tag}] loaded {time.time()-t0:.1f}s", flush=True)

    os.makedirs(a.out_dir, exist_ok=True)
    for tf in a.task_files:
        tt = time.time()
        rows = load_test(tf, a.limit)
        # choices: default Yes/No; use task choices if present
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
        # score: accuracy + MCC
        gold = {r["id"]: r["answer_short"] for r in rows}
        acc = sum(preds[i] == gold[i] for i in preds) / len(preds)
        mcc, _ = mcc_score(rows, preds, {})
        task = os.path.basename(tf).replace(".jsonl", "")
        out = os.path.join(a.out_dir, f"bioLL_{a.tag}__{task}.json")
        json.dump({"ckpt": a.ckpt, "tag": a.tag, "task": task, "n": len(rows),
                   "acc": round(acc, 4), "mcc": round(mcc, 4),
                   "elapsed_s": round(time.time()-tt, 1), "predictions": preds},
                  open(out, "w"), ensure_ascii=False, indent=2)
        print(f"[bioLL:{a.tag}] {task} (n={len(rows)}, {time.time()-tt:.0f}s): "
              f"acc={acc:.4f} mcc={mcc:.4f}", flush=True)
    print(f"[bioLL:{a.tag}] DONE {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
