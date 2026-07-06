"""Phase-1 ④CoT diagnostics: does bio-CPT change *reasoning behavior* (not accuracy)?

Same-question, same 8-shot CoT scaffold across checkpoints, sampled k times.
Per generation we measure behavioral signals (NOT correctness):
  - reasoning_len : #tokens of the chain
  - backtrack     : self-correction markers (wait/actually/reconsider/however/but...)
  - hedge         : uncertainty markers (maybe/possibly/might/uncertain/I think...)
  - self_consist  : agreement rate of the final answer across k samples (per question)
Also records accuracy as a secondary sanity signal.

Backends: GSM8K (math reasoning, ground-truth) is the primary probe.
"""
import argparse, json, os, re, sys, time
from collections import Counter, defaultdict

import transformers.integrations.moe as _moe
_moe._can_use_grouped_mm = lambda *a, **k: False
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BACKTRACK = re.compile(r"\b(wait|actually|reconsider|re-?examine|hold on|on second thought|"
                       r"let me reconsider|hmm|but wait|however|actually,|correction)\b", re.I)
HEDGE = re.compile(r"\b(maybe|perhaps|possibly|might|could be|i think|i believe|"
                   r"not sure|uncertain|approximately|roughly|around)\b", re.I)
GSM_ANS = re.compile(r"(-?\d[\d,]*\.?\d*)")

FEWSHOT_HEADER = (
    "Solve the math problem. Think step by step, then give the final answer after '####'.\n\n"
    "Question: Natalia sold clips to 48 friends in April, and half as many in May. "
    "How many clips did she sell altogether?\n"
    "Answer: In April she sold 48. In May she sold 48/2 = 24. Total = 48+24 = 72. #### 72\n\n"
    "Question: Weng earns $12 an hour for babysitting. Yesterday she did 50 minutes. "
    "How much did she earn?\n"
    "Answer: Per minute she earns 12/60 = 0.2 dollars. For 50 minutes = 0.2*50 = 10. #### 10\n\n"
)


def extract_final(text):
    if "####" in text:
        tail = text.split("####")[-1]
    else:
        tail = text
    m = GSM_ANS.findall(tail.replace(",", ""))
    return m[-1] if m else (GSM_ANS.findall(text.replace(",", ""))[-1]
                            if GSM_ANS.findall(text.replace(",", "")) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-questions", type=int, default=100)
    ap.add_argument("--k-samples", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test").select(range(a.n_questions))

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.ckpt, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        a.ckpt, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    print(f"[cot:{a.tag}] loaded {time.time()-t0:.1f}s", flush=True)

    per_gen = []            # one row per (question, sample)
    finals = defaultdict(list)
    gold = {}
    for qi, ex in enumerate(ds):
        gold[qi] = ex["answer"].split("####")[-1].strip().replace(",", "")
        prompt = FEWSHOT_HEADER + f"Question: {ex['question']}\nAnswer:"
        enc = tok([prompt] * a.k_samples, return_tensors="pt", padding=True,
                  truncation=True, max_length=1536).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new_tokens,
                                 do_sample=True, temperature=0.7, top_p=0.95,
                                 pad_token_id=tok.pad_token_id)
        for o, inp in zip(out, enc["input_ids"]):
            gen = tok.decode(o[len(inp):], skip_special_tokens=True)
            gen = gen.split("Question:")[0].strip()   # cut runaway next-shot
            ntok = len(tok(gen, add_special_tokens=False).input_ids)
            fin = extract_final(gen)
            finals[qi].append(fin)
            per_gen.append({"q": qi, "len": ntok,
                            "backtrack": len(BACKTRACK.findall(gen)),
                            "hedge": len(HEDGE.findall(gen)),
                            "final": fin, "correct": (fin == gold[qi])})
        if qi % 20 == 0:
            print(f"[cot:{a.tag}] {qi}/{len(ds)}", flush=True)

    # aggregates
    n = len(per_gen)
    mean = lambda k: round(sum(r[k] for r in per_gen) / n, 3)
    # self-consistency: fraction of a question's k samples that match its modal answer
    sc = []
    for qi, fs in finals.items():
        fs2 = [f for f in fs if f is not None]
        if not fs2:
            sc.append(0.0); continue
        top = Counter(fs2).most_common(1)[0][1]
        sc.append(top / len(fs))
    agg = {"ckpt": a.ckpt, "tag": a.tag, "n_q": len(ds), "k": a.k_samples,
           "reasoning_len": mean("len"), "backtrack_per_gen": mean("backtrack"),
           "hedge_per_gen": mean("hedge"),
           "self_consistency": round(sum(sc) / len(sc), 4),
           "acc_mean": round(sum(r["correct"] for r in per_gen) / n, 4),
           "elapsed_min": round((time.time() - t0) / 60, 1)}
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump({"agg": agg, "per_gen": per_gen},
              open(os.path.join(a.out_dir, f"cot_{a.tag}.json"), "w"), indent=2)
    print(f"[cot:{a.tag}] AGG {json.dumps(agg)}", flush=True)


if __name__ == "__main__":
    main()
