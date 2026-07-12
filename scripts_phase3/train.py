#!/usr/bin/env python
# coding: utf-8
"""
Bio2NL Tier-1 GPT-2 training (full finetune, single GPU).

Two uses via env STAGE:
  warmup      : gpt2 -> continue on NL (openwebtext) for WARM_TOKENS -> save nl_base
  continue    : nl_base -> continue on a mixture (MIX json over pools) for CONT_TOKENS

Env: STAGE, INIT (model dir or 'gpt2'), MIX (json, e.g. '{"nl":1.0}' or
     '{"protein":0.5,"dna":0.5}'), TOKENS, OUT, POOL_DIR, LR, BLOCK.
Matched compute = identical TOKENS/steps/batch/optimizer across arms; only MIX differs.
"""
import os, json, struct, math
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer)

STAGE  = os.environ["STAGE"]
INIT   = os.getenv("INIT", "gpt2")
MIX    = json.loads(os.getenv("MIX", '{"nl":1.0}'))
TOKENS = int(os.getenv("TOKENS", "200000000"))
OUT    = os.environ["OUT"]
POOL   = os.getenv("POOL_DIR", "/root/autodl-tmp/bio-trans/bio2nl/tier1_gpt2/pools")
LR     = float(os.getenv("LR", "5e-5"))
BLOCK  = int(os.getenv("BLOCK", "1024"))
SEED   = 42 + int(os.getenv("SEED_OFFSET", "0"))
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)
import transformers as _tf
_tf.set_seed(SEED)

# ---- assemble token-budgeted index from pools ----
def build_index():
    rows = []
    srcs = sorted(MIX)
    binmap = {}
    for sid, s in enumerate(srcs):
        idx = np.load(f"{POOL}/{s}_index.npy")
        binmap[sid] = f"{POOL}/{s}.bin"
        want = int(TOKENS * MIX[s]); got = 0
        for k in rng.permutation(len(idx)):
            off, ln = int(idx[k][0]), int(idx[k][1])
            rows.append((sid, off, ln)); got += ln
            if got >= want:
                break
        print(f"  {s}: want {want:,} got {got:,}", flush=True)
    rng.shuffle(rows)
    return srcs, binmap, np.array(rows, dtype=np.int64)

SRCS, BINMAP, INDEX = build_index()
print(f"[{STAGE}] init={INIT} mix={MIX} tokens={int(INDEX[:,2].sum()):,} chunks={len(INDEX)}", flush=True)

class PoolDS(Dataset):
    def __init__(self): self.f = {}
    def __len__(self): return len(INDEX)
    def __getitem__(self, i):
        sid, off, ln = (int(x) for x in INDEX[i])
        h = self.f.get(sid)
        if h is None:
            h = open(BINMAP[sid], "rb"); self.f[sid] = h
        h.seek(off + 4)
        ids = np.frombuffer(h.read(ln*4), dtype=np.uint32).astype(np.int64).tolist()[:BLOCK]
        return {"input_ids": ids, "labels": ids}

class Collate:
    def __init__(self, pad): self.pad = pad
    def __call__(self, b):
        m = min(max(len(x["input_ids"]) for x in b), BLOCK)
        ii, ll, am = [], [], []
        for x in b:
            v = x["input_ids"][:m]; p = m - len(v)
            ii.append(v + [self.pad]*p); ll.append(v + [-100]*p); am.append([1]*len(v)+[0]*p)
        return {"input_ids": torch.tensor(ii), "labels": torch.tensor(ll),
                "attention_mask": torch.tensor(am)}

tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(INIT)
model.config.use_cache = False

BS, ACC = 8, 8   # eff batch 64 * 1024 tok = 65536 tok/step
steps = max(1, int(INDEX[:, 2].sum()) // (BS*ACC*BLOCK))
args = TrainingArguments(
    output_dir=f"{OUT}/_ck", max_steps=steps,
    per_device_train_batch_size=BS, gradient_accumulation_steps=ACC,
    learning_rate=LR, lr_scheduler_type="cosine", warmup_ratio=0.02,
    weight_decay=0.01, bf16=True, logging_steps=50, save_strategy="no",
    report_to="none", dataloader_num_workers=4, remove_unused_columns=False)
print(f"[{STAGE}] steps={steps} eff_batch_tok={BS*ACC*BLOCK}", flush=True)

Trainer(model=model, args=args, train_dataset=PoolDS(),
        data_collator=Collate(tok.pad_token_id)).train()

model.save_pretrained(OUT); tok.save_pretrained(OUT)
json.dump({"stage": STAGE, "init": INIT, "mix": MIX,
           "tokens": int(INDEX[:, 2].sum()), "steps": steps, "lr": LR},
          open(f"{OUT}/train_meta.json", "w"), indent=2)
print(f"[{STAGE}] SAVED -> {OUT}", flush=True)
