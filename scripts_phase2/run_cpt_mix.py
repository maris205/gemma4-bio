#!/usr/bin/env python
# coding: utf-8
"""
Phase-II parameterized QLoRA CPT.

Assembles a MIX_TOKENS-budget training index by sampling chunk-indices from
per-source pools at a requested ratio, then runs QLoRA CPT (r=64, unfreeze
embedding + MoE router) exactly as omnigene_v2/scripts/cpt/2-run_cpt.py, on a
single 96GB GPU. sm_120 grouped-mm guard applied.

Config via env:
  TAG          model tag (e.g. M2-bio20)
  MIX          json ratio over sources, e.g. '{"text":0.8,"protein":0.14,"dna":0.04,"biolit":0.02}'
  MIX_TOKENS   total token budget (default 100_000_000)
  POOL_DIR, MODEL_DIR, OUT_ROOT
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sm120_guard  # noqa: F401  dtype-safe MoE fallback; MUST precede model load

import json, struct
import numpy as np
import torch

from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, Trainer)
from peft import LoraConfig, inject_adapter_in_model

TAG        = os.environ["TAG"]
MIX        = json.loads(os.environ["MIX"])
MIX_TOKENS = int(os.getenv("MIX_TOKENS", "100000000"))
POOL_DIR   = os.getenv("POOL_DIR", "/root/autodl-tmp/dnagpt/bio-trans2/data/pools")
MODEL_DIR  = os.getenv("MODEL_DIR", "/autodl-fs/data/omnigene_v2/models/gemma-4-26B-A4B-it-bio")
OUT_ROOT   = os.getenv("OUT_ROOT", "/root/autodl-tmp/dnagpt/bio-trans2/models")
MAX_LENGTH = 1024
SEED = 42

OUT_DIR = os.path.join(OUT_ROOT, TAG)
CKPT_DIR = os.path.join(OUT_ROOT, "_checkpoints", TAG)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)

# ---------- assemble mixed index from pools ----------
def build_mix_index():
    rows = []          # (source_id, offset, length)
    sources = sorted(MIX.keys())
    src_bins = {}
    for sid, src in enumerate(sources):
        idx = np.load(f"{POOL_DIR}/{src}_index.npy")   # [[offset,length],...]
        src_bins[sid] = f"{POOL_DIR}/{src}.bin"
        want_tok = int(MIX_TOKENS * MIX[src])
        # shuffle pool order, take chunks until token budget met
        order = rng.permutation(len(idx))
        got = 0
        for k in order:
            off, ln = int(idx[k][0]), int(idx[k][1])
            rows.append((sid, off, ln))
            got += ln
            if got >= want_tok:
                break
        avail = int(idx[:, 1].sum())
        print(f"  {src:8s} want={want_tok:>12,} got={got:>12,} "
              f"(pool avail={avail:,}{' *SHORT*' if got < want_tok*0.98 else ''})", flush=True)
    rng.shuffle(rows)
    return sources, src_bins, np.array(rows, dtype=np.int64)

print(f"=== {TAG}  budget={MIX_TOKENS:,} tok  mix={MIX} ===", flush=True)
SOURCES, SRC_BINS, MIX_INDEX = build_mix_index()
actual_tok = int(MIX_INDEX[:, 2].sum())
print(f"  assembled {len(MIX_INDEX):,} chunks, {actual_tok:,} tokens", flush=True)

class MixDataset(Dataset):
    def __init__(self):
        self.files = {}
    def __len__(self):
        return len(MIX_INDEX)
    def __getitem__(self, i):
        sid, off, ln = (int(x) for x in MIX_INDEX[i])
        f = self.files.get(sid)
        if f is None:
            f = open(SRC_BINS[sid], "rb"); self.files[sid] = f
        f.seek(off + 4)
        data = f.read(ln * 4)
        ids = np.frombuffer(data, dtype=np.uint32).astype(np.int64).tolist()[:MAX_LENGTH]
        return {"input_ids": ids, "labels": ids}

class Collator:
    def __init__(self, tok):
        self.pad = tok.pad_token_id or 0
    def __call__(self, feats):
        m = min(max(len(f["input_ids"]) for f in feats), MAX_LENGTH)
        ii, ll, am = [], [], []
        for f in feats:
            x = f["input_ids"][:m]; p = m - len(x)
            ii.append(x + [self.pad]*p)
            ll.append(f["labels"][:m] + [-100]*p)
            am.append([1]*len(x) + [0]*p)
        return {"input_ids": torch.tensor(ii), "labels": torch.tensor(ll),
                "attention_mask": torch.tensor(am),
                "mm_token_type_ids": torch.zeros(len(feats), m, dtype=torch.long)}

# ---------- model ----------
print("Loading model (4-bit)...", flush=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, quantization_config=bnb,
    device_map={"": 0})
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

lora = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, bias="none",
    target_modules=['q_proj','k_proj','v_proj','o_proj',
                    'gate_proj','up_proj','down_proj','router.proj'])
for p in model.parameters():
    p.requires_grad = False
inject_adapter_in_model(lora, model.model.language_model, adapter_name="default")
for p in model.get_input_embeddings().parameters():
    p.requires_grad = True
    p.data = p.data.to(torch.float32)
model._hf_peft_config_loaded = True
trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
tot = sum(p.numel() for p in model.parameters())
print(f"Trainable {trn:,}/{tot:,} = {100*trn/tot:.3f}%", flush=True)

args = TrainingArguments(
    output_dir=CKPT_DIR, num_train_epochs=1,
    per_device_train_batch_size=2, gradient_accumulation_steps=16,   # eff batch 32
    optim="paged_adamw_8bit", learning_rate=2e-5, lr_scheduler_type="cosine",
    warmup_ratio=0.03, weight_decay=0.01, bf16=True, max_grad_norm=1.0,
    logging_steps=25, save_strategy="no", report_to="none",
    dataloader_num_workers=4, dataloader_pin_memory=True,
    remove_unused_columns=False)

trainer = Trainer(model=model, args=args, train_dataset=MixDataset(),
                  data_collator=Collator(tok))
eff = args.per_device_train_batch_size * args.gradient_accumulation_steps
print(f"Steps ~{len(MIX_INDEX)//eff}  eff_batch={eff}", flush=True)
trainer.train()

# ---------- save adapter + embeddings (not merged; merge at eval as needed) ----------
print(f"Saving to {OUT_DIR}...", flush=True)
lora_sd = {k: v for k, v in model.state_dict().items() if "lora_" in k}
torch.save(lora_sd, os.path.join(OUT_DIR, "lora_weights.pt"))
torch.save(model.get_input_embeddings().weight.data,
           os.path.join(OUT_DIR, "embedding_weights.pt"))
tok.save_pretrained(OUT_DIR)
json.dump({"tag": TAG, "mix": MIX, "budget_tokens": MIX_TOKENS,
           "actual_tokens": actual_tok, "base_model": MODEL_DIR,
           "lora_r": 64, "trainable": trn},
          open(os.path.join(OUT_DIR, "cpt_meta.json"), "w"), indent=2)
print(f"[{TAG}] DONE", flush=True)
