#!/usr/bin/env python
# coding: utf-8
"""
Scaling experiment (review2.md "必做2"): does the Phase-2 capability-shaping
result (bio50 preserves general axis, lifts biology axis, at matched 100M
tokens vs a text-only 100M control) hold at smaller, DENSE Gemma-4 scales
(E2B ~2.3B, E4B ~4.5B -- confirmed enable_moe_block=False, so this is a size
mirror, not a router experiment)?

Reuses the exact Phase-2 protocol: LoRA CPT (bf16, no 4-bit needed at this
scale) over attention+MLP, embeddings unfrozen, 100M tokens, from the model's
own -it checkpoint (E2B-it / E4B-it are the size-matched counterparts of the
26B-A4B-it lineage's origin). No MoE router target (these models have none).

Config via env:
  TAG        model tag (e.g. E2B-text100, E2B-bio50, E4B-text100, E4B-bio50)
  MODEL_DIR  gemma-4-E2B-it or gemma-4-E4B-it path
  MIX        json ratio over {text,protein,dna,biolit}
  MIX_TOKENS default 100_000_000
  POOL_DIR   gemma4_small_pools (tokenized with the E2B/E4B native tokenizer)
  OUT_ROOT
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sm120_guard  # noqa: F401  dtype-safe MoE fallback; harmless no-op for dense models

import json, struct
import numpy as np
import torch

from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer)
from peft import LoraConfig, inject_adapter_in_model

TAG        = os.environ["TAG"]
MODEL_DIR  = os.environ["MODEL_DIR"]
MIX        = json.loads(os.environ["MIX"])
MIX_TOKENS = int(os.getenv("MIX_TOKENS", "100000000"))
POOL_DIR   = os.getenv("POOL_DIR", "/autodl-fs/data/bt2_phase2/gemma4_small_pools")
OUT_ROOT   = os.getenv("OUT_ROOT", "/autodl-fs/data/bt2_phase2/gemma4_small_models")
MAX_LENGTH = 1024
SEED = 42

OUT_DIR = os.path.join(OUT_ROOT, TAG)
CKPT_DIR = os.path.join(OUT_ROOT, "_checkpoints", TAG)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)


def build_mix_index():
    rows = []
    sources = sorted(MIX.keys())
    src_bins = {}
    for sid, src in enumerate(sources):
        idx = np.load(f"{POOL_DIR}/{src}_index.npy")
        src_bins[sid] = f"{POOL_DIR}/{src}.bin"
        want_tok = int(MIX_TOKENS * MIX[src])
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


print(f"=== {TAG}  model={MODEL_DIR}  budget={MIX_TOKENS:,} tok  mix={MIX} ===", flush=True)
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
                "attention_mask": torch.tensor(am)}


print("Loading model (bf16)...", flush=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16,
    device_map={"": 0})
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

lora = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, bias="none",
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'])  # no router: dense model
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
    per_device_train_batch_size=4, gradient_accumulation_steps=8,   # eff batch 32
    optim="adamw_torch", learning_rate=2e-5, lr_scheduler_type="cosine",
    warmup_ratio=0.03, weight_decay=0.01, bf16=True, max_grad_norm=1.0,
    logging_steps=25, save_strategy="no", report_to="none",
    dataloader_num_workers=4, dataloader_pin_memory=True,
    remove_unused_columns=False)

trainer = Trainer(model=model, args=args, train_dataset=MixDataset(),
                  data_collator=Collator(tok))
eff = args.per_device_train_batch_size * args.gradient_accumulation_steps
print(f"Steps ~{len(MIX_INDEX)//eff}  eff_batch={eff}", flush=True)
trainer.train()

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
