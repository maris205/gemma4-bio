#!/usr/bin/env python
# coding: utf-8
"""
Scaling experiment (review2.md "必做2") data prep.

Tokenizes text/protein/dna sources with the E2B/E4B NATIVE tokenizer (262144
vocab, shared identically by both -- confirmed via direct comparison), since
they do NOT have the bio-expanded 290172 vocab used by the 26B-A4B-it-bio
lineage in Phase-2. Pools are separate from bt2_phase2/pools for this reason.

Mirrors bio-trans2/scripts/prepare_pools.py (Phase-2) / bio2nl prepare_pools.py
(Part III) exactly, just re-tokenized for this vocab.
"""
import os, random, struct
import numpy as np
from transformers import AutoTokenizer

DATA_DIR = os.getenv("DATA_DIR", "/root/autodl-tmp/dnagpt/data")
MODEL_DIR = os.getenv("MODEL_DIR", "/autodl-fs/data/bt2_phase2/gemma4_small/gemma-4-E2B-it")
POOL_DIR = os.getenv("POOL_DIR", "/autodl-fs/data/bt2_phase2/gemma4_small_pools")
os.makedirs(POOL_DIR, exist_ok=True)
random.seed(42)
MAX_LENGTH = 1024
MIN_LENGTH = 64

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
print(f"tokenizer vocab={len(tok)}", flush=True)

# Scaling models need only 2 arms (text100 vs bio50) x 2 sizes (E2B, E4B),
# each 100M tokens like Phase-2's M0/M3 -- so max per-source demand:
#   text: 100M (text-only arm)      -> pool 250M (shared across E2B+E4B runs)
#   protein: 0.35*100M = 35M/arm    -> pool 100M
#   dna: 0.10*100M = 10M/arm        -> pool 40M
#   biolit: 0.05*100M = 5M/arm      -> pool 20M
SOURCES = {
    "text":    {"files": [("openwebtext.txt", 1.0)],           "target_tok": 250_000_000},
    "protein": {"files": [("protein_uni_16.txt", 0.5), ("protein_lucaone_15g.txt", 0.5)], "target_tok": 100_000_000},
    "dna":     {"files": [("dna_32g.txt", 1.0)],               "target_tok": 40_000_000},
    "biolit":  {"files": [("/autodl-fs/data/s2orc_biology_text.txt", 1.0)], "target_tok": 20_000_000},
}


def resolve(fp):
    return fp if fp.startswith("/") else os.path.join(DATA_DIR, fp)


def build(src, spec):
    binp = f"{POOL_DIR}/{src}.bin"
    if os.path.exists(binp):
        os.remove(binp)
    got = 0
    target = spec["target_tok"]
    with open(binp, "ab") as fout:
        for fname, frac in spec["files"]:
            budget = int(target * frac)
            sub = 0
            buf, blen = [], 0
            with open(resolve(fname), "r", errors="ignore") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    buf.append(line); blen += len(line)
                    if blen >= MAX_LENGTH * 4:
                        ids = tok.encode(" ".join(buf), add_special_tokens=False)
                        for j in range(0, len(ids), MAX_LENGTH):
                            ch = ids[j:j+MAX_LENGTH]
                            if len(ch) >= MIN_LENGTH:
                                a = np.array(ch, dtype=np.uint32)
                                fout.write(struct.pack("I", len(ch))); fout.write(a.tobytes())
                                got += len(ch); sub += len(ch)
                        buf, blen = [], 0
                        if got % 20_000_000 < MAX_LENGTH:
                            print(f"  {src}: {got/1e6:.0f}M tok", flush=True)
                    if sub >= budget:
                        break
    idx, off = [], 0
    with open(binp, "rb") as f:
        while True:
            h = f.read(4)
            if not h or len(h) < 4:
                break
            ln = struct.unpack("I", h)[0]
            idx.append((off, ln)); off += 4 + ln*4; f.seek(off)
    np.save(f"{POOL_DIR}/{src}_index.npy", np.array(idx, dtype=np.int64))
    print(f"[{src}] DONE {got:,} tok, {len(idx)} chunks ({os.path.getsize(binp)/1e9:.2f}GB)", flush=True)
    return got


if __name__ == "__main__":
    only = os.getenv("ONLY")
    for s, sp in SOURCES.items():
        if only and s != only:
            continue
        build(s, sp)
    print("GEMMA-SMALL POOLS DONE", flush=True)
