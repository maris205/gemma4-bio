#!/usr/bin/env python
# coding: utf-8
"""
Phase-II data prep: tokenize each SOURCE into its own binary pool ONCE.
Mixtures are then assembled at train time by sampling chunk-indices from
pools at the requested ratio -- so we tokenize each corpus a single time
and reuse it across all 7 models.

Pools written to $POOL_DIR/{source}.bin + {source}_index.npy
Sources: text, biolit, protein, dna   (structure folded into protein pool
optionally; kept separate here for clean composition ablation).

Reuses the streaming tokenize logic of omnigene_v2/scripts/cpt/1-prepare_cpt_data.py.
"""
import os, random, struct, json
import numpy as np
from transformers import AutoTokenizer

DATA_DIR = os.getenv("DATA_DIR", "/root/autodl-tmp/dnagpt/data")
MODEL_DIR = os.getenv("MODEL_DIR", "/autodl-fs/data/omnigene_v2/models/gemma-4-26B-A4B-it-bio")
POOL_DIR  = os.getenv("POOL_DIR", "/root/autodl-tmp/dnagpt/bio-trans2/data/pools")
os.makedirs(POOL_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
MAX_LENGTH = 1024
MIN_LENGTH = 64

# per-source target raw bytes to tokenize into a pool.
# We need enough tokens that the largest single-model demand for that source
# can be met without repetition. Largest demand per source across 7 models:
#   text:    100M tok (M0-base = 100% text)
#   protein: 0.7*0.5*100M = 35M (M3-bio50) ; C1-protein = 20M -> 35M max
#   dna:     20M (C2-dna) ; mixture dna share <= 0.2*0.5*100M=10M -> 20M max
#   biolit:  0.1*0.5*100M = 5M -> 10M safe
# tokens ~= bytes/4 for text, ~bytes/3 for sequence. Over-provision 1.5x.
SOURCES = {
    "text":    {"files": [("openwebtext.txt", 1.0)],           "target_gb": 1.2},
    "biolit":  {"files": [("/autodl-fs/data/s2orc_biology_text.txt", 1.0)], "target_gb": 0.3},
    "protein": {"files": [("protein_uni_16.txt", 0.5), ("protein_lucaone_15g.txt", 0.5)], "target_gb": 0.7},
    "dna":     {"files": [("dna_32g.txt", 1.0)],               "target_gb": 0.5},
}

print("Loading tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
print(f"  vocab={len(tok)}", flush=True)


def resolve(fp):
    return fp if fp.startswith("/") else os.path.join(DATA_DIR, fp)


def stream_pool(source, spec):
    bin_file = f"{POOL_DIR}/{source}.bin"
    if os.path.exists(bin_file):
        os.remove(bin_file)
    target_bytes = int(spec["target_gb"] * 1024**3)
    total_chunks = 0
    total_tokens = 0
    per_file_bytes = target_bytes // len(spec["files"]) if len(spec["files"]) > 1 else target_bytes
    with open(bin_file, "ab") as fout:
        for fname, frac in spec["files"]:
            budget = int(target_bytes * frac)
            path = resolve(fname)
            print(f"\n[{source}] {path} budget={budget/1024**3:.2f}GB", flush=True)
            written = 0
            buf, buf_len = [], 0
            with open(path, "r", errors="ignore") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    buf.append(line); buf_len += len(line)
                    if buf_len >= MAX_LENGTH * 4:
                        ids = tok.encode(" ".join(buf), add_special_tokens=False)
                        for j in range(0, len(ids), MAX_LENGTH):
                            chunk = ids[j:j+MAX_LENGTH]
                            if len(chunk) >= MIN_LENGTH:
                                arr = np.array(chunk, dtype=np.uint32)
                                fout.write(struct.pack("I", len(chunk)))
                                fout.write(arr.tobytes())
                                total_chunks += 1; total_tokens += len(chunk)
                        written += buf_len; buf, buf_len = [], 0
                        if total_chunks % 20000 == 0:
                            print(f"  {source}: {total_chunks} chunks, {written/1024**3:.2f}GB", flush=True)
                    if written >= budget:
                        break
    # index
    index = []
    offset = 0
    with open(bin_file, "rb") as f:
        while True:
            h = f.read(4)
            if not h or len(h) < 4:
                break
            length = struct.unpack("I", h)[0]
            index.append((offset, length))
            offset += 4 + length * 4
            f.seek(offset)
    np.save(f"{POOL_DIR}/{source}_index.npy", np.array(index, dtype=np.int64))
    print(f"[{source}] DONE {total_chunks} chunks {total_tokens:,} tokens "
          f"({os.path.getsize(bin_file)/1024**3:.2f}GB)", flush=True)
    return {"chunks": total_chunks, "tokens": int(total_tokens)}


if __name__ == "__main__":
    only = os.getenv("ONLY_SOURCE")  # optional: prep a single source
    stats = {}
    for src, spec in SOURCES.items():
        if only and src != only:
            continue
        stats[src] = stream_pool(src, spec)
    meta_path = f"{POOL_DIR}/pools_meta.json"
    if os.path.exists(meta_path):
        stats = {**json.load(open(meta_path)), **stats}
    json.dump(stats, open(meta_path, "w"), indent=2)
    print("\nPools ready:", json.dumps(stats, indent=2), flush=True)
