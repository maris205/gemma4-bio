#!/usr/bin/env python
# coding: utf-8
"""
Bio2NL Tier-1 data prep: tokenize each source into a GPT-2-tokenized pool ONCE.
Sources: nl (openwebtext), protein (uni+lucaone), dna.
Pools reused across warmup + both arms (NL-arm continuation, Bio-arm continuation).

Binary format per chunk: [uint32 length][uint32 token_ids...], + <src>_index.npy.
GPT-2 context = 1024; we pack to BLOCK tokens.
"""
import os, random, struct
import numpy as np
from transformers import AutoTokenizer

DATA = os.getenv("DATA_DIR", "/root/autodl-tmp/dnagpt/data")
POOL = os.getenv("POOL_DIR", "/root/autodl-tmp/bio-trans/bio2nl/tier1_gpt2/pools")
os.makedirs(POOL, exist_ok=True)
random.seed(42)
BLOCK = 1024
MINLEN = 128

tok = AutoTokenizer.from_pretrained("gpt2")
print(f"gpt2 vocab={len(tok)}", flush=True)

# target tokens per pool (over-provision vs demand):
#   nl warmup 300M + nl-arm 200M = 500M NL demand -> pool 600M
#   bio-arm 200M (protein+dna) -> protein pool 250M + dna pool 250M
SOURCES = {
    "nl":      {"files": [("openwebtext.txt", 1.0)],           "target_tok": 600_000_000},
    "protein": {"files": [("protein_uni_16.txt", 0.5), ("protein_lucaone_15g.txt", 0.5)], "target_tok": 260_000_000},
    "dna":     {"files": [("dna_32g.txt", 1.0)],               "target_tok": 260_000_000},
}


def resolve(fp):
    return fp if fp.startswith("/") else os.path.join(DATA, fp)


def build(src, spec):
    binp = f"{POOL}/{src}.bin"
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
                    if blen >= BLOCK * 4:
                        ids = tok.encode(" ".join(buf), add_special_tokens=False)
                        for j in range(0, len(ids), BLOCK):
                            ch = ids[j:j+BLOCK]
                            if len(ch) >= MINLEN:
                                a = np.array(ch, dtype=np.uint32)
                                fout.write(struct.pack("I", len(ch))); fout.write(a.tobytes())
                                got += len(ch); sub += len(ch)
                        buf, blen = [], 0
                        if got % 20_000_000 < BLOCK:
                            print(f"  {src}: {got/1e6:.0f}M tok", flush=True)
                    if sub >= budget:
                        break
    # index
    idx, off = [], 0
    with open(binp, "rb") as f:
        while True:
            h = f.read(4)
            if not h or len(h) < 4:
                break
            ln = struct.unpack("I", h)[0]
            idx.append((off, ln)); off += 4 + ln*4; f.seek(off)
    np.save(f"{POOL}/{src}_index.npy", np.array(idx, dtype=np.int64))
    print(f"[{src}] DONE {got:,} tok, {len(idx)} chunks ({os.path.getsize(binp)/1e9:.2f}GB)", flush=True)
    return got


if __name__ == "__main__":
    only = os.getenv("ONLY")
    for s, sp in SOURCES.items():
        if only and s != only:
            continue
        build(s, sp)
    print("POOLS DONE", flush=True)
