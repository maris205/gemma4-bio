#!/usr/bin/env python
# coding: utf-8
"""
review2.md "必做1": Router analysis across the Phase-2 checkpoint sweep.

Question: does biological CPT reshape expert-routing allocation for
biological inputs while leaving natural-language routing (largely) stable --
the router-level mechanism behind "general preserved, biology lifted"?

For each checkpoint (base it/it-bio/bio + Phase-2 mixture/composition models),
loads it (same load path as run_eval.py's bt2_load, for the 7 Phase-2 models;
plain load for the 3 base checkpoints), installs forward hooks on all 30
routers, feeds 3 prompt categories (protein, dna, natural_language, 40 prompts
each), records top-8 expert activations per layer, computes:
  - per-layer JS divergence: protein_vs_nl, dna_vs_nl, protein_vs_dna
  - a summary scalar: mean bio-vs-nl JS (avg of protein_vs_nl, dna_vs_nl)

If the "structure not content" story holds, bio-vs-nl JS should INCREASE with
biological share across M0->M1->M2->M3, while whatever routing NL prompts get
should stay comparatively stable across the same sweep (checked separately by
comparing nl-prompt expert distributions across models via JS as well).

Usage: CUDA_VISIBLE_DEVICES=<n> python run_router_analysis.py --tag <tag>
"""
import argparse, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sm120_guard  # noqa: F401
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

N_LAYERS, N_EXPERTS, TOP_K = 30, 128, 8
N_PROMPTS = 40
OUT_DIR = "/root/autodl-tmp/dnagpt/bio-trans2/results_router"
os.makedirs(OUT_DIR, exist_ok=True)
random.seed(42)

BASE_CKPTS = {
    "it":      "/autodl-fs/data/omnigene_v2/models/gemma-4-26B-A4B-it",
    "it-bio":  "/autodl-fs/data/omnigene_v2/models/gemma-4-26B-A4B-it-bio",
    "bio":     "/autodl-fs/data/omnigene_v2/models/gemma-4-26B-A4B-bio",
}
PHASE2_ROOT = "/autodl-fs/data/bt2_phase2/models"
PHASE2_BASE = "/autodl-fs/data/omnigene_v2/models/gemma-4-26B-A4B-it-bio"


def load_model(tag):
    if tag in BASE_CKPTS:
        path = BASE_CKPTS[tag]
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
            device_map={"": 0})
        return model, tok, {"tag": tag, "mix": None}
    # Phase-2 model: base it-bio + LoRA + trained embedding, merged
    from peft import LoraConfig, inject_adapter_in_model
    from peft.tuners.lora import LoraLayer
    d = os.path.join(PHASE2_ROOT, tag)
    meta = json.load(open(os.path.join(d, "cpt_meta.json")))
    tok = AutoTokenizer.from_pretrained(d)
    model = AutoModelForCausalLM.from_pretrained(PHASE2_BASE, torch_dtype=torch.bfloat16,
        device_map={"": 0})
    lora = LoraConfig(r=meta.get("lora_r", 64), lora_alpha=2*meta.get("lora_r", 64),
        lora_dropout=0.0, bias="none",
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj', 'router.proj'])
    inject_adapter_in_model(lora, model.model.language_model, adapter_name="default")
    emb = torch.load(os.path.join(d, "embedding_weights.pt"), map_location="cpu")
    with torch.no_grad():
        model.get_input_embeddings().weight.copy_(emb.to(model.get_input_embeddings().weight.dtype))
    lora_sd = torch.load(os.path.join(d, "lora_weights.pt"), map_location="cpu")
    missing, unexpected = model.load_state_dict(lora_sd, strict=False)
    loaded = len(lora_sd) - len(unexpected)
    print(f"[router:{tag}] LoRA tensors loaded={loaded}/{len(lora_sd)}", flush=True)
    assert loaded > 0
    for module in model.modules():
        if isinstance(module, LoraLayer):
            module.merge()
    return model, tok, meta


def build_prompts():
    # protein: homology-pair prompts (real biological content)
    rows = [json.loads(l) for l in open(
        "/root/autodl-tmp/dnagpt/biopaws2/data/protein_homology_std.jsonl") if l.strip()]
    prot = [r["messages"][0]["content"] for r in random.sample(rows, N_PROMPTS)]

    # dna: real DNA chunks sampled from the Phase-2 dna pool
    import struct
    idx = np.load("/autodl-fs/data/bt2_phase2/pools/dna_index.npy")
    with open("/autodl-fs/data/bt2_phase2/pools/dna.bin", "rb") as f:
        picks = random.sample(range(len(idx)), N_PROMPTS)
        dna_tok_ids = []
        for k in picks:
            off, ln = int(idx[k][0]), int(idx[k][1])
            f.seek(off + 4)
            ids = np.frombuffer(f.read(ln*4), dtype=np.uint32).astype(np.int64).tolist()
            dna_tok_ids.append(ids[:200])  # short chunk

    # natural language: biology-flavored English questions (varied, not repeated)
    nl_templates = [
        "What is the function of {p} in the cell?",
        "Explain how {p} contributes to cellular metabolism.",
        "Describe the role of {p} in gene regulation.",
        "What happens when {p} is mutated?",
        "How does {p} interact with other proteins in a signaling pathway?",
    ]
    proteins_for_nl = ["hemoglobin", "insulin", "p53", "kinases", "ribosomes",
                       "mitochondria", "actin", "collagen", "myosin", "cytochrome c",
                       "DNA polymerase", "RNA polymerase", "histones", "chaperones",
                       "transcription factors", "ATP synthase", "hemoglobin subunits",
                       "topoisomerase", "helicase", "ligase"]
    nl = []
    for p in proteins_for_nl:
        for t in nl_templates:
            nl.append(t.format(p=p))
    nl = random.sample(nl, N_PROMPTS)
    return prot, dna_tok_ids, nl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    model, tok, meta = load_model(a.tag)
    model.eval()
    print(f"[router:{a.tag}] loaded mix={meta.get('mix')} VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB",
          flush=True)

    expert_counts = {}
    current = {"task": None}

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            if not isinstance(output, tuple) or len(output) < 3:
                return
            idx_ = output[2].detach().cpu().numpy()
            task = current["task"]
            if task is None:
                return
            if task not in expert_counts:
                expert_counts[task] = np.zeros((N_LAYERS, N_EXPERTS), dtype=np.float32)
            for tokpos in range(idx_.shape[0]):
                for k in range(idx_.shape[1]):
                    e = int(idx_[tokpos, k])
                    if 0 <= e < N_EXPERTS:
                        expert_counts[task][layer_idx, e] += 1
        return hook

    hooks = []
    lm = model.model.language_model
    for i, layer in enumerate(lm.layers):
        if hasattr(layer, "router"):
            hooks.append(layer.router.register_forward_hook(make_hook(i)))
    print(f"[router:{a.tag}] attached {len(hooks)}/{len(lm.layers)} routers", flush=True)

    prot_prompts, dna_tok_lists, nl_prompts = build_prompts()

    @torch.no_grad()
    def run_text(prompt, task):
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(model.device)
        current["task"] = task
        model(input_ids=ids, use_cache=False)

    @torch.no_grad()
    def run_ids(id_list, task):
        ids = torch.tensor([id_list], dtype=torch.long, device=model.device)
        current["task"] = task
        model(input_ids=ids, use_cache=False)

    for i, p in enumerate(prot_prompts):
        run_text(p, "protein")
    for i, ids in enumerate(dna_tok_lists):
        run_ids(ids, "dna")
    for i, p in enumerate(nl_prompts):
        run_text(p, "natural_language")

    for h in hooks:
        h.remove()
    print(f"[router:{a.tag}] collected: " +
          ", ".join(f"{k}={v.sum():.0f}" for k, v in expert_counts.items()), flush=True)

    def js(p, q):
        p = p / max(p.sum(), 1); q = q / max(q.sum(), 1)
        m = 0.5 * (p + q)
        def kl(a, b):
            mask = (a > 0) & (b > 0)
            return float((a[mask] * np.log(a[mask] / b[mask])).sum())
        return 0.5 * kl(p, m) + 0.5 * kl(q, m)

    modalities = sorted(expert_counts.keys())
    per_layer = {}
    for i, t1 in enumerate(modalities):
        for j, t2 in enumerate(modalities):
            if i >= j:
                continue
            per_layer[f"{t1}_vs_{t2}"] = [js(expert_counts[t1][L], expert_counts[t2][L])
                                          for L in range(N_LAYERS)]
    layer_avg = {k: float(np.mean(v)) for k, v in per_layer.items()}

    out = {
        "tag": a.tag, "mix": meta.get("mix"),
        "modalities": modalities,
        "layer_averaged_js": layer_avg,
        "per_layer_js": per_layer,
    }
    for task, mat in expert_counts.items():
        np.savez_compressed(os.path.join(OUT_DIR, f"routing_counts_{a.tag}_{task}.npz"), counts=mat)
    json.dump(out, open(os.path.join(OUT_DIR, f"routing_{a.tag}.json"), "w"), indent=2)
    print(f"[router:{a.tag}] layer_avg_js={json.dumps(layer_avg)}", flush=True)
    print(f"[router:{a.tag}] DONE", flush=True)


if __name__ == "__main__":
    main()
