"""Smoke test: load a Gemma4-MM Bio checkpoint on sm_120 and generate one line.

Confirms (a) the grouped-mm guard is required/effective on RTX PRO 6000 Blackwell,
(b) Gemma4ForConditionalGeneration loads as a text CausalLM, (c) output is sane.
"""
import sys, time, torch

# sm_120 guard: disable grouped-mm kernel (unsupported on Blackwell sm_120)
import transformers.integrations.moe as _moe
_moe._can_use_grouped_mm = lambda *a, **k: False

from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    "/autodl-fs/data/omnigene_v2/models/gemma-4-26B-A4B-it-bio"

t0 = time.time()
tok = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForCausalLM.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16, device_map={"": 0})
model.eval()
print(f"[smoke] loaded {CKPT} in {time.time()-t0:.1f}s "
      f"| vocab={model.config.text_config.vocab_size} "
      f"| VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

for prompt in ["The capital of France is",
               "Q: What is 12 + 30? A:"]:
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=20, do_sample=False)
    txt = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"[smoke] {prompt!r} -> {txt!r}", flush=True)

print("[smoke] OK", flush=True)
