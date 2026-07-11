"""Reconstruct a Phase-2 CPT model from its saved adapter for evaluation.

Phase-2 training saves (per model dir):
  lora_weights.pt       - state dict of LoRA A/B tensors ("...lora_...")
  embedding_weights.pt  - float32 trained input embedding (incl. new bio tokens)
  cpt_meta.json         - {tag, mix, base_model, lora_r, ...}

We rebuild: base (it-bio, bf16) -> inject same LoRA config -> load LoRA weights
-> overwrite input embedding with the trained one (cast to bf16) -> merge LoRA
into base -> return a plain bf16 CausalLM suitable for lm-eval HFLM.

Since the merged model is pure bf16 (no float32 activations), only the plain
grouped-mm guard is needed here (installed by the importer before model load).
"""
import json, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, inject_adapter_in_model

OUT_ROOT = os.getenv("OUT_ROOT", "/autodl-fs/data/bt2_phase2/models")
BASE_DEFAULT = "/autodl-fs/data/omnigene_v2/models/gemma-4-26B-A4B-it-bio"


def load_cpt_model(tag, device=0, merge=True):
    d = os.path.join(OUT_ROOT, tag)
    meta = json.load(open(os.path.join(d, "cpt_meta.json")))
    base = meta.get("base_model", BASE_DEFAULT)
    r = meta.get("lora_r", 64)

    tok = AutoTokenizer.from_pretrained(d)  # tokenizer saved with the model
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map={"": device})

    lora = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, bias="none",
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj', 'router.proj'])
    inject_adapter_in_model(lora, model.model.language_model, adapter_name="default")

    # load trained embedding (float32 on disk -> bf16 for eval)
    emb = torch.load(os.path.join(d, "embedding_weights.pt"), map_location="cpu")
    with torch.no_grad():
        model.get_input_embeddings().weight.copy_(
            emb.to(model.get_input_embeddings().weight.dtype))

    # load LoRA weights
    lora_sd = torch.load(os.path.join(d, "lora_weights.pt"), map_location="cpu")
    missing, unexpected = model.load_state_dict(lora_sd, strict=False)
    loaded = len(lora_sd) - len(unexpected)
    print(f"[bt2_load:{tag}] LoRA tensors loaded={loaded}/{len(lora_sd)} "
          f"unexpected={len(unexpected)}", flush=True)
    assert loaded > 0, f"no LoRA tensors matched for {tag}"

    if merge:
        # merge adapter into base weights for clean, fast eval
        from peft.tuners.lora import LoraLayer
        for module in model.modules():
            if isinstance(module, LoraLayer):
                module.merge()
    model.eval()
    return model, tok, meta
