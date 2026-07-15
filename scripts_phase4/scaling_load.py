"""Reconstruct a scaling-experiment (E2B/E4B) CPT model from its saved adapter.

Same pattern as bt2_load.py, but these are DENSE Gemma-4 models (no MoE
router), so the LoRA target_modules list omits 'router.proj'. base_model path
comes from each model's own cpt_meta.json (E2B-it or E4B-it), not a fixed
26B default.
"""
import json, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, inject_adapter_in_model

OUT_ROOT = os.getenv("SCALING_OUT_ROOT", "/autodl-fs/data/bt2_phase2/gemma4_small_models")


def load_scaling_model(tag, device=0, merge=True):
    d = os.path.join(OUT_ROOT, tag)
    meta = json.load(open(os.path.join(d, "cpt_meta.json")))
    base = meta["base_model"]
    r = meta.get("lora_r", 64)

    tok = AutoTokenizer.from_pretrained(d)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map={"": device})

    lora = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, bias="none",
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'])  # no router: dense model
    inject_adapter_in_model(lora, model.model.language_model, adapter_name="default")

    emb = torch.load(os.path.join(d, "embedding_weights.pt"), map_location="cpu")
    with torch.no_grad():
        model.get_input_embeddings().weight.copy_(
            emb.to(model.get_input_embeddings().weight.dtype))

    lora_sd = torch.load(os.path.join(d, "lora_weights.pt"), map_location="cpu")
    missing, unexpected = model.load_state_dict(lora_sd, strict=False)
    loaded = len(lora_sd) - len(unexpected)
    print(f"[scaling_load:{tag}] LoRA tensors loaded={loaded}/{len(lora_sd)} "
          f"unexpected={len(unexpected)}", flush=True)
    assert loaded > 0, f"no LoRA tensors matched for {tag}"

    if merge:
        from peft.tuners.lora import LoraLayer
        for module in model.modules():
            if isinstance(module, LoraLayer):
                module.merge()
    model.eval()
    return model, tok, meta
