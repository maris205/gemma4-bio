"""sm_120 (Blackwell RTX PRO 6000) MoE guard for Gemma-4.

Two problems on sm_120 when TRAINING (float32 embeddings/LoRA feed float32
activations into bf16 MoE experts):
  1. the torch grouped_mm kernel is unsupported -> force the python fallback.
  2. the library dispatcher only casts input->weight.dtype on the *kernel*
     path; the fallback custom-op path (torch.ops.transformers.grouped_mm_fallback)
     receives raw float32 input against bf16 weight and crashes in torch.mm
     with "expected mat1 and mat2 to have the same dtype".

Fix: replace _moe._grouped_mm with a version that always casts input to
weight.dtype before dispatch (mirroring the kernel path). The cast is a
differentiable op, so float32 grads still route back to the float32 embedding.
Import this module BEFORE loading any Gemma-4 model.
"""
import torch
import transformers.integrations.moe as _moe

# 1. force the fallback path (no grouped_mm kernel on sm_120)
_moe._can_use_grouped_mm = lambda *a, **k: False


# 2. dtype-safe dispatcher: cast input to weight dtype, then use the fallback op
def _grouped_mm_safe(input, weight, offs):
    return torch.ops.transformers.grouped_mm_fallback(
        input.to(weight.dtype), weight, offs=offs)


_moe._grouped_mm = _grouped_mm_safe
