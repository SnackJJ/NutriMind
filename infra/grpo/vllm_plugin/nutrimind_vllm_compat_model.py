"""Expose the existing Qwen3.5 text model's hybrid-cache metadata.

The cache calculations are the same upstream methods used by its multimodal
wrapper. Newer vLLM releases put these on the text base class itself. Model
forward, weight loading and all attention kernels remain upstream implementations.
"""

import torch
from vllm.model_executor.models.interfaces import IsHybrid, SupportsMRoPE
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
)


class Qwen3_5TextForCausalLM(Qwen3_5ForCausalLM, IsHybrid, SupportsMRoPE):
    get_mamba_state_dtype_from_config = Qwen3_5ForConditionalGeneration.__dict__["get_mamba_state_dtype_from_config"]
    get_mamba_state_shape_from_config = Qwen3_5ForConditionalGeneration.__dict__["get_mamba_state_shape_from_config"]
    get_mamba_state_copy_func = Qwen3_5ForConditionalGeneration.__dict__["get_mamba_state_copy_func"]

    def get_mrope_input_positions(self, input_tokens, mm_features):
        if mm_features:
            raise ValueError("NutriMind uses a text-only checkpoint")
        # Same text positions as vLLM 0.29's Qwen3_5ForCausalLMBase.
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1), 0
