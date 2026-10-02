"""Register vLLM 0.24's existing text-only Qwen3.5 implementation.

0.24 contains the class but omits its registry entry and hybrid-cache interface. Our SFT export is a
Qwen3_5ForCausalLM, not a multimodal Qwen3_5ForConditionalGeneration. No model
code or weights are changed; the standard plugin runs in all vLLM processes.
"""


def register():
    from vllm import ModelRegistry

    if "Qwen3_5ForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("Qwen3_5ForCausalLM",
            "nutrimind_vllm_compat_model:Qwen3_5TextForCausalLM")
