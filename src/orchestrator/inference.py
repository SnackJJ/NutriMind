from typing import List, Dict, Any
from abc import ABC, abstractmethod

class InferenceBackend(ABC):
    @abstractmethod
    def generate(self, messages: List[Dict[str, Any]]) -> str:
        pass

class MockBackend(InferenceBackend):
    def generate(self, messages: List[Dict[str, Any]]) -> str:
        # Simple mock response that will act based on the last user message and the mocked logic.
        # This is strictly a placeholder for Phase 1 Smoke Test.
        return "<think>Mocking</think>Final mock answer based on context."

class VLLMBackend(InferenceBackend):
    def __init__(self, server_url: str, model_name: str, max_tokens: int, temperature: float):
        import httpx
        self.server_url = server_url
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = httpx.Client()
        
    def generate(self, messages: List[Dict[str, Any]]) -> str:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        res = self.client.post(f"{self.server_url}/chat/completions", json=payload)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]

class LocalTransformersBackend(InferenceBackend):
    """Zero-overhead Transformers inference backend for local Testing."""
    def __init__(self, model_path: str, max_tokens: int, temperature: float):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        import torch
        import json
        from pathlib import Path
        
        # Load adapter config to find base model
        adapter_config_path = Path(model_path) / "adapter_config.json"
        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)
        base_model_path = adapter_config.get("base_model_name_or_path")
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
        )
        
        # Load Base
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map='auto',
            trust_remote_code=True,
            local_files_only=True,
        )
        
        # Load Adapter
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()
        
        self.max_tokens = max_tokens
        self.temperature = temperature



        
    def generate(self, messages: List[Dict[str, Any]]) -> str:
        import torch
        # Apply chat template (with reasoning support if enabled)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True, # Critical for SFT-v2
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.9,
                do_sample=True if self.temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        # Safely extract response by manually slicing off the Qwen3 end token
        # This absolutely guarantees <think> and <tool_call> are preserved even if forced to special tokens
        return raw_text.split("<|im_end|>")[0].strip()

