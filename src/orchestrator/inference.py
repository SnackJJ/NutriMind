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
