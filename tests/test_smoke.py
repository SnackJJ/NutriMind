import unittest
from src.orchestrator.orchestrator import orchestrate
from src.orchestrator.inference import MockBackend
import src.orchestrator.orchestrator as orch_module

class CustomMockBackend(MockBackend):
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0
        
    def generate(self, messages):
        response = self.responses[self.call_count]
        self.call_count += 1
        return response

class TestOrchestrator(unittest.TestCase):
    def test_t1_single_step(self):
        responses = [
            "<think>T1 lookup</think>\n<tool_call>{\"name\": \"get_food_nutrition\", \"arguments\": {\"foods\": [{\"food_name\": \"chicken breast\", \"amount_grams\": 100}]}}</tool_call>",
            "<think>Got it</think>\nChicken breast has 165 kcal."
        ]
        
        original_get_backend = orch_module.get_backend
        orch_module.get_backend = lambda: CustomMockBackend(responses)
        
        ans = orchestrate("How much calories in chicken breast?")
        self.assertIn("165 kcal", ans)
        
        orch_module.get_backend = original_get_backend

    def test_t4_safety(self):
        responses = [
            "<think>Dialysis means safety declaration</think>\nYour situation involves complex medical nutrition management that exceeds my safe service boundary. Please consult your physician."
        ]
        
        original_get_backend = orch_module.get_backend
        orch_module.get_backend = lambda: CustomMockBackend(responses)
        
        ans = orchestrate("I'm on dialysis.")
        self.assertIn("safe service boundary", ans)
        
        orch_module.get_backend = original_get_backend

if __name__ == '__main__':
    unittest.main()
