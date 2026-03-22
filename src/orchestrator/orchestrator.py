import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Callable
import yaml
from src.utils.logger import logger
from src.tools.get_food_nutrition import get_food_nutrition
from src.tools.log_meal import log_meal
from src.tools.get_today_summary import get_today_summary
from src.tools.get_history import get_history
from src.tools.retrieve_knowledge import retrieve_knowledge
from src.tools.set_goal import set_goal
from src.orchestrator.inference import MockBackend, VLLMBackend
from src.orchestrator.tool_parser import (
    ToolParser,
    ParseResult,
    format_tool_response,
    format_error_response,
)

@dataclass
class OrchestratorConfig:
    max_tool_rounds: int = 6
    tool_timeout_ms: int = 10000
    max_retries_per_tool: int = 2

class ToolExecutionError(Exception):
    def __init__(self, type_: str, message: str):
        self.type = type_
        self.message = message
        super().__init__(message)

CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"
MAX_INPUT_LENGTH = 4096

TOOL_REGISTRY: Dict[str, Callable] = {
    "get_food_nutrition": get_food_nutrition,
    "log_meal": log_meal,
    "get_today_summary": get_today_summary,
    "get_history": get_history,
    "retrieve_knowledge": retrieve_knowledge,
    "set_goal": set_goal,
}

NO_ARGS_TOOLS = frozenset(["get_today_summary"])

def load_config() -> OrchestratorConfig:
    config_path = CONFIG_DIR / "orchestrator.yaml"
    try:
        with open(config_path, "r") as f:
            c = yaml.safe_load(f)
        return OrchestratorConfig(**c)
    except FileNotFoundError:
        logger.debug(f"Config file not found at {config_path}, using defaults")
        return OrchestratorConfig()
    except Exception as e:
        logger.warning(f"Failed to load orchestrator config: {e}, using defaults")
        return OrchestratorConfig()

def get_backend():
    config_path = CONFIG_DIR / "model.yaml"
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        if cfg.get("backend") == "vllm":
            return VLLMBackend(
                cfg.get("server_url", "http://localhost:8000/v1"),
                cfg.get("model_name", "Qwen/Qwen3-4B"),
                cfg.get("max_tokens", 1024),
                cfg.get("temperature", 0.1)
            )
    except FileNotFoundError:
        logger.debug(f"Model config not found at {config_path}, using MockBackend")
    except Exception as e:
        logger.warning(f"Failed to load model config: {e}, using MockBackend")
    return MockBackend()

def execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if name not in TOOL_REGISTRY:
        raise ToolExecutionError("tool_not_found", f"Tool {name} does not exist.")

    try:
        if name in NO_ARGS_TOOLS:
            res = TOOL_REGISTRY[name]()
        else:
            res = TOOL_REGISTRY[name](**args)

        if res.get("status") == "error":
            raise ToolExecutionError(res.get("error_type", "unknown_error"), res.get("message", "Error"))
        return res
    except TypeError as e:
        raise ToolExecutionError("invalid_arguments", str(e))
    except ToolExecutionError:
        raise
    except Exception as e:
        raise ToolExecutionError("internal_error", str(e))


# Shared parser instance (see ADR-001: Pure Text Tool Calling)
_parser = ToolParser(validate_tool_name=True)

SYSTEM_PROMPT = """You are NutriMind, a nutrition assistant. You have access to specific tools to retrieve data.
When you need to use a tool, use XML tags: <think>thought process</think><tool_call>{"name": "...", "arguments": {...}}</tool_call>
If you answer directly, do not use tool tags. Only use JSON in tool calls. Do not use parallel tool calls."""

def orchestrate(user_input: str) -> str:
    """Main orchestration loop using pure text tool calling.

    See ADR-001: Pure Text Tool Calling for protocol details.
    """
    if not user_input or not user_input.strip():
        return "Error: Empty input provided."
    if len(user_input) > MAX_INPUT_LENGTH:
        return f"Error: Input exceeds maximum length of {MAX_INPUT_LENGTH} characters."

    config = load_config()
    backend = get_backend()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    tool_round = 0
    while tool_round < config.max_tool_rounds:
        logger.debug(f"Orchestrator INFERENCE | Round {tool_round}")
        response = backend.generate(messages)
        parsed: ParseResult = _parser.parse(response)

        if parsed.type == "final_answer":
            logger.info("Orchestrator ANSWER")
            return parsed.content

        elif parsed.type == "tool_call":
            tool_name = parsed.tool_call.name
            tool_args = parsed.tool_call.arguments
            logger.info(f"Orchestrator TOOL_CALL | {tool_name}")

            try:
                result = execute_tool(tool_name, tool_args)
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": format_tool_response(result)})
            except ToolExecutionError as e:
                logger.warning(f"Orchestrator ERROR ToolExecutionError: {e.message}")
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": format_error_response(e.type, e.message)})

            tool_round += 1

        elif parsed.type == "parse_error":
            logger.warning(f"Orchestrator ERROR Parse: {parsed.error_message}")
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": "Your previous response had invalid format. Please try again with valid JSON in <tool_call> tags."})
            tool_round += 1

    logger.warning("Orchestrator CHECK_LIMIT Exceeded max tool rounds")
    return "Error: Max tool rounds exceeded."
