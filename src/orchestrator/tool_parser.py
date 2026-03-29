"""
Tool call parser for student model inference.

This module parses the pure text output from the student model (Qwen3-4B via vLLM),
extracting <tool_call> and <think> blocks for the agent environment to process.

See ADR-001 for architecture details:
- Teacher (data collection): Uses API function calling (stable output)
- Student (inference): Outputs raw text, parsed by THIS module

Used by:
- Student inference (src/orchestrator/orchestrator.py)
- SFT validation (src/training/sft/validate_rules.py) - to validate student output format

NOT used by:
- Trajectory collection (src/training/sft/collect_trajectories.py) - uses API directly
"""

import re
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any

# Valid tools for schema validation
VALID_TOOLS = frozenset([
    "get_food_nutrition",
    "log_meal",
    "get_today_summary",
    "get_history",
    "retrieve_knowledge",
    "set_goal",
])

# Regex pattern for extracting tool calls (end tag optional for truncated outputs)
TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*\})\s*(?:</tool_call>)?",
    re.DOTALL
)

# Regex pattern for extracting think blocks
THINK_PATTERN = re.compile(
    r"<think>(.*?)</think>",
    re.DOTALL
)


@dataclass
class ToolCall:
    """Parsed tool call with name and arguments."""
    name: str
    arguments: Dict[str, Any]


@dataclass
class ParseResult:
    """Result of parsing model output.

    Attributes:
        type: One of "tool_call", "final_answer", "parse_error"
        think: Extracted <think> content, if present
        tool_call: Parsed tool call, if type == "tool_call"
        content: Final answer text, if type == "final_answer"
        raw: Original text, if type == "parse_error"
        error_message: Description of parse error, if type == "parse_error"
    """
    type: str
    think: Optional[str] = None
    tool_call: Optional[ToolCall] = None
    content: Optional[str] = None
    raw: Optional[str] = None
    error_message: Optional[str] = None


class ToolParser:
    """Parser for extracting and validating tool calls from model output.

    This parser implements the pure text tool calling protocol:
    - Detects <tool_call>...</tool_call> tags
    - Extracts and validates JSON payload
    - Enforces single tool per turn (first match only)
    - Returns structured ParseResult

    Usage:
        parser = ToolParser()
        result = parser.parse(model_output)

        if result.type == "tool_call":
            name = result.tool_call.name
            args = result.tool_call.arguments
            # Execute tool...
        elif result.type == "final_answer":
            answer = result.content
        else:
            # Handle parse error
            ...
    """

    def __init__(self, validate_tool_name: bool = True):
        """Initialize parser.

        Args:
            validate_tool_name: If True, reject tool calls with unknown names.
                               Set to False during collection if teacher may
                               propose tools not in VALID_TOOLS.
        """
        self.validate_tool_name = validate_tool_name

    def parse(self, text: str) -> ParseResult:
        """Parse model output into structured result.

        Args:
            text: Raw model output text

        Returns:
            ParseResult with type, content, and optional tool call / think
        """
        if not text or not text.strip():
            return ParseResult(
                type="parse_error",
                raw=text,
                error_message="Empty response"
            )

        # Extract think block
        think_match = THINK_PATTERN.search(text)
        think_content = think_match.group(1).strip() if think_match else None

        # Extract tool call (first match only - sequential enforcement)
        tool_match = TOOL_CALL_PATTERN.search(text)

        if tool_match:
            return self._parse_tool_call(text, tool_match, think_content)
        else:
            return self._parse_final_answer(text, think_content)

    def _parse_tool_call(
        self,
        text: str,
        tool_match: re.Match,
        think_content: Optional[str]
    ) -> ParseResult:
        """Parse a tool call from regex match."""
        json_str = tool_match.group(1)

        try:
            payload = json.loads(json_str)
        except json.JSONDecodeError as e:
            return ParseResult(
                type="parse_error",
                think=think_content,
                raw=text,
                error_message=f"Invalid JSON in <tool_call>: {e}"
            )

        # Validate required field: name (support both "name" and "function" keys)
        name = payload.get("name") or payload.get("function")
        if not name:
            return ParseResult(
                type="parse_error",
                think=think_content,
                raw=text,
                error_message="Missing 'name' or 'function' field in tool call"
            )

        # Validate tool name if enabled
        if self.validate_tool_name and name not in VALID_TOOLS:
            return ParseResult(
                type="parse_error",
                think=think_content,
                raw=text,
                error_message=f"Unknown tool: {name}"
            )

        # Normalize arguments: ensure dict (support both "arguments" and "parameters" keys)
        arguments = payload.get("arguments") or payload.get("parameters") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return ParseResult(
                    type="parse_error",
                    think=think_content,
                    raw=text,
                    error_message="Invalid JSON in 'arguments' field"
                )

        return ParseResult(
            type="tool_call",
            think=think_content,
            tool_call=ToolCall(name=name, arguments=arguments)
        )

    def _parse_final_answer(
        self,
        text: str,
        think_content: Optional[str]
    ) -> ParseResult:
        """Parse as final answer (no tool call found)."""
        # Remove think blocks to get clean answer
        answer = THINK_PATTERN.sub("", text).strip()

        if answer:
            return ParseResult(
                type="final_answer",
                think=think_content,
                content=answer
            )
        else:
            # Only think block, no answer - incomplete
            return ParseResult(
                type="parse_error",
                think=think_content,
                raw=text,
                error_message="Response contains only <think> block, no answer"
            )

    def extract_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract tool call as dict, or None if not found.

        Simplified interface for collection scripts.

        Returns:
            {"name": str, "arguments": dict} or None
        """
        result = self.parse(text)
        if result.type == "tool_call" and result.tool_call:
            return {
                "name": result.tool_call.name,
                "arguments": result.tool_call.arguments
            }
        return None


def format_tool_response(result: Dict[str, Any], indent: int = 2) -> str:
    """Format tool execution result as <tool_response> block.

    Args:
        result: Tool execution result dict
        indent: JSON indentation (default 2)

    Returns:
        Formatted string with <tool_response> tags
    """
    json_str = json.dumps(result, ensure_ascii=False, indent=indent, default=str)
    return f"<tool_response>\n{json_str}\n</tool_response>"


def format_error_response(error_type: str, message: str) -> str:
    """Format error as <tool_response> block.

    Args:
        error_type: Error type identifier (e.g., "invalid_json", "unknown_tool")
        message: Human-readable error message

    Returns:
        Formatted error response with <tool_response> tags
    """
    error_dict = {
        "status": "error",
        "error_type": error_type,
        "message": message
    }
    return format_tool_response(error_dict)
