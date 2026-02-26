# Orchestrator Specification

The Orchestrator is the central state machine that manages the agentic loop between the 3B model and tools.

## State Machine

```
                    ┌─────────────┐
                    │    START    │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  INFERENCE  │◄────────────────┐
                    │  (3B Model) │                 │
                    └──────┬──────┘                 │
                           │                        │
              ┌────────────┼────────────┐           │
              │            │            │           │
              ▼            ▼            ▼           │
        ┌──────────┐ ┌──────────┐ ┌──────────┐      │
        │TOOL_CALL │ │  ANSWER  │ │  ERROR   │      │
        └────┬─────┘ └────┬─────┘ └────┬─────┘      │
             │            │            │            │
             ▼            │            │            │
       ┌───────────┐      │            │            │
       │ TOOL_EXEC │      │            │            │
       └─────┬─────┘      │            │            │
             │            │            │            │
             ▼            │            │            │
      ┌────────────┐      │            │            │
      │CHECK_LIMIT │──────┼────────────┘            │
      └─────┬──────┘      │                         │
            │             │                         │
    ┌───────┴───────┐     │                         │
    │ within limit  │     │                         │
    └───────┬───────┘     │                         │
            │             │                         │
            └─────────────┼─────────────────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │     END     │
                    └─────────────┘
```

## States

| State | Description |
|-------|-------------|
| `START` | Receive user input, initialize context |
| `INFERENCE` | Call 3B model, parse output |
| `TOOL_CALL` | Detected `<tool_call>` in output |
| `TOOL_EXEC` | Execute tool, get result |
| `CHECK_LIMIT` | Check if max rounds exceeded |
| `ANSWER` | Model produced final answer (no tool call) |
| `ERROR` | Unrecoverable error occurred |
| `END` | Return response to user |

## Configuration

```python
@dataclass
class OrchestratorConfig:
    max_tool_rounds: int = 8          # Max sequential tool calls (8 atomic tools)
    tool_timeout_ms: int = 10000      # Per-tool timeout
    max_retries_per_tool: int = 2     # Retry count on tool failure
```

## Main Loop (Pseudocode)

```python
def orchestrate(user_input: str, user_context: dict) -> str:
    messages = build_initial_messages(user_input, user_context)
    tool_round = 0

    while tool_round < config.max_tool_rounds:
        # 1. Call 3B model
        response = inference_3b(messages)

        # 2. Parse response
        parsed = parse_model_output(response)

        # 3. Branch based on output type
        if parsed.type == "final_answer":
            return parsed.content

        elif parsed.type == "tool_call":
            tool_name = parsed.tool_call.name
            tool_args = parsed.tool_call.arguments

            # 4. Execute tool
            try:
                result = execute_tool(tool_name, tool_args)

                # 5. Inject tool response
                messages.append({
                    "role": "assistant",
                    "content": response
                })
                messages.append({
                    "role": "user",
                    "content": format_tool_response(result)
                })

            except ToolExecutionError as e:
                # Handle tool failure
                messages.append({
                    "role": "user",
                    "content": format_error_response(e)
                })

            tool_round += 1

        elif parsed.type == "parse_error":
            # Invalid format - attempt recovery
            messages.append({
                "role": "user",
                "content": "Your previous response had invalid format. Please try again with valid JSON in <tool_call> tags."
            })
            tool_round += 1

    # Max rounds exceeded - force answer
    return force_final_answer(messages)
```

## Termination Conditions

| Condition | Action |
|-----------|--------|
| Model outputs final answer (no `<tool_call>`) | Return answer |
| `max_tool_rounds` exceeded | Force model to answer with current context |
| Unrecoverable tool error (after retries) | Return error message to user |
| Safety check fails | Return safe fallback response |

## Parsing Logic

```python
import re
import json

def parse_model_output(response: str) -> ParsedOutput:
    # Extract think block (optional capture)
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else None

    # Check for tool call
    tool_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', response, re.DOTALL)

    if tool_match:
        try:
            tool_json = json.loads(tool_match.group(1))
            return ParsedOutput(
                type="tool_call",
                think=think_content,
                tool_call=ToolCall(
                    name=tool_json["name"],
                    arguments=tool_json.get("arguments", {})
                )
            )
        except json.JSONDecodeError:
            return ParsedOutput(type="parse_error", raw=response)

    # No tool call - extract final answer
    # Remove think tags, return remaining content
    answer = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    if answer:
        return ParsedOutput(type="final_answer", content=answer)

    return ParsedOutput(type="parse_error", raw=response)
```

## Tool Response Formatting

```python
def format_tool_response(result: dict) -> str:
    return f"<tool_response>\n{json.dumps(result, indent=2)}\n</tool_response>"

def format_error_response(error: ToolExecutionError) -> str:
    return f"""<tool_response>
{{"status": "error", "error_type": "{error.type}", "message": "{error.message}"}}
</tool_response>"""
```

## Safety Check (Expert Responses)

```python
def safety_check_expert_response(result: dict, user_context: dict) -> dict:
    response_text = result.get("data", {}).get("response", "")
    user_allergies = user_context.get("allergies", [])

    # Check for allergens
    for allergen in user_allergies:
        if allergen.lower() in response_text.lower():
            # Flag and modify
            result["safety_warning"] = f"Response mentions allergen: {allergen}"
            result["data"]["response"] = add_allergen_warning(response_text, allergen)

    # Check for extreme values
    calories = extract_daily_calories(response_text)
    if calories and (calories < 800 or calories > 5000):
        result["safety_warning"] = f"Extreme calorie value: {calories}"
        result["blocked"] = True

    return result
```

## Error Recovery Strategies

| Error Type | Strategy |
|------------|----------|
| `food_not_found` | Inject error response, let model retry with different query |
| `invalid_json` | Prompt model to regenerate with correct format |
| `no_relevant_results` | Model answers from internal knowledge + notes limitation |
| `max_rounds_exceeded` | Force final answer with current context |

## Message Format

The orchestrator maintains a message list compatible with chat-style inference:

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_input},
    {"role": "assistant", "content": "<think>...</think><tool_call>...</tool_call>"},
    {"role": "user", "content": "<tool_response>...</tool_response>"},
    {"role": "assistant", "content": "<think>...</think><tool_call>...</tool_call>"},
    {"role": "user", "content": "<tool_response>...</tool_response>"},
    {"role": "assistant", "content": "Final answer to user..."}
]
```

Note: Tool responses are injected as "user" role messages to maintain alternating turn structure.

## Agentic Complexity Alignment

The orchestrator handles all four tiers of agentic complexity defined in the PRD:

| Tier | Orchestrator Behavior |
|------|----------------------|
| **T1** | Single INFERENCE → TOOL_CALL → TOOL_EXEC → INFERENCE → ANSWER loop |
| **T2** | Multiple INFERENCE → TOOL_CALL → TOOL_EXEC cycles (2-3 atomic tools, data dependency) |
| **T3** | Multiple cycles with conditional branching (next tool depends on intermediate result) |
| **T4** | INFERENCE → ANSWER directly (no tool calls). Model outputs safety boundary disclaimer. |

> **T4 Note**: Since T4 produces no `<tool_call>` tags, the orchestrator treats it
> the same as a `final_answer`. No special routing needed — the model itself decides
> not to call any tool and produces a direct safety declaration.
