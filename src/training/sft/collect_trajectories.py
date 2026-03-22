"""
Trajectory collection for SFT training.

Uses API function calling for teacher (stable output), then converts to pure text
format for SFT training. See ADR-001 for rationale.

Architecture:
- Teacher: API function calling with enable_thinking=True
- Conversion: api_response_to_sft_text() transforms to <tool_call> format
- Student: ToolParser (src/orchestrator/tool_parser.py) - NOT used here
"""
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.tools.get_food_nutrition import get_food_nutrition
from src.tools.log_meal import log_meal
from src.tools.get_today_summary import get_today_summary
from src.tools.get_history import get_history
from src.tools.retrieve_knowledge import retrieve_knowledge
from src.config import settings

TEACHER_MODEL_T0 = "qwen3.5-flash"
TEACHER_MODEL_OTHERS = "qwen3.5-397b-a17b"
# Default for direct calls outside batch_collect
TEACHER_MODEL = TEACHER_MODEL_OTHERS 

# Global upper limit to prevent infinite loops; actual quality gating is done in validation
MAX_TURNS = 20

# Max characters for tool result to prevent context overflow
MAX_TOOL_RESULT_CHARS = 4000

client = OpenAI(
    api_key=settings.qwen_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# =============================================================================
# API Function Calling Tools Definition
# =============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_food_nutrition",
            "description": "Look up nutrition for one or more foods from the USDA database. Returns detailed nutrients per food (calories, protein, fat, carbs, fiber, sugars, sodium, vitamins) + totals + macro_ratio. Each item includes match_confidence (high/medium/low).",
            "parameters": {
                "type": "object",
                "properties": {
                    "foods": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "food_name": {"type": "string", "description": "Name of the food to look up"},
                                "amount_grams": {"type": "number", "description": "Amount in grams"}
                            },
                            "required": ["food_name", "amount_grams"]
                        },
                        "description": "List of foods with amounts"
                    }
                },
                "required": ["foods"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "log_meal",
            "description": "Persist a single meal record to user history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "meal_type": {
                        "type": "string",
                        "enum": ["breakfast", "lunch", "dinner", "snack"],
                        "description": "Type of meal"
                    },
                    "foods": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "food_name": {"type": "string"},
                                "amount_grams": {"type": "number"}
                            },
                            "required": ["food_name", "amount_grams"]
                        },
                        "description": "List of foods in the meal"
                    },
                    "timestamp": {
                        "type": "string",
                        "description": "ISO-8601 timestamp (optional)"
                    }
                },
                "required": ["meal_type", "foods"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_today_summary",
            "description": "Retrieve today's nutritional intake summary and remaining calorie budget.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_history",
            "description": "Query multi-day nutritional history and trends. Set compare_to_goal=true when user asks about goal progress or adherence rate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Number of days to query (default 7, max 90)"
                    },
                    "metric": {
                        "type": "string",
                        "enum": ["calories", "protein", "fat", "carbs", "all"],
                        "description": "Which metric to retrieve (default 'all')"
                    },
                    "compare_to_goal": {
                        "type": "boolean",
                        "description": "Whether to compare against user's goals (default false)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_knowledge",
            "description": "Search the authoritative nutrition knowledge base (RAG). Use for dietary guidelines, medical nutrition principles, supplement information, sports nutrition, micronutrients, and life stage recommendations. Do NOT use for food facts/macros — use get_food_nutrition instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for the knowledge base"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["hybrid", "semantic", "keyword"],
                        "description": "Retrieval strategy. Use 'hybrid' for general questions, 'semantic' for concepts, 'keyword' for specific terms after a hybrid failure."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default 3, max 5)"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


# =============================================================================
# API Response to SFT Text Conversion
# =============================================================================

def api_response_to_sft_text(
    thinking_content: str | None,
    tool_calls: list[dict[str, Any]] | None,
    text_content: str | None
) -> str:
    """Convert API response components to pure text SFT format.

    This function transforms the structured API response into the text format
    that the student model will learn to produce.

    Args:
        thinking_content: The thinking/reasoning content from API (may be None)
        tool_calls: List of tool call dicts from API (may be None or empty)
        text_content: Plain text content from API (may be None)

    Returns:
        Formatted string with <think> and <tool_call> tags as needed

    Examples:
        >>> api_response_to_sft_text("analyzing...", [{"name": "get_food_nutrition", "arguments": {"foods": [...]}}], None)
        '<think>\nanalyzing...\n</think>\n<tool_call>\n{"name": "get_food_nutrition", "arguments": {"foods": [...]}}\n</tool_call>'

        >>> api_response_to_sft_text("synthesis...", None, "The answer is...")
        '<think>\nsynthesis...\n</think>\nThe answer is...'
    """
    parts = []

    # 1. Add thinking block if present
    if thinking_content and thinking_content.strip():
        parts.append(f"<think>\n{thinking_content.strip()}\n</think>")

    # 2. Add tool calls if present
    if tool_calls:
        for tc in tool_calls:
            # Handle both OpenAI object format and dictionary format
            if hasattr(tc, "function"):
                name = tc.function.name
                args_str = tc.function.arguments
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    args = {}
            else:
                # Fallback for dictionary format
                func = tc.get("function", tc)
                name = func.get("name", "")
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

            tool_json = json.dumps(
                {"name": name, "arguments": args},
                ensure_ascii=False,
                indent=2
            )
            parts.append(f"<tool_call>\n{tool_json}\n</tool_call>")

    # 3. Add text content if present (final answer)
    if text_content and text_content.strip():
        parts.append(text_content.strip())

    return "\n".join(parts)


def format_tool_response(result: dict[str, Any], indent: int | None = 2) -> str:
    """Format tool execution result as <tool_response> block.

    Args:
        result: Tool execution result dict
        indent: JSON indentation (None for compact)

    Returns:
        Formatted string with <tool_response> tags
    """
    json_str = json.dumps(result, ensure_ascii=False, indent=indent, default=str)
    return f"<tool_response>\n{json_str}\n</tool_response>"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def call_api_with_retry(**kwargs):
    """Wrapper for API calls with exponential backoff retry."""
    return client.chat.completions.create(**kwargs)


SYSTEM_PROMPT = """You are NutriMind, a specialized AI nutrition assistant.

## LANGUAGE RULES
- ALL output must be in English.
- User may write in Chinese or English — always respond in English.

## BEHAVIOR GUIDELINES

1. **Analyze before acting**: Determine what information you need before calling a tool. If the request is general, conversational, or answerable from your own knowledge, respond directly without tools.

2. **Analyze results**: When you receive a tool result, check for data quality issues (anomalous values, semantic mismatches, low confidence) before using it in your response.

3. **Concise answers**: Provide clear, actionable nutrition advice. Distill and synthesize tool results into your own words — do NOT copy-paste raw content from retrieved passages.

4. **Capability awareness**: If a request exceeds your toolset (e.g., edit/delete a logged meal), inform the user and suggest alternatives.

## HARD CONSTRAINTS

1. **One tool per turn**: Call only one tool at a time, then wait for the result.

2. **Formatting rules**:
   - Simple queries (single food lookup, today's summary, etc.): 1-2 plain paragraphs.
   - Use Markdown tables ONLY for direct side-by-side data comparisons.
   - Maximum one level of header. No nested bullet points.
   - Complex advisory queries may use a short structured format, but keep it concise.

3. **Data audit**: If any returned value exceeds reasonable bounds (calories > 10,000 kcal/day, protein > 1,000 g/day), flag it as a data anomaly, do NOT use the data, and ask the user to verify their logs.

## SAFETY BOUNDARY

If the user's situation involves any of the following, do NOT call any tools. Respond directly with the boundary statement below:
- Dialysis or post-organ transplant nutrition
- Post-surgery recovery nutrition
- Active cancer treatment
- Medications that interact with food (warfarin, chemotherapy, immunosuppressants, MAOIs)

Response: "Your situation involves complex medical nutrition management that exceeds my safe service boundary. Please consult your physician or a registered dietitian."

## TOOL USAGE NOTES

### get_food_nutrition
Look up nutrition data for one or more foods.
- If `match_confidence` is "low": check whether the returned `food_name` semantically matches your query. If it does not match (e.g., you queried "potato, raw" but got "Bread, potato"), retry once with USDA-standard naming (e.g., "Potatoes, flesh and skin, raw"). If the second attempt also fails, ask the user to clarify.
- If `match_confidence` is "high" but the returned `food_name` is clearly a different food from what was queried, treat it as a mismatch and retry as above.

### log_meal
Record a meal to user history.
- Only use when the user explicitly wants to log/record a meal.
- You cannot edit or delete existing entries.

### get_today_summary
Today's intake summary and remaining budget.
- Use for questions about today's totals, remaining calories/macros, or current progress.

### get_history
Multi-day nutritional trends and goal adherence.
- Use `compare_to_goal=true` when the user asks about goal progress or adherence rates.
- Do NOT use for today-only queries (use `get_today_summary` instead).

### retrieve_knowledge
RAG search over nutrition knowledge base (dietary guidelines, supplements, medical nutrition principles).
- Do NOT use for food nutrition facts — use `get_food_nutrition` instead.
- **Retrieval strategy**:
  - Start with `mode: "hybrid"` (default).
  - Check `top_relevance_score` AND examine the passage `source`/`section` to judge relevance.
  - If score > 0.7 AND passage content clearly addresses your query: use the result.
  - If score < 0.4 OR passage topic is off-topic (e.g., you asked about VLCD risks but got a passage about calcium): reformulate your query OR switch mode (try "keyword" for precise terms, "semantic" for conceptual queries).
  - After three attempts with poor results: stop searching. Answer from your own knowledge and state: "My nutrition knowledge base does not currently cover this topic in detail. Here is general information based on my training.\""""

TOOL_DISPATCH = {
    "get_food_nutrition": get_food_nutrition,
    "log_meal": log_meal,
    "get_today_summary": get_today_summary,
    "get_history": get_history,
    "retrieve_knowledge": retrieve_knowledge,
}


# Removed normalize_tier as model selection is now uniform


def execute_tool(name: str, arguments: dict) -> dict:
    try:
        fn = TOOL_DISPATCH.get(name)
        if fn is None:
            return {"status": "error", "message": f"Unknown tool: {name}"}
        return fn(**arguments)
    except Exception as e:
        return {"status": "error", "error_type": type(e).__name__, "message": str(e)}


def truncate_tool_result(result: dict, max_chars: int = MAX_TOOL_RESULT_CHARS) -> dict:
    """Truncate tool result to prevent context overflow."""
    result_str = json.dumps(result, ensure_ascii=False, default=str)
    if len(result_str) <= max_chars:
        return result

    # For list results (e.g., get_food_nutrition matches), truncate the list
    if isinstance(result.get("data"), list):
        truncated = result.copy()
        data = truncated["data"]
        while len(json.dumps(truncated, ensure_ascii=False, default=str)) > max_chars and len(data) > 1:
            data = data[:-1]
            truncated["data"] = data
        truncated["_truncated"] = True
        return truncated

    # For non-list results, truncate the JSON string and wrap as summary
    return {
        "status": result.get("status", "success"),
        "_truncated": True,
        "_original_length": len(result_str),
        "data_summary": result_str[:max_chars],
    }


def is_error_result(content: str) -> bool:
    """Check if tool result indicates an error via JSON parsing."""
    try:
        result = json.loads(content)
        return result.get("status") == "error"
    except (json.JSONDecodeError, TypeError, AttributeError):
        return False


def infer_tier(messages: list) -> str:
    """Infer trajectory tier by counting <tool_call> occurrences in assistant messages."""
    tool_call_count = 0
    has_tool_error = False

    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "") or ""
        tool_calls_in_msg = re.findall(r"<tool_call>", content)
        tool_call_count += len(tool_calls_in_msg)

    # Check tool results for errors
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "") or ""
            if "<tool_response>" in content:
                inner = re.search(r"<tool_response>\s*(.*?)\s*</tool_response>", content, re.DOTALL)
                if inner and is_error_result(inner.group(1)):
                    has_tool_error = True

    t4_keywords = ["dialysis", "post-surgery", "active cancer", "organ transplant", "warfarin", "chemotherapy"]
    
    if tool_call_count == 0:
        # Check for T4 safety boundary response
        final_answers = [m.get("content", "").lower() for m in messages if m.get("role") == "assistant"]
        # Only label T4 if it hits the strict prompt-injected boundary
        if any(word in ans for ans in final_answers for word in t4_keywords) or any("safe service boundary" in ans for ans in final_answers):
            return "T4"
        return "T0-qa"

    if has_tool_error:
        return "error_recovery"
    if tool_call_count == 1:
        return "T1"
    if tool_call_count <= 3:
        return "T2"
    return "T3"


def simulate_trajectory(query: str, model: str = TEACHER_MODEL) -> dict:
    """Collect a single trajectory using API function calling.

    Uses DashScope API with tools parameter for stable output,
    then converts to pure text format for SFT training.
    """

    # API messages use standard format (no <tool_call> tags)
    api_messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": query}
    ]

    # SFT messages will have converted text format
    sft_messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": query}
    ]

    for turn in range(MAX_TURNS):
        response = call_api_with_retry(
            model=model,
            messages=api_messages,
            tools=TOOLS,
            extra_body={"enable_thinking": True},
        )

        message = response.choices[0].message
        # DashScope API uses "reasoning_content" field for thinking (not "thinking_content")
        reasoning_content = getattr(message, "reasoning_content", None)
        tool_calls = message.tool_calls
        text_content = message.content

        # Convert API response to SFT text format
        sft_content = api_response_to_sft_text(reasoning_content, tool_calls, text_content)

        if not sft_content.strip():
            print(f"[WARNING] Empty response at turn {turn}, nudging model")
            api_messages.append({"role": "user", "content": "Please provide your answer."})
            continue

        # Add to SFT messages (converted format)
        sft_messages.append({"role": "assistant", "content": sft_content})

        # Check if this is a tool call or final answer
        if not tool_calls:
            # No tool call — this is the final answer
            # Also add to API messages for completeness
            api_messages.append({"role": "assistant", "content": text_content or ""})
            break

        # Process tool call
        tc = tool_calls[0]  # Sequential: one tool per turn
        func = tc.function
        name = func.name
        try:
            args = json.loads(func.arguments) if isinstance(func.arguments, str) else func.arguments
        except json.JSONDecodeError:
            args = {}

        # Add assistant message to API conversation (for next turn context)
        api_messages.append({
            "role": "assistant",
            "content": text_content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": name, "arguments": func.arguments}
                }
            ]
        })

        # Execute tool
        real_result = execute_tool(name, args)
        truncated_result = truncate_tool_result(real_result)
        result_json = json.dumps(truncated_result, ensure_ascii=False, default=str)

        # Add tool result to API messages (standard format)
        api_messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result_json
        })

        # Add tool result to SFT messages (our format)
        sft_messages.append({
            "role": "user",
            "content": format_tool_response(truncated_result, indent=None)
        })

    # Check if trajectory is complete
    last_sft = sft_messages[-1]
    last_role = last_sft.get("role", "")
    last_content = last_sft.get("content", "") or ""

    is_tool_response = last_role == "user" and "<tool_response>" in last_content
    is_think_only = last_role == "assistant" and "<tool_call>" not in last_content and "<think>" in last_content and not last_content.replace("<think>", "").replace("</think>", "").strip()

    if is_tool_response or is_think_only:
        print(f"[WARNING] Incomplete trajectory — forcing final completion")

        if is_think_only:
            sft_messages.pop()
            api_messages.pop()

        try:
            # Force final answer (no tools)
            final_response = call_api_with_retry(
                model=model,
                messages=api_messages,
                extra_body={"enable_thinking": True},
            )

            final_msg = final_response.choices[0].message
            final_thinking = getattr(final_msg, "reasoning_content", None)
            final_content = final_msg.content

            if not final_content or not final_content.strip():
                raise ValueError("Forced completion produced empty response")

            _LOW_QUALITY_SIGNALS = [
                "i recommend consulting",
                "refer to the tool",
                "based on the information i've gathered",
                "i apologize, but i encountered",
                "please consult a",
                "i wasn't able to find",
            ]

            if any(sig in final_content.lower() for sig in _LOW_QUALITY_SIGNALS):
                raise ValueError(f"Low-quality forced completion: {final_content[:120]}")

            sft_final = api_response_to_sft_text(final_thinking, None, final_content)
            sft_messages.append({"role": "assistant", "content": sft_final})

        except Exception as e:
            raise ValueError(f"Forced completion failed for '{query[:60]}': {e}") from e

    tier = infer_tier(sft_messages)
    return {
        "query": query,
        "tier": tier,
        "messages": sft_messages,  # Return SFT format messages
        "metadata": {
            "teacher_model": model,
            "real_tool_executed": True,
            "collection_mode": "api_function_calling",
            "turns": sum(
                1 for m in sft_messages
                if m["role"] == "user" and "<tool_response>" in (m.get("content") or "")
            ),
        }
    }


def batch_collect(queries: list, output_path: str, model: str = None, workers: int = 5, requests_per_minute: int = 60):
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    failed_file = output_file.parent / f"{output_file.stem}_failed.jsonl"

    # Load already-collected queries to support resuming
    done_queries = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done_queries.add(json.loads(line)["query"])
                    except Exception:
                        pass
    if done_queries:
        print(f"[INFO] Skipping {len(done_queries)} already-collected queries")

    pending = [item for item in queries if item["query"] not in done_queries]
    if not pending:
        print("✅ All queries already collected, nothing to do.")
        return

    print(f"[INFO] Collecting {len(pending)} queries with {workers} workers")
    write_lock = threading.Lock()
    stats = {"completed": 0, "failed": 0}

    # Simple rate limiting: min interval between requests per worker
    min_interval = 60.0 / requests_per_minute * workers

    def collect_one(item: dict):
        query = item["query"]
        start_time = time.time()
        # Determine which teacher model to use for this query
        if model is not None:
            effective_model = model
        else:
            tier = item.get("tier") or item.get("tier_hint") or ""
            if tier.startswith("T0"):
                effective_model = TEACHER_MODEL_T0
            else:
                effective_model = TEACHER_MODEL_OTHERS
        
        result = simulate_trajectory(query, model=effective_model)
        # Respect rate limit
        elapsed = time.time() - start_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        return query, result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(collect_one, item): item for item in pending}
        for future in as_completed(futures):
            item = futures[future]
            try:
                query, traj = future.result()
                with write_lock:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(traj, ensure_ascii=False, default=str) + "\n")
                    stats["completed"] += 1
                    print(f"[{stats['completed']}/{len(pending)}] {query[:50]}... -> {traj['tier']}")
            except Exception as e:
                with write_lock:
                    # Save failed queries for retry
                    with open(failed_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "query": item["query"],
                            "error": str(e),
                            "error_type": type(e).__name__
                        }, ensure_ascii=False) + "\n")
                    stats["failed"] += 1
                    print(f"[ERROR] {item['query'][:50]}... | {e}")

    print(f"\n✅ Collected {stats['completed']} trajectories -> {output_path}")
    if stats["failed"] > 0:
        print(f"⚠️  {stats['failed']} failed queries saved to {failed_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=str, default="data/queries/sft_candidate_pool.jsonl")
    parser.add_argument("--output", type=str, default="data/trajectories/real_trajectories.jsonl")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel collection threads")
    parser.add_argument("--tier", type=str, help="Filter queries by tier (e.g., T1, T2)")
    parser.add_argument("--model", type=str, default=None, help="Teacher model name. If None (default), use dynamic logic: T0=flash, others=397b.")
    parser.add_argument("--dry-run", action="store_true", help="Print queries that would be collected without calling API")
    args = parser.parse_args()

    # Create dummy query file if it doesn't exist to test
    if not Path(args.queries).exists():
        Path(args.queries).parent.mkdir(parents=True, exist_ok=True)
        dummy_queries = [
            {"query": "How much protein is in 100g chicken breast?", "tier_hint": "T1"},
            {"query": "Log my lunch. I had 200g of rice and 150g of chicken breast.", "tier_hint": "T2"},
            {"query": "I am on dialysis. What should I eat for dinner?", "tier_hint": "T4"}
        ]
        with open(args.queries, 'w') as f:
            for q in dummy_queries:
                f.write(json.dumps(q) + "\n")

    with open(args.queries, 'r') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]

    if args.tier:
        data = [item for item in data if (item.get("tier") or "").startswith(args.tier)]
        print(f"[INFO] Filtered by source tier '{args.tier}': {len(data)} queries found")

    if args.limit > 0:
        data = data[:args.limit]

    if args.dry_run:
        print("\n[DRY RUN] Queries to be collected:")
        for i, item in enumerate(data):
            print(f"{i+1}. [{item.get('tier')}] {item['query']}")
        print(f"\nTotal: {len(data)} queries. No API calls will be made.")
    else:
        batch_collect(data, args.output, model=args.model, workers=args.workers)
