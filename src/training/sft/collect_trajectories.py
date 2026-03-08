import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.tools.get_food_nutrition import get_food_nutrition
from src.tools.log_meal import log_meal
from src.tools.get_today_summary import get_today_summary
from src.tools.get_history import get_history
from src.tools.retrieve_knowledge import retrieve_knowledge
from src.tools.get_goal_adherence import get_goal_adherence
from src.config import settings

TEACHER_MODELS = {
    "T0-qa": "qwen-plus",
    "T1": "qwen-plus",
    "T2": "qwen-plus",
    "T3": "qwen-plus",
    "T4": "qwen-plus",
    "error_recovery": "qwen-plus",
}

# Per-tier max turns to avoid wasting API calls
MAX_TURNS = {
    "T0-qa": 3,          # Direct answer, minimal interaction
    "T1": 5,             # Single tool + buffer for retry
    "T2": 8,             # Multi-step, 2-3 tools + retries
    "T3": 12,            # Complex conditional flows
    "T4": 3,             # Should refuse immediately
    "error_recovery": 10, # Extra turns for error handling
}

# Max characters for tool result to prevent context overflow
MAX_TOOL_RESULT_CHARS = 4000

client = OpenAI(
    api_key=settings.qwen_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def call_api_with_retry(**kwargs):
    """Wrapper for API calls with exponential backoff retry."""
    return client.chat.completions.create(**kwargs)

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_food_nutrition",
            "description": "Look up nutrition information for one or more foods from USDA database. Returns detailed nutrients (calories, protein, fat, carbs, fiber, sugars, sodium, vitamins, etc.) for each food, plus totals if multiple foods. Use for ANY food nutrition query - single food or full meal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "foods": {
                        "type": "array",
                        "description": "List of foods to look up. For single food query, just provide one item.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "food_name": {"type": "string", "description": "Food name (English preferred)"},
                                "amount_grams": {
                                    "type": "number",
                                    "description": "Amount in grams. Estimate from natural units: '1 large egg'->50, '1 medium banana'->118, '1 cup milk'->244, '1 slice bread'->30"
                                }
                            },
                            "required": ["food_name", "amount_grams"]
                        }
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
                    "meal_type": {"type": "string", "enum": ["breakfast", "lunch", "dinner", "snack"]},
                    "foods": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "food_name": {"type": "string"},
                                "amount_grams": {"type": "number"}
                            },
                            "required": ["food_name", "amount_grams"]
                        }
                    },
                    "timestamp": {"type": "string"}
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
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_history",
            "description": "Query multi-day nutritional history and trends.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "default": 7},
                    "metric": {"type": "string", "enum": ["calories", "protein", "fat", "carbs", "all"], "default": "all"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_knowledge",
            "description": "Retrieve nutrition knowledge from authoritative English-language RAG knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 3},
                    "domain": {
                        "type": "string",
                        "enum": [
                            "micronutrients",
                            "dietary_guidelines",
                            "sports_nutrition",
                            "life_stage",
                            "meal_planning",
                            "food_safety",
                            "supplements",
                            "medical_nutrition",
                            "weight_management"
                        ]
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_goal_adherence",
            "description": "Analyze adherence to nutrition goals over a specified period.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "default": 7, "description": "Number of days to analyze (max 90)"},
                    "metric": {"type": "string", "enum": ["calories", "protein", "fat", "carbs", "all"], "default": "all"}
                }
            }
        }
    }
]

# Inject thought requirement into every tool schema natively so the API forces CoT before execution
for tool in TOOLS_SCHEMA:
    props = tool["function"]["parameters"]["properties"]
    props["thought"] = {
        "type": "string",
        "description": "Step-by-step reasoning in English about what data is needed and why. MUST ALWAYS BE FILLED FIRST."
    }
    req = tool["function"]["parameters"].setdefault("required", [])
    if "thought" not in req:
        req.insert(0, "thought")

SYSTEM_PROMPT = """You are NutriMind, a specialized AI nutrition assistant.

## LANGUAGE RULES
- ALL output must be in English: reasoning, tool parameters, and final answers.
- User may write in Chinese or English — always respond in English.

## BEHAVIOR
Before every tool call, reason step-by-step in <think>...</think> tags.
For specific food data, facts, macros, or medical knowledge, you MUST ALWAYS use the available tools. For general knowledge or simple questions (e.g. "what are empty calories?"), you may answer directly. 
CRITICAL RULE: NEVER MAKE PARALLEL/MULTIPLE TOOL CALLS. ONLY call ONE tool at a time. Wait for the tool result before proceeding.

## SAFETY BOUNDARY (T4)
If the user mentions: dialysis, post-surgery recovery, active cancer, organ transplant, or is taking medications that interact with food (e.g. warfarin/anticoagulants, chemotherapy drugs, immunosuppressants, MAOIs) — DO NOT use tools.
Instead, respond directly:
"Your situation involves complex medical nutrition management that exceeds my safe service boundary. Please consult your physician or a registered dietitian."

## AVAILABLE TOOLS
- get_food_nutrition(foods[{food_name, amount_grams}]): Look up nutrition for one or more foods. Returns detailed nutrients per food + totals. Use for ANY food query.
- log_meal(meal_type, foods[]): Record a meal to user history
- get_today_summary(): Today's intake and remaining calorie budget
- get_history(days, metric): Multi-day nutrition trends
- retrieve_knowledge(query, top_k, domain): Search authoritative nutrition knowledge base
- get_goal_adherence(days, metric): Analyze adherence to nutrition goals over a period"""

TOOL_DISPATCH = {
    "get_food_nutrition": get_food_nutrition,
    "log_meal": log_meal,
    "get_today_summary": get_today_summary,
    "get_history": get_history,
    "retrieve_knowledge": retrieve_knowledge,
    "get_goal_adherence": get_goal_adherence,
}

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

    # For list results (e.g., search_food matches), truncate the list
    if isinstance(result.get("data"), list):
        truncated = result.copy()
        data = truncated["data"]
        while len(json.dumps(truncated, ensure_ascii=False, default=str)) > max_chars and len(data) > 1:
            data = data[:-1]
            truncated["data"] = data
        truncated["_truncated"] = True
        return truncated

    # For other results, add truncation marker
    result["_truncated"] = True
    result["_original_length"] = len(result_str)
    return result


def is_error_result(content: str) -> bool:
    """Check if tool result indicates an error via JSON parsing."""
    try:
        result = json.loads(content)
        return result.get("status") == "error"
    except (json.JSONDecodeError, TypeError, AttributeError):
        return False

def infer_tier(messages: list) -> str:
    tool_calls = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tool_calls.extend([tc["function"]["name"] for tc in msg["tool_calls"]])

    if not tool_calls:
        final_answers = [m.get("content", "") for m in messages if m.get("role") == "assistant" and not m.get("tool_calls")]
        safety_keywords = ["safe service boundary", "registered dietitian", "consult your physician"]
        if any(kw in ans for ans in final_answers for kw in safety_keywords):
            return "T4"
        return "T0-qa"

    # Use proper JSON parsing to detect errors
    if any(m.get("role") == "tool" and is_error_result(m.get("content", "")) for m in messages):
        return "error_recovery"
    if len(tool_calls) == 1:
        return "T1"
    if len(tool_calls) <= 3:
        return "T2"
    return "T3"

def simulate_trajectory(query: str, tier_hint: str = "T2") -> dict:
    model = TEACHER_MODELS.get(tier_hint, TEACHER_MODELS["T2"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": query}
    ]

    max_turns = MAX_TURNS.get(tier_hint, MAX_TURNS["T2"])
    for turn in range(max_turns):
        response = call_api_with_retry(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        # We need to serialize msg and exclude None carefully, as pydantic object serialization might differ based on SDK version
        msg_dict = msg.model_dump(exclude_none=True)

        # Wrap pre-tool-call reasoning as <think> block for SFT training
        # Qwen returns reasoning in content field (not a separate reasoning_content)
        # When there are tool_calls and content has reasoning text, wrap it
        content = msg_dict.get("content", "") or ""
        if msg.tool_calls and content.strip():
            msg_dict["content"] = f"<think>{content.strip()}</think>"

        messages.append(msg_dict)

        # Exit loop only if: (1) no tool_calls AND (2) has actual content
        # This prevents exiting on empty responses
        content = msg_dict.get("content", "") or ""
        if not msg.tool_calls and content.strip():
            break

        # If model returned empty response (no tool_calls, no content), retry once
        if not msg.tool_calls and not content.strip():
            print(f"[WARNING] Empty response at turn {turn}, retrying with explicit instruction")
            # Add a nudge to get the model to respond
            messages.pop()  # Remove the empty response
            messages.append({"role": "user", "content": "Please provide your answer based on the information gathered."})
            retry_response = call_api_with_retry(
                model=model,
                messages=messages,
                # REMOVED tools and tool_choice to force pure text response safely
            )
            retry_msg = retry_response.choices[0].message
            retry_dict = retry_msg.model_dump(exclude_none=True)
            # Remove the nudge message and add the retry response
            messages.pop()
            messages.append(retry_dict)
            break

        # Filter out empty/invalid tool calls (DashScope sometimes returns name='', arguments='', or id='')
        valid_tool_calls = [
            tc for tc in msg.tool_calls
            if tc.function.name and tc.function.name.strip() and tc.id and tc.id.strip()
        ]

        if not valid_tool_calls:
            # All tool calls were invalid/empty, treat as final answer
            if "tool_calls" in messages[-1]:
                del messages[-1]["tool_calls"]
            break
            
        # FORCE SINGLE TOOL CALL constraint for 3B learning
        if len(valid_tool_calls) > 1:
            valid_tool_calls = [valid_tool_calls[0]]
            
        # Also truncate the saved assistant message so the trajectory reflects valid single call
        valid_ids = {tc.id for tc in valid_tool_calls}
        messages[-1]["tool_calls"] = [tc_dict for tc_dict in messages[-1].get("tool_calls", []) if tc_dict.get("id") in valid_ids]

        # Extract thought and execute tool
        for idx, tc_dict in enumerate(messages[-1]["tool_calls"]):
            name = tc_dict["function"]["name"]
            try:
                args = json.loads(tc_dict["function"]["arguments"]) if tc_dict["function"]["arguments"] else {}
                
                # Extract 'thought' into 'content' to restore <think> behavior
                thought_content = args.pop("thought", "")
                if thought_content:
                    # Append it (in case multiple tool calls existed before truncation, we just populate content once)
                    if not messages[-1].get("content") or messages[-1]["content"].strip() == "":
                        messages[-1]["content"] = f"<think>\n{thought_content}\n</think>\n"
                
                # Re-serialize arguments purely for training
                tc_dict["function"]["arguments"] = json.dumps(args, ensure_ascii=False)
                
            except json.JSONDecodeError as e:
                args = {}
                print(f"[WARNING] Invalid JSON in args for {name}: {e}")
                
            real_result = execute_tool(name, args)
            # Truncate result to prevent context overflow
            truncated_result = truncate_tool_result(real_result)

            messages.append({
                "role": "tool",
                "tool_call_id": tc_dict["id"],
                "name": name,
                "content": json.dumps(truncated_result, ensure_ascii=False, default=str)
            })

    # Check if we need to force a final answer
    # Cases: (1) last msg is tool response, (2) last msg is empty assistant, (3) last msg has tool_calls
    last_msg = messages[-1]
    needs_final = (
        last_msg.get("role") == "tool" or
        (last_msg.get("role") == "assistant" and not (last_msg.get("content") or "").strip()) or
        (last_msg.get("role") == "assistant" and last_msg.get("tool_calls"))
    )

    if needs_final:
        reason = "tool response" if last_msg.get("role") == "tool" else \
                 "empty assistant" if not (last_msg.get("content") or "").strip() else \
                 "assistant with pending tool_calls"
        print(f"[WARNING] Incomplete trajectory ({reason}) — forcing final completion")

        # If last msg is empty assistant or has tool_calls, remove it before forcing
        if last_msg.get("role") == "assistant":
            messages.pop()

        try:
            final_response = call_api_with_retry(
                model=model,
                messages=messages,
                # REMOVED tools and tool_choice to force pure text response safely
            )
            final_msg = final_response.choices[0].message
            final_dict = final_msg.model_dump(exclude_none=True)

            # Strip out any tool calls the model might have hallucinated despite missing tools
            if "tool_calls" in final_dict:
                del final_dict["tool_calls"]

            # Validate the forced response has content
            if not (final_dict.get("content") or "").strip():
                print(f"[WARNING] Forced completion also empty, adding fallback")
                final_dict["content"] = "Based on the information I've gathered, I recommend consulting the specific data provided by the tools above for accurate nutritional guidance."

            messages.append(final_dict)
        except Exception as e:
            print(f"[WARNING] Failed to get forced final answer: {e}")
            # Add a minimal fallback to prevent completely broken trajectory
            messages.append({
                "role": "assistant",
                "content": "I apologize, but I encountered an issue completing my analysis. Please refer to the tool results above for the nutritional information."
            })

    tier = infer_tier(messages)
    return {
        "query": query,
        "tier": tier,
        "messages": messages,
        "metadata": {
            "teacher_model": model,
            "real_tool_executed": True,
            "turns": len([m for m in messages if m["role"] == "tool"]),
        }
    }

def batch_collect(queries: list, output_path: str, workers: int = 5, requests_per_minute: int = 60):
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
        tier_hint = item.get("tier_hint") or item.get("tier", "T2")
        start_time = time.time()
        result = simulate_trajectory(query, tier_hint)
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
                            "tier_hint": item.get("tier_hint") or item.get("tier", "T2"),
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
    parser.add_argument("--queries", type=str, default="data/queries/collection_queries.json")
    parser.add_argument("--output", type=str, default="data/trajectories/real_trajectories.jsonl")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel collection threads")
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
            json.dump(dummy_queries, f)

    with open(args.queries, 'r') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]

    if args.limit > 0:
        data = data[:args.limit]

    batch_collect(data, args.output, workers=args.workers)
