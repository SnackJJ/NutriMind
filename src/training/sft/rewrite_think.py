"""
Rewrite degraded think blocks in multi-turn trajectories.

For each detected issue:
1. Extract context: user query, previous tool response, current tool call
2. Call teacher model to generate analytical reasoning
3. Replace the degraded think with the new one
4. Output fixed trajectories
"""
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config import settings

# Teacher model for rewriting
REWRITE_MODEL = "qwen3.5-plus"

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


REWRITE_SYSTEM_PROMPT = """You are a reasoning expert helping to improve AI agent training data.

Your task: Write a detailed <think> block that explains the agent's reasoning process.

Context you'll receive:
1. User's original question
2. Previous tool call and its result (what the agent already tried)
3. Next tool call the agent is about to make

Your <think> block MUST:
- Analyze what the previous tool result provided
- Explain what information is still missing or insufficient
- Justify WHY the next tool call is needed and HOW the query was refined
- Be written in English
- Be 80-200 words

DO NOT:
- Use generic phrases like "I should call X to get the required information"
- Simply describe what the tool does
- Repeat the tool parameters without reasoning

Output ONLY the content inside <think>...</think> tags, without the tags themselves."""


def build_rewrite_prompt(
    user_query: str,
    prev_tool_name: str,
    prev_tool_args: dict,
    prev_tool_response: str,
    next_tool_name: str,
    next_tool_args: dict,
) -> str:
    """Build the rewrite prompt with full context."""

    # Truncate tool response if too long
    if len(prev_tool_response) > 2000:
        prev_tool_response = prev_tool_response[:2000] + "... [truncated]"

    return f"""## User's Original Question
{user_query}

## Previous Tool Call
Tool: {prev_tool_name}
Arguments: {json.dumps(prev_tool_args, ensure_ascii=False)}

## Previous Tool Result
{prev_tool_response}

## Next Tool Call (agent decided to make this call)
Tool: {next_tool_name}
Arguments: {json.dumps(next_tool_args, ensure_ascii=False)}

Write the reasoning that explains WHY the agent made this next tool call after seeing the previous result."""


def extract_tool_call_info(msg: dict) -> tuple[str, dict] | None:
    """Extract tool name and arguments from assistant message."""
    content = msg.get("content", "")

    # Try <tool_call> format first
    tc_match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.DOTALL)
    if tc_match:
        try:
            tc_json = json.loads(tc_match.group(1))
            return tc_json.get("name"), tc_json.get("arguments", {})
        except json.JSONDecodeError:
            pass

    # Try OpenAI tool_calls format
    tool_calls = msg.get("tool_calls", [])
    if tool_calls:
        tc = tool_calls[0]
        name = tc.get("function", {}).get("name")
        args_str = tc.get("function", {}).get("arguments", "{}")
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            args = {}
        return name, args

    return None


def rewrite_single_think(
    user_query: str,
    messages: list[dict],
    issue: dict,
) -> str:
    """
    Rewrite a single degraded think block.

    Args:
        user_query: Original user question
        messages: Full message list
        issue: Issue dict with msg_idx, prev_tool_response

    Returns:
        New think content (without <think> tags)
    """
    msg_idx = issue["msg_idx"]
    current_msg = messages[msg_idx]

    # Extract current tool call info
    current_tc_info = extract_tool_call_info(current_msg)
    if not current_tc_info:
        raise ValueError(f"Cannot extract tool call from message at index {msg_idx}")

    next_tool_name, next_tool_args = current_tc_info

    # Find previous tool call and response
    prev_tool_name = None
    prev_tool_args = {}
    prev_tool_response = issue.get("prev_tool_response", "")

    # Look backwards for the previous assistant message with tool call
    for j in range(msg_idx - 1, -1, -1):
        msg = messages[j]
        if msg.get("role") == "assistant":
            tc_info = extract_tool_call_info(msg)
            if tc_info:
                prev_tool_name, prev_tool_args = tc_info
                break

    if not prev_tool_name:
        # First tool call in conversation - use a simpler prompt
        raise ValueError("No previous tool call found - this shouldn't be flagged as an issue")

    prompt = build_rewrite_prompt(
        user_query=user_query,
        prev_tool_name=prev_tool_name,
        prev_tool_args=prev_tool_args,
        prev_tool_response=prev_tool_response,
        next_tool_name=next_tool_name,
        next_tool_args=next_tool_args,
    )

    response = call_api_with_retry(
        model=REWRITE_MODEL,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=500,
    )

    new_think = response.choices[0].message.content.strip()

    # Clean up any accidental <think> tags
    new_think = re.sub(r"^<think>\s*", "", new_think)
    new_think = re.sub(r"\s*</think>$", "", new_think)

    return new_think


def replace_think_in_message(msg: dict, new_think: str) -> dict:
    """Replace think content in a message, preserving tool call."""
    msg = msg.copy()
    content = msg.get("content", "")

    # Replace existing think
    if "<think>" in content:
        new_content = re.sub(
            r"<think>.*?</think>",
            f"<think>\n{new_think}\n</think>",
            content,
            flags=re.DOTALL
        )
    else:
        # Prepend think if missing
        new_content = f"<think>\n{new_think}\n</think>\n{content}"

    msg["content"] = new_content
    return msg


def rewrite_trajectory(traj: dict, issues: list[dict]) -> dict:
    """
    Rewrite all degraded thinks in a trajectory.

    Returns:
        Fixed trajectory dict
    """
    traj = json.loads(json.dumps(traj))  # Deep copy
    messages = traj["messages"]
    user_query = traj.get("query", "")

    # If no query field, extract from first user message
    if not user_query:
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # Skip tool responses
                if not content.startswith("<tool_response>"):
                    user_query = content
                    break

    # Process issues in reverse order to maintain indices
    for issue in sorted(issues, key=lambda x: x["msg_idx"], reverse=True):
        try:
            new_think = rewrite_single_think(user_query, messages, issue)
            messages[issue["msg_idx"]] = replace_think_in_message(
                messages[issue["msg_idx"]], new_think
            )
        except Exception as e:
            print(f"[WARNING] Failed to rewrite issue at msg_idx {issue['msg_idx']}: {e}")
            # Keep original if rewrite fails

    traj["messages"] = messages
    traj["metadata"] = traj.get("metadata", {})
    traj["metadata"]["think_rewritten"] = True
    return traj


def batch_rewrite(
    input_path: str,
    issues_path: str,
    output_path: str,
    workers: int = 5,
    requests_per_minute: int = 30,
) -> dict:
    """
    Batch rewrite trajectories with detected issues.

    Args:
        input_path: Original trajectories JSONL
        issues_path: Issues JSONL from detect_think_quality.py
        output_path: Output path for fixed trajectories
        workers: Number of parallel workers
        requests_per_minute: Rate limit

    Returns:
        Stats dict
    """
    # Load issues
    issues_by_line = {}
    with open(issues_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            issues_by_line[item["line_num"]] = item["issues"]

    print(f"[INFO] Loaded {len(issues_by_line)} trajectories with issues to fix")

    # Load all trajectories
    trajectories = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))

    print(f"[INFO] Loaded {len(trajectories)} total trajectories")

    # Setup output
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    stats = {"total": len(trajectories), "rewritten": 0, "failed": 0, "unchanged": 0}
    write_lock = threading.Lock()
    min_interval = 60.0 / requests_per_minute * workers

    def process_one(item: tuple[int, dict]) -> tuple[int, dict, bool]:
        line_num, traj = item
        start_time = time.time()

        if line_num in issues_by_line:
            issues = issues_by_line[line_num]
            fixed_traj = rewrite_trajectory(traj, issues)
            rewritten = True
        else:
            fixed_traj = traj
            rewritten = False

        # Rate limiting
        elapsed = time.time() - start_time
        if elapsed < min_interval and rewritten:
            time.sleep(min_interval - elapsed)

        return line_num, fixed_traj, rewritten

    # Process all trajectories
    fixed_trajectories = [None] * len(trajectories)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_one, (i, traj)): i
            for i, traj in enumerate(trajectories)
        }

        for future in as_completed(futures):
            try:
                line_num, fixed_traj, rewritten = future.result()
                fixed_trajectories[line_num] = fixed_traj

                if rewritten:
                    stats["rewritten"] += 1
                    print(f"[{stats['rewritten']}/{len(issues_by_line)}] Fixed line {line_num}")
                else:
                    stats["unchanged"] += 1
            except Exception as e:
                line_num = futures[future]
                # Keep original on failure
                fixed_trajectories[line_num] = trajectories[line_num]
                stats["failed"] += 1
                print(f"[ERROR] Line {line_num}: {e}")

    # Write output
    with open(output_file, "w", encoding="utf-8") as f:
        for traj in fixed_trajectories:
            if traj is not None:
                f.write(json.dumps(traj, ensure_ascii=False) + "\n")

    print(f"\n✅ Rewrite complete:")
    print(f"  Total: {stats['total']}")
    print(f"  Rewritten: {stats['rewritten']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Unchanged: {stats['unchanged']}")
    print(f"  Output: {output_path}")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rewrite degraded think blocks")
    parser.add_argument("--input", type=str, required=True, help="Input trajectories JSONL")
    parser.add_argument("--issues", type=str, required=True, help="Issues JSONL from detection")
    parser.add_argument("--output", type=str, required=True, help="Output fixed trajectories JSONL")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers")
    parser.add_argument("--rpm", type=int, default=30, help="Requests per minute")
    args = parser.parse_args()

    batch_rewrite(
        input_path=args.input,
        issues_path=args.issues,
        output_path=args.output,
        workers=args.workers,
        requests_per_minute=args.rpm,
    )
