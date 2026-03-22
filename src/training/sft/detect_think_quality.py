"""
Detect think quality issues in multi-turn trajectories.

Issue types:
1. Template think: "I should call X to get the required information." (len < 80)
2. Repeated think: consecutive thinks with >80% similarity
3. Missing analysis: intermediate thinks don't reference previous tool results

Output: JSONL with line numbers and specific turn positions needing rewrite.
"""
import json
import re
from pathlib import Path
from difflib import SequenceMatcher


# Known template patterns that indicate low-quality think
TEMPLATE_PATTERNS = [
    r"i should call \w+ to get the required information",
    r"let me call \w+ to get",
    r"i need to call \w+ to",
    r"i'll use \w+ to get",
]

MIN_QUALITY_THINK_LEN = 80  # Minimum chars for a quality think


def extract_thinks(messages: list[dict]) -> list[dict]:
    """Extract think blocks from assistant messages with metadata."""
    thinks = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if think_match:
            thinks.append({
                "msg_idx": i,
                "content": think_match.group(1).strip(),
                "has_tool_call": "<tool_call>" in content or bool(msg.get("tool_calls")),
            })
    return thinks


def is_template_think(think_content: str) -> bool:
    """Check if think is a low-effort template."""
    lower = think_content.lower()
    for pattern in TEMPLATE_PATTERNS:
        if re.search(pattern, lower):
            return True
    return False


def similarity_ratio(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_prev_tool_response(messages: list[dict], before_idx: int) -> str | None:
    """
    Find previous tool response before given message index.

    Handles two formats:
    1. Standard: role="tool" with content
    2. Wrapped: role="user" with <tool_response>...</tool_response>
    """
    for j in range(before_idx - 1, -1, -1):
        msg = messages[j]
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Standard tool role
        if role == "tool":
            return content

        # Wrapped format: <tool_response> in user message
        if role == "user" and "<tool_response>" in content:
            # Extract content between <tool_response> tags
            match = re.search(r"<tool_response>\s*(.*?)\s*</tool_response>", content, re.DOTALL)
            if match:
                return match.group(1).strip()
            # If no closing tag, return everything after <tool_response>
            start = content.find("<tool_response>") + len("<tool_response>")
            return content[start:].strip()

    return None


def detect_issues(trajectory: dict) -> list[dict]:
    """
    Detect think quality issues in a trajectory.

    Returns list of issues, each with:
    - msg_idx: index in messages array
    - turn: 1-indexed turn number (among assistant messages)
    - issue_type: 'template' | 'too_short' | 'repeated'
    - think_content: the problematic think content
    - prev_tool_response: previous tool response (for rewrite context)
    """
    messages = trajectory.get("messages", [])
    thinks = extract_thinks(messages)
    issues = []

    for turn_idx, think in enumerate(thinks):
        # Skip first turn - usually has good quality from initial reasoning
        if turn_idx == 0:
            continue

        # Skip final answer thinks (no tool call)
        if not think["has_tool_call"]:
            continue

        content = think["content"]
        msg_idx = think["msg_idx"]

        # Issue 1: Template think
        if is_template_think(content):
            prev_tool_response = find_prev_tool_response(messages, msg_idx)

            issues.append({
                "msg_idx": msg_idx,
                "turn": turn_idx + 1,
                "issue_type": "template",
                "think_content": content,
                "prev_tool_response": prev_tool_response,
            })
            continue

        # Issue 2: Too short
        if len(content) < MIN_QUALITY_THINK_LEN:
            prev_tool_response = find_prev_tool_response(messages, msg_idx)

            issues.append({
                "msg_idx": msg_idx,
                "turn": turn_idx + 1,
                "issue_type": "too_short",
                "think_content": content,
                "prev_tool_response": prev_tool_response,
            })
            continue

        # Issue 3: Repeated think (>80% similar to previous)
        if turn_idx > 0:
            prev_think = thinks[turn_idx - 1]["content"]
            if similarity_ratio(content, prev_think) > 0.8:
                prev_tool_response = find_prev_tool_response(messages, msg_idx)

                issues.append({
                    "msg_idx": msg_idx,
                    "turn": turn_idx + 1,
                    "issue_type": "repeated",
                    "think_content": content,
                    "prev_tool_response": prev_tool_response,
                })

    return issues


def detect_file(input_path: str, output_path: str | None = None) -> dict:
    """
    Scan trajectories and output issues report.

    Returns stats dict with counts by tier and issue type.
    """
    input_file = Path(input_path)
    issues_data = []
    stats = {
        "total": 0,
        "multi_turn": 0,
        "with_issues": 0,
        "total_issues": 0,
        "by_tier": {},
        "by_issue_type": {"template": 0, "too_short": 0, "repeated": 0},
    }

    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue

            stats["total"] += 1
            traj = json.loads(line)
            tier = traj.get("tier", "unknown")

            # Initialize tier stats
            if tier not in stats["by_tier"]:
                stats["by_tier"][tier] = {"total": 0, "multi_turn": 0, "with_issues": 0}
            stats["by_tier"][tier]["total"] += 1

            # Check if multi-turn
            assistant_count = sum(1 for m in traj.get("messages", []) if m.get("role") == "assistant")
            if assistant_count < 2:
                continue

            stats["multi_turn"] += 1
            stats["by_tier"][tier]["multi_turn"] += 1

            # Detect issues
            issues = detect_issues(traj)
            if issues:
                stats["with_issues"] += 1
                stats["total_issues"] += len(issues)
                stats["by_tier"][tier]["with_issues"] += 1

                for issue in issues:
                    stats["by_issue_type"][issue["issue_type"]] += 1

                issues_data.append({
                    "line_num": line_num,
                    "tier": tier,
                    "query": traj.get("query", "")[:100],
                    "issues": issues,
                })

    # Write issues to output file
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for item in issues_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Issues written to {output_path}")

    return stats, issues_data


def print_report(stats: dict) -> None:
    """Print detection report."""
    print("\n" + "=" * 60)
    print("THINK QUALITY DETECTION REPORT")
    print("=" * 60)

    print(f"\nOverall Statistics:")
    print(f"  Total trajectories: {stats['total']}")
    print(f"  Multi-turn (≥2 turns): {stats['multi_turn']}")
    print(f"  With issues: {stats['with_issues']} ({stats['with_issues']/stats['multi_turn']*100:.1f}% of multi-turn)")
    print(f"  Total issues to fix: {stats['total_issues']}")

    print(f"\nBy Issue Type:")
    for issue_type, count in stats["by_issue_type"].items():
        print(f"  {issue_type}: {count}")

    print(f"\nBy Tier:")
    print(f"  {'Tier':<20} {'Multi-turn':<12} {'With Issues':<12} {'Rate':<10}")
    print(f"  {'-'*54}")
    for tier, tier_stats in sorted(stats["by_tier"].items()):
        mt = tier_stats["multi_turn"]
        wi = tier_stats["with_issues"]
        rate = wi / mt * 100 if mt > 0 else 0
        print(f"  {tier:<20} {mt:<12} {wi:<12} {rate:.1f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect think quality issues in trajectories")
    parser.add_argument("--input", type=str, required=True, help="Input trajectory JSONL file")
    parser.add_argument("--output", type=str, default=None, help="Output issues JSONL file")
    args = parser.parse_args()

    stats, _ = detect_file(args.input, args.output)
    print_report(stats)
