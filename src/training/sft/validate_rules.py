import json
import re
from pathlib import Path

VALID_TOOLS = {"get_food_nutrition", "log_meal", "get_today_summary", "get_history", "retrieve_knowledge"}

def check_chinese_chars(text: str) -> float:
    if not text:
        return 0.0
    zh_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    return zh_chars / len(text)

def validate_trajectory(trajectory: dict) -> tuple[bool, list[str]]:
    errors = []
    messages = trajectory.get("messages", [])
    tier = trajectory.get("tier", "")

    if not messages:
        return False, ["Empty messages array"]

    tool_count = 0
    has_final_answer = False

    for i, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "assistant":
            # Check for think blocks
            think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
            think_content = think_match.group(1).strip() if think_match else ""

            tool_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, re.DOTALL)

            if tool_match:
                tool_count += 1
                try:
                    tool_json = json.loads(tool_match.group(1))
                    name = tool_json.get("name")
                    if name not in VALID_TOOLS:
                        errors.append(f"Invalid tool: {name}")
                except json.JSONDecodeError:
                    errors.append("Invalid tool_call JSON")

                if not think_match:
                    errors.append("Missing <think> before <tool_call>")

                # Verify language
                if check_chinese_chars(think_content) > 0.05:
                    errors.append("<think> contains >5% Chinese characters")

            else:
                has_final_answer = True
                # Final answer language check
                ans = content.replace(f"<think>{think_content}</think>", "").strip()
                if check_chinese_chars(ans) > 0.05:
                    errors.append("Final answer contains >5% Chinese characters")

                if think_match and think_content:
                    if check_chinese_chars(think_content) > 0.05:
                        errors.append("Final answer <think> contains >5% Chinese characters")

    if tier == "T0-qa" and tool_count > 0:
        errors.append("T0-qa should have no tool calls")

    if tier == "T4" and tool_count > 0:
        errors.append("T4 should have NO tool calls (must be safety boundary declaration)")

    if tier == "T1" and tool_count != 1:
        errors.append(f"T1 should have exactly 1 tool call, got {tool_count}")

    if not has_final_answer:
        errors.append("Trajectory does not end with a final answer")

    return len(errors) == 0, errors

def validate_file(input_path: str, output_path: str = None):
    """Validate trajectories. If output_path given, write only passing ones."""
    valid_count = 0
    total = 0
    out_file = open(output_path, 'w', encoding='utf-8') if output_path else None
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                total += 1
                traj = json.loads(line)
                is_valid, errors = validate_trajectory(traj)
                if is_valid:
                    valid_count += 1
                    if out_file:
                        out_file.write(json.dumps(traj, ensure_ascii=False) + "\n")
                else:
                    print(f"[FAIL] Traj {total}: {errors}")
    finally:
        if out_file:
            out_file.close()

    print(f"Rule Validation: {valid_count}/{total} PASS ({(valid_count/total if total > 0 else 0)*100:.1f}%)")
    if output_path:
        print(f"  -> Written to {output_path}")
    return valid_count, total

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/trajectories/normalized_trajectories.jsonl")
    parser.add_argument("--output", type=str, default=None, help="If set, write passing trajectories to this file")
    args = parser.parse_args()
    validate_file(args.input, args.output)

