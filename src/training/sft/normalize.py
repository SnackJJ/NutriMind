import json
import re
from pathlib import Path

def convert_to_sft_format(traj: dict) -> dict:
    messages = traj["messages"]
    normalized_messages = []
    
    # Track if we need to merge consecutive tool outputs (if any)
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            normalized_messages.append({"role": "system", "content": msg["content"]})
        elif role == "user":
            normalized_messages.append({"role": "user", "content": msg["content"]})
        elif role == "assistant":
            # Some teacher models might put think inside content, and args in tool_calls.
            content = msg.get("content", "") or ""
            
            # Format: <think>...</think>
            # <tool_call>{"name": "...", "arguments": {...}}</tool_call>
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    # We assume single sequential tool calls for our agent.
                    name = tc["function"]["name"]
                    args = tc["function"]["arguments"]
                    
                    if not content.strip().endswith("</think>") and not "<think>" in content:
                        # Fallback if model forgot <think> tags but generated some reasoning.
                        if content.strip():
                            content = f"<think>{content.strip()}</think>\n"
                        else:
                            content = f"<think>I should call {name} to get the required information.</think>\n"
                    
                    # Ensure properly formatted <tool_call> JSON string
                    # The args from API is a string, we parse and dump to ensure canonical format, no markdown fences 
                    try:
                        args_dict = json.loads(args)
                        args_str = json.dumps(args_dict, ensure_ascii=False)
                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"[WARNING] Invalid JSON in tool arguments for {name}, using raw string: {e}")
                        args_str = args
                        
                    tool_call_str = f'<tool_call>{{"name": "{name}", "arguments": {args_str}}}</tool_call>'
                    
                    if content:
                        normalized_content = f"{content.strip()}\n{tool_call_str}"
                    else:
                        normalized_content = tool_call_str
                        
                    normalized_messages.append({"role": "assistant", "content": normalized_content})
            else:
                normalized_messages.append({"role": "assistant", "content": content})
                
        elif role == "tool":
            # For SFT, the orchestrator expects tool outputs inside <tool_response> wrapper
            # But the dataset format is {"role": "tool", "content": '{"status": "success", ...}'}
            # The template handles applying im_start|tool, so here we mostly just ensure raw content is correct.
            # But wait, Orcehstrator injects <tool_response>{...}</tool_response> as a "user" role!
            # Let's align with what orchestrator sees, or what SFT framework expects.
            # The PRD v2 mentions: Role "user" with <tool_response>...</tool_response>
            
            # In docs/specs/orchestrator.md:
            # messages.append({"role": "user", "content": format_tool_response(result)})
            
            content = msg.get("content", "")
            # Ensure it's valid JSON without markdown
            content = re.sub(r'```json\s*(.*?)\s*```', r'\1', content, flags=re.DOTALL).strip()
            
            tool_resp_str = f"<tool_response>\n{content}\n</tool_response>"
            normalized_messages.append({"role": "user", "content": tool_resp_str})

    return {
        "tier": traj.get("tier", "unknown"),
        "messages": normalized_messages,
        "metadata": traj.get("metadata", {})
    }

def normalize_file(input_path: str, output_path: str):
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"File not found: {input_path}")
        return
        
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
               
    processed = 0
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip(): continue
            try:
                traj = json.loads(line)
                norm_traj = convert_to_sft_format(traj)
                fout.write(json.dumps(norm_traj, ensure_ascii=False) + "\n")
                processed += 1
            except Exception as e:
                print(f"Error processing line: {e}")
                
    print(f"Normalized {processed} records to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/trajectories/real_trajectories.jsonl")
    parser.add_argument("--output", type=str, default="data/trajectories/normalized_trajectories.jsonl")
    args = parser.parse_args()
    normalize_file(args.input, args.output)
