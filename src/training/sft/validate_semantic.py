"""Semantic validation for SFT trajectories using LLM judge.

Features:
- Resume support: skips already-validated trajectories
- Parallel validation: configurable workers
- Failed records: saves rejected trajectories with reasons

Usage:
    # Validate with 5 workers (default)
    python -m src.training.sft.validate_semantic

    # Resume interrupted validation
    python -m src.training.sft.validate_semantic  # automatically resumes

    # Custom workers and paths
    python -m src.training.sft.validate_semantic --workers 3 --input custom.jsonl
"""

import json
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config import settings

# Use Gemini as judge model (faster and cheaper than qwen-max)
client = OpenAI(
    api_key=settings.gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
JUDGE_MODEL = "gemini-2.5-flash"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _call_judge_api(prompt: str) -> str:
    res = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return res.choices[0].message.content.strip()


SEMANTIC_PROMPT_T1_T2 = """
Is this trajectory suitable for training a nutrition assistant? Check:

1. Tool usage pattern: Does the agent use tools appropriately (not hallucinating data, using real tool results)?
2. Final answer: Is it helpful and based on the tool results (not making up numbers)?
3. Format: Is the conversation flow reasonable (user question → tool call → answer)?

PASS if: The trajectory demonstrates good assistant behavior that a model should learn.
FAIL only if: The agent hallucinates data, ignores tool results, or the answer is completely wrong/harmful.

Respond PASS or FAIL with brief reason.
Trajectory: {traj}
"""

SEMANTIC_PROMPT_T3_T4 = """
Is this trajectory suitable for training? This is a complex/safety case (T3/T4).

1. For T4 (safety boundary - dialysis, cancer, surgery, drug interactions): Does the agent REFUSE to give specific advice and recommend professional help? No tools should be called.
2. For T3 (complex reasoning): Does the agent reach a reasonable conclusion through multi-step tool usage?
3. Is the overall behavior something we want the model to learn?

PASS if: T4 correctly refuses OR T3 shows reasonable multi-step problem solving.
FAIL only if: T4 gives dangerous medical advice, or T3 is completely incoherent.

Respond PASS or FAIL with brief reason.
Trajectory: {traj}
"""

SEMANTIC_PROMPT_QA = """
Is this no-tool response suitable for training? (T0-qa: general knowledge questions)

1. Is it reasonable that no tool was needed? (General nutrition concepts don't require database lookup)
2. Is the response helpful and not obviously wrong?

PASS if: The response is reasonable general nutrition guidance.
FAIL only if: The response contains dangerous misinformation or should have used tools but didn't.

Respond PASS or FAIL with brief reason.
Trajectory: {traj}
"""


def get_traj_hash(traj: dict) -> str:
    """Generate a stable hash for a trajectory based on messages content."""
    # Use user message as the key (first non-system message)
    for msg in traj.get("messages", []):
        if msg.get("role") == "user":
            return hashlib.md5(msg.get("content", "").encode()).hexdigest()[:16]
    # Fallback to full messages hash
    return hashlib.md5(json.dumps(traj.get("messages", []), sort_keys=True).encode()).hexdigest()[:16]


def judge_trajectory(traj: dict) -> tuple[bool, str]:
    """Judge a single trajectory using LLM."""
    tier = traj.get("tier", "T1")
    if tier in ["T1", "T2"]:
        prompt = SEMANTIC_PROMPT_T1_T2.format(traj=json.dumps(traj, ensure_ascii=False))
    elif tier in ["T3", "T4", "error_recovery"]:
        prompt = SEMANTIC_PROMPT_T3_T4.format(traj=json.dumps(traj, ensure_ascii=False))
    else:
        prompt = SEMANTIC_PROMPT_QA.format(traj=json.dumps(traj, ensure_ascii=False))

    try:
        content = _call_judge_api(prompt)
        is_pass = "PASS" in content[:20].upper()
        return is_pass, content
    except Exception as e:
        return False, f"API_ERROR: {type(e).__name__}: {e}"


def validate_semantic_file(
    input_path: str,
    output_path: str,
    workers: int = 5,
    requests_per_minute: int = 60,
):
    """Validate trajectories with resume support and parallel processing."""
    input_file = Path(input_path)
    output_file = Path(output_path)
    failed_file = output_file.parent / f"{output_file.stem}_failed.jsonl"

    if not input_file.exists():
        print(f"[ERROR] Input file not found: {input_file}")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load already-validated trajectory hashes for resume support
    done_hashes: set[str] = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        traj = json.loads(line)
                        done_hashes.add(get_traj_hash(traj))
                    except json.JSONDecodeError:
                        pass

    # Also load failed hashes to avoid re-validating
    if failed_file.exists():
        with open(failed_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        traj = json.loads(line)
                        done_hashes.add(get_traj_hash(traj))
                    except json.JSONDecodeError:
                        pass

    if done_hashes:
        print(f"[INFO] Skipping {len(done_hashes)} already-validated trajectories (resume mode)")

    # Load pending trajectories
    pending: list[dict] = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traj = json.loads(line)
                if get_traj_hash(traj) not in done_hashes:
                    pending.append(traj)

    if not pending:
        print("All trajectories already validated, nothing to do.")
        return

    print(f"[INFO] Validating {len(pending)} trajectories with {workers} workers")

    # Thread-safe file writing
    write_lock = threading.Lock()
    stats = {"passed": 0, "failed": 0}

    def validate_one(traj: dict) -> tuple[dict, bool, str]:
        is_pass, reason = judge_trajectory(traj)
        traj["semantic_metadata"] = {
            "judge_model": JUDGE_MODEL,
            "passed": is_pass,
            "reason": reason,
        }
        return traj, is_pass, reason

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(validate_one, traj): traj for traj in pending}

        for future in as_completed(futures):
            try:
                traj, is_pass, reason = future.result()
                traj_hash = get_traj_hash(traj)

                with write_lock:
                    if is_pass:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(traj, ensure_ascii=False) + "\n")
                        stats["passed"] += 1
                        tier = traj.get("tier", "?")
                        print(f"[PASS {stats['passed']}/{len(pending)}] {tier} - {traj_hash}")
                    else:
                        with open(failed_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(traj, ensure_ascii=False) + "\n")
                        stats["failed"] += 1
                        tier = traj.get("tier", "?")
                        reason_short = reason[:100].replace("\n", " ")
                        print(f"[FAIL {stats['failed']}] {tier} - {reason_short}...")

            except Exception as e:
                stats["failed"] += 1
                print(f"[ERROR] Validation exception: {e}")

    total = stats["passed"] + stats["failed"]
    pass_rate = stats["passed"] / total * 100 if total > 0 else 0
    print(f"\n{'='*50}")
    print(f"Semantic Validation Complete:")
    print(f"  Passed: {stats['passed']}/{total} ({pass_rate:.1f}%)")
    print(f"  Failed: {stats['failed']}")
    print(f"  Output: {output_file}")
    if stats["failed"] > 0:
        print(f"  Failed records: {failed_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Semantic validation for SFT trajectories")
    parser.add_argument(
        "--input",
        type=str,
        default="data/trajectories/rule_validated_trajectories.jsonl",
        help="Input trajectory file (JSONL)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/trajectories/validated_trajectories.jsonl",
        help="Output file for passed trajectories (JSONL)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel validation workers",
    )
    args = parser.parse_args()

    validate_semantic_file(args.input, args.output, workers=args.workers)
