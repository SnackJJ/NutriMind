# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.9,<4"]
# ///
"""Plot the public experiment snapshot; --refresh rebuilds it from local raw data."""

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/nutrimind-v2"
TAGS = ["base", "sft_b12", "rl_step10", "rl_step17", "rl_step34"]
BLUE, TEAL, ORANGE, GRAY = "#3563A6", "#087F8C", "#D07928", "#64748B"


def collect():
    sources = {}

    def read(relative, jsonl=False):
        path = ROOT / relative
        raw = path.read_bytes()
        sources[relative] = hashlib.sha256(raw).hexdigest()
        return [json.loads(s) for s in raw.splitlines() if s.strip()] if jsonl else json.loads(raw)

    snapshot = {"schema": "nutrimind-report/1", "as_of": "2026-10-04", "exam": {},
                "sources_sha256": sources}
    contract = None
    ids = None
    fields = ["split", "contract", "prompt_fingerprint", "scorer_version", "loop_version",
              "temperature", "extra_body", "parse_error_policy", "context_limit"]
    rates_by_task = {}
    for tag in TAGS:
        runs = [read(f"data/eval/exam/exam8_20261003/{tag}/run{i}.json") for i in range(1, 9)]
        for run in runs:
            current = {k: run[k] for k in fields}
            if contract is None:
                contract = current
            assert current == contract, (tag, "evaluation contract differs")
            current_ids = {t["task_id"] for t in run["tasks"]}
            if ids is None:
                ids = current_ids
            assert current_ids == ids and len(run["tasks"]) == len(ids) == 63, tag
            assert run["total_tasks"] == 63 and run["void_count"] == 0, tag
            assert run["passed_tasks"] == sum(t["passed"] for t in run["tasks"]), tag
        tasks = [t for run in runs for t in run["tasks"]]
        rates = [100 * run["passed_tasks"] / 63 for run in runs]
        per_task = {tid: statistics.mean(t["passed"] for t in tasks if t["task_id"] == tid)
                    for tid in sorted(ids)}
        rates_by_task[tag] = per_task
        snapshot["exam"][tag] = {
            "passes": [run["passed_tasks"] for run in runs], "runs": 8, "tasks": 63,
            "mean_pct": statistics.mean(rates), "sd_pp": statistics.stdev(rates),
            "observed_pass8_pct": 100 * sum(v > 0 for v in per_task.values()) / 63,
            "per_task_pass_rate": per_task,
            "families": {family: {"tasks": sum(t["family"] == family for t in runs[0]["tasks"]),
                                  "mean_pct": 100 * statistics.mean(t["passed"] for t in tasks if t["family"] == family)}
                         for family in sorted({t["family"] for t in tasks})},
            "failure_tags": {tag_: sum(t["score_tag"] == tag_ for t in tasks if not t["passed"])
                             for tag_ in sorted({t["score_tag"] for t in tasks if not t["passed"]})},
            "allergen_flagged_episodes": sum(t["allergen_violated"] for t in tasks),
            "protocol_violations": sum(run["total_protocol_violations"] for run in runs),
            "mean_completion_tokens": statistics.mean(t["total_completion_tokens"] for t in tasks),
            "transfer": {prefix: {"passed": sum(t["passed"] for t in tasks if t["task_id"].startswith(prefix)),
                                   "episodes": sum(t["task_id"].startswith(prefix) for t in tasks)}
                         for prefix in ["adr29-buy", "adr29-dish"]},
        }
    snapshot["evaluation_contract"] = contract
    snapshot["comparisons"] = {}
    for before, after in [("base", "sft_b12"), ("sft_b12", "rl_step34"), ("rl_step17", "rl_step34")]:
        diffs = [rates_by_task[after][tid] - rates_by_task[before][tid] for tid in sorted(ids)]
        rng = random.Random(0)
        draws = sorted(100 * statistics.mean(rng.choices(diffs, k=63)) for _ in range(10000))
        snapshot["comparisons"][f"{after} - {before}"] = {
            "gain_pp": 100 * statistics.mean(diffs), "task_bootstrap_ci95_pp": [draws[250], draws[9750]],
            "resamples": 10000, "seed": 0,
        }
    teacher = [read(f"data/eval/exam/teacher_deepseek-flash_v1.1_high/run{i}.json") for i in range(1, 4)]
    teacher_rates = [100 * r["passed_tasks"] / r["total_tasks"] for r in teacher]
    snapshot["teacher"] = {"passes": [r["passed_tasks"] for r in teacher], "runs": 3, "tasks": 63,
                           "mean_pct": statistics.mean(teacher_rates), "sd_pp": statistics.stdev(teacher_rates),
                           "temperature": teacher[0]["temperature"], "reasoning_effort": teacher[0]["reasoning_effort"]}
    manifest = read("data/student/models/sft_v2_lora_b12/run_manifest.json")
    state = read("data/student/models/sft_v2_lora_b12/trainer_state.json")
    snapshot["sft"] = {"train_raw": manifest["data"]["train"]["n_records"],
                       "train_kept": manifest["data"]["train"]["n_kept"],
                       "loss_val": manifest["data"]["loss_val"]["n_kept"],
                       "train_problem_identities": None, "steps": state["global_step"],
                       "config": manifest["config"], "metrics": manifest["metrics"],
                       "nutrimind_revision": manifest["nutrimind_rev"],
                       "nutrimind_dirty": manifest["nutrimind_dirty"],
                       "nutrienv_revision": manifest["nutrienv_rev_installed"],
                       "log_history": state["log_history"]}
    rl_manifest = read("data/rl/grpo_v2/manifest.json")
    snapshot["rl_dataset"] = {"train_tasks": rl_manifest["artifacts"]["train"]["tasks"],
                              "val_tasks": rl_manifest["artifacts"]["val"]["tasks"],
                              "removed_sft_seen": len(rl_manifest["holdout_audit"]["sft_seen_task_ids"]),
                              "removed_duplicates": len(rl_manifest["holdout_audit"]["duplicate_holdout_task_ids"]),
                              "catalog_sha256": rl_manifest["catalog_sha256"],
                              "adapter_sha256": rl_manifest["adapter_sha256"]}
    snapshot["sft"]["train_problem_identities"] = len(rl_manifest["sft_train_problem_sha256"])
    probe = read("data/rl_probes/sft_b12_pass8/report.json")
    snapshot["probe"] = {k: probe[k] for k in ["k", "completed_episodes", "completed_groups", "mixed_groups",
                                             "pass_at_1_mean", "pass_at_k"]}
    rows = read("data/rl_runs/metrics/nutrimind-v2-grpo/4090x2-async-bs16.jsonl", jsonl=True)
    steps = [row["step"] for row in rows]
    assert steps == list(range(11, 47)), "Saved async segment must cover steps 11-46 without gaps"
    snapshot["rl_async"] = []
    for row in rows:
        d = row["data"]
        item = {"step": row["step"], "raw_pass_rate": d["training/filter_groups/raw/avg@n"],
                "selected_pass_rate": d["train/passrate/avg_passrate"],
                "mixed_group_fraction": d["training/filter_groups/raw/passrate/mid"],
                "entropy": d["actor/entropy"], "grad_norm": d["actor/grad_norm"],
                "step_seconds": d["timing_s/step"]}
        if "val-core/nutrimind_v2/reward/mean@1" in d:
            item["validation_pass_rate"] = d["val-core/nutrimind_v2/reward/mean@1"]
        snapshot["rl_async"].append(item)
    snapshot["rl_preflight"] = read("data/rl_runs/logs/grpo_v2_4090x2_async/preflight.json")
    # The exact launcher captures the A800 -> asynchronous resume, not a new run from SFT.
    launcher = "data/eval/exam/exam8_20261003/launch_after_bench.sh"
    sources[launcher] = hashlib.sha256((ROOT / launcher).read_bytes()).hexdigest()
    snapshot["checkpoint_lineage"] = "A800 step10 -> separate_async resume -> steps17/34; saved metrics end at step46"
    return snapshot


def finish(fig, name):
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    for suffix in ["png", "svg"]:
        path = OUT / f"{name}.{suffix}"
        fig.savefig(path, dpi=180, facecolor="white",
                    metadata={"Date": None} if suffix == "svg" else None)
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip(" \t") for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def plot(snapshot):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 10, "svg.hashsalt": "nutrimind-v2"})
    exam = snapshot["exam"]
    fig, ax = plt.subplots(figsize=(8.6, 4.4), layout="constrained")
    tags = ["base", "sft_b12", "rl_step34"]
    for x, tag, color in zip(range(3), tags, [GRAY, BLUE, TEAL]):
        d = exam[tag]
        ax.bar(x, d["mean_pct"], width=0.55, color=color, alpha=0.2)
        ax.errorbar(x, d["mean_pct"], yerr=d["sd_pp"], fmt="o", color=color, capsize=5, lw=2)
        jitter = [(i - 3.5) * 0.046 for i in range(8)]
        ax.scatter([x + j for j in jitter], [100 * p / 63 for p in d["passes"]], s=26, color=color, alpha=0.65)
        ax.text(x, d["mean_pct"] + d["sd_pp"] + 3, f'{d["mean_pct"]:.1f}% ± {d["sd_pp"]:.1f}', ha="center", color=color)
    ax.set(xticks=range(3), xticklabels=["Original Qwen3.5-2B", "+ SFT", "+ GRPO"],
           ylabel="Pass@1", ylim=(0, 67), title="NutriEnv v1.1 · 63 tasks · 8 decoding repetitions per checkpoint")
    ax.yaxis.set_major_formatter(PercentFormatter())
    fig.supxlabel("Dots: individual runs. Error bars: run SD, not a confidence interval. GRPO: checkpoint 34.", fontsize=9)
    finish(fig, "benchmark_results")

    history = snapshot["sft"]["log_history"]
    train = [r for r in history if "loss" in r]
    val = [r for r in history if "eval_loss" in r]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), layout="constrained")
    for ax, train_key, val_key, label in zip(axes, ["loss", "mean_token_accuracy"],
                                           ["eval_loss", "eval_mean_token_accuracy"], ["Assistant-token loss", "Assistant-token accuracy"]):
        ax.plot([r["step"] for r in train], [r[train_key] for r in train], color=BLUE, lw=1.7, label="Train · logged intervals")
        ax.plot([r["step"] for r in val], [r[val_key] for r in val], "o--", color=ORANGE, lw=1.5, label="Loss validation · 79 traces")
        ax.set(xlabel="SFT optimizer step", ylabel=label)
        for step in [53, 106, 159]:
            ax.axvline(step, color=GRAY, alpha=0.2, lw=1)
        ax.legend(frameon=False, fontsize=8)
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    fig.suptitle("SFT · 418 retained traces · 3 epochs · seed 42")
    fig.supxlabel("Validation at epoch boundaries. Token accuracy is not task Pass. No smoothing applied.", fontsize=9)
    finish(fig, "sft_learning_curves")

    rows = snapshot["rl_async"]
    steps = [r["step"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    ax = axes[0, 0]
    ax.plot(steps, [r["raw_pass_rate"] for r in rows], color=TEAL, lw=1.5, label="Before group filtering")
    ax.plot(steps, [r["selected_pass_rate"] for r in rows], color=BLUE, lw=1.3, alpha=0.75, label="Selected training groups")
    ax.set(ylabel="Training rollout Pass", ylim=(0, 1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.legend(frameon=False, fontsize=8)
    for ax, key, label, color in [(axes[0, 1], "mixed_group_fraction", "Raw groups with both Pass and Fail", TEAL),
                                   (axes[1, 0], "entropy", "Actor entropy · logged metric", ORANGE),
                                   (axes[1, 1], "grad_norm", "Actor gradient norm", BLUE)]:
        ax.plot(steps, [r[key] for r in rows], color=color, lw=1.5)
        ax.set(ylabel=label)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    for ax in axes.flat:
        ax.set_xlabel("Cumulative RL optimizer step")
        for step in [17, 34]:
            ax.axvline(step, color=GRAY, alpha=0.2, lw=1)
    fig.suptitle("GRPO async segment · recorded steps 11–46 · resumed from A800 step 10")
    fig.supxlabel("Raw per-step metrics; no smoothing. Training-pool Pass is not benchmark performance.", fontsize=9)
    finish(fig, "grpo_training_curves")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    positions = [10, 17, 34]
    checkpoint_tags = ["rl_step10", "rl_step17", "rl_step34"]
    axes[0].errorbar(positions, [exam[t]["mean_pct"] for t in checkpoint_tags],
                     yerr=[exam[t]["sd_pp"] for t in checkpoint_tags], fmt="o--", color=TEAL, capsize=5)
    axes[0].axhline(exam["sft_b12"]["mean_pct"], color=BLUE, ls=":", label="SFT mean")
    axes[0].set(title="Official benchmark · 63 tasks · 8 runs", ylim=(0, 65), ylabel="Pass@1 · mean ± run SD")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].axvline(10.5, color=GRAY, ls=":", alpha=0.65)
    axes[0].text(11.5, 4, "Async resume + IS correction", fontsize=8, color=GRAY)
    validation = [r for r in rows if "validation_pass_rate" in r]
    axes[1].plot([r["step"] for r in validation], [100 * r["validation_pass_rate"] for r in validation], "o--", color=ORANGE)
    for r in validation:
        rate = r["validation_pass_rate"]
        axes[1].annotate(f"{round(rate * 104)}/104", (r["step"], 100 * rate), xytext=(0, 9), textcoords="offset points", ha="center")
    axes[1].set(title="Internal validation · 104 tasks · 1 rollout", ylim=(0, 100), ylabel="Pass · veRL validation settings")
    for ax in axes:
        ax.set(xlabel="Cumulative RL optimizer step", xticks=positions, xlim=(7, 37))
        ax.yaxis.set_major_formatter(PercentFormatter())
    fig.supxlabel("The two panels use different tasks and rollout settings. Dashed lines connect measured checkpoints only.", fontsize=9)
    finish(fig, "checkpoint_evaluation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Rebuild the summary from gitignored local raw artifacts")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "metrics.json"
    if args.refresh:
        snapshot = collect()
        path.write_text(json.dumps(snapshot, indent=2) + "\n")
    else:
        snapshot = json.loads(path.read_text())
    plot(snapshot)
    print(f"Generated 4 PNG figures and 4 SVG figures in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
