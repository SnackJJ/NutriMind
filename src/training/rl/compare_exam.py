"""Paired comparison of two ``eval_exam`` output dirs (e.g. SFT vs baseline).

``python -m src.training.rl.compare_exam --baseline DIR --candidate DIR``

Refuses unless both dirs ran the same exam revision (exam blob, lab HEAD,
loop version) on the same task set. Per task, the pass rate over that dir's
k runs; the headline is the mean candidate − baseline difference with a paired
bootstrap 95% CI over tasks. Exact McNemar on per-task majority pass, and on
run i vs run i. Indeterminate counts as not-pass (as in the eval report).
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import defaultdict
from collections.abc import Sequence

from src.training.rl.eval_exam import (
    MANIFEST,
    bootstrap_mean_ci,
    load_rows,
    per_task_matrix,
)

__all__ = ["CompareRefused", "compare_dirs", "mcnemar_exact", "main"]

_SAME_EXAM = ("exam_blob", "lab_head", "loop_version", "task_ids_sha256")


class CompareRefused(RuntimeError):
    """The two evals are not on the same exam revision / task set."""


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p: binomial(b + c, 0.5) on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def _load(out_dir: pathlib.Path) -> tuple[dict, dict[str, list[dict]]]:
    manifest = json.loads((out_dir / MANIFEST).read_text(encoding="utf-8"))
    return manifest, per_task_matrix(manifest, load_rows(out_dir))


def _passes(rows: Sequence[dict]) -> list[bool]:
    return [row["status"] == "pass" for row in rows]


def compare_dirs(baseline_dir: pathlib.Path, candidate_dir: pathlib.Path) -> dict:
    base_m, base = _load(pathlib.Path(baseline_dir))
    cand_m, cand = _load(pathlib.Path(candidate_dir))
    diff = [k for k in _SAME_EXAM if base_m["exam"].get(k) != cand_m["exam"].get(k)]
    if diff:
        raise CompareRefused(
            "exam revision differs on "
            + ", ".join(f"{k} ({base_m['exam'].get(k)} vs {cand_m['exam'].get(k)})" for k in diff)
        )
    if sorted(base_m["task_ids"]) != sorted(cand_m["task_ids"]):
        raise CompareRefused("task sets differ")

    task_ids = base_m["task_ids"]
    families = base_m["families"]
    kb, kc = base_m["runs"], cand_m["runs"]
    rate_b = {t: sum(_passes(base[t])) / kb for t in task_ids}
    rate_c = {t: sum(_passes(cand[t])) / kc for t in task_ids}
    deltas = [rate_c[t] - rate_b[t] for t in task_ids]

    maj_b = {t: 2 * sum(_passes(base[t])) > kb for t in task_ids}
    maj_c = {t: 2 * sum(_passes(cand[t])) > kc for t in task_ids}
    gained = sorted(t for t in task_ids if maj_c[t] and not maj_b[t])
    lost = sorted(t for t in task_ids if maj_b[t] and not maj_c[t])

    per_run = []
    for r in range(min(kb, kc)):
        b = sum(1 for t in task_ids if base[t][r]["status"] == "pass" and cand[t][r]["status"] != "pass")
        c = sum(1 for t in task_ids if base[t][r]["status"] != "pass" and cand[t][r]["status"] == "pass")
        per_run.append({"run": r + 1, "baseline_only": b, "candidate_only": c, "p_exact": mcnemar_exact(b, c)})

    by_family: dict[str, list[str]] = defaultdict(list)
    for t in task_ids:
        by_family[families[t]].append(t)
    family_rows = {
        fam: {
            "n_tasks": len(ts),
            "baseline": sum(rate_b[t] for t in ts) / len(ts),
            "candidate": sum(rate_c[t] for t in ts) / len(ts),
            "delta": sum(rate_c[t] - rate_b[t] for t in ts) / len(ts),
        }
        for fam, ts in sorted(by_family.items())
    }

    return {
        "exam": {k: base_m["exam"].get(k) for k in ("exam_path", *_SAME_EXAM)},
        "baseline": {"dir": str(baseline_dir), "model": base_m["model"], "adapter": base_m.get("adapter"), "k": kb},
        "candidate": {"dir": str(candidate_dir), "model": cand_m["model"], "adapter": cand_m.get("adapter"), "k": kc},
        "n_tasks": len(task_ids),
        "pass_rate": {
            "baseline": sum(rate_b.values()) / len(task_ids),
            "candidate": sum(rate_c.values()) / len(task_ids),
            "mean_delta": sum(deltas) / len(deltas),
            "delta_ci95_paired_bootstrap": list(bootstrap_mean_ci(deltas)),
        },
        "mcnemar_majority": {
            "baseline_only": len(lost),
            "candidate_only": len(gained),
            "p_exact": mcnemar_exact(len(lost), len(gained)),
        },
        "mcnemar_per_run": per_run,
        "flipped_to_pass": gained,
        "flipped_to_fail": lost,
        "by_family": family_rows,
        "per_task": {t: {"baseline": rate_b[t], "candidate": rate_c[t]} for t in task_ids},
    }


def render_markdown(result: dict) -> str:
    pr, mm = result["pass_rate"], result["mcnemar_majority"]
    lo, hi = pr["delta_ci95_paired_bootstrap"]
    pct = lambda v: f"{100 * v:.1f}%"  # noqa: E731
    lines = [
        f"# Exam compare — {result['candidate']['model']} vs {result['baseline']['model']}",
        "",
        f"- exam `{result['exam']['exam_path']}` blob `{result['exam']['exam_blob']}`, lab `{result['exam']['lab_head']}`",
        f"- {result['n_tasks']} tasks; k baseline {result['baseline']['k']}, candidate {result['candidate']['k']}",
        "",
        "| metric | value |",
        "|---|---|",
        f"| mean per-task pass rate: baseline / candidate | {pct(pr['baseline'])} / {pct(pr['candidate'])} |",
        f"| mean delta (95% paired bootstrap CI) | {pct(pr['mean_delta'])} ({pct(lo)} to {pct(hi)}) |",
        f"| McNemar majority-pass: baseline-only / candidate-only, exact p | {mm['baseline_only']} / {mm['candidate_only']}, p={mm['p_exact']:.4g} |",
        *[
            f"| McNemar run {r['run']}: baseline-only / candidate-only, exact p | {r['baseline_only']} / {r['candidate_only']}, p={r['p_exact']:.4g} |"
            for r in result["mcnemar_per_run"]
        ],
        "",
        "## By family",
        "",
        "| family | n | baseline | candidate | delta |",
        "|---|---|---|---|---|",
        *[
            f"| {fam} | {row['n_tasks']} | {pct(row['baseline'])} | {pct(row['candidate'])} | {pct(row['delta'])} |"
            for fam, row in result["by_family"].items()
        ],
        "",
        f"Flipped to pass ({len(result['flipped_to_pass'])}): {', '.join(result['flipped_to_pass']) or '—'}",
        "",
        f"Flipped to fail ({len(result['flipped_to_fail'])}): {', '.join(result['flipped_to_fail']) or '—'}",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--out-dir", default=None, help="default: <candidate>/compare_vs_<baseline name>")
    args = p.parse_args(argv)
    baseline, candidate = pathlib.Path(args.baseline), pathlib.Path(args.candidate)
    try:
        result = compare_dirs(baseline, candidate)
    except CompareRefused as exc:
        raise SystemExit(f"compare refused: {exc}") from None
    out = pathlib.Path(args.out_dir) if args.out_dir else candidate / f"compare_vs_{baseline.resolve().name}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (out / "compare.md").write_text(render_markdown(result), encoding="utf-8")
    print(f"[compare_exam] mean delta {result['pass_rate']['mean_delta']:+.3f} "
          f"CI95 {result['pass_rate']['delta_ci95_paired_bootstrap']}; "
          f"McNemar p={result['mcnemar_majority']['p_exact']:.4g}; {out / 'compare.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
